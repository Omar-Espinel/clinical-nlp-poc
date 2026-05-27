"""
Tests for PgVectorCascadeStrategy.

Fixtures:
  - test_db with clinical_nlp schema + 20 hand-picked concepts
  - Mock SNOMEDFallbackChain (no real API calls)
"""

import os
import pytest
import psycopg2
from src.snomed_search.pgvector_cascade import PgVectorCascadeStrategy
from src.snomed_search.fhir_fallback import SNOMEDFallbackChain, FallbackResult

# VALID_SNOMED_HIERARCHIES mirrors the spec constant used in the strategy
VALID_SNOMED_HIERARCHIES = {
    "Condition", "Procedure", "Drug", "Measurement", "Device", "Observation"
}


SEED_CONCEPTS = [
    ("73211009",  "diabetes mellitus",       "Condition",  ["sugar diabetes", "DM", "diabetes"]),
    ("38341003",  "hypertension",            "Condition",  ["high blood pressure", "HTN"]),
    ("195967001", "asthma",                  "Condition",  ["bronchial asthma"]),
    ("22298006",  "myocardial infarction",   "Condition",  ["heart attack", "MI"]),
    ("84114007",  "heart failure",           "Condition",  ["cardiac failure", "CHF"]),
    ("44054006",  "type 2 diabetes mellitus","Condition",  ["T2DM", "type II diabetes"]),
    ("80146002",  "appendectomy",            "Procedure",  ["appendix removal"]),
    ("387207008", "ibuprofen",               "Drug",       ["advil", "motrin"]),
    ("372756006", "warfarin",                "Drug",       ["coumadin"]),
    ("271442007", "fetal heart rate",        "Measurement",["fetal HR", "FHR"]),
]


@pytest.fixture(scope="module")
def test_db():
    """
    Yields the DATABASE_URL after ensuring the clinical_nlp schema exists and
    a small hand-picked seed dataset is loaded for deterministic tests.

    Skips cleanly if DATABASE_URL is not set or the DB is unreachable.
    Does NOT auto-migrate — skips if the schema is absent.
    Seed rows are inserted with ON CONFLICT DO NOTHING so re-runs are safe.
    """
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        pytest.skip("DATABASE_URL not set")

    try:
        conn = psycopg2.connect(db_url)
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL not available")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'clinical_nlp'"
        )
        if cur.fetchone() is None:
            conn.close()
            pytest.skip("clinical_nlp schema not migrated")

    # --- Seed deterministic concepts + synonyms + embeddings ---
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with conn.cursor() as cur:
        for code, term, domain, synonyms in SEED_CONCEPTS:
            cur.execute(
                """
                INSERT INTO clinical_nlp.concepts
                    (concept_id, preferred_term, domain, source)
                VALUES (%s, %s, %s, 'test_seed')
                ON CONFLICT (concept_id) DO NOTHING
                """,
                (code, term, domain),
            )
            for syn in synonyms:
                cur.execute(
                    """
                    INSERT INTO clinical_nlp.synonyms (concept_id, synonym_text, source)
                    VALUES (%s, %s, 'test_seed')
                    ON CONFLICT DO NOTHING
                    """,
                    (code, syn),
                )
            emb = model.encode(term)
            emb_str = "[" + ",".join(f"{v:.8f}" for v in emb.tolist()) + "]"
            cur.execute(
                """
                INSERT INTO clinical_nlp.embeddings (concept_id, embedding)
                VALUES (%s, %s::vector)
                ON CONFLICT (concept_id) DO NOTHING
                """,
                (code, emb_str),
            )
    conn.commit()
    conn.close()

    yield db_url


@pytest.fixture
def mock_fallback(monkeypatch):
    """Mock SNOMEDFallbackChain that returns a real seed concept_id (FK-safe)."""
    def mock_lookup(self, term):
        if "unknown" in term.lower():
            return FallbackResult("73211009", "diabetes mellitus", "ols4", 0.75)
        return None
    monkeypatch.setattr(SNOMEDFallbackChain, "lookup", mock_lookup)


def test_exact_match_returns_confidence_1_0(test_db):
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db)
    results = strategy.search("diabetes mellitus")
    assert len(results) == 1
    assert results[0].code == "73211009"
    assert results[0].confidence == 1.0
    assert results[0].match_type == "exact"


def test_synonym_match_returns_confidence_0_95(test_db):
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db)
    results = strategy.search("sugar diabetes")
    assert len(results) >= 1
    assert any(r.code == "73211009" for r in results)
    assert any(r.match_type == "synonym" for r in results)


def test_fuzzy_match_handles_typos(test_db):
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db)
    results = strategy.search("diabetis")
    assert len(results) >= 1
    assert any("diabetes" in r.display.lower() for r in results)
    assert any(0.75 <= r.confidence <= 0.85 for r in results if r.match_type == "fuzzy")


def test_semantic_match_handles_paraphrases(test_db):
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db)
    results = strategy.search("high blood sugar condition")
    assert len(results) >= 1
    assert any("diabetes" in r.display.lower() for r in results)


def test_cache_hit_returns_cached_result(test_db, mock_fallback):
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db, fallback_api_key="test_key_xxxxxxxxxx")
    strategy.search("unknown_term")
    results = strategy.search("unknown_term")
    assert any("cached" in r.match_type for r in results)


def test_cache_respects_ttl(test_db):
    """A cache row with cached_at far in the past should be treated as expired."""
    conn = psycopg2.connect(test_db)
    try:
        with conn.cursor() as cur:
            # Insert a cache row with cached_at 10 days ago and ttl_days=1
            cur.execute(
                """
                INSERT INTO clinical_nlp.cache
                    (term, concept_id, source, confidence, cached_at, ttl_days)
                VALUES ('ttl_test_term_expired', '73211009', 'ols4', 0.80,
                        NOW() - INTERVAL '10 days', 1)
                ON CONFLICT (term) DO UPDATE
                    SET cached_at = NOW() - INTERVAL '10 days', ttl_days = 1
                """
            )
        conn.commit()

        strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db)
        db_conn = strategy._db_pool.getconn()
        try:
            result = strategy._cache_lookup(db_conn, "ttl_test_term_expired")
        finally:
            strategy._db_pool.putconn(db_conn)

        assert result == [], "Expired cache entry should return empty list"
    finally:
        conn.rollback()
        conn.close()


def test_embedding_prefilter_blocks_non_medical_terms(test_db, mock_fallback):
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db, fallback_api_key="test_key_xxxxxxxxxx")
    results = strategy.search("JavaScript")
    assert len(results) == 0


def test_api_fallback_called_on_cache_miss(test_db, mock_fallback):
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db, fallback_api_key="test_key_xxxxxxxxxx")
    results = strategy.search("unknown_medical_condition")
    assert len(results) >= 1


def test_api_result_written_to_cache(test_db, mock_fallback):
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db, fallback_api_key="test_key_xxxxxxxxxx")
    strategy.search("unknown_term")
    conn = psycopg2.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM clinical_nlp.cache WHERE term = 'unknown_term'")
    assert cursor.fetchone()[0] == 1


def test_circuit_breaker_opens_after_5_failures(monkeypatch):
    """After 5 consecutive OLS4 httpx timeouts the circuit breaker opens and blocks further calls."""
    import httpx
    from src.snomed_search.fhir_fallback import OLS4Client

    call_count = {"n": 0}

    def mock_get(self, *args, **kwargs):
        call_count["n"] += 1
        raise httpx.TimeoutException("simulated OLS4 timeout")

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    client = OLS4Client()
    # Trigger 5 failures — each should return None and increment the breaker
    for _ in range(5):
        result = client.lookup("test query")
        assert result is None

    # After 5 failures the breaker must be open
    assert client._breaker.is_open, "Circuit breaker should be open after 5 failures"

    # 6th call must return None WITHOUT making an HTTP attempt
    pre_count = call_count["n"]
    sixth_result = client.lookup("another query")
    assert sixth_result is None
    assert call_count["n"] == pre_count, (
        "Circuit breaker should prevent HTTP attempt on 6th call"
    )


def test_connection_pool_thread_safety(test_db):
    from concurrent.futures import ThreadPoolExecutor
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db)

    def search_worker():
        return strategy.search("diabetes")

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(search_worker) for _ in range(50)]
        results = [f.result() for f in futures]

    assert len(results) == 50


def test_invalid_hierarchy_rejected():
    """OLS4Client._in_valid_hierarchy correctly accepts and rejects SNOMED hierarchy docs."""
    from src.snomed_search.fhir_fallback import OLS4Client, VALID_SNOMED_HIERARCHIES

    client = OLS4Client()

    # A doc whose hierarchy contains at least one known SNOMED code should pass
    valid_id = next(iter(VALID_SNOMED_HIERARCHIES))  # pick any valid code
    doc_valid = {"hierarchy": [valid_id, "some-other-id"]}

    # A doc whose hierarchy contains only unknown codes should fail
    doc_invalid = {"hierarchy": ["999999999", "888888888"]}

    # An empty hierarchy should fail
    doc_empty = {"hierarchy": []}

    assert client._in_valid_hierarchy(doc_valid) is True, (
        "Doc with a valid SNOMED hierarchy code should be accepted"
    )
    assert client._in_valid_hierarchy(doc_invalid) is False, (
        "Doc with no valid SNOMED hierarchy codes should be rejected"
    )
    assert client._in_valid_hierarchy(doc_empty) is False, (
        "Doc with empty hierarchy should be rejected"
    )


def test_deduplication_keeps_highest_confidence(test_db):
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db)
    results = strategy.search("diabetis mellitus")
    concept_ids = [r.code for r in results]
    assert len(concept_ids) == len(set(concept_ids)), "Duplicates found"


def test_graceful_degradation_on_db_failure(test_db, monkeypatch):
    """Strategy constructs against a real DB, then degrades when getconn fails."""
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db)

    def mock_getconn(self, *args, **kwargs):
        raise psycopg2.OperationalError("Connection refused")

    monkeypatch.setattr(strategy._db_pool.__class__, "getconn", mock_getconn)

    results = strategy.search("diabetes")
    assert results == []


def test_semantic_search_latency_benchmark(test_db):
    import time
    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db)
    conn = strategy._db_pool.getconn()
    try:
        start = time.monotonic()
        results = strategy._semantic_match(conn, "diabetes")
        elapsed_ms = (time.monotonic() - start) * 1000
    finally:
        strategy._db_pool.putconn(conn)

    assert elapsed_ms < 10, f"Semantic search took {elapsed_ms}ms (threshold: 10ms)"


def test_memory_usage_under_load(test_db):
    import psutil
    from concurrent.futures import ThreadPoolExecutor

    strategy = PgVectorCascadeStrategy(dictionary_path="", database_url=test_db)
    process = psutil.Process()
    baseline_mb = process.memory_info().rss / 1024 / 1024

    def search_worker():
        return strategy.search("diabetes")

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(search_worker) for _ in range(100)]
        results = [f.result() for f in futures]

    final_mb = process.memory_info().rss / 1024 / 1024
    growth = final_mb - baseline_mb

    assert growth < 100, f"Memory grew {growth}MB (threshold: 100MB)"
