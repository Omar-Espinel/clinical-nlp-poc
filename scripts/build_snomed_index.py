"""
build_snomed_index.py — Populate clinical_nlp PostgreSQL schema from OLS4 (EMBL-EBI).

OLS4 is the sole ingest source. Concepts are fetched via the OLS4 REST API across
5 SNOMED query terms, paginated, deduplicated, and inserted into clinical_nlp.concepts
and clinical_nlp.synonyms. Embeddings are generated with all-MiniLM-L6-v2.

Usage:
    python scripts/build_snomed_index.py            # full build
    python scripts/build_snomed_index.py --test     # 500 concepts (100 per term)
    python scripts/build_snomed_index.py --embeddings-only   # skip ingest; generate missing embeddings only
    python scripts/build_snomed_index.py --force    # skip confirmation prompt
    python scripts/build_snomed_index.py --resume   # resume interrupted build
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

# ---------------------------------------------------------------------------
# Logging — HIPAA: never log concept_name, preferred_term, or user input
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional tqdm — fall back to plain iteration if not installed
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)

except ImportError:  # pragma: no cover
    log.info("tqdm not installed; using plain iteration with periodic print")

    def _progress(iterable, total=None, desc=""):  # type: ignore[misc]
        for i, item in enumerate(iterable):
            if i % 500 == 0:
                log.info("%s — processed %d / %s", desc, i, total or "?")
            yield item


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

OLS4_BASE_URL = "https://www.ebi.ac.uk/ols4/api/search"

OLS4_QUERY_TERMS = [
    {"term": "clinical finding", "domain": "Condition", "expected": 50000},
    {"term": "procedure", "domain": "Procedure", "expected": 30000},
    {"term": "pharmaceutical product", "domain": "Drug", "expected": 25000},
    {"term": "measurement", "domain": "Measurement", "expected": 15000},
    {"term": "device", "domain": "Device", "expected": 8000},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_url(url: str) -> str:
    """Mask password in a database URL before logging."""
    return re.sub(r":[^@/]+@", ":***@", url)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clinical_nlp SNOMED index from Athena + OLS4."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Load only 100 concepts (fast dev/CI mode).",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        dest="embeddings_only",
        help="Skip OLS4 ingest; only generate missing embeddings for existing concepts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt when schema already has data.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted build using progress stored in clinical_nlp.metadata.",
    )
    return parser.parse_args()


def _validate_env(args: argparse.Namespace) -> str:
    """Return DATABASE_URL or exit 1."""
    load_dotenv(REPO_ROOT / ".env")
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        log.error("DATABASE_URL is not set. Add it to .env or export it.")
        sys.exit(1)
    log.info("DATABASE_URL: %s", _mask_url(db_url))

    return db_url


def _connect(db_url: str) -> psycopg2.extensions.connection:
    """Connect to PostgreSQL, verify migration, register pgvector."""
    try:
        conn = psycopg2.connect(db_url)
    except Exception as exc:
        log.error(
            "Cannot connect to database (%s: %s). URL: %s",
            type(exc).__name__,
            exc,
            _mask_url(db_url),
        )
        sys.exit(1)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'clinical_nlp'"
        )
        if cur.fetchone() is None:
            log.error(
                "Schema 'clinical_nlp' not found. "
                "Run: psql $DATABASE_URL -f db/migrations/001_create_snomed_schema.sql"
            )
            conn.close()
            sys.exit(1)

    register_vector(conn)
    log.info("Connected to PostgreSQL; pgvector registered.")
    return conn


def _idempotency_check(conn: psycopg2.extensions.connection, args: argparse.Namespace) -> None:
    """If table has data and not --force/--test, prompt; TRUNCATE on confirm."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM clinical_nlp.concepts")
        row_count = cur.fetchone()[0]

    if row_count == 0:
        return

    if not args.test and not args.force:
        try:
            answer = input(f"Schema has data ({row_count:,} rows). Drop and rebuild? (y/n): ")
        except EOFError:
            answer = "n"
        if answer.strip().lower() != "y":
            log.info("Aborted by user.")
            sys.exit(0)

    log.info("Truncating clinical_nlp.concepts CASCADE …")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE clinical_nlp.concepts CASCADE")
    conn.commit()


def _insert_concepts(conn: psycopg2.extensions.connection, rows: list) -> None:
    """Batch-insert concepts using execute_values with ON CONFLICT DO NOTHING."""
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO clinical_nlp.concepts
                (concept_id, preferred_term, domain, source, hierarchy)
            VALUES %s
            ON CONFLICT (concept_id) DO NOTHING
            """,
            [
                (r["concept_id"], r["preferred_term"], r["domain"], r["source"], r["hierarchy"])
                for r in rows
            ],
            template="(%s, %s, %s, %s, %s::text[])",
            page_size=1000,
        )
    conn.commit()


def _insert_synonyms(conn: psycopg2.extensions.connection, rows: list) -> None:
    """
    Derive synonyms from preferred_term:
      - the term itself
      - lowercase variant (deduplicated)
    Both inserted with source='derived'.
    """
    syn_rows = []
    for r in rows:
        term = r["preferred_term"]  # already lowercased
        cid = r["concept_id"]
        seen = set()
        for variant in [term, term.lower()]:
            if variant not in seen:
                seen.add(variant)
                syn_rows.append((cid, variant, "derived"))

    if not syn_rows:
        return

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO clinical_nlp.synonyms (concept_id, synonym_text, source)
            VALUES %s
            """,
            syn_rows,
            page_size=1000,
        )
    conn.commit()


def _insert_ols4_synonyms(conn: psycopg2.extensions.connection, rows: list) -> None:
    """
    Batch-insert OLS4-provided synonyms from the '_synonyms' field on each row.
    Deduplicates per concept (case-insensitive, stripped). Source = 'ols4'.
    Uses execute_batch with ON CONFLICT DO NOTHING.
    """
    syn_rows = []
    for r in rows:
        ols4_synonyms = r.get("_synonyms", [])
        if not ols4_synonyms:
            continue
        cid = r["concept_id"]
        seen: set = set()
        for raw in ols4_synonyms:
            normalised = raw.strip().lower()
            if normalised and normalised not in seen:
                seen.add(normalised)
                syn_rows.append((cid, normalised, "ols4"))

    if not syn_rows:
        return

    # Process in capped batches of 1000
    BATCH_SIZE = 1000
    with conn.cursor() as cur:
        for start in range(0, len(syn_rows), BATCH_SIZE):
            batch = syn_rows[start: start + BATCH_SIZE]
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO clinical_nlp.synonyms (concept_id, synonym_text, source)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                batch,
                page_size=BATCH_SIZE,
            )
    conn.commit()


def _load_build_progress(conn: psycopg2.extensions.connection) -> dict:
    """Load build_progress from clinical_nlp.metadata, or return empty dict."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT value FROM clinical_nlp.metadata WHERE key = 'build_progress'"
            )
            row = cur.fetchone()
        if row:
            return row[0] if isinstance(row[0], dict) else json.loads(row[0])
    except Exception as exc:
        log.warning("Could not read build_progress (%s); starting fresh.", type(exc).__name__)
    return {}


def _save_build_progress(
    conn: psycopg2.extensions.connection,
    progress: dict,
) -> None:
    """Upsert build_progress JSON into clinical_nlp.metadata."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO clinical_nlp.metadata (key, value)
                VALUES ('build_progress', %s::jsonb)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (json.dumps(progress),),
            )
        conn.commit()
    except Exception as exc:
        log.warning("Could not save build_progress (%s).", type(exc).__name__)
        conn.rollback()


def _fetch_ols4(
    conn: psycopg2.extensions.connection,
    args: argparse.Namespace,
) -> int:
    """
    Fetch OLS4 concepts for all 5 query terms with pagination.
    Supports --resume to skip completed terms and continue from last offset.
    Returns total count of new concepts inserted.
    """
    log.info("Fetching OLS4 concepts for %d query terms …", len(OLS4_QUERY_TERMS))

    rows_per_page = 100
    total_inserted = 0

    # Load resume state if requested
    progress: dict = {}
    if args.resume:
        progress = _load_build_progress(conn)
        log.info("Resume mode: loaded progress for %d term(s).", len(progress))

    # Collect existing concept_ids to dedupe
    with conn.cursor() as cur:
        cur.execute("SELECT concept_id FROM clinical_nlp.concepts")
        existing_ids = {row[0] for row in cur.fetchall()}

    for term_spec in OLS4_QUERY_TERMS:
        term = term_spec["term"]
        domain = term_spec["domain"]

        term_progress = progress.get(term, {})
        if term_progress.get("completed"):
            log.info("Skipping term '%s' (already completed in resume state).", term)
            total_inserted += term_progress.get("inserted", 0)
            continue

        # In test mode fetch 100 per term (500 total across 5 terms)
        max_rows = 100 if args.test else None

        start_offset = term_progress.get("last_offset", 0)
        term_inserted = term_progress.get("inserted", 0)
        offset = start_offset

        log.info(
            "Fetching OLS4 term '%s' → domain=%s, starting at offset=%d",
            term,
            domain,
            offset,
        )

        while True:
            if max_rows is not None and term_inserted >= max_rows:
                break

            url = (
                f"{OLS4_BASE_URL}"
                f"?ontology=snomed&q={requests.utils.quote(term)}"
                f"&rows={rows_per_page}&start={offset}"
            )
            try:
                time.sleep(0.1)  # 10 req/s rate limit
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                docs = data.get("response", {}).get("docs", [])
            except Exception as exc:
                log.warning(
                    "OLS4 fetch failed for term '%s' at offset=%d (%s) — stopping this term.",
                    term,
                    offset,
                    type(exc).__name__,
                )
                break

            if not docs:
                break

            new_rows = []
            for doc in docs:
                obo_id = doc.get("obo_id", "").strip()
                label = doc.get("label", "").strip()
                synonyms = doc.get("synonym", [])
                if not obo_id or not label or obo_id in existing_ids:
                    continue
                existing_ids.add(obo_id)
                new_rows.append({
                    "concept_id": obo_id,
                    "preferred_term": label.lower(),
                    "domain": domain,
                    "source": "ols4",
                    "hierarchy": [],
                    "_synonyms": [s.lower() for s in synonyms if s],
                })

            if new_rows:
                try:
                    _insert_concepts(conn, new_rows)
                    _insert_synonyms(conn, new_rows)
                    _insert_ols4_synonyms(conn, new_rows)
                    term_inserted += len(new_rows)
                    total_inserted += len(new_rows)
                except Exception as exc:
                    log.warning(
                        "OLS4 insert error for term '%s' (%s) — skipping batch.",
                        term,
                        type(exc).__name__,
                    )
                    conn.rollback()

            offset += rows_per_page

            # Persist progress after each page
            progress[term] = {
                "completed": False,
                "last_offset": offset,
                "inserted": term_inserted,
            }
            _save_build_progress(conn, progress)

            if len(docs) < rows_per_page:
                # Last page
                break

        # Mark term complete
        progress[term] = {
            "completed": True,
            "last_offset": offset,
            "inserted": term_inserted,
        }
        _save_build_progress(conn, progress)
        log.info("OLS4 term '%s': %d concepts inserted.", term, term_inserted)

    log.info("OLS4 fetch complete: %d total new concepts inserted.", total_inserted)
    return total_inserted


def _generate_embeddings(conn: psycopg2.extensions.connection) -> None:
    """Encode concepts without embeddings and insert into clinical_nlp.embeddings."""
    log.info("Loading sentence-transformer model: %s", EMBEDDING_MODEL)
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:
        log.error(
            "Failed to load embedding model '%s' (%s). "
            "Manually download with: python -c \"from sentence_transformers import SentenceTransformer; "
            "SentenceTransformer('%s')\"",
            EMBEDDING_MODEL,
            type(exc).__name__,
            EMBEDDING_MODEL,
        )
        sys.exit(1)

    # Fetch concepts lacking embeddings
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.concept_id, c.preferred_term
            FROM clinical_nlp.concepts c
            LEFT JOIN clinical_nlp.embeddings e USING (concept_id)
            WHERE e.concept_id IS NULL
            """
        )
        rows = cur.fetchall()

    if not rows:
        log.info("All concepts already have embeddings.")
        return

    log.info("Generating embeddings for %d concepts …", len(rows))
    concept_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    BATCH = 64
    all_embeddings = []

    for start in _progress(
        range(0, len(texts), BATCH),
        total=(len(texts) + BATCH - 1) // BATCH,
        desc="Embedding batches",
    ):
        batch_texts = texts[start: start + BATCH]
        batch_embs = model.encode(batch_texts, batch_size=BATCH, show_progress_bar=False)
        for emb in batch_embs:
            assert emb.shape == (EMBEDDING_DIM,), (
                f"Expected embedding shape ({EMBEDDING_DIM},), got {emb.shape}"
            )
        all_embeddings.extend(zip(concept_ids[start: start + BATCH], batch_embs))

    # Batch insert
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO clinical_nlp.embeddings (concept_id, embedding, model)
            VALUES %s
            ON CONFLICT (concept_id) DO NOTHING
            """,
            [(cid, emb.tolist(), EMBEDDING_MODEL) for cid, emb in all_embeddings],
            page_size=200,
        )
    conn.commit()
    log.info("Embeddings inserted: %d", len(all_embeddings))


def _write_metadata(
    conn: psycopg2.extensions.connection,
    *,
    ols4_count: int,
    test_mode: bool,
) -> None:
    """Upsert build metadata into clinical_nlp.metadata."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM clinical_nlp.concepts")
        total_count = cur.fetchone()[0]

    metadata = {
        "build_date": datetime.utcnow().isoformat(),
        "ols4_count": ols4_count,
        "total_concepts": total_count,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIM,
        "test_mode": test_mode,
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO clinical_nlp.metadata (key, value)
            VALUES ('build_info', %s::jsonb)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            (json.dumps(metadata),),
        )
    conn.commit()
    log.info("Metadata written: build_date=%s total_concepts=%d", metadata["build_date"], total_count)


def _run_validation_checks(
    conn: psycopg2.extensions.connection,
    test_mode: bool,
) -> None:
    """Run 5 spec validation checks and print PASS/FAIL for each."""
    results = []

    # Check 1: HNSW index exists
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM pg_indexes
            WHERE schemaname = 'clinical_nlp' AND indexname = 'idx_embeddings_hnsw'
            """
        )
        count = cur.fetchone()[0]
    passed = count == 1
    results.append(("HNSW index exists", passed, f"count={count}"))

    # Check 2: Concept count in expected range
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM clinical_nlp.concepts")
        concept_count = cur.fetchone()[0]
    if test_mode:
        lo, hi = 100, 600
    else:
        lo, hi = 80_000, 150_000
    passed = lo <= concept_count <= hi
    results.append((
        f"Concept count in range [{lo:,}–{hi:,}]",
        passed,
        f"count={concept_count:,}",
    ))

    # Check 3: Embedding count == concept count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM clinical_nlp.embeddings")
        embedding_count = cur.fetchone()[0]
    passed = embedding_count == concept_count
    results.append((
        "Embedding count == concept count",
        passed,
        f"embeddings={embedding_count:,} concepts={concept_count:,}",
    ))

    # Check 4: All embeddings are 384-dim
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM clinical_nlp.embeddings WHERE vector_dims(embedding) != 384"
        )
        bad_dim_count = cur.fetchone()[0]
    passed = bad_dim_count == 0
    results.append(("All embeddings 384-dim", passed, f"bad_dim_count={bad_dim_count}"))

    # Check 5: Avg synonyms per concept >= 2.0 (skip in test mode — too noisy)
    if not test_mode:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT AVG(syn_count) FROM (
                    SELECT COUNT(*) AS syn_count
                    FROM clinical_nlp.synonyms
                    GROUP BY concept_id
                ) sub
                """
            )
            avg_synonyms = cur.fetchone()[0] or 0.0
        passed = float(avg_synonyms) >= 2.0
        results.append((
            "Avg synonyms per concept >= 2.0",
            passed,
            f"avg={float(avg_synonyms):.2f}",
        ))

    # Print results
    log.info("=== Validation Checks ===")
    all_passed = True
    for label, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        log.info("[%s] %s (%s)", status, label, detail)
        if not passed:
            all_passed = False

    if all_passed:
        log.info("All validation checks PASSED.")
    else:
        log.warning("One or more validation checks FAILED — review output above.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.monotonic()
    args = _parse_args()

    log.info(
        "build_snomed_index starting — test=%s embeddings_only=%s force=%s resume=%s",
        args.test,
        args.embeddings_only,
        args.force,
        args.resume,
    )

    # 1. Validate environment
    db_url = _validate_env(args)

    # 2. Connect to PostgreSQL
    conn = _connect(db_url)

    ols4_count = 0

    try:
        if not args.embeddings_only:
            # 3. Idempotency check / TRUNCATE
            _idempotency_check(conn, args)

            # 4. OLS4 fetch (all 5 query terms with pagination)
            ols4_count = _fetch_ols4(conn, args)

        # 5. Generate embeddings
        _generate_embeddings(conn)

        # 6. Write metadata
        _write_metadata(
            conn,
            ols4_count=ols4_count,
            test_mode=args.test,
        )

        # 10. Validation checks (5 spec checks)
        _run_validation_checks(conn, test_mode=args.test)

    finally:
        conn.close()

    # 7. Summary log — no concept names or user data
    elapsed = time.monotonic() - start_time
    log.info(
        "Build complete in %.1fs — ols4=%d",
        elapsed,
        ols4_count,
    )


if __name__ == "__main__":
    main()
