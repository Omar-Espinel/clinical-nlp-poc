from __future__ import annotations

import os
import re
import logging
import time
from typing import Optional

import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

from src.snomed_search.base import SNOMEDMatch, SNOMEDSearchStrategy
from src.snomed_search.fhir_fallback import SNOMEDFallbackChain, FallbackResult

log = logging.getLogger(__name__)

_STOP_CONCEPTS: frozenset[str] = frozenset({"169230002"})


class PgVectorCascadeStrategy:
    """SNOMED search strategy backed by PostgreSQL + pgvector with API fallback.

    Candidate extraction approach (search docstring):
    1. Exact match via LOWER() equality on preferred_term.
    2. Synonym match via JOIN to synonyms table.
    3. Fuzzy match via pg_trgm similarity > 0.5.
    4. Semantic match via pgvector cosine distance on sentence embeddings.
    5. Cache lookup for previously resolved API results.
    6. FHIR API fallback (OLS4 / BioPortal) when local confidence is too low.
    """

    def __init__(
        self,
        dictionary_path: str,         # kept for Protocol compat, ignored
        fallback_api_key: Optional[str] = None,
        database_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Resolve DATABASE_URL
        if database_url is None:
            database_url = os.environ.get("DATABASE_URL", "")
        if not database_url:
            raise ValueError(
                "DATABASE_URL is required. Set the environment variable or pass "
                "database_url= to the constructor."
            )

        safe_url = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", database_url)
        log.info("Initialising PgVectorCascadeStrategy with DSN: %s", safe_url)

        # Resolve fallback API key
        if fallback_api_key is None:
            fallback_api_key = os.environ.get("BIOPORTAL_API_KEY", "")

        # Create connection pool (hard-capped at 5 per spec security constraint #4)
        self._db_pool: psycopg2.pool.ThreadedConnectionPool = (
            psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=database_url,
            )
        )

        # Register pgvector extension with one bootstrapping connection
        _boot_conn = self._db_pool.getconn()
        try:
            register_vector(_boot_conn)
        finally:
            self._db_pool.putconn(_boot_conn)

        # Load sentence transformer model (eager, per Protocol contract)
        self._model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")

        # Configure FHIR fallback chain.
        # OLS4 requires no key — always instantiate so OLS4 is always available.
        # BioPortalClient self-guards: returns None immediately when key is absent
        # or shorter than 16 chars, so passing an empty string is safe.
        self._fallback: SNOMEDFallbackChain = SNOMEDFallbackChain(
            fallback_api_key or ""
        )
        log.info("OLS4 fallback enabled (no key required)")
        if fallback_api_key and len(fallback_api_key) >= 16:
            log.info("BioPortal fallback enabled")
        else:
            log.warning(
                "BioPortal fallback disabled — "
                "BIOPORTAL_API_KEY missing or shorter than 16 chars"
            )

        # Lightweight in-memory maps for the autocomplete Tier-1/Tier-2 prefix &
        # fuzzy scan and Tier-3 gate. The 90k pgvector SQL search path does NOT
        # use these — they exist only so AutocompleteIndex (which reads
        # strategy._exact_index / _synonym_index / _alias_dict / semantic_available)
        # functions when pgvector is the active strategy.
        self.semantic_available = True
        self._build_autocomplete_indices(dictionary_path)

    # ------------------------------------------------------------------
    # Protocol interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "pgvector_cascade"

    def health_check(self) -> dict:
        try:
            conn = self._db_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                # Close the implicit transaction opened by SELECT so the
                # connection returns to the pool in idle state, not idle-in-txn
                # (otherwise the next borrower's set_session() raises).
                conn.commit()
            finally:
                try:
                    conn.rollback()
                except Exception:
                    pass
                self._db_pool.putconn(conn)
            _ = self._model
        except Exception as e:  # noqa: BLE001
            return {"ready": False, "error_type": type(e).__name__}

        # Attempt concept count — failure here does NOT fail the health check.
        # Uses an explicit transaction (autocommit=False) so SET LOCAL is scoped
        # to this single query; conn.commit()/rollback() are tracked by psycopg2
        # and end the transaction cleanly even if the implicit BEGIN's state
        # differs from the driver's view.
        concept_count = None
        count_conn = self._db_pool.getconn()
        try:
            count_conn.set_session(autocommit=False)
            with count_conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = 1000")
                cur.execute("SELECT COUNT(*) FROM clinical_nlp.concepts")
                row = cur.fetchone()
                if row is not None:
                    concept_count = int(row[0])
            count_conn.commit()
        except Exception:  # noqa: BLE001
            try:
                count_conn.rollback()
            except Exception:
                pass
        finally:
            self._db_pool.putconn(count_conn)

        return {"ready": True, "strategy": "pgvector_cascade", "concept_count": concept_count}

    def search(self, query: str) -> list[SNOMEDMatch]:
        """Return SNOMED matches for *query* using the cascade strategy.

        Candidate extraction:
          exact → synonym → fuzzy → semantic → cache → (optional) API fallback.
        All SQL queries use parameterised %s placeholders (no f-string interpolation).
        Query text is never logged (HIPAA constraint).
        """
        # --- Validation (spec constraints #5 / #8) ---
        if len(query) > 500:
            raise ValueError("Query exceeds maximum length")
        if not query or not query.strip():
            return []

        t0 = time.monotonic()
        try:
            conn = self._db_pool.getconn()
        except (psycopg2.OperationalError, psycopg2.pool.PoolError) as e:
            log.error(
                "Unable to acquire DB connection (error_type=%s)",
                type(e).__name__,
            )
            return []
        results: list[SNOMEDMatch] = []

        try:
            # Defensive: clear any inherited transaction state on the pooled
            # connection so set_session() is legal (it errors inside a txn).
            try:
                conn.rollback()
            except Exception:
                pass
            conn.set_session(readonly=True, autocommit=True)

            # Step 1: exact match — return immediately if found
            exact = self._exact_match(conn, query)
            if exact:
                log.debug(
                    "exact match returned %d result(s) in %.1f ms",
                    len(exact),
                    (time.monotonic() - t0) * 1000,
                )
                results = self._deduplicate(exact)
                results = [r for r in results if r.code not in _STOP_CONCEPTS]
                return results

            # Step 2: synonym match — return immediately if found
            synonym = self._synonym_match(conn, query)
            if synonym:
                log.debug(
                    "synonym match returned %d result(s) in %.1f ms",
                    len(synonym),
                    (time.monotonic() - t0) * 1000,
                )
                results = self._deduplicate(synonym)
                results = [r for r in results if r.code not in _STOP_CONCEPTS]
                return results

            # Step 3–5: accumulate fuzzy + semantic + cache results
            results.extend(self._fuzzy_match(conn, query))
            results.extend(self._semantic_match(conn, query))
            results.extend(self._cache_lookup(conn, query))

            # If any accumulated result has confidence >= 0.82 → done
            if any(r.confidence >= 0.82 for r in results):
                log.debug(
                    "high-confidence local result(s) (%d) found in %.1f ms",
                    len(results),
                    (time.monotonic() - t0) * 1000,
                )
                results = self._deduplicate(results)
                results = [r for r in results if r.code not in _STOP_CONCEPTS]
                return results

        except psycopg2.OperationalError as e:
            log.error(
                "DB operational error during search (error_type=%s)",
                type(e).__name__,
            )
            return []
        finally:
            # Reset session flags before returning to pool so the next caller
            # (e.g. cache write on a recycled conn) doesn't inherit readonly.
            try:
                conn.set_session(readonly=False, autocommit=False)
            except Exception:
                pass
            self._db_pool.putconn(conn)

        # --- API fallback path (needs a writable connection) ---
        write_conn = self._db_pool.getconn()
        try:
            # Ensure writable + transactional in case conn was recycled with stale state.
            write_conn.set_session(readonly=False, autocommit=False)
            fallback_result: Optional[FallbackResult] = self._fallback.lookup(query)
            if fallback_result is not None:
                self._write_cache(write_conn, query, fallback_result)
                write_conn.commit()
                results.append(
                    SNOMEDMatch(
                        code=fallback_result.concept_id,
                        display=fallback_result.preferred_term,
                        match_type=fallback_result.source,
                        confidence=fallback_result.confidence,
                        original_text=query,
                        span=self._make_span(query),
                    )
                )
                log.debug(
                    "API fallback returned concept_id=%s via source=%s",
                    fallback_result.concept_id,
                    fallback_result.source,
                )
        except Exception as e:  # noqa: BLE001
            log.error(
                "API fallback error (error_type=%s)", type(e).__name__
            )
            write_conn.rollback()
        finally:
            self._db_pool.putconn(write_conn)

        results = self._deduplicate(results)
        results = [r for r in results if r.code not in _STOP_CONCEPTS]
        log.debug(
            "search complete: %d deduplicated result(s) in %.1f ms",
            len(results),
            (time.monotonic() - t0) * 1000,
        )
        return results

    def get_top_neighbors(
        self,
        query: str,
        n: int = 15,
        low_threshold: float = 0.42,
    ) -> list[SNOMEDMatch]:
        """Return up to n SNOMEDMatch objects whose cosine similarity to query
        falls at or above low_threshold.

        Called by EmbeddingAmbiguityGate (Layer 2). NOT part of SNOMEDSearchStrategy
        Protocol — detected via duck-typing (hasattr).

        HIPAA: query text NOT logged. Only n_returned and latency_ms logged on success.
        Error handling: any exception is caught, logged by error_type only, returns [].
        """
        if not query or not query.strip():
            return []
        if len(query) > 500:
            return []

        t0 = time.monotonic()
        try:
            embedding = self._model.encode(query)
            embedding_str = (
                "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"
            )
        except Exception as e:  # noqa: BLE001
            log.error(
                "get_top_neighbors: embedding error (error_type=%s)",
                type(e).__name__,
            )
            return []

        conn = None
        try:
            conn = self._db_pool.getconn()
            # Defensive: clear any inherited transaction state on the pooled
            # connection. set_session() is illegal mid-txn; conn.commit() on a
            # connection in autocommit=True (from a prior search()) silently
            # no-ops but leaves the driver's notion of state intact.
            try:
                conn.rollback()
            except Exception:
                pass
            # Explicit autocommit=False so SET LOCAL binds to a real transaction
            # that conn.commit()/rollback() track at the driver level. Raw
            # cur.execute("BEGIN") + cur.execute("COMMIT") would not be tracked
            # by psycopg2 and could desync the pool's view of session state.
            conn.set_session(autocommit=False)
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = 5000")
                cur.execute(
                    """
                    SELECT e.concept_id, c.preferred_term,
                           1 - (e.embedding <=> %s::vector) AS similarity
                    FROM clinical_nlp.embeddings e
                    JOIN clinical_nlp.concepts c ON e.concept_id = c.concept_id
                    WHERE 1 - (e.embedding <=> %s::vector) >= %s
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (embedding_str, embedding_str, low_threshold, n),
                )
                rows = cur.fetchall()
            conn.commit()

            matches = [
                SNOMEDMatch(
                    code=row[0],
                    display=row[1],
                    match_type="semantic",
                    confidence=round(float(row[2]), 4),
                    original_text=query,
                    span=(0, max(1, len(query))),
                    negated=False,
                )
                for row in rows
            ]
            log.debug(
                "get_top_neighbors: n_returned=%d latency_ms=%.1f",
                len(matches),
                (time.monotonic() - t0) * 1000,
            )
            return matches

        except Exception as e:  # noqa: BLE001
            log.error(
                "get_top_neighbors: DB error (error_type=%s)",
                type(e).__name__,
            )
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return []
        finally:
            if conn is not None:
                # Belt-and-suspenders: rollback before putback so a successful
                # commit followed by some other failure can't leak an open txn
                # into the pool. No-op if no txn is open.
                try:
                    conn.rollback()
                except Exception:
                    pass
                self._db_pool.putconn(conn)

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _build_autocomplete_indices(self, csv_path: str) -> None:
        import csv as _csv
        from src.snomed_search.hybrid_cascade import ALIAS_DICTIONARY

        self._exact_index: dict[str, dict] = {}
        self._synonym_index: dict[str, dict] = {}
        self._alias_dict: dict[str, str] = {}
        try:
            with open(csv_path, encoding="utf-8") as fh:
                for row in _csv.DictReader(fh):
                    preferred_term = (row.get("preferred_term") or "").strip().lower()
                    if not preferred_term:
                        continue
                    record = {
                        "concept_id": (row.get("concept_id") or "").strip(),
                        "preferred_term": preferred_term,
                    }
                    self._exact_index[preferred_term] = record
                    for syn in (row.get("synonyms") or "").split("|"):
                        syn_clean = syn.strip().lower()
                        if syn_clean:
                            self._synonym_index[syn_clean] = record
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "PgVectorCascadeStrategy: autocomplete index build failed (%s) "
                "— Tier-1/2 prefix degraded", type(exc).__name__,
            )
            return
        for alias, target in ALIAS_DICTIONARY.items():
            alias_key = alias.lower().strip()
            target_key = target.lower().strip()
            if target_key in self._exact_index:
                self._alias_dict[alias_key] = target_key
        log.info(
            "PgVectorCascadeStrategy: autocomplete maps built (terms=%d synonyms=%d aliases=%d)",
            len(self._exact_index), len(self._synonym_index), len(self._alias_dict),
        )

    def _exact_match(self, conn, query: str) -> list[SNOMEDMatch]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT concept_id, preferred_term
                FROM clinical_nlp.concepts
                WHERE LOWER(preferred_term) = LOWER(%s)
                LIMIT 1
                """,
                (query,),
            )
            row = cur.fetchone()
        if row is None:
            return []
        return [
            SNOMEDMatch(
                code=row[0],
                display=row[1],
                match_type="exact",
                confidence=1.0,
                original_text=query,
                span=self._make_span(query),
            )
        ]

    def _synonym_match(self, conn, query: str) -> list[SNOMEDMatch]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.concept_id, c.preferred_term
                FROM clinical_nlp.concepts c
                JOIN clinical_nlp.synonyms s ON s.concept_id = c.concept_id
                WHERE LOWER(s.synonym_text) = LOWER(%s)
                LIMIT 5
                """,
                (query,),
            )
            rows = cur.fetchall()
        return [
            SNOMEDMatch(
                code=row[0],
                display=row[1],
                match_type="synonym",
                confidence=0.95,
                original_text=query,
                span=self._make_span(query),
            )
            for row in rows
        ]

    def _fuzzy_match(self, conn, query: str) -> list[SNOMEDMatch]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT concept_id, preferred_term,
                       similarity(preferred_term, %s) AS sim
                FROM clinical_nlp.concepts
                WHERE similarity(preferred_term, %s) > 0.5
                ORDER BY sim DESC
                LIMIT 10
                """,
                (query, query),
            )
            rows = cur.fetchall()
        return [
            SNOMEDMatch(
                code=row[0],
                display=row[1],
                match_type="fuzzy",
                confidence=0.70 + (float(row[2]) * 0.20),
                original_text=query,
                span=self._make_span(query),
            )
            for row in rows
        ]

    def _semantic_match(self, conn, query: str) -> list[SNOMEDMatch]:
        try:
            embedding = self._model.encode(query)
            if embedding.shape != (384,):
                log.error(
                    "Unexpected embedding shape %s; skipping semantic match",
                    embedding.shape,
                )
                return []
            embedding_str = "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"
            with conn.cursor() as cur:
                # Distance < 0.40 = similarity > 0.60. The pipeline's downstream
                # MIN_CONFIDENCE=0.60 in the assembler and OPTION_MIN_CONFIDENCE=0.70
                # in sufficiency_gate apply the real quality bars; pre-filtering at
                # 0.82 here under-recalled lay-term ↔ SNOMED mappings (e.g.
                # "Lung Cancer" → "screening for malignant neoplasm of lung" at
                # similarity 0.7147 was being dropped at SQL level).
                cur.execute(
                    """
                    SELECT e.concept_id, c.preferred_term,
                           e.embedding <=> %s::vector AS distance
                    FROM clinical_nlp.embeddings e
                    JOIN clinical_nlp.concepts c ON e.concept_id = c.concept_id
                    WHERE e.embedding <=> %s::vector < 0.40
                    ORDER BY distance
                    LIMIT 10
                    """,
                    (embedding_str, embedding_str),
                )
                rows = cur.fetchall()
            return [
                SNOMEDMatch(
                    code=row[0],
                    display=row[1],
                    match_type="semantic",
                    confidence=max(0.0, min(1.0, 1.0 - float(row[2]))),
                    original_text=query,
                    span=self._make_span(query),
                )
                for row in rows
            ]
        except Exception as e:  # noqa: BLE001
            log.error(
                "Semantic match error (error_type=%s)", type(e).__name__
            )
            return []

    def _cache_lookup(self, conn, query: str) -> list[SNOMEDMatch]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.concept_id, c.preferred_term, ca.source, ca.confidence
                FROM clinical_nlp.cache ca
                JOIN clinical_nlp.concepts c ON ca.concept_id = c.concept_id
                WHERE LOWER(ca.term) = LOWER(%s)
                  AND ca.cached_at + (ca.ttl_days || ' days')::INTERVAL > CURRENT_TIMESTAMP
                LIMIT 1
                """,
                (query,),
            )
            row = cur.fetchone()
        if row is None:
            return []
        return [
            SNOMEDMatch(
                code=row[0],
                display=row[1],
                match_type=f"cached_{row[2]}",
                confidence=float(row[3]),
                original_text=query,
                span=self._make_span(query),
            )
        ]

    def _write_cache(self, conn, term: str, result: FallbackResult) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO clinical_nlp.cache
                    (term, concept_id, source, confidence, cached_at, ttl_days)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, 30)
                ON CONFLICT (term) DO UPDATE
                    SET concept_id  = EXCLUDED.concept_id,
                        source      = EXCLUDED.source,
                        confidence  = EXCLUDED.confidence,
                        cached_at   = EXCLUDED.cached_at,
                        ttl_days    = EXCLUDED.ttl_days
                """,
                (term, result.concept_id, result.source, result.confidence),
            )

    def _deduplicate(self, results: list[SNOMEDMatch]) -> list[SNOMEDMatch]:
        """Group by code, keep highest confidence, drop negated=True, sort DESC."""
        best: dict[str, SNOMEDMatch] = {}
        for match in results:
            if match.negated:
                continue
            existing = best.get(match.code)
            if existing is None or match.confidence > existing.confidence:
                best[match.code] = match
        return sorted(best.values(), key=lambda m: m.confidence, reverse=True)

    def _make_span(self, query: str) -> tuple[int, int]:
        """Return a span tuple that satisfies SNOMEDMatch validation constraints."""
        return (0, max(1, len(query)))
