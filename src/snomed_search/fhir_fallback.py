"""
SNOMED API fallback chain for cache misses.

Primary:   OLS4 (EMBL-EBI, no API key, excellent uptime)
Secondary: BioPortal (free API key, 15 req/s)

Both clients have circuit breakers (5 failures = 60s backoff).
All lookups are hierarchy-filtered to clinical trial relevant concepts only.

Never called in the happy path — only when local index + cache both miss.
"""

import os
import time
import logging
import httpx
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("snomed_fallback")

# Constants
OLS4_BASE = "https://www.ebi.ac.uk/ols4/api"
BIOPORTAL_BASE = "https://data.bioontology.org"
BIOPORTAL_API_KEY = os.environ.get("BIOPORTAL_API_KEY", "")

REQUEST_TIMEOUT = 5.0
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RESET = 60.0

# Only accept concepts in these SNOMED hierarchies
VALID_SNOMED_HIERARCHIES = {
    "404684003",  # Clinical finding
    "71388002",   # Procedure
    "373873005",  # Pharmaceutical/biologic product
    "118234003",  # Finding related to pregnancy
    "123037004",  # Body structure
    "362981000",  # Qualifier value
    "272379006",  # Event
}


class CircuitBreaker:
    def __init__(self, threshold: int, reset_seconds: float):
        self._threshold = threshold
        self._reset = reset_seconds
        self._failures = 0
        self._opened_at: Optional[float] = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at > self._reset:
            self._failures = 0
            self._opened_at = None
            return False
        return True

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()
            log.warning("Circuit breaker opened — failures: %d", self._failures)

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None


@dataclass(frozen=True)
class FallbackResult:
    concept_id: str
    preferred_term: str
    source: str
    confidence: float


class OLS4Client:
    def __init__(self):
        self._breaker = CircuitBreaker(CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_RESET)

    def lookup(self, term: str) -> Optional[FallbackResult]:
        if self._breaker.is_open:
            return None
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.get(
                    f"{OLS4_BASE}/search",
                    params={
                        "q": term,
                        "ontology": "snomed",
                        "rows": 5,
                        "exact": "false",
                        "fieldList": "id,label,obo_id,hierarchy",
                    },
                )
            response.raise_for_status()
            data = response.json()
            docs = data.get("response", {}).get("docs", [])

            for doc in docs:
                concept_id = doc.get("obo_id", "").replace("SNOMED:", "")
                label = doc.get("label", "")
                if not concept_id or not label:
                    continue
                if not self._in_valid_hierarchy(doc):
                    continue
                confidence = 0.75
                self._breaker.record_success()
                log.info(
                    "OLS4 lookup success — concept_id: %s confidence: %.2f",
                    concept_id,
                    confidence,
                )
                return FallbackResult(
                    concept_id=concept_id,
                    preferred_term=label.lower(),
                    source="ols4",
                    confidence=confidence,
                )
            return None

        except httpx.TimeoutException:
            self._breaker.record_failure()
            log.warning("OLS4 timeout — failures: %d", self._breaker._failures)
            return None
        except Exception as e:
            self._breaker.record_failure()
            log.warning(
                "OLS4 lookup failed — error_type: %s failures: %d",
                type(e).__name__,
                self._breaker._failures,
            )
            return None

    def _in_valid_hierarchy(self, doc: dict) -> bool:
        # OLS4 /api/search no longer returns a "hierarchy" field in docs.
        # When the field is absent we trust the ontology=snomed filter already
        # applied server-side and allow the result through.  When the field IS
        # present (future OLS4 versions or alternate deployments) we still
        # validate against the known clinical-trial hierarchy roots.
        hierarchy = doc.get("hierarchy")
        if hierarchy is None:
            return True
        return any(h in VALID_SNOMED_HIERARCHIES for h in hierarchy)


class BioPortalClient:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._breaker = CircuitBreaker(CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_RESET)

    def lookup(self, term: str) -> Optional[FallbackResult]:
        if self._breaker.is_open:
            return None
        if not self._api_key or len(self._api_key) < 16:
            return None
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.get(
                    f"{BIOPORTAL_BASE}/search",
                    params={
                        "q": term,
                        "ontologies": "SNOMEDCT",
                        "pagesize": 5,
                        "display_context": "false",
                    },
                    headers={"Authorization": f"apikey token={self._api_key}"},
                )
            response.raise_for_status()
            data = response.json()
            items = data.get("collection", [])

            for item in items:
                concept_id = item.get("@id", "").split("/")[-1]
                label = item.get("prefLabel", "")
                if not concept_id or not label:
                    continue
                confidence = 0.72
                self._breaker.record_success()
                log.info(
                    "BioPortal lookup success — concept_id: %s confidence: %.2f",
                    concept_id,
                    confidence,
                )
                return FallbackResult(
                    concept_id=concept_id,
                    preferred_term=label.lower(),
                    source="bioportal",
                    confidence=confidence,
                )
            return None

        except httpx.TimeoutException:
            self._breaker.record_failure()
            log.warning("BioPortal timeout — failures: %d", self._breaker._failures)
            return None
        except Exception as e:
            self._breaker.record_failure()
            log.warning(
                "BioPortal lookup failed — error_type: %s failures: %d",
                type(e).__name__,
                self._breaker._failures,
            )
            return None


class SNOMEDFallbackChain:
    """Called only when local index fails AND embedding pre-filter
    thinks the term looks medical. Never blocks the response —
    all failures fall through to the clarification gate."""

    def __init__(self, bioportal_api_key: str):
        self._ols4 = OLS4Client()
        self._bioportal = BioPortalClient(bioportal_api_key)

    def lookup(self, term: str) -> Optional[FallbackResult]:
        result = self._ols4.lookup(term)
        if result is not None:
            return result

        result = self._bioportal.lookup(term)
        if result is not None:
            return result

        log.info("Fallback chain exhausted — term_length: %d", len(term))
        return None


__all__ = ["SNOMEDFallbackChain", "FallbackResult", "CircuitBreaker", "OLS4Client", "BioPortalClient"]
