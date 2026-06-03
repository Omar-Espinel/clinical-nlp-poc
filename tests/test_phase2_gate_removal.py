"""tests/test_phase2_gate_removal.py — Phase 2 gate-removal acceptance tests.

Spec: All 8 ACs must assert result.type == "search".

After Phase 2 the pipeline must ALWAYS return NLPOutput (type:"search") for any
query that passes the preprocessor safety check AND preflight.  ClarificationOutput
may ONLY be returned for preprocessor safety rejections and genuine preflight
failures.

Uses the hybrid_cascade strategy to avoid needing a live Postgres/pgvector DB.
The pipeline is built once per module (expensive startup amortized).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on path for direct invocation
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force hybrid_cascade before any pipeline import so get_strategy() returns it.
# This is the same approach used by batch_eval.py and smoke_step10.py.
os.environ.setdefault("SNOMED_SEARCH_STRATEGY", "hybrid_cascade")

PROJECT_ROOT = Path(__file__).parent.parent
SNOMED_CSV   = str(PROJECT_ROOT / "data" / "snomed_clinical_trials.csv")


# ---------------------------------------------------------------------------
# Module-scoped pipeline fixture — expensive to build, shared across all ACs.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    """Real NLPPipeline with hybrid_cascade strategy (no DB required)."""
    from src.snomed_search.hybrid_cascade import HybridCascadeStrategy
    from src.pipeline import NLPPipeline

    strategy = HybridCascadeStrategy(dictionary_path=SNOMED_CSV)
    return NLPPipeline(snomed_strategy=strategy)


def _run(pipeline, query: str):
    """Run a fresh single-turn session and return the result."""
    from src.conversation import ConversationSession
    session = ConversationSession.new()
    return pipeline.run_with_session(query, session)


# ---------------------------------------------------------------------------
# AC-1: "Gout studies in Michigan"
# Previously could fail preflight before Phase 1 Signal D fix.
# ---------------------------------------------------------------------------

def test_ac1_gout_michigan(pipeline):
    """AC-1: 'Gout studies in Michigan' must return type='search'."""
    result = _run(pipeline, "Gout studies in Michigan")
    assert result.type == "search", (
        f"AC-1 expected type='search', got type={result.type!r}"
    )


# ---------------------------------------------------------------------------
# AC-2: "Studies by doctor michaels on leukemia phase ii"
# Previously embedding gate could have fired; metric gate could block.
# ---------------------------------------------------------------------------

def test_ac2_doctor_michaels_leukemia(pipeline):
    """AC-2: 'Studies by doctor michaels on leukemia phase ii' must return type='search'."""
    result = _run(pipeline, "Studies by doctor michaels on leukemia phase ii")
    assert result.type == "search", (
        f"AC-2 expected type='search', got type={result.type!r}"
    )


# ---------------------------------------------------------------------------
# AC-3: Multi-filter with phase range
# ---------------------------------------------------------------------------

def test_ac3_lung_cancer_new_york_dr_holtz(pipeline):
    """AC-3: Complex query with location, investigator, and multi-phase must return type='search'."""
    result = _run(
        pipeline,
        "Lung cancer research in New York by Dr Holtz phase 2 or 3",
    )
    assert result.type == "search", (
        f"AC-3 expected type='search', got type={result.type!r}"
    )


# ---------------------------------------------------------------------------
# AC-4: Natural language with phase range and geography
# ---------------------------------------------------------------------------

def test_ac4_heart_attack_bay_area_martinez(pipeline):
    """AC-4: Long natural-language query must return type='search'."""
    result = _run(
        pipeline,
        "Can you pull up any Phase 2 or Phase 3 heart attack studies being run out "
        "of the Bay Area by Dr. Martinez",
    )
    assert result.type == "search", (
        f"AC-4 expected type='search', got type={result.type!r}"
    )


# ---------------------------------------------------------------------------
# AC-5: Same query as AC-4 — not a clarification (idempotency check)
# ---------------------------------------------------------------------------

def test_ac5_same_as_ac4_not_clarification(pipeline):
    """AC-5: Identical query to AC-4 must still return type='search', not clarification."""
    result = _run(
        pipeline,
        "Can you pull up any Phase 2 or Phase 3 heart attack studies being run out "
        "of the Bay Area by Dr. Martinez",
    )
    assert result.type == "search", (
        f"AC-5 expected type='search', got type={result.type!r}"
    )


# ---------------------------------------------------------------------------
# AC-6: Bare condition + phase — previously could hit ambiguous_trigger
# ---------------------------------------------------------------------------

def test_ac6_cancer_phase_2(pipeline):
    """AC-6: 'cancer phase 2' must return type='search' (no clarification gate)."""
    result = _run(pipeline, "cancer phase 2")
    assert result.type == "search", (
        f"AC-6 expected type='search', got type={result.type!r}"
    )


# ---------------------------------------------------------------------------
# AC-7: Investigator prefix with phase — previously name-ambiguity could block
# ---------------------------------------------------------------------------

def test_ac7_phase_ii_dr_holmes(pipeline):
    """AC-7: 'Phase ii studies by Dr Holmes' must return type='search'."""
    result = _run(pipeline, "Phase ii studies by Dr Holmes")
    assert result.type == "search", (
        f"AC-7 expected type='search', got type={result.type!r}"
    )


# ---------------------------------------------------------------------------
# AC-8: Misspelled condition + location + phase
# ---------------------------------------------------------------------------

def test_ac8_breast_cancer_misspelled_boston_phase3(pipeline):
    """AC-8: Misspelled 'Breast Canccer' with city + phase must return type='search'."""
    result = _run(
        pipeline,
        "Find me all clinical trials for Breast Canccer in Boston that are currently "
        "in Phase 3",
    )
    assert result.type == "search", (
        f"AC-8 expected type='search', got type={result.type!r}"
    )
