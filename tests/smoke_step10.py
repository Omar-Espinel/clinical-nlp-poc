"""Smoke test for STEP 10 of the deterministic-NLP rework.

Runs 6 canonical queries through the pipeline and prints a one-line
verdict per query. Light: no full batch eval. HIPAA: prints query text
ONLY in this smoke context (developer use, not production logging).
"""

import os
import sys
import logging
from pathlib import Path

# Quiet down chatty modules
logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import NLPPipeline
from src.conversation import ConversationSession
from src.assembler import NLPOutput, ClarificationOutput


def fmt(out) -> str:
    if isinstance(out, NLPOutput):
        inv = out.filters.investigator_name.value
        site = out.filters.site_name.value
        city = out.filters.city.value
        state = out.filters.state.values
        phase = out.filters.phase.value
        snomed_n = len(out.snomed_terms)
        return (
            f"SEARCH inv={inv!r} site={site!r} city={city!r} "
            f"state={state} phase={phase!r} snomed={snomed_n}"
        )
    if isinstance(out, ClarificationOutput):
        return f"CLARIFY q={out.question[:60]!r} options={len(out.options)}"
    return f"UNKNOWN type={type(out).__name__}"


def main() -> int:
    print("Loading pipeline...")
    pipeline = NLPPipeline()
    print(f"Strategy: {pipeline._snomed.name}\n")

    queries = [
        "Dr. Smith phase 3 diabetes trials in Boston",
        "trials at Mayo Clinic",
        "phase 3 in Boston",
        "active trials east coast",
        "trials washington",
        "MD Anderson phase 2 diabetes",
    ]

    for q in queries:
        session = ConversationSession.new()
        try:
            out = pipeline.run_with_session(q, session)
            print(f"  Q: {q!r}")
            print(f"  ==> {fmt(out)}\n")
        except Exception as exc:
            print(f"  Q: {q!r}")
            print(f"  ==> ERROR type={type(exc).__name__} msg={exc!s}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
