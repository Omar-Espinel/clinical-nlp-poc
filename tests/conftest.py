"""
tests/conftest.py — Shared fixtures and configuration for the test suite.

Rapidfuzz version assertion: Group 3 fuzzy thresholds (75–80) in test_metric_filters.py
are tuned to rapidfuzz 3.10.0. Fail fast if a different version is installed.
"""

import importlib.metadata

_expected_rapidfuzz = "3.10.0"
_installed = importlib.metadata.version("rapidfuzz")
if _installed != _expected_rapidfuzz:
    raise RuntimeError(
        f"rapidfuzz version mismatch: expected {_expected_rapidfuzz}, got {_installed}. "
        "Fuzzy thresholds in tests/test_metric_filters.py are tuned to 3.10.0. "
        "Re-tune thresholds or pin rapidfuzz==3.10.0."
    )
