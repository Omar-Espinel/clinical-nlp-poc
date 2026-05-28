"""
Tests for NameExtractor.
Run: python -m pytest tests/test_name_extractor.py -v --tb=short
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.extractors.names import NameExtractor

PROJECT_ROOT = Path(__file__).parent.parent
INST_KW_PATH = str(PROJECT_ROOT / "data" / "institution_keywords.json")
GEO_PATH = str(PROJECT_ROOT / "data" / "geo_canonical.json")


@pytest.fixture(scope="module")
def extractor() -> NameExtractor:
    return NameExtractor(
        institution_keywords_path=INST_KW_PATH,
        geo_json_path=GEO_PATH,
    )


# ---------------------------------------------------------------------------
# Test 1: person prefix (Dr.)
# ---------------------------------------------------------------------------
def test_dr_prefix(extractor: NameExtractor) -> None:
    result = extractor.extract("dr johnson diabetes phase 2")
    assert result.investigator_name == "Johnson"
    assert result.investigator_confidence >= 0.95
    assert result.site_name is None
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 2: person suffix (MD)
# ---------------------------------------------------------------------------
def test_person_suffix_md(extractor: NameExtractor) -> None:
    result = extractor.extract("john smith md diabetes")
    assert result.investigator_name == "John Smith"
    assert result.investigator_confidence >= 0.93
    assert result.site_name is None
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 3: context site "at" + lowercase site name
# ---------------------------------------------------------------------------
def test_context_site_at_northwestern(extractor: NameExtractor) -> None:
    result = extractor.extract("trials at northwestern")
    assert result.site_name is not None
    assert "Northwestern" in result.site_name
    assert result.site_confidence >= 0.80
    assert result.investigator_name is None
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 4: institution primary keyword "hospital" + preceding name
# ---------------------------------------------------------------------------
def test_mayo_hospital(extractor: NameExtractor) -> None:
    result = extractor.extract("mayo hospital diabetes")
    assert result.site_name is not None
    assert "Mayo" in result.site_name
    assert result.site_confidence >= 0.85
    assert result.investigator_name is None
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 5: context person "by" + single name
# ---------------------------------------------------------------------------
def test_by_single_name(extractor: NameExtractor) -> None:
    result = extractor.extract("trials by johnson")
    assert result.investigator_name == "Johnson"
    assert result.investigator_confidence >= 0.80
    assert result.site_name is None
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 6: context person "by" + two-token name
# ---------------------------------------------------------------------------
def test_by_two_names(extractor: NameExtractor) -> None:
    result = extractor.extract("trials by john smith")
    assert result.investigator_name == "John Smith"
    assert result.investigator_confidence >= 0.85
    assert result.site_name is None
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 7: MD Anderson (multiword, MD Anderson exception)
# ---------------------------------------------------------------------------
def test_md_anderson(extractor: NameExtractor) -> None:
    result = extractor.extract("md anderson phase 2 diabetes")
    assert result.site_name is not None
    assert "MD Anderson" in result.site_name or "Anderson" in result.site_name
    assert result.site_confidence >= 0.90
    assert result.investigator_name is None
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 8: ambiguous — both johnson and northwestern, no clear role
# ---------------------------------------------------------------------------
def test_ambiguous_johnson_northwestern(extractor: NameExtractor) -> None:
    result = extractor.extract("johnson northwestern diabetes")
    ambig_texts = [a.text.lower() for a in result.ambiguous_names]
    assert len(result.ambiguous_names) >= 2
    assert any("johnson" in t for t in ambig_texts)
    assert any("northwestern" in t for t in ambig_texts)


# ---------------------------------------------------------------------------
# Test 9: no signals at all
# ---------------------------------------------------------------------------
def test_no_signals(extractor: NameExtractor) -> None:
    result = extractor.extract("trials smith")
    assert result.investigator_name is None
    assert result.site_name is None
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 10: "washington" is geo state alias — no person/site extracted
# ---------------------------------------------------------------------------
def test_washington_geo_alias(extractor: NameExtractor) -> None:
    result = extractor.extract("trials washington", excluded_spans=[])
    # "washington" is lowercase — no title-case patterns fire
    # It may or may not be in ambiguous; main assertion: no firm extraction
    assert result.investigator_name is None
    assert result.site_name is None
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 11: structured parse "investigator, site"
# ---------------------------------------------------------------------------
def test_structured_parse_full(extractor: NameExtractor) -> None:
    result = extractor.extract("johnson investigator, northwestern site")
    assert result.investigator_name == "Johnson"
    assert result.investigator_confidence >= 0.95
    assert result.site_name == "Northwestern"
    assert result.site_confidence >= 0.95
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 12: structured parse compact "inv,site"
# ---------------------------------------------------------------------------
def test_structured_parse_compact(extractor: NameExtractor) -> None:
    result = extractor.extract("johnson inv,northwestern site")
    assert result.investigator_name == "Johnson"
    assert result.investigator_confidence >= 0.95
    assert result.site_name == "Northwestern"
    assert result.site_confidence >= 0.95
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 13: "dr washington" — prefix forces person interpretation despite geo
# ---------------------------------------------------------------------------
def test_dr_washington(extractor: NameExtractor) -> None:
    result = extractor.extract("dr washington phase 2")
    assert result.investigator_name == "Washington"
    assert result.investigator_confidence >= 0.95
    assert result.site_name is None
    assert result.ambiguous_names == []


# ---------------------------------------------------------------------------
# Test 14: "boston children's" — primary keyword hit with preceding title-case
# ---------------------------------------------------------------------------
def test_boston_childrens(extractor: NameExtractor) -> None:
    result = extractor.extract("phase 3 boston children's")
    assert result.site_name is not None
    assert "Children" in result.site_name or "children" in result.site_name.lower()
    assert result.site_confidence >= 0.85
    assert result.investigator_name is None
