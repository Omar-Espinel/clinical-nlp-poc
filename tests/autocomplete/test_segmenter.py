import pytest
from src.autocomplete.segmenter import QuerySegmenter, ContextType

GEO_KEYS = frozenset({
    "boston", "new york city", "california", "bay area",
    "los angeles", "chicago", "massachusetts",
})


@pytest.fixture
def segmenter():
    return QuerySegmenter(GEO_KEYS)


def test_phase_prefix_extracted(segmenter):
    result = segmenter.segment("Phase 3 diab")
    assert "Phase 3" in result.context_tokens
    assert result.active_prefix == "diab"
    assert ContextType.PHASE in result.context_types


def test_bare_prefix_no_context(segmenter):
    result = segmenter.segment("diabetes")
    assert result.context_tokens == []
    assert result.active_prefix == "diabetes"


def test_geo_prefix_extracted(segmenter):
    result = segmenter.segment("boston diab")
    assert any("boston" in t.lower() for t in result.context_tokens)
    assert result.active_prefix == "diab"
    assert ContextType.GEO in result.context_types


def test_short_prefix_returns_empty_active(segmenter):
    result = segmenter.segment("di")
    assert result.active_prefix == ""


def test_phase_only_no_trailing_prefix(segmenter):
    result = segmenter.segment("Phase 3")
    assert result.active_prefix == "" or len(result.active_prefix) < 3


def test_case_insensitive_phase(segmenter):
    result = segmenter.segment("phase 2 card")
    assert result.active_prefix == "card"
    assert ContextType.PHASE in result.context_types


def test_two_char_geo_keys_filtered():
    seg = QuerySegmenter(frozenset({"ca", "ny", "boston"}))
    result = seg.segment("ca diab")
    assert ContextType.GEO not in result.context_types


def test_multiword_geo_extracted(segmenter):
    result = segmenter.segment("bay area diab")
    assert ContextType.GEO in result.context_types
    assert result.active_prefix == "diab"


SNOMED_TERMS = frozenset({
    "malignant neoplasm of esophagus",
    "myocardial infarction",
    "type 2 diabetes mellitus",
})


def make_segmenter():
    return QuerySegmenter(GEO_KEYS, snomed_known_terms=SNOMED_TERMS)


def test_comma_boundary_isolates_active_prefix():
    seg = make_segmenter()
    result = seg.segment("Malignant Neoplasm of esophagus, phas")
    assert result.active_prefix.lower() == "phas"
    assert "malignant neoplasm of esophagus," in result.committed_prefix_raw.lower()


def test_in_boundary_isolates_geo_prefix():
    seg = make_segmenter()
    result = seg.segment("Malignant Neoplasm of esophagus, phase 3 in Michig")
    assert result.active_prefix.lower() == "michig"
    assert ContextType.PHASE in result.context_types


def test_committed_prefix_raw_preserved_verbatim():
    seg = make_segmenter()
    result = seg.segment("Malignant Neoplasm of esophagus, phase 3 in Michig")
    assert result.committed_prefix_raw.endswith("in ")


def test_bare_single_term_unchanged():
    seg = make_segmenter()
    result = seg.segment("Esophagus")
    assert result.active_prefix == "Esophagus"
    assert result.committed_prefix_raw == ""


def test_filler_prefix_preserved_in_committed_raw():
    seg = make_segmenter()
    result = seg.segment("Studies on Malignant Neoplasm of esophagus, phas")
    assert result.active_prefix.lower() == "phas"
    assert result.committed_prefix_raw.lower().startswith("studies on")


def test_strategy_b_last_token_split_on_known_snomed():
    seg = make_segmenter()
    result = seg.segment("Myocardial Infarction Michig")
    assert result.active_prefix.lower() == "michig"


def test_strategy_b_two_word_lookback():
    seg = make_segmenter()
    result = seg.segment("Type 2 Diabetes Mellitus Bost")
    assert result.active_prefix.lower() == "bost"


def test_snomed_context_type_emitted_when_snomed_committed():
    seg = make_segmenter()
    result = seg.segment("Malignant Neoplasm of esophagus, phas")
    assert ContextType.SNOMED in result.context_types


def test_trailing_comma_returns_empty_active_prefix():
    seg = make_segmenter()
    result = seg.segment("esophagus,")
    assert result.active_prefix == ""


def test_short_active_prefix_returns_empty():
    seg = make_segmenter()
    result = seg.segment("Malignant Neoplasm of esophagus, ph")
    assert result.active_prefix == ""


def test_no_false_split_on_for_connector():
    seg = make_segmenter()
    result = seg.segment("trials for diabet")
    # "for" is intentionally not a boundary, so the whole phrase remains the
    # active prefix and nothing is committed (verbatim reconstruction intact).
    assert result.active_prefix.lower() == "trials for diabet"
    assert result.committed_prefix_raw == ""


def test_compound_phase_and_detected():
    seg = make_segmenter()
    result = seg.normalize_active_prefix("Phase 2 and 3")
    assert result.is_compound_phase is True
    assert "Phase 2" in result.compound_phases
    assert "Phase 3" in result.compound_phases
    assert result.search_text == "Phase 2"


def test_compound_phase_or_detected():
    seg = make_segmenter()
    result = seg.normalize_active_prefix("Phase 2 or 3")
    assert result.is_compound_phase is True
    assert "Phase 2" in result.compound_phases
    assert "Phase 3" in result.compound_phases


def test_compound_phases_detected_case_insensitive():
    seg = make_segmenter()
    result = seg.normalize_active_prefix("phase 2 AND 3")
    assert result.is_compound_phase is True


def test_noise_suffix_only_stripped():
    seg = make_segmenter()
    result = seg.normalize_active_prefix("gout only")
    assert result.search_text == "gout"
    assert result.is_compound_phase is False


def test_noise_suffix_us_sites_only_stripped():
    seg = make_segmenter()
    result = seg.normalize_active_prefix("gout US Sites only")
    assert result.search_text == "gout"


def test_noise_suffix_sites_only_stripped():
    seg = make_segmenter()
    result = seg.normalize_active_prefix("diabetes sites only")
    assert result.search_text == "diabetes"


def test_no_noise_no_change():
    seg = make_segmenter()
    result = seg.normalize_active_prefix("diabetes")
    assert result.search_text == "diabetes"
    assert result.original == "diabetes"
    assert result.is_compound_phase is False


def test_filler_a_study_in_stripped():
    seg = make_segmenter()
    result = seg.segment("a study in gout")
    assert result.active_prefix.lower() == "gout"


def test_filler_a_trial_in_stripped():
    seg = make_segmenter()
    result = seg.segment("a trial in diabetes")
    assert result.active_prefix.lower() == "diabetes"


def test_filler_looking_for_stripped():
    seg = make_segmenter()
    result = seg.segment("looking for myocardial")
    assert result.active_prefix.lower() == "myocardial"


def test_full_complex_query_segments_correctly():
    seg = make_segmenter()
    result = seg.segment("a study in gout US Sites only, Phase 2 and 3")
    assert result.active_prefix.lower() == "phase 2 and 3"
    normalized = seg.normalize_active_prefix(result.active_prefix)
    assert normalized.is_compound_phase is True
    assert len(normalized.compound_phases) == 2
