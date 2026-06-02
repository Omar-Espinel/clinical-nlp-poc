from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.autocomplete.index import AutocompleteSuggestion
    from src.autocomplete.segmenter import SegmentResult

from src.autocomplete.segmenter import ContextType

_TIER_WEIGHTS: dict[str, float] = {
    "prefix": 1.00,
    "alias": 0.93,
    "fuzzy": 0.85,
    "semantic": 0.80,
}

_CATEGORY_BOOSTS: dict[str, float] = {
    "snomed": 1.00,
    "phase": 1.05,
    "geo_city": 0.95,
    "geo_state": 0.90,
    "geo_region": 0.92,
}


def _specificity_boost(display: str, prefix: str) -> float:
    if len(prefix) > 5 and prefix.lower() in display.lower():
        return 1.05
    return 1.0


def _context_bonus(category: str, context_types: list) -> float:
    """Return a score multiplier based on what context is already committed.

    When the user has already typed a SNOMED term, phase and geo suggestions
    are boosted because they are the logical next filter. When phase or geo
    context is already present, suggestions of the same category get a
    smaller boost (rare but valid — e.g. adding a second geo refinement).
    """
    if ContextType.SNOMED in context_types:
        if category == "phase":
            return 1.15
        if category.startswith("geo"):
            return 1.10
    if ContextType.PHASE in context_types and category == "phase":
        return 1.05
    if ContextType.GEO in context_types and category.startswith("geo"):
        return 1.05
    return 1.0


def _compute_score(suggestion: "AutocompleteSuggestion", segment: "SegmentResult") -> float:
    return (
        suggestion.raw_score
        * _TIER_WEIGHTS.get(suggestion.match_type, 0.80)
        * _CATEGORY_BOOSTS.get(suggestion.category, 1.0)
        * _specificity_boost(suggestion.display, segment.active_prefix)
        * _context_bonus(suggestion.category, segment.context_types)
    )


def deduplicate(suggestions: list["AutocompleteSuggestion"]) -> list["AutocompleteSuggestion"]:
    """Remove duplicates in two passes.

    Pass 1 deduplicates by SNOMED concept_id keeping highest raw_score.
    Pass 2 deduplicates by normalized display string keeping highest raw_score.
    Pass 2 catches synonyms that resolve to different concept_ids but
    produce the same preferred_term display string.
    """
    best_by_code: dict[str, "AutocompleteSuggestion"] = {}
    no_code: list["AutocompleteSuggestion"] = []
    for s in suggestions:
        if s.snomed_code:
            existing = best_by_code.get(s.snomed_code)
            if existing is None or s.raw_score > existing.raw_score:
                best_by_code[s.snomed_code] = s
        else:
            no_code.append(s)

    seen_displays: dict[str, "AutocompleteSuggestion"] = {}
    for s in list(best_by_code.values()) + no_code:
        key = s.display.lower().strip()
        existing = seen_displays.get(key)
        if existing is None or s.raw_score > existing.raw_score:
            seen_displays[key] = s

    return list(seen_displays.values())


def rank_and_trim(
    suggestions: list["AutocompleteSuggestion"],
    segment: "SegmentResult",
    limit: int = 7,
) -> list["AutocompleteSuggestion"]:
    """Deduplicate, score, apply hierarchy collapse, return top limit results.

    Hierarchy collapse suppresses a parent term when a more specific child
    term containing the parent as a substring is already in the results and
    their raw relevance scores are within 0.05 of each other. This prevents
    redundant generic suggestions like 'diabetes mellitus' appearing alongside
    'type 2 diabetes mellitus' when the user is clearly being specific.

    The window compares raw_score, not the ranking score: the specificity
    boost that lifts a child above its parent must not also widen the gap
    enough to spare the parent from collapse — that would invert the intent.
    """
    deduped = deduplicate(suggestions)
    scored = [(s, _compute_score(s, segment)) for s in deduped]
    scored.sort(key=lambda x: x[1], reverse=True)

    final: list[tuple["AutocompleteSuggestion", float]] = []
    for candidate, candidate_score in scored:
        suppressed = any(
            candidate.display.lower() in kept.display.lower()
            and candidate.display.lower() != kept.display.lower()
            and abs(candidate.raw_score - kept.raw_score) < 0.05
            for kept, kept_score in final
        )
        if not suppressed:
            final.append((candidate, candidate_score))
        if len(final) >= limit:
            break

    return [s for s, _ in final]
