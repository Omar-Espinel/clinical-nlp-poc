"""Geography normalization for city and state values extracted from clinical queries."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process as rf_process

logger = logging.getLogger(__name__)


@dataclass
class GeoResult:
    """Normalized geography result.

    states is always a list:
      - empty list  → no state resolved
      - 1 element   → specific city or single-state metro region
      - 2+ elements → multi-state geographic region (east coast, midwest, etc.)
    """
    city: Optional[str]
    states: list[str]
    country: Optional[str]
    confidence: float
    is_region: bool
    original_city: Optional[str]
    original_state: Optional[str]


class GeoNormalizer:
    """Normalizes raw city/state text to canonical geographic names."""

    def __init__(self, json_path: str) -> None:
        """Load geo canonical JSON and build lookup indexes."""
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._cities: dict[str, dict] = data.get("cities", {})
        self._states: dict[str, str] = data.get("states", {})
        self._regions: dict[str, dict] = data.get("regions", {})
        logger.info(
            "GeoNormalizer loaded: %d cities, %d states, %d regions.",
            len(self._cities), len(self._states), len(self._regions),
        )

    def normalize(self, city: Optional[str], state: Optional[str]) -> GeoResult:
        """Normalize raw city and state strings to canonical values.

        For multi-state regions (east coast, west coast, midwest, etc.) the
        returned GeoResult.states list contains all covered states.
        For specific cities or single-state metro areas it contains one state.
        Returns states=[] and confidence=0.0 if nothing can be resolved.
        """
        if city is None and state is None:
            return GeoResult(
                city=None, states=[], country=None,
                confidence=0.0, is_region=False,
                original_city=None, original_state=None,
            )

        original_city = city
        original_state = state

        city_entry: Optional[dict] = None
        is_region = False
        city_confidence = 0.0

        if city:
            city_entry = self._lookup_city(city)
            if city_entry:
                is_region = city_entry.get("type") == "region"
                city_confidence = 0.95 if is_region else 1.0
            else:
                city_entry = self._fuzzy_city_match(city)
                if city_entry:
                    city_confidence = city_entry.get("_fuzzy_confidence", 0.80)
                    is_region = city_entry.get("type") == "region"

        resolved_city: Optional[str] = None
        resolved_states: list[str] = []
        resolved_country: Optional[str] = None
        confidence = 0.0

        if city_entry:
            resolved_country = city_entry.get("country")
            confidence = city_confidence

            if is_region and "region_states" in city_entry:
                # Multi-state region — return full state list, no primary city
                resolved_city = None
                resolved_states = list(city_entry["region_states"])
            elif is_region:
                # Single-state metro region (bay area, research triangle, etc.)
                resolved_city = city_entry.get("primary_city")
                s = city_entry.get("state")
                if s:
                    resolved_states = [s]
            else:
                # Specific city
                resolved_city = city_entry.get("canonical")
                s = city_entry.get("state")
                if s:
                    resolved_states = [s]

        # State override / inference from explicit state input
        if state:
            looked_up = self._lookup_state(state)
            if looked_up:
                if not is_region:
                    # Explicit state always wins for non-region results
                    resolved_states = [looked_up]
                elif not resolved_states:
                    resolved_states = [looked_up]
                # For multi-state regions, don't override region_states with a
                # single state — the region definition is more informative.
                if not city_entry:
                    # State-only query
                    confidence = 0.85 if len(state.strip()) > 2 else 0.70
                else:
                    confidence = city_confidence

        if resolved_city is None and not resolved_states:
            return GeoResult(
                city=None, states=[], country=None,
                confidence=0.0, is_region=False,
                original_city=original_city, original_state=original_state,
            )

        return GeoResult(
            city=resolved_city,
            states=resolved_states,
            country=resolved_country,
            confidence=round(confidence, 4),
            is_region=is_region,
            original_city=original_city,
            original_state=original_state,
        )

    def _lookup_city(self, city: str) -> Optional[dict]:
        """Exact lookup in cities dict, then regions dict."""
        key = city.lower().strip()
        return self._cities.get(key) or self._regions.get(key)

    def _fuzzy_city_match(self, city: str) -> Optional[dict]:
        """Fuzzy match against cities dict keys using rapidfuzz token_sort_ratio."""
        all_keys = list(self._cities.keys())
        result = rf_process.extractOne(
            city.lower().strip(),
            all_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=82,
        )
        if result is None:
            return None
        matched_key, score, _ = result
        entry = dict(self._cities[matched_key])
        entry["_fuzzy_confidence"] = round((score / 100.0) * 0.90, 4)
        return entry

    def _lookup_state(self, state: str) -> Optional[str]:
        """Look up a state abbreviation or full name, returning the canonical full name."""
        for key in (state.strip().upper(), state.strip().lower(), state.strip().title()):
            if key in self._states:
                return self._states[key]
        return None
