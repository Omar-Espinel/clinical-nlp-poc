"""Filter-only LLM extraction for clinical research queries."""

import json
import logging
from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict

from src.exceptions import ExtractionError
from src.llm_provider.base import LLMProvider

logger = logging.getLogger(__name__)


class FilterField(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: Optional[str]
    confidence: float


class StateFilter(BaseModel):
    model_config = ConfigDict(frozen=True)
    values: list[str]
    confidence: float
    is_region: bool


class ExtractedFilters(BaseModel):
    model_config = ConfigDict(frozen=True)
    investigator_name: FilterField
    site_name: FilterField
    city: FilterField
    state: StateFilter
    phase: FilterField
    raw_response_length: int


_REQUIRED_KEYS = frozenset(
    {"investigator_name", "site_name", "city", "state", "phase"}
)


class FilterExtractor:
    SYSTEM_PROMPT: ClassVar[str] = (
        "You extract structured filter fields from clinical research queries.\n"
        "Return JSON only. Fields:\n"
        "- investigator_name (string or null)\n"
        "- site_name (string or null)\n"
        "- city (string or null)\n"
        "- state (string or null) — US state or Canadian province name; null for multi-state regions\n"
        "- phase (string or null) — normalized\n"
        "- per-field confidence (float 0.0–1.0)\n"
        "\n"
        "CRITICAL RULES (do not skip):\n"
        "\n"
        "1. Do NOT extract medical conditions or diseases. They are handled separately. If the\n"
        "   query contains medical terms, ignore them — return null for any field they don't fit.\n"
        "\n"
        "2. State extraction — explicit only:\n"
        "   - Extract state ONLY from explicit state mentions (\"California\", \"in TX\", \"Texas trials\").\n"
        "   - Do NOT infer state from a city name. \"Kansas City\" is in Missouri, NOT Kansas.\n"
        "     \"Oklahoma City\" is in Oklahoma but you do not know that here — leave state=null.\n"
        "     \"New York\" alone is the city — return city=\"New York\", state=null. The downstream\n"
        "     geo normalizer resolves this.\n"
        "   - Do NOT infer state from an institution name. \"Massachusetts General Hospital\"\n"
        "     does not imply state=Massachusetts when a different city is specified.\n"
        "   - For multi-state regions (e.g., \"east coast\", \"midwest\"): state=null. The geo\n"
        "     normalizer expands these.\n"
        "\n"
        "3. Phase normalization (return EXACTLY one of these forms):\n"
        "   - \"phase iii\", \"p3\", \"phase-3\", \"pivotal\" → \"Phase 3\"\n"
        "   - \"first in human\", \"fih\" → \"Phase 1\"\n"
        "   - \"phase 1/2\", \"p1/2\", \"phase i/ii\" → \"Phase 1/2\"\n"
        "   - \"phase 2b\" → \"Phase 2b\"\n"
        "   - Generic phases: \"Phase 1\", \"Phase 2\", \"Phase 3\", \"Phase 4\"\n"
        "\n"
        "4. Investigator names: lastname-only is acceptable. Title prefixes (Dr., Prof.) stripped.\n"
        "\n"
        "5. Site names: full institution name as written. Do NOT split city out of site name.\n"
        "\n"
        "6. Confidence scoring: high (0.9+) for explicit unambiguous mentions. Lower for inferred.\n"
        "   Null fields have confidence 0.0.\n"
        "\n"
        "Output JSON schema (all fields required, value may be null):\n"
        "{\n"
        "  \"investigator_name\": {\"value\": ..., \"confidence\": ...},\n"
        "  \"site_name\":         {\"value\": ..., \"confidence\": ...},\n"
        "  \"city\":              {\"value\": ..., \"confidence\": ...},\n"
        "  \"state\":             {\"value\": ..., \"confidence\": ...},\n"
        "  \"phase\":             {\"value\": ..., \"confidence\": ...}\n"
        "}"
    )

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def extract(self, canonical_query: str) -> ExtractedFilters:
        """Call the LLM provider and return structured filter fields.

        Raises ExtractionError on JSON parse failure or missing required keys.
        Never logs query text or response content (HIPAA).
        """
        raw = self._provider.complete(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=canonical_query,
            max_tokens=512,
            temperature=0.0,
            json_mode=True,
        )

        parse_ok = False
        try:
            parsed = json.loads(raw)
            parse_ok = True
        except json.JSONDecodeError as e:
            logger.error(
                "Filter extraction JSON parse failed (provider=%s, response_length=%d)",
                self._provider.name,
                len(raw),
            )
            raise ExtractionError("Could not parse extraction response") from e
        finally:
            if not parse_ok:
                logger.debug(
                    "parse_success=False provider=%s", self._provider.name
                )

        result = self._validate(parsed, raw_response_length=len(raw))
        logger.info(
            "Filter extraction complete: provider=%s response_length=%d "
            "investigator_name_conf=%.2f site_name_conf=%.2f city_conf=%.2f "
            "state_conf=%.2f phase_conf=%.2f",
            self._provider.name,
            len(raw),
            result.investigator_name.confidence,
            result.site_name.confidence,
            result.city.confidence,
            result.state.confidence,
            result.phase.confidence,
        )
        return result

    def _validate(self, parsed: dict, raw_response_length: int) -> ExtractedFilters:
        """Build ExtractedFilters from the raw LLM dict.

        Silently drops medical_terms if the LLM accidentally included it (rule 1).
        Defends against the legacy "null" string leak that appeared in extractor.py.
        """
        if not isinstance(parsed, dict):
            raise ExtractionError("Could not parse extraction response")

        missing = _REQUIRED_KEYS - parsed.keys()
        if missing:
            raise ExtractionError("Could not parse extraction response")

        def _scalar(key: str) -> FilterField:
            raw_field = parsed[key]
            if not isinstance(raw_field, dict):
                raise ExtractionError("Could not parse extraction response")
            raw_val = raw_field.get("value")
            # Defend against literal "null" string (legacy bug in extractor.py)
            if isinstance(raw_val, str) and (
                raw_val.strip().lower() == "null" or raw_val.strip() == ""
            ):
                raw_val = None
            return FilterField(
                value=raw_val,
                confidence=float(raw_field.get("confidence", 0.0)),
            )

        raw_state = parsed["state"]
        if not isinstance(raw_state, dict):
            raise ExtractionError("Could not parse extraction response")

        state_val = raw_state.get("value")
        if isinstance(state_val, str) and (
            state_val.strip().lower() == "null" or state_val.strip() == ""
        ):
            state_val = None

        if state_val is None:
            state_filter = StateFilter(values=[], confidence=0.0, is_region=False)
        else:
            state_filter = StateFilter(
                values=[state_val],
                confidence=float(raw_state.get("confidence", 0.0)),
                is_region=False,
            )

        return ExtractedFilters(
            investigator_name=_scalar("investigator_name"),
            site_name=_scalar("site_name"),
            city=_scalar("city"),
            state=state_filter,
            phase=_scalar("phase"),
            raw_response_length=raw_response_length,
        )
