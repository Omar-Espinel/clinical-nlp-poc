"""Filter-only LLM extraction for clinical research queries."""

import json
import logging
from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.exceptions import ExtractionError
from src.llm_provider.base import LLMProvider
from src.normalizers.metric import MetricMatch, MetricFilterOutput, MetricFilterNormalizer

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
    metric_fields: dict[str, MetricFilterOutput] = Field(default_factory=dict)


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

    def __init__(
        self,
        provider: LLMProvider,
        known_metric_fields: Optional[frozenset[str]] = None,
    ) -> None:
        self._provider = provider
        self._known_metric_fields: Optional[frozenset[str]] = known_metric_fields

    def extract(
        self,
        canonical_query: str,
        metric_matches: Optional[list[MetricMatch]] = None,
    ) -> ExtractedFilters:
        """Call the LLM provider and return structured filter fields.

        Raises ExtractionError on JSON parse failure or missing required keys.
        Never logs query text or response content (HIPAA).

        metric_matches: if non-empty, appends a metric extraction section to the
        system prompt and parses metric_fields from the LLM response.
        """
        system = self.SYSTEM_PROMPT
        if metric_matches:
            system = system + "\n\n" + self._build_metric_section(metric_matches)

        raw = self._provider.complete(
            system_prompt=system,
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

        result = self._validate(parsed, raw_response_length=len(raw), metric_matches=metric_matches)
        logger.info(
            "Filter extraction complete: provider=%s response_length=%d "
            "investigator_name_conf=%.2f site_name_conf=%.2f city_conf=%.2f "
            "state_conf=%.2f phase_conf=%.2f metric_fields_count=%d",
            self._provider.name,
            len(raw),
            result.investigator_name.confidence,
            result.site_name.confidence,
            result.city.confidence,
            result.state.confidence,
            result.phase.confidence,
            len(result.metric_fields),
        )
        return result

    def _build_metric_section(self, metric_matches: list[MetricMatch]) -> str:
        """Build the metric extraction prompt section.

        Asserts every match's canonical_field is in the known allowlist (rev2 S6).
        Drops unknown fields silently with a count log.
        """
        known = self._known_metric_fields
        if known is not None:
            safe_matches = [m for m in metric_matches if m.canonical_field in known]
            if len(safe_matches) < len(metric_matches):
                logger.info(
                    "_build_metric_section: dropped unknown fields count=%d",
                    len(metric_matches) - len(safe_matches),
                )
        else:
            safe_matches = list(metric_matches)

        if not safe_matches:
            return ""

        # rev2 §3.4: include data_type per field so the LLM can distinguish numeric vs date.
        fields_block = "\n".join(
            f"  - Field key: {m.canonical_field} ({m.canonical_label}) — data_type={m.data_type}"
            for m in safe_matches
        )

        # Build the JSON example showing snake_case keys verbatim
        example_field = safe_matches[0].canonical_field
        example_json = (
            "{\n"
            '  "metric_fields": {\n'
            f'    "{example_field}": {{\n'
            '      "operator_text": "under",\n'
            '      "value": "5",\n'
            '      "value_end": null\n'
            "    }\n"
            "  }\n"
            "}"
        )

        return (
            "ADDITIONAL TASK — Metric fields detected:\n\n"
            "For each field below, extract from the query:\n\n"
            "Fields:\n"
            f"{fields_block}\n\n"
            'Respond with a JSON object containing a "metric_fields" key, using EXACTLY the\n'
            "snake_case field keys shown above. Example:\n\n"
            f"{example_json}\n\n"
            'Include "metric_fields" in your JSON response even if all values are null.\n'
            "Do NOT invent values not present in the query.\n"
            "Do NOT use camelCase keys.\n"
            'If the user expressed "No preference" for a field, emit operator_text="any" and value=null for that field.\n'
            'If data_type=date, value MUST be in MM/YYYY format (e.g. "01/2026"). '
            'Convert relative phrases like "since June 2025" to "06/2025". '
            "If you cannot determine an explicit month/year, emit value=null.\n"
            'When operator is "between", emit both value (lower bound) and value_end '
            "(upper bound) — same data_type. For all other operators, value_end MUST be null."
        )

    def _validate(
        self,
        parsed: dict,
        raw_response_length: int,
        metric_matches: Optional[list[MetricMatch]] = None,
    ) -> ExtractedFilters:
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

        # ── Parse metric_fields from LLM response ────────────────────────────
        metric_fields_dict: dict[str, MetricFilterOutput] = {}
        if metric_matches:
            raw_metric = parsed.get("metric_fields", {})
            if isinstance(raw_metric, dict):
                # Build a quick lookup: canonical_field → MetricMatch
                match_by_field: dict[str, MetricMatch] = {
                    m.canonical_field: m for m in metric_matches
                }
                # Determine the expected field set (from known allowlist or from matches)
                known = self._known_metric_fields
                expected_keys = (
                    set(match_by_field.keys()) if known is None
                    else set(match_by_field.keys()) & known
                )

                unknown_field_count = 0
                for field_key, raw_entry in raw_metric.items():
                    if field_key not in expected_keys:
                        unknown_field_count += 1
                        continue
                    if not isinstance(raw_entry, dict):
                        continue

                    match = match_by_field.get(field_key)
                    if match is None:
                        continue

                    data_type = match.data_type
                    operator_text = raw_entry.get("operator_text") or ""
                    operator = MetricFilterNormalizer.normalize_operator(operator_text)
                    # If LLM returned "any" but match has an implied operator, use implied
                    if operator == "any" and match.implied_operator != "any":
                        operator = match.implied_operator

                    raw_value = raw_entry.get("value")
                    # LLM may return numeric primitives — coerce to str for normalize_value.
                    if raw_value is not None and not isinstance(raw_value, str):
                        raw_value = str(raw_value)
                    value = MetricFilterNormalizer.normalize_value(
                        raw_value, data_type, operator
                    )

                    # rev2 §3.3b: extract value_end too (used only for operator='between').
                    raw_value_end = raw_entry.get("value_end")
                    if operator == "between" and raw_value_end is not None:
                        if not isinstance(raw_value_end, str):
                            raw_value_end = str(raw_value_end)
                        value_end = MetricFilterNormalizer.normalize_value(
                            raw_value_end, data_type, operator
                        )
                    else:
                        value_end = None

                    # unit: server-derived — for v1 all 12 fields are numeric without
                    # specific units; set to None per spec §4 rev2
                    server_unit = None

                    try:
                        mfo = MetricFilterOutput(
                            field=field_key,
                            canonical_label=match.canonical_label,
                            operator=operator,
                            data_type=data_type,
                            value=value,
                            value_end=value_end,
                            original_text=match.matched_text,
                            confidence=match.confidence,
                            unit=server_unit,
                        )
                        metric_fields_dict[field_key] = mfo
                    except ValidationError as exc:
                        logger.info(
                            "_validate metric_fields: skipped field error_count=%d",
                            exc.error_count(),
                            # NOT logged: str(exc), repr(exc), field values
                        )
                        continue

                if unknown_field_count:
                    logger.info(
                        "_validate metric_fields: skipped unknown fields count=%d",
                        unknown_field_count,
                    )

        return ExtractedFilters(
            investigator_name=_scalar("investigator_name"),
            site_name=_scalar("site_name"),
            city=_scalar("city"),
            state=state_filter,
            phase=_scalar("phase"),
            raw_response_length=raw_response_length,
            metric_fields=metric_fields_dict,
        )
