"""Groq LLM extraction of medical terms and structured filters from clinical queries."""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import groq

from src.preprocessor import PreprocessedInput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical research query parser specialized in
extracting structured information from natural language queries about clinical trials.

EXTRACTION RULES:
1. Extract ALL medical/clinical terms that could map to SNOMED CT concepts. Include: diseases, conditions,
   procedures, substances, symptoms, body parts in clinical context.

2. Expand ALL medical abbreviations before including them:
   T2DM or T2D → type 2 diabetes mellitus
   T1DM or T1D → type 1 diabetes mellitus
   NSCLC → non-small cell lung carcinoma
   SCLC → small cell lung carcinoma
   HCC → hepatocellular carcinoma
   CRC → colorectal carcinoma
   RA → rheumatoid arthritis
   COPD → chronic obstructive pulmonary disease
   CKD → chronic kidney disease
   HF or CHF → heart failure
   HTN → hypertension
   MI → myocardial infarction
   AFib or AF → atrial fibrillation
   AFL → atrial flutter
   PAD → peripheral arterial disease
   DVT → deep vein thrombosis
   PE → pulmonary embolism
   NHL → non-hodgkin lymphoma
   HL → hodgkin lymphoma
   AML → acute myeloid leukemia
   CML → chronic myeloid leukemia
   ALL → acute lymphoblastic leukemia
   CLL → chronic lymphocytic leukemia
   MM → multiple myeloma
   NASH → non-alcoholic steatohepatitis
   NAFLD → non-alcoholic fatty liver disease
   IBD → inflammatory bowel disease
   UC → ulcerative colitis
   CD → crohn disease
   PSA → prostate specific antigen measurement
   AD → alzheimer disease
   PD → parkinson disease
   ALS → amyotrophic lateral sclerosis
   SLE → systemic lupus erythematosus
   SSc → systemic sclerosis
   AS → ankylosing spondylitis
   PsA → psoriatic arthritis
   GBM → glioblastoma multiforme
   MS → multiple sclerosis (only when clearly medical context)

   IMPORTANT: Only expand abbreviations when context clearly indicates a medical meaning. If ambiguous, do not expand.

3. Mark negated: true if query contains "no", "not", "without", "excluding", "except", "non-" before the term.

4. Investigator names: Last name minimum.
   "Dr. Sarah Johnson" → "Sarah Johnson" (no title in value).
   "Dr. Smith" → "Smith".
   "J. Rodriguez" → "J. Rodriguez".

5. Cities: extract raw text only. Do not normalize.

6. States: extract raw text only, including abbreviations.

7. Phase normalization:
   "phase 3", "p3", "phase iii", "phase-3" → "Phase 3"
   "phase 1/2", "p1/2", "phase i/ii" → "Phase 1/2"
   "phase 2b" → "Phase 2b"
   "pivotal" → "Phase 3"
   "first in human" or "fih" → "Phase 1"
   "phase 1" or "phase i" or "p1" → "Phase 1"
   "phase 2" or "phase ii" or "p2" → "Phase 2"
   "phase 4" or "phase iv" or "p4" → "Phase 4"

8. Site names: hospitals, clinics, universities, cancer centers, research institutions only.

9. Missing fields: null value, 0.0 confidence.

10. Confidence: 0.95-1.0 explicit, 0.80-0.94 implied, 0.60-0.79 uncertain, below 0.60 very uncertain.

11. English only. US and Canada geography only.

OUTPUT: Valid JSON only. No markdown fences. No explanation. No text before or after the JSON object.

{
  "medical_terms": [
    {"term": "string", "negated": boolean, "confidence": number}
  ],
  "filters": {
    "investigator_name": {"value": "string or null", "confidence": number},
    "site_name": {"value": "string or null", "confidence": number},
    "city": {"value": "string or null", "confidence": number},
    "state": {"value": "string or null", "confidence": number},
    "phase": {"value": "string or null", "confidence": number}
  }
}"""


@dataclass
class MedicalTerm:
    """A medical/clinical term extracted from a query."""
    term: str
    negated: bool
    confidence: float


@dataclass
class FilterField:
    """A structured filter field with value and confidence."""
    value: Optional[str]
    confidence: float


@dataclass
class ExtractionResult:
    """Full extraction result from the LLM."""
    medical_terms: list[MedicalTerm] = field(default_factory=list)
    investigator_name: FilterField = field(default_factory=lambda: FilterField(None, 0.0))
    site_name: FilterField = field(default_factory=lambda: FilterField(None, 0.0))
    city: FilterField = field(default_factory=lambda: FilterField(None, 0.0))
    state: FilterField = field(default_factory=lambda: FilterField(None, 0.0))
    phase: FilterField = field(default_factory=lambda: FilterField(None, 0.0))
    raw_response: str = ""


class ExtractionError(Exception):
    """Raised when LLM extraction fails."""
    pass


class Extractor:
    """Calls Groq LLM to extract medical terms and filters from a preprocessed query."""

    MODEL = "llama-3.1-8b-instant"
    TEMPERATURE = 0.0
    MAX_TOKENS = 1024
    TIMEOUT = 10.0
    MAX_RETRIES = 3

    def __init__(self, api_key: str) -> None:
        self._client = groq.Groq(api_key=api_key)

    def extract(self, preprocessed: PreprocessedInput) -> ExtractionResult:
        """Run Groq extraction on a preprocessed input.

        Raises ExtractionError on parse failure or rate limiting.
        """
        prompt = preprocessed.text
        raw = self._call_groq(prompt)
        result = self._parse_response(raw)
        result.raw_response = raw
        return result

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API with exponential backoff retries.

        Raises ExtractionError on rate limit or exhausted retries.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.MODEL,
                    temperature=self.TEMPERATURE,
                    max_tokens=self.MAX_TOKENS,
                    timeout=self.TIMEOUT,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except groq.RateLimitError as exc:
                raise ExtractionError(
                    "Rate limit reached. Please wait 60 seconds and try again."
                ) from exc
            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning("Groq attempt %d failed: %s. Retrying in %ds.", attempt + 1, exc, wait)
                time.sleep(wait)

        raise ExtractionError(
            "Failed to parse LLM response as valid JSON. "
            "Please rephrase your query and try again."
        ) from last_error

    def _parse_response(self, response: str) -> ExtractionResult:
        """Parse JSON response from the LLM into an ExtractionResult.

        Raises ExtractionError if JSON is invalid or required keys are missing.
        """
        text = response.strip()
        # Strip markdown fences if the model added them despite instructions
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # Log error type and response length only — never log response content (may echo query)
            logger.error("JSON decode error: %s (response length: %d chars)", exc, len(response))
            raise ExtractionError(
                "Failed to parse LLM response as valid JSON. "
                "Please rephrase your query and try again."
            ) from exc

        if not self._validate_json(data):
            raise ExtractionError(
                "Failed to parse LLM response as valid JSON. "
                "Please rephrase your query and try again."
            )

        medical_terms = [
            MedicalTerm(
                term=t["term"],
                negated=bool(t["negated"]),
                confidence=float(t["confidence"]),
            )
            for t in data["medical_terms"]
        ]

        filters = data["filters"]

        def _field(key: str) -> FilterField:
            f = filters.get(key, {})
            return FilterField(
                value=f.get("value") or None,
                confidence=float(f.get("confidence", 0.0)),
            )

        return ExtractionResult(
            medical_terms=medical_terms,
            investigator_name=_field("investigator_name"),
            site_name=_field("site_name"),
            city=_field("city"),
            state=_field("state"),
            phase=_field("phase"),
        )

    def _validate_json(self, data: dict) -> bool:
        """Return True if the parsed dict has all required keys and structure."""
        if not isinstance(data, dict):
            return False
        if "medical_terms" not in data or not isinstance(data["medical_terms"], list):
            return False
        if "filters" not in data or not isinstance(data["filters"], dict):
            return False
        required_filter_keys = {"investigator_name", "site_name", "city", "state", "phase"}
        if not required_filter_keys.issubset(data["filters"].keys()):
            return False
        for f_key in required_filter_keys:
            fv = data["filters"][f_key]
            if not isinstance(fv, dict) or "value" not in fv or "confidence" not in fv:
                return False
        for term in data["medical_terms"]:
            if not isinstance(term, dict):
                return False
            if not {"term", "negated", "confidence"}.issubset(term.keys()):
                return False
        return True
