"""Orchestrates the full Clinical NLP pipeline."""

import logging
import time
from pathlib import Path
from typing import Optional

from src.preprocessor import Preprocessor, PreprocessedInput, PreprocessorError
from src.extractor import Extractor, ExtractionResult, ExtractionError
from src.snomed_resolver import SNOMEDResolver, SNOMEDMatch
from src.geo_normalizer import GeoNormalizer, GeoResult
from src.assembler import ResponseAssembler, NLPOutput

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class NLPPipeline:
    """Single entry point for the full Clinical Research NLP pipeline."""

    def __init__(
        self,
        groq_api_key: str,
        snomed_csv_path: Optional[str] = None,
        geo_json_path: Optional[str] = None,
    ) -> None:
        """Initialize all pipeline components.

        Default data file paths resolve relative to the project root.
        """
        csv_path = snomed_csv_path or str(DATA_DIR / "snomed_clinical_trials.csv")
        json_path = geo_json_path or str(DATA_DIR / "geo_canonical.json")

        self.preprocessor = Preprocessor()
        self.extractor = Extractor(api_key=groq_api_key)
        self.resolver = SNOMEDResolver(csv_path=csv_path)
        self.normalizer = GeoNormalizer(json_path=json_path)
        self.assembler = ResponseAssembler()

        logger.info("NLPPipeline initialized.")

    def run(self, raw_query: str) -> NLPOutput:
        """Run the full pipeline on a raw query string.

        Raises:
            PreprocessorError: if the query is invalid or contains no clinical content.
            ExtractionError: if the LLM call fails.
        """
        start_time = time.time()

        preprocessed: PreprocessedInput = self.preprocessor.process(raw_query)
        # Log only length, never raw query text (HIPAA: avoid logging potential PHI)
        logger.info("Preprocessed query: %d chars", preprocessed.char_count)

        extraction: ExtractionResult = self.extractor.extract(preprocessed)

        # ── Clinical intent validation ────────────────────────────────────
        # Reject queries that yield zero medical terms AND zero structured filters.
        # This is the primary defense against non-clinical / data-dump queries that
        # slip past the pattern check (e.g. "give me all results", "hello", etc.).
        has_medical_terms = len(extraction.medical_terms) > 0
        has_filters = any([
            extraction.investigator_name.value,
            extraction.site_name.value,
            extraction.city.value,
            extraction.state.value,
            extraction.phase.value,
        ])
        if not has_medical_terms and not has_filters:
            raise PreprocessorError(
                "No clinical content found. Please enter a query about a medical "
                "condition, investigator, research site, location, or study phase."
            )

        # Log only counts, never the extracted values themselves (HIPAA)
        filter_count = sum(1 for v in [
            extraction.investigator_name.value,
            extraction.site_name.value,
            extraction.city.value,
            extraction.state.value,
            extraction.phase.value,
        ] if v)
        logger.info(
            "Extracted %d medical terms, %d non-null filters",
            len(extraction.medical_terms),
            filter_count,
        )

        snomed_matches: list[SNOMEDMatch] = []
        for term in extraction.medical_terms:
            match = self.resolver.resolve(term.term, term.negated)
            if match:
                snomed_matches.append(match)

        geo: GeoResult = self.normalizer.normalize(
            city=extraction.city.value,
            state=extraction.state.value,
        )
        # Log geo confidence only, not the resolved place name
        logger.info("Geo normalization confidence: %.2f", geo.confidence)

        output = self.assembler.assemble(
            extraction=extraction,
            snomed_matches=snomed_matches,
            geo=geo,
            start_time=start_time,
        )
        logger.info(
            "Pipeline complete: %d SNOMED terms, %dms",
            output.metadata.total_snomed_matches,
            output.metadata.processing_time_ms,
        )
        return output
