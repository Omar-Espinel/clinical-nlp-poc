"""HybridCascadeStrategy: 4-stage cascade SNOMED search (exact → synonym → fuzzy → semantic)."""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process as rf_process

from src.snomed_search.base import SNOMEDMatch, SNOMEDSearchStrategy

logger = logging.getLogger(__name__)

# Canonical location per architect decision D4. snomed_resolver.py re-exports from here.
ALIAS_DICTIONARY: dict[str, str] = {
    "diabetes type 2": "type 2 diabetes mellitus",
    "type 2 diabetes": "type 2 diabetes mellitus",
    "type ii diabetes": "type 2 diabetes mellitus",
    "adult onset diabetes": "type 2 diabetes mellitus",
    "insulin resistance": "type 2 diabetes mellitus",
    "t2 diabetes": "type 2 diabetes mellitus",
    "type 1 diabetes": "type 1 diabetes mellitus",
    "type i diabetes": "type 1 diabetes mellitus",
    "juvenile diabetes": "type 1 diabetes mellitus",
    "blood sugar": "blood glucose measurement",
    "blood sugar management": "glucose monitoring",
    "glucose management": "glucose monitoring",
    "glucose control": "glucose monitoring",
    "sugar control": "glucose monitoring",
    "heart attack": "myocardial infarction",
    "congestive heart failure": "heart failure",
    "high blood pressure": "hypertension",
    "elevated blood pressure": "hypertension",
    "stroke": "cerebrovascular accident",
    "brain stroke": "cerebrovascular accident",
    "irregular heartbeat": "atrial fibrillation",
    "a fib": "atrial fibrillation",
    "afib": "atrial fibrillation",
    "blood clot": "thrombosis",
    "leg clot": "deep vein thrombosis",
    "arterial disease": "peripheral arterial disease",
    "cancer": "malignant neoplasm",
    "tumor": "neoplasm",
    "breast cancer": "malignant neoplasm of breast",
    "lung cancer": "malignant neoplasm of lung",
    "colon cancer": "malignant neoplasm of colon",
    "colorectal cancer": "malignant neoplasm of colon",
    "liver cancer": "hepatocellular carcinoma",
    "pancreatic cancer": "malignant neoplasm of pancreas",
    "prostate cancer": "malignant neoplasm of prostate",
    "ovarian cancer": "malignant neoplasm of ovary",
    "cervical cancer": "malignant neoplasm of cervix uteri",
    "brain cancer": "malignant neoplasm of brain",
    "brain tumor": "neoplasm of brain",
    "skin cancer": "malignant melanoma",
    "melanoma": "malignant melanoma",
    "blood cancer": "leukemia",
    "bone cancer": "malignant neoplasm of bone",
    "kidney cancer": "malignant neoplasm of kidney",
    "bladder cancer": "malignant neoplasm of urinary bladder",
    "thyroid cancer": "malignant neoplasm of thyroid gland",
    "stomach cancer": "malignant neoplasm of stomach",
    "esophageal cancer": "malignant neoplasm of esophagus",
    "lymphoma": "malignant lymphoma",
    "alzheimers": "alzheimer disease",
    "alzheimer's": "alzheimer disease",
    "parkinsons": "parkinson disease",
    "parkinson's": "parkinson disease",
    "multiple sclerosis": "multiple sclerosis",
    "als": "amyotrophic lateral sclerosis",
    "lou gehrig disease": "amyotrophic lateral sclerosis",
    "seizures": "epilepsy",
    "copd": "chronic obstructive pulmonary disease",
    "emphysema": "chronic obstructive pulmonary disease",
    "lung disease": "lung disorder",
    "pulmonary fibrosis": "idiopathic pulmonary fibrosis",
    "chemo": "chemotherapy",
    "bone marrow transplant": "stem cell transplant",
    "radiation": "radiation therapy",
    "surgery": "surgical procedure",
    "biopsy": "biopsy procedure",
    "blood test": "blood specimen collection",
    "mri": "magnetic resonance imaging",
    "ct scan": "computed tomography",
    "pet scan": "positron emission tomography",
    "rheumatoid arthritis": "rheumatoid arthritis",
    "lupus": "systemic lupus erythematosus",
    "ibd": "inflammatory bowel disease",
    "crohns": "crohn disease",
    "crohn's": "crohn disease",
    "colitis": "ulcerative colitis",
    "psoriatic arthritis": "psoriatic arthritis",
    "hiv": "human immunodeficiency virus infection",
    "aids": "acquired immunodeficiency syndrome",
    "hepatitis b": "hepatitis b",
    "hepatitis c": "hepatitis c",
    "hbv": "hepatitis b",
    "hcv": "hepatitis c",
    "tuberculosis": "tuberculosis",
    "covid": "covid-19",
    "covid-19": "covid-19",
    "coronavirus": "covid-19",
}

FUZZY_CUTOFF_DEFAULT = 88
SEMANTIC_THRESHOLD_DEFAULT = 0.82
EMBEDDING_MODEL_DEFAULT = "all-MiniLM-L6-v2"
MIN_RESIDUAL_LEN = 2
MAX_NGRAM_WINDOW = 4


class HybridCascadeStrategy:
    """
    Hybrid cascade: exact → synonym → fuzzy → semantic.
    Production default. Ports existing snomed_resolver.py behavior.

    Candidate extraction:
      Stages 1+2 (exact/synonym): enumerate all 1..MAX_NGRAM_WINDOW token windows.
      Stages 3+4 (fuzzy/semantic): operate ONLY on character spans NOT already matched
        (residual spans). This prevents fuzzy/semantic from re-matching terms that
        already have a higher-quality exact/synonym hit.

    All instance attributes after __init__ are read-only:
      - _exact_index, _synonym_index, _alias_dict: dict lookups, no mutation
      - _embedder: SentenceTransformer.encode() is documented thread-safe
      - _collection (ChromaDB EphemeralClient): query() is thread-safe per ChromaDB docs
      - _np_embeddings: numpy array, read-only after construction
      - _np_terms: list, read-only after construction
      - _fuzzy_cutoff, _semantic_threshold: primitive floats

    No locks required. Multiple concurrent search() calls are safe.
    """

    name = "hybrid_cascade"

    def __init__(
        self,
        dictionary_path: str,
        fuzzy_cutoff: int = FUZZY_CUTOFF_DEFAULT,
        semantic_threshold: float = SEMANTIC_THRESHOLD_DEFAULT,
        embedding_model: str = EMBEDDING_MODEL_DEFAULT,
        **kwargs,
    ) -> None:
        self._fuzzy_cutoff = fuzzy_cutoff
        self._semantic_threshold = semantic_threshold
        self._exact_index: dict[str, dict] = {}
        self._synonym_index: dict[str, dict] = {}
        self._alias_dict: dict[str, str] = {}
        self.semantic_available: bool = False
        self._collection = None
        self._embedder = None
        self._np_embeddings = None
        self._np_terms: list[str] = []

        self._load_csv(dictionary_path)
        self._validate_aliases()
        self._build_semantic_index(embedding_model)
        self._ready = True

    def _load_csv(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for _, row in df.iterrows():
            concept_id = row["concept_id"].strip()
            preferred_term = row["preferred_term"].strip().lower()
            synonyms_raw = row["synonyms"].strip()
            record = {"concept_id": concept_id, "preferred_term": preferred_term}
            self._exact_index[preferred_term] = record
            for syn in synonyms_raw.split("|"):
                syn_clean = syn.strip().lower()
                if syn_clean:
                    self._synonym_index[syn_clean] = record
        logger.info(
            "HybridCascadeStrategy: loaded %d terms, %d synonyms",
            len(self._exact_index), len(self._synonym_index),
        )

    def _validate_aliases(self) -> None:
        for alias, target in ALIAS_DICTIONARY.items():
            alias_key = alias.lower().strip()
            target_key = target.lower().strip()
            if target_key in self._exact_index:
                self._alias_dict[alias_key] = target_key
            else:
                logger.warning(
                    "HybridCascadeStrategy: alias target not in CSV — alias skipped "
                    "(alias_len=%d, target_len=%d)",
                    len(alias_key), len(target_key),
                )

    def _build_semantic_index(self, embedding_model: str) -> None:
        """Tries ChromaDB first; falls back to numpy cosine similarity.
        Never raises. Sets self.semantic_available = True on success of either backend.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(embedding_model)
        except Exception as exc:
            self.semantic_available = False
            logger.warning(
                "HybridCascadeStrategy: sentence-transformers unavailable — "
                "semantic disabled (%s)", type(exc).__name__,
            )
            return

        terms = list(self._exact_index.keys())

        try:
            import chromadb
            client = chromadb.EphemeralClient()
            self._collection = client.create_collection(
                name="snomed_clinical_trials_v1",
                metadata={"hnsw:space": "cosine"},
            )
            ids = [
                self._exact_index[t]["concept_id"] + "_" + str(i)
                for i, t in enumerate(terms)
            ]
            embeddings = self._embedder.encode(terms, show_progress_bar=False).tolist()
            self._collection.add(documents=terms, embeddings=embeddings, ids=ids)
            self.semantic_available = True
            self._np_embeddings = None
            logger.info(
                "HybridCascadeStrategy: ChromaDB index built (%d terms)", len(terms),
            )
            return
        except Exception as exc:
            logger.warning(
                "HybridCascadeStrategy: ChromaDB unavailable (%s) — numpy fallback",
                type(exc).__name__,
            )

        try:
            import numpy as np
            raw_embeddings = self._embedder.encode(terms, show_progress_bar=False)
            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._np_embeddings = (raw_embeddings / norms).astype("float32")
            self._np_terms = terms
            self.semantic_available = True
            logger.info(
                "HybridCascadeStrategy: numpy index built (%d terms)", len(terms),
            )
        except Exception as exc:
            self.semantic_available = False
            logger.warning(
                "HybridCascadeStrategy: numpy fallback also failed (%s) — "
                "semantic disabled", type(exc).__name__,
            )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def search(self, query: str) -> list[SNOMEDMatch]:
        """
        Candidate extraction (documented per SNOMEDSearchStrategy contract):
          Stage 1+2: enumerate all 1..MAX_NGRAM_WINDOW token windows over query.
          Stage 3:   rapidfuzz token_sort_ratio on residual character spans.
          Stage 4:   sentence-transformer cosine similarity on still-residual spans.
        Returns ALL candidates from stages 1-4; does NOT pre-filter by MIN_CONFIDENCE.
        """
        hits: list[SNOMEDMatch] = []

        exact_syn_hits = self._exact_synonym_pass(query)
        hits.extend(exact_syn_hits)

        after_stage_2 = self._compute_residual_spans(query, hits)
        fuzzy_hits = self._fuzzy_pass(query, after_stage_2)
        hits.extend(fuzzy_hits)

        after_stage_3 = self._compute_residual_spans(query, hits)
        semantic_hits = self._semantic_pass(query, after_stage_3)
        hits.extend(semantic_hits)

        return self._dedup_longest_match(hits)

    def health_check(self) -> dict:
        return {
            "ready": self._ready,
            "name": self.name,
            "dictionary_size": len(self._exact_index),
            "synonym_size": len(self._synonym_index),
            "semantic_available": self.semantic_available,
        }

    def get_top_neighbors(
        self,
        query: str,
        n: int = 15,
        low_threshold: float = 0.42,
    ) -> list[SNOMEDMatch]:
        """Return up to n SNOMEDMatch objects whose cosine similarity to query
        falls at or above low_threshold.

        Called by EmbeddingAmbiguityGate (Layer 2). NOT part of SNOMEDSearchStrategy
        Protocol — detected via duck-typing (hasattr).

        Returns [] if semantic is unavailable. Emits one INFO log on first call
        when unavailable (flag suppresses repeated logging).
        Thread-safe: uses same read-only embedder/index as search().
        HIPAA: query text NOT logged.
        """
        if not self.semantic_available or self._embedder is None:
            if not getattr(self, "_top_neighbors_unavailable_logged", False):
                logger.info(
                    "get_top_neighbors: semantic unavailable — returning [] "
                    "(this message logged once per strategy instance)"
                )
                self._top_neighbors_unavailable_logged = True
            return []

        try:
            query_vec = self._embedder.encode([query], show_progress_bar=False)

            if self._collection is not None:
                embedding = query_vec.tolist()
                results = self._collection.query(
                    query_embeddings=embedding,
                    n_results=min(n, len(self._exact_index)),
                    include=["documents", "distances"],
                )
                if not results["documents"] or not results["documents"][0]:
                    return []
                matches: list[SNOMEDMatch] = []
                for doc, dist in zip(
                    results["documents"][0], results["distances"][0]
                ):
                    similarity = 1.0 - dist
                    if similarity < low_threshold:
                        continue
                    record = self._exact_index.get(doc)
                    if record is None:
                        continue
                    matches.append(SNOMEDMatch(
                        code=record["concept_id"],
                        display=record["preferred_term"],
                        match_type="semantic",
                        confidence=round(similarity, 4),
                        original_text=query,
                        span=(0, len(query)),
                        negated=False,
                    ))
                return matches

            if self._np_embeddings is not None and self._np_terms:
                import numpy as np
                norm = np.linalg.norm(query_vec)
                if norm == 0:
                    return []
                query_norm = (query_vec / norm).astype("float32")
                similarities = self._np_embeddings @ query_norm.T
                sims_flat = similarities.ravel()
                # Get indices sorted by similarity descending
                top_indices = int(np.argsort(sims_flat)[::-1].ravel()[0].__class__(0))
                sorted_indices = list(np.argsort(sims_flat)[::-1])
                matches = []
                for idx in sorted_indices:
                    sim = float(sims_flat[idx])
                    if sim < low_threshold:
                        break  # sorted descending — no need to continue
                    if len(matches) >= n:
                        break
                    term = self._np_terms[idx]
                    record = self._exact_index.get(term)
                    if record is None:
                        continue
                    matches.append(SNOMEDMatch(
                        code=record["concept_id"],
                        display=record["preferred_term"],
                        match_type="semantic",
                        confidence=round(sim, 4),
                        original_text=query,
                        span=(0, len(query)),
                        negated=False,
                    ))
                return matches

        except Exception as exc:
            logger.warning(
                "get_top_neighbors: exception (%s) — returning []",
                type(exc).__name__,
            )
        return []

    # -------------------------------------------------------------------------
    # Stage 1+2: exact + synonym via n-gram windows
    # -------------------------------------------------------------------------

    def _exact_synonym_pass(self, query: str) -> list[SNOMEDMatch]:
        """
        Tokenize query with character offsets.
        Enumerate all 1..MAX_NGRAM_WINDOW token windows.
        For each window: exact_index lookup, then alias lookup, then synonym_index lookup.
        Add match on first hit per window; do NOT add both exact and synonym for same window.
        """
        tokens = self._tokenize_with_offsets(query)
        hits: list[SNOMEDMatch] = []

        for n in range(MAX_NGRAM_WINDOW, 0, -1):
            for i in range(len(tokens) - n + 1):
                window = tokens[i: i + n]
                window_text = " ".join(t[0] for t in window).lower()
                span_start = window[0][1]
                span_end = window[-1][2]

                if window_text in self._exact_index:
                    record = self._exact_index[window_text]
                    hits.append(SNOMEDMatch(
                        code=record["concept_id"],
                        display=record["preferred_term"],
                        match_type="exact",
                        confidence=0.99,
                        original_text=window_text,
                        span=(span_start, span_end),
                        negated=False,
                    ))
                    continue

                alias_target = self._alias_dict.get(window_text)
                if alias_target:
                    record = self._exact_index[alias_target]
                    hits.append(SNOMEDMatch(
                        code=record["concept_id"],
                        display=record["preferred_term"],
                        match_type="synonym",
                        confidence=0.97,
                        original_text=window_text,
                        span=(span_start, span_end),
                        negated=False,
                    ))
                    continue

                if window_text in self._synonym_index:
                    record = self._synonym_index[window_text]
                    hits.append(SNOMEDMatch(
                        code=record["concept_id"],
                        display=record["preferred_term"],
                        match_type="synonym",
                        confidence=0.95,
                        original_text=window_text,
                        span=(span_start, span_end),
                        negated=False,
                    ))

        return hits

    def _tokenize_with_offsets(self, query: str) -> list[tuple[str, int, int]]:
        """
        Returns list of (token_text, char_start, char_end_exclusive).
        Splits on whitespace sequences. Preserves original character offsets in query.
        Does NOT lowercase tokens here; lowercasing happens at comparison site.

        M1 FIX: replaced query.split() + query.index(token, cursor) with
        re.finditer(r'\\S+', query). The old approach fails for tokens with attached
        punctuation or repeated substrings. re.finditer yields offsets directly, O(n).
        """
        return [
            (m.group(), m.start(), m.end())
            for m in re.finditer(r'\S+', query)
        ]

    # -------------------------------------------------------------------------
    # Residual span computation
    # -------------------------------------------------------------------------

    def _compute_residual_spans(
        self,
        query: str,
        matched_so_far: list[SNOMEDMatch],
    ) -> list[tuple[int, int]]:
        """
        Returns character spans NOT covered by any match in matched_so_far.
        Algorithm (from architect spec §3.7.4):
          1. Sort matches by span[0] ascending.
          2. Walk matches; for each: residual = (cursor, match.span[0]) if non-empty.
          3. Advance cursor = max(cursor, match.span[1]).
          4. After loop: residual = (cursor, len(query)) if non-empty.
          5. Drop residuals with length < MIN_RESIDUAL_LEN (noise filter).
        """
        if not matched_so_far:
            return [(0, len(query))] if len(query) >= MIN_RESIDUAL_LEN else []

        sorted_matches = sorted(matched_so_far, key=lambda m: m.span[0])
        residuals: list[tuple[int, int]] = []
        cursor = 0

        for m in sorted_matches:
            if m.span[0] > cursor:
                gap_start = cursor
                gap_end = m.span[0]
                if gap_end - gap_start >= MIN_RESIDUAL_LEN:
                    residuals.append((gap_start, gap_end))
            cursor = max(cursor, m.span[1])

        tail_start = cursor
        tail_end = len(query)
        if tail_end - tail_start >= MIN_RESIDUAL_LEN:
            residuals.append((tail_start, tail_end))

        return residuals

    # -------------------------------------------------------------------------
    # Stage 3: fuzzy pass on residual spans
    # -------------------------------------------------------------------------

    def _fuzzy_pass(
        self,
        query: str,
        residual_spans: list[tuple[int, int]],
    ) -> list[SNOMEDMatch]:
        """
        For each residual span:
          1. Extract substring from query.
          2. Run rapidfuzz token_sort_ratio against ALL preferred_terms (exact_index keys).
          3. If best score >= self._fuzzy_cutoff: create SNOMEDMatch with
             confidence = score / 100.0, match_type = "fuzzy".
          4. span = (span_start, span_end) of the residual character span.

        NOT logged: extracted substring text.
        """
        all_preferred_terms = list(self._exact_index.keys())
        hits: list[SNOMEDMatch] = []

        for span_start, span_end in residual_spans:
            substring = query[span_start:span_end].strip()
            if not substring:
                continue

            result = rf_process.extractOne(
                substring,
                all_preferred_terms,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self._fuzzy_cutoff,
            )
            if result is None:
                continue

            matched_term, score, _ = result
            record = self._exact_index.get(matched_term)
            if record is None:
                continue

            hits.append(SNOMEDMatch(
                code=record["concept_id"],
                display=record["preferred_term"],
                match_type="fuzzy",
                confidence=round(score / 100.0, 4),
                original_text=substring,
                span=(span_start, span_end),
                negated=False,
            ))

        return hits

    # -------------------------------------------------------------------------
    # Stage 4: semantic pass on residual spans
    # -------------------------------------------------------------------------

    def _semantic_pass(
        self,
        query: str,
        residual_spans: list[tuple[int, int]],
    ) -> list[SNOMEDMatch]:
        """
        For each residual span:
          1. Extract substring; encode with sentence-transformer.
          2. Query ChromaDB (if available) or compute numpy cosine similarity.
          3. If best similarity >= self._semantic_threshold: create SNOMEDMatch.
          4. span = residual span bounds.

        Returns [] immediately if self.semantic_available is False.
        All exceptions caught internally. NOT logged: substring text.
        """
        if not self.semantic_available or self._embedder is None:
            return []

        hits: list[SNOMEDMatch] = []
        for span_start, span_end in residual_spans:
            substring = query[span_start:span_end].strip()
            if not substring:
                continue
            match = self._semantic_match_single(substring, span_start, span_end)
            if match is not None:
                hits.append(match)
        return hits

    def _semantic_match_single(
        self,
        substring: str,
        span_start: int,
        span_end: int,
    ) -> Optional[SNOMEDMatch]:
        try:
            if self._collection is not None:
                embedding = self._embedder.encode([substring], show_progress_bar=False).tolist()
                results = self._collection.query(
                    query_embeddings=embedding,
                    n_results=1,
                    include=["documents", "distances"],
                )
                if not results["documents"] or not results["documents"][0]:
                    return None
                doc = results["documents"][0][0]
                similarity = 1.0 - results["distances"][0][0]
                if similarity < self._semantic_threshold:
                    return None
                record = self._exact_index.get(doc)
                if record is None:
                    return None
                return SNOMEDMatch(
                    code=record["concept_id"],
                    display=record["preferred_term"],
                    match_type="semantic",
                    confidence=round(similarity, 4),
                    original_text=substring,
                    span=(span_start, span_end),
                    negated=False,
                )

            if self._np_embeddings is not None and self._np_terms:
                import numpy as np
                query_vec = self._embedder.encode([substring], show_progress_bar=False)
                norm = np.linalg.norm(query_vec)
                if norm == 0:
                    return None
                query_norm = (query_vec / norm).astype("float32")
                similarities = self._np_embeddings @ query_norm.T
                best_idx = int(np.argmax(similarities))
                similarity = float(similarities[best_idx])
                if similarity < self._semantic_threshold:
                    return None
                matched_term = self._np_terms[best_idx]
                record = self._exact_index.get(matched_term)
                if record is None:
                    return None
                return SNOMEDMatch(
                    code=record["concept_id"],
                    display=record["preferred_term"],
                    match_type="semantic",
                    confidence=round(similarity, 4),
                    original_text=substring,
                    span=(span_start, span_end),
                    negated=False,
                )

        except Exception as exc:
            logger.warning(
                "_semantic_match_single: exception (%s) — skipping span",
                type(exc).__name__,
            )
        return None

    # -------------------------------------------------------------------------
    # Deduplication: longest-match-wins
    # -------------------------------------------------------------------------

    def _dedup_longest_match(self, hits: list[SNOMEDMatch]) -> list[SNOMEDMatch]:
        """
        For any two hits where hit_A.span fully contains hit_B.span (or equals it),
        keep hit_A and discard hit_B.
        Tie-break: confidence DESC, then span_start ASC.
        Two hits with overlapping but non-containing spans: both kept.

        Algorithm:
          1. Sort: primary=span_length DESC, secondary=confidence DESC, tertiary=span[0] ASC.
          2. Walk sorted list. Track kept_spans as list of (start, end).
          3. For each candidate: if its span is fully contained within any kept span, discard.
             Otherwise keep and add to kept_spans.
        """
        if not hits:
            return []

        sorted_hits = sorted(
            hits,
            key=lambda m: (-(m.span[1] - m.span[0]), -m.confidence, m.span[0]),
        )

        kept: list[SNOMEDMatch] = []
        kept_spans: list[tuple[int, int]] = []

        for candidate in sorted_hits:
            cs, ce = candidate.span
            contained = any(
                ks <= cs and ce <= ke
                for ks, ke in kept_spans
            )
            if not contained:
                kept.append(candidate)
                kept_spans.append((cs, ce))

        return kept
