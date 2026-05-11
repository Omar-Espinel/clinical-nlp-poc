# OBSOLETE-AT-SCALE: shim — remove after pipeline.py migrates to SNOMEDSearchStrategy
"""SNOMED CT term resolution using exact, synonym, fuzzy, and semantic matching."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process as rf_process

from src.snomed_search.hybrid_cascade import ALIAS_DICTIONARY  # noqa: F401 — re-exported

logger = logging.getLogger(__name__)


@dataclass
class SNOMEDMatch:
    """A resolved SNOMED CT concept."""
    code: str
    display: str
    match_type: str
    confidence: float
    original_text: str
    negated: bool


class SNOMEDResolver:
    """Resolves free-text medical terms to SNOMED CT concepts via a 4-step cascade."""

    CONFIDENCE_THRESHOLD = 0.60
    SEMANTIC_THRESHOLD = 0.82

    def __init__(self, csv_path: str) -> None:
        """Load SNOMED CSV and build indexes. Semantic index built if libraries available."""
        self.exact_index: dict[str, dict] = {}
        self.synonym_index: dict[str, dict] = {}
        self.semantic_available: bool = False
        self._collection = None
        self._embedder = None
        self._np_embeddings = None
        self._np_terms: list[str] = []
        self._validated_aliases: dict[str, str] = {}

        self._load_csv(csv_path)
        self._validate_aliases()
        self._build_chroma_index()

    def _load_csv(self, csv_path: str) -> None:
        """Build exact_index and synonym_index from the SNOMED CSV."""
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for _, row in df.iterrows():
            record = {
                "concept_id": row["concept_id"].strip(),
                "preferred_term": row["preferred_term"].strip().lower(),
                "synonyms": row["synonyms"].strip(),
            }
            key = record["preferred_term"]
            self.exact_index[key] = record
            for syn in record["synonyms"].split("|"):
                syn = syn.strip().lower()
                if syn:
                    self.synonym_index[syn] = record
        logger.info("Loaded %d SNOMED terms, %d synonyms.", len(self.exact_index), len(self.synonym_index))

    def _validate_aliases(self) -> None:
        """Validate alias targets exist in exact_index; skip invalid ones with a warning."""
        for alias, target in ALIAS_DICTIONARY.items():
            target_lower = target.lower().strip()
            if target_lower in self.exact_index:
                self._validated_aliases[alias.lower().strip()] = target_lower
            else:
                logger.warning("Alias target not found in CSV — skipping alias '%s' → '%s'", alias, target)

    def _build_chroma_index(self) -> None:
        """Build semantic index: tries ChromaDB first, falls back to numpy cosine similarity.

        Sets self.semantic_available = True on success of either backend.
        Never raises — all exceptions handled internally.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            self.semantic_available = False
            logger.warning("sentence-transformers unavailable — semantic search disabled: %s", exc)
            return

        terms = list(self.exact_index.keys())

        # Try ChromaDB backend first
        try:
            import chromadb
            client = chromadb.EphemeralClient()
            self._collection = client.create_collection(
                name="snomed_clinical_trials_v1",
                metadata={"hnsw:space": "cosine"},
            )
            ids = [self.exact_index[t]["concept_id"] + "_" + str(i) for i, t in enumerate(terms)]
            embeddings = self._embedder.encode(terms, show_progress_bar=False).tolist()
            self._collection.add(documents=terms, embeddings=embeddings, ids=ids)
            self.semantic_available = True
            self._np_embeddings = None
            self._np_terms: list[str] = []
            logger.info("ChromaDB semantic index built with %d terms.", len(terms))
            return
        except Exception as exc:
            logger.warning("ChromaDB unavailable (%s) — falling back to numpy cosine similarity.", exc)

        # Numpy cosine similarity fallback (no C++ compilation required)
        try:
            import numpy as np
            raw = self._embedder.encode(terms, show_progress_bar=False)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
            self._np_embeddings = (raw / norms).astype("float32")
            self._np_terms = terms
            self.semantic_available = True
            logger.info("Numpy semantic index built with %d terms.", len(terms))
        except Exception as exc:
            self.semantic_available = False
            logger.warning("Numpy semantic fallback also failed — semantic search disabled: %s", exc)

    def resolve(self, term: str, negated: bool) -> Optional[SNOMEDMatch]:
        """Resolve a term to a SNOMEDMatch using a 4-step cascade.

        Returns None if no match meets the confidence threshold.
        """
        normalized = term.strip().lower()

        match = (
            self._exact_match(normalized)
            or self._synonym_match(normalized)
            or self._fuzzy_synonym_match(normalized)
            or self._semantic_match(normalized)
        )

        if match is None:
            logger.info("No match found for \"%s\" — skipping", term)
            return None

        if match.confidence < self.CONFIDENCE_THRESHOLD:
            logger.info("Match for \"%s\" below threshold (%.2f) — skipping", term, match.confidence)
            return None

        match.negated = negated
        match.original_text = term
        logger.info(
            "Resolved \"%s\" → %s via %s (%.2f)",
            term, match.code, match.match_type, match.confidence,
        )
        return match

    def _exact_match(self, term: str) -> Optional[SNOMEDMatch]:
        """Check for an exact preferred_term match."""
        row = self.exact_index.get(term)
        if row:
            return SNOMEDMatch(
                code=row["concept_id"],
                display=row["preferred_term"],
                match_type="exact",
                confidence=0.99,
                original_text=term,
                negated=False,
            )
        return None

    def _synonym_match(self, term: str) -> Optional[SNOMEDMatch]:
        """Check alias dictionary first, then synonym_index."""
        # Alias dictionary lookup
        alias_target = self._validated_aliases.get(term)
        if alias_target:
            row = self.exact_index.get(alias_target)
            if row:
                return SNOMEDMatch(
                    code=row["concept_id"],
                    display=row["preferred_term"],
                    match_type="synonym",
                    confidence=0.97,
                    original_text=term,
                    negated=False,
                )

        # Raw synonym index lookup
        row = self.synonym_index.get(term)
        if row:
            return SNOMEDMatch(
                code=row["concept_id"],
                display=row["preferred_term"],
                match_type="synonym",
                confidence=0.95,
                original_text=term,
                negated=False,
            )
        return None

    def _fuzzy_synonym_match(self, term: str) -> Optional[SNOMEDMatch]:
        """Fuzzy match against all synonym and preferred term keys using rapidfuzz."""
        all_keys = list(self.synonym_index.keys()) + list(self.exact_index.keys())
        result = rf_process.extractOne(
            term,
            all_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=88,
        )
        if result is None:
            return None
        matched_key, score, _ = result
        row = self.synonym_index.get(matched_key) or self.exact_index.get(matched_key)
        if row is None:
            return None
        return SNOMEDMatch(
            code=row["concept_id"],
            display=row["preferred_term"],
            match_type="fuzzy",
            confidence=score / 100.0,
            original_text=term,
            negated=False,
        )

    def _semantic_match(self, term: str) -> Optional[SNOMEDMatch]:
        """Vector similarity search using ChromaDB or numpy fallback.

        Returns None if semantic_available is False or no match exceeds SEMANTIC_THRESHOLD.
        """
        if not self.semantic_available or self._embedder is None:
            return None
        try:
            # ChromaDB path
            if self._collection is not None:
                embedding = self._embedder.encode([term], show_progress_bar=False).tolist()
                results = self._collection.query(
                    query_embeddings=embedding,
                    n_results=1,
                    include=["documents", "distances"],
                )
                if not results["documents"] or not results["documents"][0]:
                    return None
                doc = results["documents"][0][0]
                similarity = 1.0 - results["distances"][0][0]
                if similarity < self.SEMANTIC_THRESHOLD:
                    return None
                row = self.exact_index.get(doc)
                if row is None:
                    return None
                logger.info("Resolved \"%s\" → %s via semantic/chroma (%.2f)", term, row["concept_id"], similarity)
                return SNOMEDMatch(
                    code=row["concept_id"],
                    display=row["preferred_term"],
                    match_type="semantic",
                    confidence=round(similarity, 4),
                    original_text=term,
                    negated=False,
                )

            # Numpy fallback path
            if self._np_embeddings is not None and self._np_terms:
                import numpy as np
                query = self._embedder.encode([term], show_progress_bar=False)
                norm = np.linalg.norm(query)
                if norm == 0:
                    return None
                query_norm = (query / norm).astype("float32")
                similarities = self._np_embeddings @ query_norm.T
                best_idx = int(np.argmax(similarities))
                similarity = float(similarities[best_idx])
                if similarity < self.SEMANTIC_THRESHOLD:
                    return None
                matched_term = self._np_terms[best_idx]
                row = self.exact_index.get(matched_term)
                if row is None:
                    return None
                logger.info("Resolved \"%s\" → %s via semantic/numpy (%.2f)", term, row["concept_id"], similarity)
                return SNOMEDMatch(
                    code=row["concept_id"],
                    display=row["preferred_term"],
                    match_type="semantic",
                    confidence=round(similarity, 4),
                    original_text=term,
                    negated=False,
                )

        except Exception as exc:
            logger.warning("Semantic match failed for \"%s\": %s", term, exc)
        return None
