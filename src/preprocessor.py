"""Input sanitization and validation for clinical NLP queries."""

import re
from dataclasses import dataclass


@dataclass
class PreprocessedInput:
    """Holds cleaned and validated query text."""
    text: str
    original: str
    char_count: int


class PreprocessorError(Exception):
    """Raised when input fails validation."""
    pass


class Preprocessor:
    """Sanitizes and validates raw query strings before LLM extraction."""

    MIN_LENGTH = 3
    MAX_LENGTH = 500

    # Patterns that are never legitimate clinical queries.
    # Split into logical groups for maintainability.
    _INJECTION_PATTERNS = [
        # ── Prompt injection ──────────────────────────────────────────────
        r"ignore previous instructions",
        r"ignore all previous",
        r"system prompt",
        r"you are now",
        r"disregard",
        r"forget everything",
        r"jailbreak",
        r"act as",
        # ── Code / script injection ───────────────────────────────────────
        r"<script",
        r"SELECT \* FROM",
        r"DROP TABLE",
        r"\{\{",
        r"\}\}",
        r"<!--",
        # ── Data exfiltration attempts ────────────────────────────────────
        # "show/tell/get/list everything"
        r"show\s+me\s+everything",
        r"tell\s+me\s+everything",
        r"get\s+me\s+everything",
        r"list\s+everything",
        r"give\s+me\s+everything",
        # "give/show/return/fetch me all <data noun>"
        r"give\s+me\s+all\s+(results|data|records|information|entries|everything)",
        r"show\s+me\s+all\s+(results|data|records|information|entries|everything)",
        r"return\s+all\s+(results|data|records|information|entries)",
        r"fetch\s+all\s+(results|data|records|information|entries)",
        # "dump / export all"
        r"(dump|export)\s+(all|every)",
        # "all your data / your database"
        r"all\s+your\s+(data|records|information|results|entries)",
        r"your\s+(database|records|data|index|knowledge)",
        # "what data do you have / are you storing"
        r"what\s+(data|information)\s+(do\s+you\s+have|have\s+you\s+stored|are\s+you\s+storing)",
        # "reveal all"
        r"reveal\s+(all|everything|your)",
        # ── Social engineering ────────────────────────────────────────────
        r"pretend\s+(you\s+are|to\s+be)",
        r"roleplay",
        r"hypothetically",
        r"as\s+a\s+(different|new)\s+(ai|model|assistant|system)",
        r"bypass",
        r"override\s+(your\s+)?(rules|instructions|guidelines|safety)",
    ]

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self._INJECTION_PATTERNS
        ]

    def process(self, raw_input: str) -> PreprocessedInput:
        """Validate and sanitize a raw query string.

        Raises PreprocessorError if the input is too short, too long,
        or contains injection/exfiltration patterns.
        """
        self._validate_length(raw_input)
        if self._check_injection(raw_input):
            raise PreprocessorError("Invalid query detected")
        sanitized = self._sanitize(raw_input)
        return PreprocessedInput(
            text=sanitized,
            original=raw_input,
            char_count=len(sanitized),
        )

    def _sanitize(self, text: str) -> str:
        """Strip leading/trailing whitespace and collapse internal whitespace."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _check_injection(self, text: str) -> bool:
        """Return True if any injection or exfiltration pattern is detected."""
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                return True
        return False

    def _validate_length(self, text: str) -> None:
        """Raise PreprocessorError if text length is outside allowed bounds."""
        stripped = text.strip()
        if len(stripped) < self.MIN_LENGTH:
            raise PreprocessorError("Query must be at least 3 characters")
        if len(stripped) > self.MAX_LENGTH:
            raise PreprocessorError("Query must be under 500 characters")
