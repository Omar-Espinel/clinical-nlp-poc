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
        r"show\s+me\s+everything",
        r"tell\s+me\s+everything",
        r"get\s+me\s+everything",
        r"list\s+everything",
        r"give\s+me\s+everything",
        r"give\s+me\s+all\s+(results|data|records|information|entries|everything)",
        r"show\s+me\s+all\s+(results|data|records|information|entries|everything)",
        r"return\s+all\s+(results|data|records|information|entries)",
        r"fetch\s+all\s+(results|data|records|information|entries)",
        r"(dump|export)\s+(all|every)",
        r"all\s+your\s+(data|records|information|results|entries)",
        r"your\s+(database|records|data|index|knowledge)",
        r"what\s+(data|information)\s+(do\s+you\s+have|have\s+you\s+stored|are\s+you\s+storing)",
        r"reveal\s+(all|everything|your)",
        # ── Social engineering ────────────────────────────────────────────
        r"pretend\s+(you\s+are|to\s+be)",
        r"roleplay",
        r"hypothetically",
        r"as\s+a\s+(different|new)\s+(ai|model|assistant|system)",
        r"bypass",
        r"override\s+(your\s+)?(rules|instructions|guidelines|safety)",
        # ── LLM special tokens ────────────────────────────────────────────────────
        r"\[\s*/?INST\s*\]",
        r"<<SYS>>",
        r"<</SYS>>",
        # ── Template / SSTI injection ─────────────────────────────────────────────
        r"\$\{[^}]*\}",
        # ── Code eval ─────────────────────────────────────────────────────────────
        r"\beval\s*\(",
        # ── HTTP header injection ─────────────────────────────────────────────────
        r"%0[ad]",
        # ── XML / tag injection ───────────────────────────────────────────────────
        r"</\w+\s*>",
        r"<\w+\s*/>",
        r"<return\b",
        # ── Path traversal (Unix and Windows) ─────────────────────────────────────
        r"\.\.[\\/]",
        # ── SQL injection additions ───────────────────────────────────────────────
        r"\bAND\s+1\s*=\s*1\b",
        r"\bOR\s+1\s*=\s*1\b",
        r"\bSLEEP\s*\(\s*\d+\s*\)",
        r"\bUNION\s+SELECT\b",
        # ── Harmful content / WMD ────────────────────────────────────────────────
        r"nerve\s+agent",
        r"\bsarin\b",
        r"bioweapon",
        r"dirty\s+bomb",
        r"chemical\s+weapon",
        r"poison\s+(water\s+supply|food\s+supply|reservoir)",
        r"\bexplosive[s]?\s+(synthesis|making|manufacture)\b",
        r"weapon\s+synthesis",
        r"(make|build|create|construct|fashion)\s+(an?\s+)?(weapon|bomb|firearm|rifle|pistol|explosive[s]?|\bied\b)",
        # ── Controlled-substance manufacturing ───────────────────────────────────
        r"\bmethamphetamine\b",
        r"manufactur\w*\s+(drug|narcotic|fentanyl|heroin|cocaine)",
        r"synthesiz\w*\s+(drug|narcotic|fentanyl|heroin|cocaine)",
        r"(cook|make|grow|produce|bake)\s+(meth|methamphetamine|heroin|cocaine|crack|fentanyl|lsd|ecstasy)",
        # ── Child safety ──────────────────────────────────────────────────────────
        r"child\s+exploitation",
        r"human\s+trafficking",
        r"child\s+abuse\s+material",
        # ── Self-harm ─────────────────────────────────────────────────────────────
        r"suicide\s+method",
        r"self[\-\s]harm\s+guide",
        r"how\s+to\s+kill\s+(myself|yourself)",
        r"how\s+to\s+commit\s+suicide",
        r"ways\s+to\s+(end|take)\s+(my|your)\s+life",
        r"how\s+to\s+(hurt|harm|injure)\s+(myself|yourself)",
        # ── Poison-as-attack-verb ─────────────────────────────────────────────────
        r"poison\s+(a\s+)?(person|someone|people|victim|target|individual)",
        # ── Cybercrime ────────────────────────────────────────────────────────────
        r"\bransomware\b",
        r"dark\s+web\s+drug",
        r"malware\s+creat\w*",
        r"(write|create|build|develop|code)\s+(an?\s+)?(malware|ransomware|botnet|exploit)",
    ]

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self._INJECTION_PATTERNS
        ]
        self._compiled_patterns.append(
            re.compile(r"\ASYSTEM\s*[:\n]", re.IGNORECASE)
        )

    def process(self, raw_input: str) -> PreprocessedInput:
        """Validate and sanitize a raw query string.

        Raises PreprocessorError if the input is too short, too long,
        or contains injection/exfiltration patterns.
        """
        raw_input = raw_input.replace("\x00", "")
        self._validate_length(raw_input)
        if self._check_injection(raw_input):
            raise PreprocessorError("Invalid query detected")
        sanitized = self._sanitize(raw_input)
        return PreprocessedInput(
            text=sanitized,
            original=raw_input,
            char_count=len(sanitized),
        )

    def assert_safe(self, text: str) -> None:
        """Check injection patterns on an already-preprocessed string.

        Does NOT enforce length limits — canonical queries from substitute/append
        may legitimately exceed MAX_LENGTH (M2 fix: called after merge, not on raw input).
        Raises PreprocessorError if any injection pattern matches.
        """
        if self._check_injection(text):
            raise PreprocessorError("Invalid query detected")

    def _sanitize(self, text: str) -> str:
        """Strip leading/trailing whitespace and collapse internal whitespace."""
        text = text.replace("\x00", "")
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
