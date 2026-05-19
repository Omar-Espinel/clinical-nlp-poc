---
name: software-architect
description: Expert guidance on architectural patterns, component design, and project-wide standards for the Clinical Research NLP platform. Use for design reviews, refactoring plans, and compliance audits.
---

# Software Architecture & Project Expert

This skill provides expert guidance for the Clinical Research NLP project, ensuring adherence to the v2 pipeline architecture, security mandates, and high-quality software engineering standards.

## Core Workflows

### 1. Architectural Review
When reviewing or proposing changes, verify against the **10-step pipeline flow**.
- **Reference:** [pipeline.md](references/pipeline.md)
- Ensure new components do not break the `canonical_query` merge logic or parallel extraction timeout.

### 2. Security & Compliance Audit
Every code change must respect HIPAA log hygiene and the multi-layer security stack.
- **Reference:** [compliance.md](references/compliance.md)
- **Mandate:** NEVER log raw text. Use `summary_for_logging()`.

### 3. Component Extension
When adding new strategies or providers:
- Use the **Pluggable Protocols** (`SNOMEDSearchStrategy`, `LLMProvider`, `FilterNormalizer`).
- Verify Pydantic V2 syntax (`model_config`, `model_dump`).
- Ensure all new dependencies are compatible with **Python 3.13**.

## Design Patterns
- **Discriminated Unions:** Always use `type` literal in output schemas.
- **Fail-Safe Fallbacks:** Use numpy cosine similarity if chromadb is unavailable.
- **Deterministic First:** Prioritize deterministic gates (SufficiencyGate, AC Automaton) over LLM calls.

## Documentation Maintenance
- Keep `CONTEXT.md` high-level and architectural.
- Move implementation logs and legacy details to `HISTORY.md`.
- Ensure `README.md` remains the primary setup guide.
