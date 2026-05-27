# Compliance & Security Standards

## HIPAA Log Hygiene
This is a hard mandate. **NEVER** log raw text or user-identifiable clinical data.

### 1. Forbidden Content
*   Raw query / User input
*   Canonical query (merged)
*   Filter values (investigator names, site names, city names)
*   SNOMED display strings (use codes only)
*   LLM responses (raw JSON)
*   Trigger values / Option text
*   Metric match text / original text

### 2. Permitted Content (Safe for Logs)
*   Counts (e.g., `snomed_match_count`)
*   Lengths (e.g., `raw_response_length`)
*   Confidence scores
*   SNOMED codes (CID)
*   Decision enums (e.g., `ambiguous_trigger`)
*   Latency (ms)
*   Session/Turn IDs

### 3. Implementation
*   Always use `session.summary_for_logging()` for session state.
*   Grep for `to_dict()` or `repr()` in new log statements to ensure compliance.

## Security Stack (9 Layers)
1.  **Preprocessor (Initial):** Length, null-byte, injection patterns.
2.  **Preprocessor (Canonical):** Defense-in-depth on merged string.
3.  **SufficiencyGate (Pre-extraction):** No-cost rejection of ambiguous inputs.
4.  **Clinical-Intent Gate:** Block non-clinical input.
5.  **Post-Extraction Safety:** orphans detection (filters without condition).
6.  **Rate Limiting:** Session-based throttling.
7.  **Max-Turns Cap:** Hard cap of 3 clarifications.
8.  **Output Sanitization:** Mandatory `html.escape()` via `ResponseAssembler`.
9.  **API Key Middleware:** (In `api.py`) X-API-Key header verification.

## Architecture Guards
*   **Kansas City Disambiguation:** State must never be inferred from city/site name by the LLM.
*   **Discriminated Unions:** Every output must have a `type` literal for safe parsing.
*   **Pydantic V2:** Use `ConfigDict` and `model_dump()`.
