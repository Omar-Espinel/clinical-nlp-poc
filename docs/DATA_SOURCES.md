# SNOMED Data Sources

## OLS4 API (Primary Source — Fastest)

**Rationale:** EMBL-EBI ontology server; synonyms are included in the same search response, which means 50% fewer API calls compared to Snowstorm (which requires a separate call per concept). Build time is 1.5–2.5 hours vs 3–4 hours for Snowstorm. No authentication required. Stable public endpoint.

**Base URL:** `https://www.ebi.ac.uk/ols4/api`

**Search Endpoint:**
```
GET /search?ontology=snomed&q={query}&rows={limit}&start={offset}
```

**Response Shape Example:**
```json
{
  "response": {
    "docs": [
      {
        "obo_id": "SNOMEDCT:73211009",
        "label": "Diabetes mellitus",
        "synonym": ["Sugar diabetes", "DM"],
        "description": ["A metabolic disorder characterised by hyperglycaemia"]
      }
    ]
  }
}
```

**Clinical Trial Query Terms:**

| Term | Domain | Expected Concepts |
|---|---|---|
| clinical finding | Condition | ~50,000 |
| procedure | Procedure | ~30,000 |
| pharmaceutical product | Drug | ~25,000 |
| measurement | Measurement | ~15,000 |
| device | Device | ~8,000 |

**Total:** ~80,000–120,000 clinical-trial-relevant concepts

**Rate Limits & Retry Policy:**
- Maximum 10 requests/second — sleep 0.1s between requests
- Maximum 100 results per page
- No API key required
- On 5xx or timeout: retry 3 times with exponential backoff [1s, 2s, 4s]
- Request timeout: 30 seconds

**Expected Build Time:**
- Test mode (500 concepts, 100 per term): 2–3 minutes
- Full build (~100k concepts): 1.5–2.5 hours

---

## Snowstorm API (Fallback Source)

**Base URL:** `https://browser.ihtsdotools.org/snowstorm/snomed-ct/MAIN`

Requires 2 API calls per concept (concept fetch + description fetch), making it significantly slower than OLS4. Available only via the `--snowstorm-fallback` CLI flag; not used in standard builds.

---

## BioPortal API (Runtime Fallback Only)

Used at **runtime only** for cache misses — not during the index build process.

**Registration:** `https://bioportal.bioontology.org/account` (free, instant)

**Setup:** Add your API key to `.env`:
```
BIOPORTAL_API_KEY=your_key_here
```

**Behaviour:**
- Timeout: 5 seconds per request
- Circuit breaker: opens after 5 consecutive failures; resets after 60 seconds
- Results are written to `clinical_nlp.cache` with a 30-day TTL
