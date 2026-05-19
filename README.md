---
title: Clinical Research NLP
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# Clinical Research NLP

A natural language processing system for clinical researchers. Takes free-text
queries about clinical trials and returns structured SNOMED CT concepts plus
filters (investigator, site, city, state, phase) — or asks a clarification
question when the query is ambiguous.

The system supports multi-turn conversations: if a query is ambiguous (e.g.
bare "cancer"), it asks which type before searching. Up to 3 clarification
turns per session.

---

## Interfaces

| Interface | File | Purpose |
|---|---|---|
| REST API | `api.py` | Primary — for application consumers |
| Streamlit UI | `app.py` | Secondary — for direct human use |

---

## Example Queries

- `Phase 3 type 2 diabetes trials in NYC`
- `Dr. Smith breast cancer research at Mayo Clinic`
- `NSCLC immunotherapy Phase 2 in California`
- `Alzheimer's disease studies in the Bay Area`
- `CHF trials excluding diabetes in Boston`
- `Dr. Williams atrial fibrillation research at Johns Hopkins`
- `Multiple myeloma Phase 1/2 studies in Toronto`

---

## Quick Start — API

**1. Install**

pip install -r requirements.txt


**2. Configure**
cp .env.example .env

Edit .env: add GROQ_API_KEY and API_KEY

**3. Run**

uvicorn api:app --port 8000


**4. Test**

curl http://localhost:8000/health/ready

curl -X POST http://localhost:8000/v1/query
-H "Content-Type: application/json"
-H "X-API-Key: your_api_key"
-d "{"query": "Phase 3 diabetes trials in Boston"}"


Interactive API docs: http://localhost:8000/docs

---

## Quick Start — Streamlit UI

streamlit run app.py


Opens at http://localhost:8501

---

## Quick Start — Docker

docker build -t clinical-nlp-api .

docker run -p 8000:8000
-e GROQ_API_KEY=your_key_here
-e API_KEY=your_internal_key_here
clinical-nlp-api


---

## Output Format

Every API response is one of two types, distinguished by the `type` field.

**Search result** (`type: "search"`):

{
"result": {
"type": "search",
"snomed_terms": [
{
"code": "44054006",
"display": "type 2 diabetes mellitus",
"confidence": 0.99,
"match_type": "exact"
}
],
"filters": {
"city": {"value": "Boston", "confidence": 0.99},
"phase": {"value": "Phase 3", "confidence": 0.99},
"state": {"values": ["Massachusetts"], "confidence": 0.99, "is_region": false}
}
},
"session_id": "uuid4-here",
"processing_time_ms": 843
}


**Clarification request** (`type: "clarification"`):
{
"result": {
"type": "clarification",
"question": "Which type of cancer are you looking for?",
"options": ["Lung Cancer", "Breast Cancer", "Colorectal Cancer"],
"turn_number": 1,
"max_turns": 3
},
"session_id": "uuid4-here",
"processing_time_ms": 12
}


To continue a clarification turn, pass the same `session_id` with your next query.

---

## Architecture
REST API (api.py — FastAPI + Uvicorn)
Streamlit UI (app.py)
└── src/pipeline.py — NLPPipeline.run_with_session()
├── src/preprocessor.py — ~77 security/injection patterns
├── src/sufficiency_gate.py — ambiguity detection (no LLM cost)
├── src/filter_extractor.py — Groq LLM filter extraction
├── src/snomed_search/ — pluggable SNOMED strategies
│ ├── hybrid_cascade.py — exact → synonym → fuzzy → semantic
│ ├── aho_corasick.py
│ └── ngram_lookup.py
├── src/snomed_search/negation.py — NegEx negation detection
├── src/conversation.py — multi-turn session state
├── src/normalizers/geo.py — city/state/region normalization
└── src/assembler.py — Pydantic V2 output assembly

data/
├── snomed_clinical_trials.csv — 116 SNOMED concepts with synonyms
├── geo_canonical.json — 200+ cities, states, 59 regions
├── ambiguous_terms.json — clarification triggers + options
└── metric_filters.json — 12 Advarra metric fields


---

## Getting a Groq API Key

1. Go to https://console.groq.com
2. Sign up for a free account
3. Navigate to API Keys and click Create API Key
4. Add to `.env` as `GROQ_API_KEY=your_key_here`

Free tier: ~30 requests/minute, 500/day.

---

## Tech Stack

| Package | Version | Purpose |
|---|---|---|
| fastapi | >=0.115.0 | REST API framework |
| uvicorn | >=0.32.0 | ASGI server |
| streamlit | 1.40.0 | Chat UI |
| groq | 0.11.0 | LLM API client |
| sentence-transformers | 3.2.1 | Semantic SNOMED search |
| rapidfuzz | 3.10.0 | Fuzzy string matching |
| pyahocorasick | >=2.0.0 | Metric field AC automaton |
| pydantic | 2.9.2 | Output schema validation |
| python-dotenv | 1.0.1 | .env loading |
| python-multipart | >=0.0.9 | Required by FastAPI internals |

---

## Running Tests

pytest tests/test_sufficiency_gate.py tests/test_conversation.py
tests/test_snomed_strategies.py tests/test_negation.py
tests/test_llm_provider.py tests/test_ambiguity_coverage.py
tests/test_metric_filters.py

python tests/batch_eval.py --limit 10

python qa_testing/test_agent.py --limit 20

Full deployment instructions: see DEPLOYMENT.md