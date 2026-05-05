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

A natural language processing tool for clinical researchers to extract structured information from free-text queries about clinical trials. Given a plain-English query, the system identifies SNOMED CT medical concepts with confidence scores and extracts structured filters including investigator name, site name, city, state, and study phase.

The system uses a Groq-powered LLM (llama-3.1-8b-instant) to parse natural language, then validates and enriches extracted terms through a 4-step SNOMED matching cascade: exact match, synonym/alias lookup, fuzzy matching via rapidfuzz, and semantic vector search via ChromaDB with sentence-transformers embeddings. Geography is normalized against a curated US + Canada city/state/region database. Queries are pre-screened by a multi-pattern security layer (~70 regex patterns) that blocks injection attacks, harmful content, and prompt manipulation before any LLM call.

## Example Queries to Try

- `Phase 3 T2DM trials in NYC`
- `Dr. Smith breast cancer research at Mayo Clinic`
- `NSCLC immunotherapy Phase 2 in California`
- `Alzheimer's disease studies in the Bay Area`
- `CHF trials excluding diabetes in Boston`
- `Dr. Williams atrial fibrillation research at Johns Hopkins in Baltimore`
- `Multiple myeloma Phase 1/2 studies in Toronto`

## Setup — Running Locally

**Step 1:** Clone or download the project files

**Step 2:** Install dependencies
```bash
pip install -r requirements.txt
```

**Step 3:** Copy the environment template
```bash
cp .env.example .env
```

**Step 4:** Add your Groq API key to `.env`
```
GROQ_API_KEY=your_key_here
```

**Step 5:** Start the app
```bash
streamlit run app.py
```

**Step 6:** Open [http://localhost:8501](http://localhost:8501)

> **Note:** First load takes 30–60 seconds while sentence-transformers downloads and loads the all-MiniLM-L6-v2 model (~90 MB).

## Setup — HuggingFace Spaces

1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to **Spaces** → **New Space** → choose **Streamlit** as the SDK
3. Upload all project files
4. Go to **Settings → Secrets** and add: `GROQ_API_KEY = your_key_here`
5. The Space auto-builds and deploys in 5–10 minutes
6. Share the URL with anyone — no login required for viewers

## Getting a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to **API Keys** and click **Create API Key**
4. Copy the key and add it to `.env` or HuggingFace Secrets

Free tier: up to 30 requests/minute, 500/day.

## Architecture

```
app.py (Streamlit UI)
  └── src/pipeline.py (NLPPipeline)
        ├── src/preprocessor.py   — input sanitization, injection detection & harmful content blocking (~70 patterns)
        ├── src/extractor.py      — Groq LLM term + filter extraction
        ├── src/snomed_resolver.py — 4-step SNOMED matching cascade
        │     ├── exact match (preferred_term lookup)
        │     ├── synonym/alias match (alias dict + synonym index)
        │     ├── fuzzy match (rapidfuzz token_sort_ratio ≥ 88)
        │     └── semantic match (ChromaDB + all-MiniLM-L6-v2 ≥ 0.82)
        ├── src/geo_normalizer.py — city/state normalization (200+ entries)
        └── src/assembler.py      — Pydantic v2 output assembly

data/
  ├── snomed_clinical_trials.csv  — curated SNOMED subset (100+ concepts)
  └── geo_canonical.json          — US + Canada geo lookup (200+ cities)

tests/
  ├── test_cases.json             — 20 structured test cases
  └── run_tests.py                — test runner (exit 0 if ≥15 pass)
```

## Running Tests

```bash
python tests/run_tests.py
```

Requires `GROQ_API_KEY` in `.env`. Exit code `0` = 15 or more tests passed.
