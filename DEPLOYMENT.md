# Deployment Guide — Clinical Research NLP API

## Overview

The system exposes two interfaces:
- **REST API** (`api.py`) — FastAPI + Uvicorn, primary interface for production consumers
- **Streamlit UI** (`app.py`) — chat interface for direct human use, runs independently

Both share the same `src/` pipeline and `data/` files.

---

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq LLM authentication |
| `API_KEY` | Recommended | None (open) | Protects API endpoints with `X-API-Key` header |
| `ALLOWED_ORIGINS` | Recommended | `http://localhost:3000` | Comma-separated CORS allowlist |
| `LLM_PROVIDER` | No | `groq` | LLM provider selection |
| `SNOMED_SEARCH_STRATEGY` | No | `hybrid_cascade` | SNOMED matching strategy |
| `AMBIG_STRICT_VALIDATION` | No | `true` | Registry validation strictness |

Create a `.env` file in the project root:
GROQ_API_KEY=your_key_here
API_KEY=your_internal_api_key_here
ALLOWED_ORIGINS=http://localhost:3000,https://your-frontend.com


---

## Local Development

### Prerequisites
- Python 3.13
- pip

### Install

pip install -r requirements.txt


### Run the API

uvicorn api:app --reload --port 8000


### Run the Streamlit UI

streamlit run app.py


### API Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health/live` | None | Liveness probe — is the process alive |
| `GET` | `/health/ready` | None | Readiness probe — is pipeline loaded |
| `POST` | `/v1/query` | X-API-Key | Submit a query |
| `DELETE` | `/v1/session/{session_id}` | X-API-Key | Clear a session |

Interactive docs (when running locally):

http://localhost:8000/docs


### Test the API locally

Liveness
curl http://localhost:8000/health/live

Readiness
curl http://localhost:8000/health/ready

First query — expects clarification response for ambiguous term
curl -X POST http://localhost:8000/v1/query
-H "Content-Type: application/json"
-H "X-API-Key: your_internal_api_key_here"
-d "{"query": "cancer trials in Boston"}"

Multi-turn — paste session_id from above response
curl -X POST http://localhost:8000/v1/query
-H "Content-Type: application/json"
-H "X-API-Key: your_internal_api_key_here"
-d "{"query": "Lung Cancer", "session_id": "<paste-session-id>"}"

Clear a session
curl -X DELETE http://localhost:8000/v1/session/<paste-session-id>
-H "X-API-Key: your_internal_api_key_here"

---

## Docker

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)

### Build
docker build -t clinical-nlp-api .


First build takes 3-5 minutes (pip install + sentence-transformer model download ~90MB).
Subsequent builds use layer cache and are much faster.

### Run

docker run -p 8000:8000
-e GROQ_API_KEY=your_key_here
-e API_KEY=your_internal_api_key_here
-e ALLOWED_ORIGINS=http://localhost:3000
clinical-nlp-api


### Run with a .env file

docker run -p 8000:8000 --env-file .env clinical-nlp-api

### Verify

curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready


### Container behaviour

- Non-root user (`nlp`) inside container
- Docker HEALTHCHECK polls `/health/live` every 30s
- Pipeline loads at startup — container is not ready until `/health/ready` returns 200
- First startup takes 30-60s while the embedding model initializes
- Sessions are in-memory — cleared on container restart

---

## Production Deployment (Recommended Minimum)

### Architecture
[ALB / Cloudflare / nginx] <- TLS termination + rate limiting
|
[Docker container] <- gunicorn + uvicorn workers
|
[Redis] <- shared session store (future)


### Switch to gunicorn for production

Replace the `CMD` in `Dockerfile`:

CMD ["gunicorn", "api:app",
"--workers", "2",
"--worker-class", "uvicorn.workers.UvicornWorker",
"--threads", "4",
"--bind", "0.0.0.0:8000",
"--timeout", "60",
"--graceful-timeout", "30",
"--access-logfile", "-",
"--error-logfile", "-"]


Add to `requirements.txt`:

gunicorn>=22.0.0


**Worker memory note:** each worker loads its own pipeline instance (~120MB RAM).
2 workers = ~240MB baseline. Size your VM accordingly (minimum 1GB RAM recommended).

### Environment variable checklist for production

- [ ] `GROQ_API_KEY` — from secrets manager, not plain env var
- [ ] `API_KEY` — strong random string (e.g. `openssl rand -hex 32`)
- [ ] `ALLOWED_ORIGINS` — locked to your actual frontend domain(s)
- [ ] `AMBIG_STRICT_VALIDATION=true` — default, keep it

### Known production gaps (POC limitations)

| Gap | Impact | Fix |
|---|---|---|
| Sessions in-memory | Lost on restart, not shared across workers | Replace with Redis |
| No per-IP rate limiting | Abuse possible before hitting Python | Add nginx or API Gateway in front |
| No audit logging | Compliance gap | Add structured logging pipeline |
| No authentication beyond API key | No per-user identity | Add OAuth2 / JWT layer |
| Single container | No HA | Add orchestration (ECS, K8s, Cloud Run) |

---

## Running Tests

### Unit + integration tests (pytest)

pytest tests/test_sufficiency_gate.py
tests/test_conversation.py
tests/test_snomed_strategies.py
tests/test_negation.py
tests/test_llm_provider.py
tests/test_ambiguity_coverage.py
tests/test_metric_filters.py


### Batch evaluation

python tests/batch_eval.py
python tests/batch_eval.py --limit 10
python tests/batch_eval.py --strategy aho_corasick


### QA agent (rate-limited for Groq free tier)

python qa_testing/test_agent.py
python qa_testing/test_agent.py --limit 20
python qa_testing/test_agent.py --category injection


### Legacy regression (kept for compatibility)

python tests/run_tests.py


---

## HuggingFace Spaces (Streamlit UI only)

The Streamlit UI (`app.py`) can still be deployed to HuggingFace Spaces independently
of the API service.

1. Create a Space with **Streamlit** SDK
2. Upload all project files
3. Add `GROQ_API_KEY` as a Space Secret
4. The Space auto-builds in 5-10 minutes

Note: HuggingFace Spaces does not serve the FastAPI service — only `app.py`.