# Deployment Guide — Clinical Research NLP

## Running Locally

**Step 1:** Clone or download the project files

**Step 2:** Install all dependencies
```bash
pip install -r requirements.txt
```

**Step 3:** Copy the environment template
```bash
cp .env.example .env
```

**Step 4:** Add your Groq API key to `.env`
```
GROQ_API_KEY=your_groq_api_key_here
```

**Step 5:** Start the Streamlit application
```bash
streamlit run app.py
```

**Step 6:** Open [http://localhost:8501](http://localhost:8501) in your browser

> **Note:** First load takes 30–60 seconds while sentence-transformers
> downloads and loads the all-MiniLM-L6-v2 model (~90 MB).
> Subsequent starts are faster as the model is cached locally.

---

## Running Tests

From the project root directory:
```bash
python tests/run_tests.py
```

Requires `GROQ_API_KEY` in `.env` file.

- Exit code `0` = 15 or more tests passed
- Exit code `1` = fewer than 15 tests passed

The test runner prints `PASS` or `FAIL` with reasons for each of the 20 test cases,
then prints a final summary line.

---

## Deploying to HuggingFace Spaces (Free, Shareable URL)

**Step 1:** Create a free account at [huggingface.co](https://huggingface.co)

**Step 2:** Go to **Spaces** and click **New Space**

**Step 3:** Choose **Streamlit** as the SDK and give your Space a name

**Step 4:** Upload all project files:
- `app.py`
- `requirements.txt`
- `README.md`
- `.env.example`
- `src/` directory (all files)
- `data/` directory (both files)

> Do **not** upload `.env` — use Secrets instead (see Step 5).

**Step 5:** Go to your Space **Settings → Secrets** and add:
```
Name:  GROQ_API_KEY
Value: your_actual_groq_api_key
```

**Step 6:** The Space will auto-build and deploy (5–10 minutes)

**Step 7:** Your app is live at `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

**Step 8:** Share the URL — anyone can use it without a HuggingFace account

> **Note:** The first cold start on HuggingFace takes 2–5 minutes while
> packages install and models download. Subsequent loads within the same
> session are faster. Free-tier Spaces may sleep after inactivity.

---

## Getting a Free Groq API Key

**Step 1:** Go to [console.groq.com](https://console.groq.com)

**Step 2:** Sign up for a free account (no credit card required)

**Step 3:** Navigate to the **API Keys** section in the left sidebar

**Step 4:** Click **Create API Key** and give it a name

**Step 5:** Copy the key immediately (it is only shown once)

**Step 6:** Add to `.env` for local use:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```
Or add to HuggingFace Spaces Secrets for cloud deployment.

**Free tier limits:** Up to 30 requests per minute, 500 requests per day.
For heavier usage, upgrade to a paid Groq plan.

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key from console.groq.com |

No other environment variables are needed. All other configuration
(data file paths, model names, thresholds) is hardcoded with sensible
defaults inside the source files.

---

## Troubleshooting

**"GROQ_API_KEY not found" error:**
- Ensure `.env` exists in the project root (not inside `src/`)
- Verify the key name is exactly `GROQ_API_KEY` (case-sensitive)
- On HuggingFace, confirm the secret was saved under Settings → Secrets

**First load is slow (30–60 seconds):**
- This is expected — sentence-transformers downloads the embedding model
- Subsequent loads reuse the cached model

**"Rate limit reached" error:**
- Free Groq tier allows 30 requests/minute
- Wait 60 seconds and retry

**ChromaDB / semantic search unavailable:**
- The app degrades gracefully — exact, synonym, and fuzzy matching still work
- Check that `torch==2.2.2` and `sentence-transformers==2.6.1` are installed
- Review logs for the specific error

**Tests failing below threshold:**
- Ensure `GROQ_API_KEY` is set in `.env`
- Run from the project root: `python tests/run_tests.py` (not from `tests/`)
- LLM non-determinism may occasionally cause near-threshold failures; re-run to confirm
