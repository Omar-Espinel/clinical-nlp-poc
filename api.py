"""Clinical NLP FastAPI service.

Routes
------
    GET    /health/live              Liveness probe. No auth, no pipeline dependency.
    GET    /health/ready             Readiness probe. No auth; 503 until pipeline is initialized.
    POST   /v1/query                 Main NLP query endpoint. Auth required. Accepts
                                      {query, session_id?}; returns {result, session_id,
                                      processing_time_ms} where result is NLPOutput |
                                      ClarificationOutput (discriminated on `type`).
    DELETE /v1/session/{session_id}  Clear a conversation session. Auth required.
    GET    /v1/autocomplete          Clinical query autocomplete. Auth required. Query params:
                                      q (prefix 3 to 100 chars), limit (1 to 10, default 7).
                                      Returns ranked suggestion array. Rate: 30 per 10 seconds per IP.

Auth
----
    All routes except /health/* require the `X-API-Key` header to match the API_KEY env
    var (middleware at require_api_key). If API_KEY is unset, the check is skipped — set
    it before exposing the service.

SNOMED strategy selection
-------------------------
    The SNOMED search strategy is bound at server startup, not per-request. Selection
    priority: explicit SNOMED_SEARCH_STRATEGY env var > DEFAULT_STRATEGY in
    src/snomed_search/registry.py (currently "pgvector_cascade"). When the default
    is implicit (no env override), any init or health-check failure auto-falls back
    to "hybrid_cascade"; with an explicit override, failures propagate.
"""

import os
import re
import time
import json
import logging
import asyncio
import collections
from pathlib import Path
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Annotated, Union

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.pipeline import NLPPipeline
from src.conversation import ConversationSession
from src.assembler import NLPOutput, ClarificationOutput
from src.exceptions import PipelineError, LLMProviderError
from src.preprocessor import PreprocessorError
from src.autocomplete.orchestrator import AutocompleteOrchestrator

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root before anything reads os.environ

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("clinical_nlp_api")

# ── Constants ─────────────────────────────────────────────────────────────────

SESSION_TTL_SECONDS = 1800
MAX_SESSIONS = 1000
API_KEY = os.environ.get("API_KEY")

# ── Dev-mode auth bypass ──────────────────────────────────────────────────────
# Set SKIP_AUTH=true in .env to disable key checks on ALL non-health routes.
# Never set this in production — it opens every endpoint to unauthenticated access.
_SKIP_AUTH = os.environ.get("SKIP_AUTH", "false").strip().lower() == "true"

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000"
).split(",")

SAFE_PREPROCESSOR_MESSAGES = frozenset({
    "Query must be at least 3 characters",
    "Query must be under 500 characters",
    "Invalid query detected",
    "No clinical content found. Please enter a query about a medical condition, "
    "investigator, research site, location, or study phase.",
})

_SID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
)

# ── Session store with TTL ────────────────────────────────────────────────────

class TTLSessionStore:
    def __init__(self, max_size: int = MAX_SESSIONS):
        self._store: OrderedDict[str, tuple[ConversationSession, float]] = OrderedDict()
        self._max = max_size

    def get(self, sid: str) -> ConversationSession | None:
        if sid not in self._store:
            return None
        session, ts = self._store[sid]
        if time.monotonic() - ts > SESSION_TTL_SECONDS:
            del self._store[sid]
            return None
        self._store.move_to_end(sid)
        self._store[sid] = (session, time.monotonic())
        return session

    def set(self, sid: str, session: ConversationSession) -> bool:
        self._evict_expired()
        if len(self._store) >= self._max and sid not in self._store:
            return False
        self._store[sid] = (session, time.monotonic())
        self._store.move_to_end(sid)
        return True

    def delete(self, sid: str) -> None:
        self._store.pop(sid, None)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, ts) in self._store.items()
                   if now - ts > SESSION_TTL_SECONDS]
        for k in expired:
            del self._store[k]

sessions = TTLSessionStore()
_session_lock = asyncio.Lock()

# ── Autocomplete rate limiting ────────────────────────────────────────────────

_AUTOCOMPLETE_RATE_WINDOW: float = 10.0
_AUTOCOMPLETE_RATE_MAX: int = 30
_autocomplete_rate_store: dict[str, collections.deque] = {}
_autocomplete_rate_lock: asyncio.Lock = asyncio.Lock()


async def _check_autocomplete_rate(client_ip: str) -> bool:
    """Return True if this IP is within the autocomplete rate limit.

    Enforces 30 requests per 10-second rolling window per IP address.
    Evicts stale IP entries when the store exceeds 10000 entries to
    prevent unbounded memory growth under high-cardinality IP loads.
    HIPAA: client_ip is infrastructure metadata, not user content.
    """
    now = time.monotonic()
    async with _autocomplete_rate_lock:
        if len(_autocomplete_rate_store) > 10_000:
            cutoff = now - _AUTOCOMPLETE_RATE_WINDOW
            stale = [
                ip for ip, dq in _autocomplete_rate_store.items()
                if not dq or dq[-1] < cutoff
            ]
            for ip in stale:
                _autocomplete_rate_store.pop(ip, None)

        if client_ip not in _autocomplete_rate_store:
            _autocomplete_rate_store[client_ip] = collections.deque()

        window = _autocomplete_rate_store[client_ip]
        cutoff = now - _AUTOCOMPLETE_RATE_WINDOW
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= _AUTOCOMPLETE_RATE_MAX:
            return False

        window.append(now)
        return True

# ── Pipeline singleton ────────────────────────────────────────────────────────

pipeline: NLPPipeline | None = None
autocomplete: AutocompleteOrchestrator | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, autocomplete

    if _SKIP_AUTH:
        log.warning(
            "SKIP_AUTH=true — API key checks disabled. "
            "Do not use this setting in production."
        )

    try:
        pipeline = NLPPipeline()
        log.info("Pipeline initialized successfully")

        _geo_path = Path(__file__).parent / "data" / "geo_canonical.json"
        try:
            with open(_geo_path, encoding="utf-8") as _gf:
                _geo_data = json.load(_gf)
            autocomplete = AutocompleteOrchestrator(pipeline.snomed_strategy, _geo_data)
            log.info("AutocompleteOrchestrator initialized successfully")
        except Exception as _ac_exc:
            log.warning(
                "AutocompleteOrchestrator init failed (%s) — /v1/autocomplete will return 503",
                type(_ac_exc).__name__,
            )
            autocomplete = None
    except Exception as e:
        log.critical("Pipeline init failed: %s", type(e).__name__)
        raise
    yield

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Clinical NLP API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# ── Auth middleware ───────────────────────────────────────────────────────────

# Paths that never require an API key regardless of SKIP_AUTH.
_AUTH_EXEMPT_PATHS = frozenset({"/health/live", "/health/ready"})


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if request.url.path in _AUTH_EXEMPT_PATHS:
        return await call_next(request)

    # Dev bypass — log loudly so it's impossible to miss in stdout.
    if _SKIP_AUTH:
        return await call_next(request)

    if API_KEY:
        key = request.headers.get("X-API-Key", "")
        if key != API_KEY:
            log.warning(
                "Unauthorized request path=%s key_length_sent=%d",
                request.url.path,
                len(key),          # length only — never the key value itself
            )
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)

# ── Models ────────────────────────────────────────────────────────────────────

ResponseOutput = Annotated[
    Union[NLPOutput, ClarificationOutput],
    Field(discriminator="type")
]

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    session_id: str | None = Field(default=None)

class QueryResponse(BaseModel):
    result: ResponseOutput
    session_id: str
    processing_time_ms: int

class AutocompleteSuggestionItem(BaseModel):
    display: str
    completion: str
    snomed_code: str | None
    category: str
    confidence: float
    match_type: str
    deprecated: bool

class AutocompleteResponse(BaseModel):
    suggestions: list[AutocompleteSuggestionItem]
    processing_time_ms: int
    tier_used: str
    fallback_active: bool

# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_session_id(sid: str | None) -> str | None:
    if sid is None:
        return None
    if not _SID_PATTERN.match(sid):
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    return sid

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health/live")
def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
def readiness():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return {"status": "ready"}


@app.post("/v1/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    sid = _validate_session_id(req.session_id)

    async with _session_lock:
        session = sessions.get(sid) if sid else None
        if session is None:
            session = ConversationSession.new()

    loop = asyncio.get_event_loop()
    start = time.monotonic()
    try:
        result: Union[NLPOutput, ClarificationOutput] = await loop.run_in_executor(
            None, pipeline.run_with_session, req.query, session
        )
    except PreprocessorError as e:
        msg = str(e)
        safe_msg = msg if msg in SAFE_PREPROCESSOR_MESSAGES else "Invalid query"
        raise HTTPException(status_code=400, detail=safe_msg)
    except (LLMProviderError, PipelineError):
        log.error(
            "Pipeline error session_id=%s turn_count=%d",
            session.session_id,
            len(session.turns),
        )
        raise HTTPException(status_code=502, detail="Pipeline error")

    elapsed = int((time.monotonic() - start) * 1000)

    async with _session_lock:
        ok = sessions.set(session.session_id, session)
        if not ok:
            raise HTTPException(status_code=503, detail="Session limit reached")

    return QueryResponse(
        result=result,
        session_id=session.session_id,
        processing_time_ms=elapsed,
    )


@app.delete("/v1/session/{session_id}")
async def clear_session(session_id: str):
    sid = _validate_session_id(session_id)
    async with _session_lock:
        sessions.delete(sid)
    return {"cleared": sid}


@app.get("/v1/autocomplete", response_model=AutocompleteResponse)
async def run_autocomplete(
    request: Request,
    q: str = Query(..., min_length=3, max_length=100),
    limit: int = Query(default=7, ge=1, le=10),
):
    """Clinical query autocomplete endpoint.

    Returns ranked suggestions for the given prefix string. Returns an
    empty suggestions array on no match and never raises a 500 for
    valid input. The q parameter is never logged, only its length is.

    Auth: X-API-Key header required via existing middleware (bypassed if SKIP_AUTH=true).
    Rate: 30 requests per 10 seconds per IP address.
    HIPAA: q parameter is never logged.
    """
    if autocomplete is None:
        raise HTTPException(status_code=503, detail="Autocomplete not ready")

    client_ip = request.client.host if request.client else "unknown"
    if not await _check_autocomplete_rate(client_ip):
        log.info(
            "autocomplete rate limit exceeded client_ip_hash=%s",
            hash(client_ip) & 0xFFFF,
        )
        raise HTTPException(
            status_code=429,
            detail="Autocomplete rate limit exceeded. Please slow down.",
        )

    result = await autocomplete.run(q, limit=limit)
    return AutocompleteResponse(**result)