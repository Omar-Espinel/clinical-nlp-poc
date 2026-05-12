"""Clinical Research NLP — Streamlit chat application entry point (v2 rework)."""

import html
import logging
import os
import time
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from src.pipeline import NLPPipeline
from src.assembler import NLPOutput, ClarificationOutput
from src.conversation import ConversationSession
from src.preprocessor import PreprocessorError
from src.exceptions import ExtractionError, PipelineError, LLMProviderError

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Per-session rate limiting ─────────────────────────────────────────────────
_RATE_LIMIT_MAX = 30        # max queries per session
_RATE_LIMIT_WINDOW = 60.0   # rolling window in seconds
_RATE_LIMIT_BURST = 5       # max queries within the window

st.set_page_config(
    page_title="Clinical Research NLP",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

MATCH_TYPE_COLORS = {
    "exact": "#28a745",
    "synonym": "#007bff",
    "fuzzy": "#6f42c1",
    "semantic": "#fd7e14",
}


@st.cache_resource
def load_pipeline() -> Optional[NLPPipeline]:
    """Load all pipeline components once and cache across sessions."""
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return NLPPipeline(groq_api_key=api_key)


def _safe(value: str) -> str:
    """HTML-escape a user-derived string before embedding in markdown."""
    return html.escape(str(value))


def _badge(match_type: str) -> str:
    # match_type comes exclusively from internal resolution logic, not user input
    color = MATCH_TYPE_COLORS.get(match_type, "#6c757d")
    safe_type = html.escape(match_type)
    return (
        f'<span style="background-color:{color};color:white;'
        f'padding:2px 8px;border-radius:4px;font-size:0.75em;">'
        f"{safe_type}</span>"
    )


def _check_rate_limit() -> bool:
    """Return True if the request is allowed, False if rate-limited.

    Uses session state to track a rolling window of query timestamps.
    This is a best-effort per-session limit; Groq enforces account-level limits.
    """
    now = time.time()
    if "query_timestamps" not in st.session_state:
        st.session_state["query_timestamps"] = []
    if "query_total" not in st.session_state:
        st.session_state["query_total"] = 0

    window_start = now - _RATE_LIMIT_WINDOW
    st.session_state["query_timestamps"] = [
        t for t in st.session_state["query_timestamps"] if t > window_start
    ]

    if st.session_state["query_total"] >= _RATE_LIMIT_MAX:
        return False
    if len(st.session_state["query_timestamps"]) >= _RATE_LIMIT_BURST:
        return False

    st.session_state["query_timestamps"].append(now)
    st.session_state["query_total"] += 1
    return True


def _render_nlp_output(output: NLPOutput) -> None:
    """Render NLPOutput as two-column SNOMED + filters view."""
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🧬 SNOMED Terms")
        st.metric("Terms Found", output.metadata.total_snomed_matches)
        if not output.snomed_terms:
            st.info("No SNOMED terms matched above the confidence threshold.")
        else:
            for term in output.snomed_terms:
                with st.container():
                    st.code(term.code, language=None)
                    st.markdown(f"**{_safe(term.display)}**", unsafe_allow_html=True)
                    st.markdown(_badge(term.match_type), unsafe_allow_html=True)
                    conf_val = min(max(float(term.confidence), 0.0), 1.0)
                    st.progress(conf_val, text=f"{conf_val * 100:.0f}% confidence")
                    if term.negated:
                        st.caption("(excluded — negated)")
                    st.divider()

    with col_right:
        st.subheader("🔎 Search Filters")
        filters = output.filters
        scalar_items = [
            ("🧑 Investigator", filters.investigator_name),
            ("🏥 Site", filters.site_name),
            ("🏙️ City", filters.city),
            ("🔬 Phase", filters.phase),
        ]
        state_non_empty = bool(filters.state.values)
        non_null_count = (
            sum(1 for _, f in scalar_items if f.value)
            + (1 if state_non_empty else 0)
        )
        st.metric("Filters Found", non_null_count)

        for label, field in scalar_items:
            if field.value:
                st.markdown(f"**{label}**")
                st.markdown(f"**{_safe(field.value)}**", unsafe_allow_html=True)
                conf_val = min(max(float(field.confidence), 0.0), 1.0)
                st.progress(conf_val, text=f"{conf_val * 100:.0f}% confidence")
                st.divider()

        if state_non_empty:
            st.markdown("**📍 State**")
            conf_val = min(max(float(filters.state.confidence), 0.0), 1.0)
            if filters.state.is_region and len(filters.state.values) > 1:
                st.markdown(f"*Region — {len(filters.state.values)} states:*")
                for s in filters.state.values:
                    st.markdown(f"- {_safe(s)}", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"**{_safe(filters.state.values[0])}**", unsafe_allow_html=True
                )
            st.progress(conf_val, text=f"{conf_val * 100:.0f}% confidence")
            st.divider()

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Processing Time (ms)", output.metadata.processing_time_ms)
    m2.metric("SNOMED Matches", output.metadata.total_snomed_matches)
    non_null = sum(
        1 for f in [
            filters.investigator_name,
            filters.site_name,
            filters.city,
            filters.phase,
        ]
        if f.value
    ) + (1 if filters.state.values else 0)
    m3.metric("Filters Found", non_null)

    with st.expander("📋 Structured Output (JSON)"):
        st.json(output.model_dump())


def _run_pipeline_turn(pipeline: NLPPipeline, user_input: str) -> None:
    """Run one pipeline turn, appending result to session state; rerun on success."""
    session: ConversationSession = st.session_state["conversation"]

    if not _check_rate_limit():
        st.session_state.setdefault("pending_error", None)
        with st.chat_message("assistant"):
            st.warning(
                "Too many requests. Please wait a moment before submitting another query."
            )
        return

    with st.spinner("Analyzing query..."):
        try:
            result = pipeline.run_with_session(user_input, session)
            logger.info(
                "run_with_session completed %s",
                session.summary_for_logging(),
                # NOT logged: user_input, result content
            )
            st.rerun()
        except PreprocessorError:
            with st.chat_message("assistant"):
                st.warning("Query issue: please check your input and try again.")
            logger.info(
                "PreprocessorError type=PreprocessorError session=%s",
                session.session_id,
            )
        except (ExtractionError, PipelineError, LLMProviderError):
            with st.chat_message("assistant"):
                st.error("Analysis failed. Please try again in a moment.")
            logger.info(
                "pipeline error type=%s session=%s",
                "ExtractionError/PipelineError/LLMProviderError",
                session.session_id,
            )
        except Exception as exc:
            with st.chat_message("assistant"):
                st.error("Unexpected error. Please try again.")
            logger.error(
                "Unexpected pipeline error type=%s session=%s",
                type(exc).__name__, session.session_id,
                exc_info=False,
            )


def _render_turn_result(turn_index: int, turn) -> None:
    """Render the assistant bubble for a completed turn."""
    decision = turn.decision
    sufficient = decision is None or decision.sufficient

    if sufficient:
        # Terminal turn — the last snomed_matches / filters came from this turn.
        # NLPOutput shape is in the session via the pipeline; reconstruct display
        # from the Turn's stored snomed_matches and the latest assembler output.
        # The actual NLPOutput is stored in session_state alongside the session.
        outputs = st.session_state.get("turn_outputs", {})
        output = outputs.get(turn_index)
        if output is not None and isinstance(output, NLPOutput):
            _render_nlp_output(output)
        else:
            st.info("Results processed successfully.")
    else:
        # Clarification turn — render question + hint list; user replies via chat_input.
        clarif = st.session_state.get("turn_outputs", {}).get(turn_index)
        if clarif is not None and isinstance(clarif, ClarificationOutput):
            st.markdown(f"**{clarif.question}**")
            if clarif.options:
                items = "\n".join(f"- {opt}" for opt in clarif.options)
                st.markdown(items)
            st.markdown("*Type your answer in the box below.*")
        else:
            st.markdown("*Please clarify your query.*")


def render_sidebar(pipeline_ready: bool) -> None:
    """Render sidebar with PHI disclaimer, New Search button, and system status."""
    with st.sidebar:
        st.header("About")
        st.markdown(
            "This tool extracts **SNOMED CT concepts** and **structured filters** "
            "(investigator, site, city, state, phase) from natural language clinical "
            "research queries using Groq LLM + vector search."
        )

        st.warning(
            "**Privacy Notice:** Do not enter patient names, patient IDs, "
            "dates of birth, or any other patient-identifying information "
            "(PHI/PII) in queries. This tool is for research protocol "
            "navigation only.",
            icon="⚠️",
        )

        if st.button("🔄 New Search", key="new_search_btn"):
            st.session_state["conversation"] = ConversationSession.new()
            st.session_state["turn_outputs"] = {}
            st.session_state["query_timestamps"] = []
            st.session_state["query_total"] = 0
            st.rerun()

        st.header("System Status")
        api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
        if api_key:
            st.markdown("🟢 **API Key:** Configured")
        else:
            st.markdown("🔴 **API Key:** Not found")

        if pipeline_ready:
            st.markdown("🟢 **Pipeline:** Ready")
        else:
            st.markdown("🔴 **Pipeline:** Not loaded")

        total = st.session_state.get("query_total", 0)
        st.caption(f"Queries this session: {total}/{_RATE_LIMIT_MAX}")


def _init_session_state() -> None:
    """Initialize session state keys on first load."""
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = ConversationSession.new()
    if "turn_outputs" not in st.session_state:
        st.session_state["turn_outputs"] = {}
    if "query_timestamps" not in st.session_state:
        st.session_state["query_timestamps"] = []
    if "query_total" not in st.session_state:
        st.session_state["query_total"] = 0


def _sync_turn_outputs(pipeline: NLPPipeline) -> None:
    """After a rerun, fill any turn_outputs gaps by re-running the pipeline's
    stored turn data. In practice the pipeline mutates session turns directly,
    so we store outputs in turn_outputs at run time — this is a no-op if already filled."""
    # The actual result objects come from _run_pipeline_turn_and_capture.
    # Nothing to do here.
    pass


def main() -> None:
    """Main Streamlit application — chat UI."""
    _init_session_state()

    pipeline = load_pipeline()
    render_sidebar(pipeline_ready=pipeline is not None)

    st.title("🏥 Clinical Research NLP")
    st.markdown(
        "*Extract SNOMED CT concepts and structured filters from clinical research queries.*"
    )

    if pipeline is None:
        st.error(
            "⚠️ GROQ_API_KEY not found.\n\n"
            "To run locally: Add GROQ_API_KEY to your .env file\n\n"
            "To run on HuggingFace: Add GROQ_API_KEY to Space Secrets\n\n"
            "Get a free key at: https://console.groq.com"
        )
        st.stop()

    if "pipeline_loaded" not in st.session_state:
        with st.spinner("Loading Clinical NLP System... (first load 30-60s)"):
            _ = load_pipeline()
        st.session_state["pipeline_loaded"] = True

    session: ConversationSession = st.session_state["conversation"]
    turn_outputs: dict = st.session_state["turn_outputs"]

    # ── Render conversation history ───────────────────────────────────────────
    for i, turn in enumerate(session.turns):
        with st.chat_message("user"):
            # user_input is user-derived; escape before rendering
            st.markdown(_safe(turn.user_input), unsafe_allow_html=True)

        with st.chat_message("assistant"):
            _render_turn_result(i, turn)

    # ── Determine if conversation is still open ───────────────────────────────
    last_turn = session.turns[-1] if session.turns else None
    conversation_terminal = (
        last_turn is not None
        and (last_turn.decision is None or last_turn.decision.sufficient)
    )

    if conversation_terminal:
        st.info(
            "Search complete. Click **New Search** in the sidebar to start a new query.",
            icon="✅",
        )
        return

    # ── Chat input for next turn ──────────────────────────────────────────────
    user_input = st.chat_input("Type your clinical research query...")
    if user_input and user_input.strip():
        _run_pipeline_turn_and_capture(pipeline, user_input.strip(), session, turn_outputs)


def _run_pipeline_turn_and_capture(
    pipeline: NLPPipeline,
    user_input: str,
    session: ConversationSession,
    turn_outputs: dict,
) -> None:
    """Run one pipeline turn, capture the output, and rerun."""
    if not _check_rate_limit():
        with st.chat_message("assistant"):
            st.warning(
                "Too many requests. Please wait a moment before submitting another query."
            )
        return

    turn_index = len(session.turns)

    with st.spinner("Analyzing query..."):
        try:
            result = pipeline.run_with_session(user_input, session)
            logger.info(
                "turn_complete %s",
                session.summary_for_logging(),
                # NOT logged: user_input, result
            )
            turn_outputs[turn_index] = result
            st.session_state["turn_outputs"] = turn_outputs
            st.rerun()
        except PreprocessorError:
            logger.info(
                "PreprocessorError session_id=%s", session.session_id
            )
            with st.chat_message("assistant"):
                st.warning("Query issue: please check your input and try again.")
        except (ExtractionError, PipelineError, LLMProviderError):
            logger.info(
                "pipeline_error type=ExtractionError/PipelineError/LLMProviderError session_id=%s",
                session.session_id,
            )
            with st.chat_message("assistant"):
                st.error("Analysis failed. Please try again in a moment.")
        except Exception as exc:
            logger.error(
                "Unexpected error type=%s session_id=%s",
                type(exc).__name__, session.session_id,
                exc_info=False,
            )
            with st.chat_message("assistant"):
                st.error("Unexpected error. Please try again.")


if __name__ == "__main__":
    main()
