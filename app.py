"""Clinical Research NLP — Streamlit application entry point."""

import html
import logging
import os
import time
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from src.pipeline import NLPPipeline
from src.assembler import NLPOutput
from src.preprocessor import PreprocessorError
from src.extractor import ExtractionError

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

EXAMPLE_QUERIES = [
    "Phase 3 T2DM trials in NYC",
    "Dr. Smith breast cancer research at Mayo Clinic",
    "NSCLC immunotherapy Phase 2 in California",
    "Alzheimer's disease studies in the Bay Area",
    "CHF trials excluding diabetes in Boston",
]

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

    # Trim timestamps outside the rolling window
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


def render_sidebar() -> None:
    """Render sidebar with About, Examples, PHI notice, and System Status."""
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

        st.header("Example Queries")
        for query in EXAMPLE_QUERIES:
            if st.button(query, key=f"example_{query[:20]}"):
                st.session_state["query_input"] = query

        st.header("System Status")
        api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
        if api_key:
            st.markdown("🟢 **API Key:** Configured")
        else:
            st.markdown("🔴 **API Key:** Not found")

        total = st.session_state.get("query_total", 0)
        st.caption(f"Queries this session: {total}/{_RATE_LIMIT_MAX}")


def render_snomed_column(output: NLPOutput) -> None:
    """Render the left column showing SNOMED term results."""
    st.subheader("🧬 SNOMED Terms")
    st.metric("Terms Found", output.metadata.total_snomed_matches)

    if not output.snomed_terms:
        st.info("No SNOMED terms matched above the confidence threshold.")
        return

    for term in output.snomed_terms:
        with st.container():
            st.code(term.code, language=None)
            # Sanitize display values that originate from LLM-extracted text
            st.markdown(f"**{_safe(term.display)}**", unsafe_allow_html=True)
            st.markdown(_badge(term.match_type), unsafe_allow_html=True)
            conf_val = min(max(float(term.confidence), 0.0), 1.0)
            st.progress(conf_val, text=f"{conf_val * 100:.0f}% confidence")
            if term.negated:
                st.caption("(excluded — negated)")
            st.divider()


def render_filters_column(output: NLPOutput) -> None:
    """Render the right column showing structured filter results."""
    st.subheader("🔎 Search Filters")
    filters = output.filters

    scalar_items = [
        ("🧑 Investigator", filters.investigator_name),
        ("🏥 Site", filters.site_name),
        ("🏙️ City", filters.city),
        ("🔬 Phase", filters.phase),
    ]
    state_non_empty = bool(filters.state.values)
    non_null_count = sum(1 for _, f in scalar_items if f.value) + (1 if state_non_empty else 0)
    st.metric("Filters Found", non_null_count)

    for label, field in scalar_items:
        if field.value:
            st.markdown(f"**{label}**")
            st.markdown(f"**{_safe(field.value)}**", unsafe_allow_html=True)
            conf_val = min(max(float(field.confidence), 0.0), 1.0)
            st.progress(conf_val, text=f"{conf_val * 100:.0f}% confidence")
            st.divider()

    # State — may be one state or a list of states for regional queries
    if state_non_empty:
        st.markdown("**📍 State**")
        conf_val = min(max(float(filters.state.confidence), 0.0), 1.0)
        if filters.state.is_region and len(filters.state.values) > 1:
            st.markdown(f"*Region — {len(filters.state.values)} states:*")
            for s in filters.state.values:
                st.markdown(f"- {_safe(s)}", unsafe_allow_html=True)
        else:
            st.markdown(f"**{_safe(filters.state.values[0])}**", unsafe_allow_html=True)
        st.progress(conf_val, text=f"{conf_val * 100:.0f}% confidence")
        st.divider()


def main() -> None:
    """Main Streamlit application."""
    render_sidebar()

    st.title("🏥 Clinical Research NLP")
    st.markdown("*Extract SNOMED CT concepts and structured filters from clinical research queries.*")

    # PHI reminder inline
    st.info(
        "**Do not include patient names, MRNs, dates of birth, or any patient-identifying "
        "information in your query.** Queries are sent to a third-party LLM API for processing.",
        icon="ℹ️",
    )

    pipeline = load_pipeline()
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

    query_value = st.session_state.get("query_input", "")
    query = st.text_area(
        "Enter your clinical research query:",
        value=query_value,
        max_chars=500,
        height=100,
        key="query_text_area",
        placeholder="e.g. Phase 3 type 2 diabetes trials in New York with Dr. Johnson",
    )

    if st.button("🔍 Analyze Query", type="primary"):
        if not query or not query.strip():
            st.warning("Please enter a query before analyzing.")
            return

        if not _check_rate_limit():
            st.error(
                "Too many requests. Please wait a moment before submitting another query."
            )
            return

        with st.spinner("Analyzing query..."):
            try:
                result: NLPOutput = pipeline.run(query)
                st.session_state["last_result"] = result
            except PreprocessorError as exc:
                st.warning(f"Query issue: {exc}")
                return
            except ExtractionError as exc:
                st.error(f"Analysis failed: {exc}")
                return
            except Exception:
                st.error("Unexpected error. Please try again.")
                logger.exception("Pipeline error")
                return

    if "last_result" in st.session_state:
        output: NLPOutput = st.session_state["last_result"]

        col_left, col_right = st.columns(2)
        with col_left:
            render_snomed_column(output)
        with col_right:
            render_filters_column(output)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Processing Time (ms)", output.metadata.processing_time_ms)
        m2.metric("SNOMED Matches", output.metadata.total_snomed_matches)
        filters = output.filters
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


if __name__ == "__main__":
    main()
