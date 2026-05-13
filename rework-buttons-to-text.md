# Spec: Clarification UI — Buttons to Hint-List + chat_input

**Branch:** Arch-changes-with-response  
**Date:** 2026-05-11  
**Author:** Architect  
**Status:** DESIGN SPEC — DO NOT CODE UNTIL QA/Security sign-off

---

## 1. Files Changed

| File | Rationale |
|---|---|
| `app.py` | Only file with UI rendering logic. Two targeted removals + one replacement block inside `_render_turn_result`. The `pending_input` consumer in `main()` is also removed. No other file touches clarification rendering. |

No changes to `src/`, `data/`, `tests/`, or `qa_testing/`.

---

## 2. Removed Code

### 2a. `pending_input` consumer — `main()` lines 345–349

**Decision A:** Remove the entire block:

```python
# lines 345-349 — DELETE IN FULL
if "pending_input" in st.session_state:
    pending = st.session_state.pop("pending_input")
    _run_pipeline_turn_and_capture(pipeline, pending, session, turn_outputs)
    return
```

Rationale: this block existed solely to handle button clicks. With buttons gone, nothing ever writes `pending_input` to session state. The block becomes unreachable dead code; deleting it eliminates the dead branch and the `return` that would have skipped the rest of `main()`.

### 2b. Button rendering block — `_render_turn_result()` lines 238–249

**Decision B:** Remove the `st.columns` layout, the `for i, option` loop, and the `st.button` + `st.session_state["pending_input"]` + `st.rerun()` calls. Specifically, delete:

```python
# lines 238-249 — DELETE IN FULL
if clarif.options:
    st.markdown("*Select an option or type your own answer:*")
    cols = st.columns(min(len(clarif.options), 3))
    for i, option in enumerate(clarif.options):
        col = cols[i % len(cols)]
        with col:
            if st.button(
                _safe(option),
                key=f"opt_{turn_index}_{i}",
            ):
                st.session_state["pending_input"] = option
                st.rerun()
```

The surrounding `if clarif is not None and isinstance(clarif, ClarificationOutput):` guard (line 236) and the `else: st.markdown("*Please clarify your query.*")` fallback (lines 250–251) are retained unchanged.

---

## 3. Replacement UI

<!-- override (post-Security review 2026-05-11):
     ResponseAssembler.build_clarification() at src/assembler.py:122-124 already
     calls html.escape() on `question` and on each `option`. Applying _safe()
     again at render time double-escapes (e.g. "Hodgkin's & Non-Hodgkin's" →
     "Hodgkin&amp;#39;s &amp;amp;..." — visible UX bug for strings with & < >).
     The current app.py:237 and app.py:245 already have this latent bug; since
     we're touching the render code, fix it inline. Assembler owns escaping;
     render site trusts already-escaped strings.
     Also drop unsafe_allow_html=True from the hint list AND the question:
     escaped-entity text renders identically without the flag, and dropping it
     tightens defense-in-depth against future markdown/HTML injection. -->

**Decision C (revised):** Replace the deleted block (lines 238–249) with:

```python
if clarif.options:
    items = "\n".join(f"- {opt}" for opt in clarif.options)
    st.markdown(items)
st.markdown("*Type your answer in the box below.*")
```

**Decision C.2 (new):** Also update the question render at line 237 from
`st.markdown(f"**{_safe(clarif.question)}**", unsafe_allow_html=True)` to
`st.markdown(f"**{clarif.question}**")` — same rationale (already escaped in assembler, no raw HTML needed).

Notes on this replacement:

- Options and `clarif.question` are ALREADY `html.escape()`'d in `ResponseAssembler.build_clarification()` (`src/assembler.py:122-124`). Re-applying `_safe()` at the render site produces double-escaped output. This work item fixes that latent bug.
- `unsafe_allow_html=True` was unnecessary on these strings: HTML-entity text (`&amp;`, `&lt;`) renders correctly in Streamlit's default markdown WITHOUT the flag. The flag's only effect is to allow raw HTML tags through unsanitized — which we don't want and don't need here. Dropping it is a defense-in-depth win.
- The instruction line `"*Type your answer in the box below.*"` is a static string literal; it does not touch user-derived data and does not need escaping.
- The hint list is omitted entirely when `clarif.options` is empty (the `if clarif.options:` guard). Only the instruction line renders. See section 4.
- Markdown link syntax `[label](url)` is processed by Streamlit's markdown renderer regardless of the html flag. Current option sources (curated `ambiguous_terms.json` + SNOMED CSV `preferred_term`) do not contain `[](` sequences. This is acceptable per the threat model; if options become end-user-controllable in the future, escape `[`/`]`/`(`/`)` at the assembler level.
- No `key=` argument needed — there are no interactive widgets in this block.

Full resulting block in context (lines 236–252 after edit):

```python
if clarif is not None and isinstance(clarif, ClarificationOutput):
    st.markdown(f"**{clarif.question}**")
    if clarif.options:
        items = "\n".join(f"- {opt}" for opt in clarif.options)
        st.markdown(items)
    st.markdown("*Type your answer in the box below.*")
else:
    st.markdown("*Please clarify your query.*")
```

---

## 4. Empty-Options Edge Case

**Decision D:** When `clarif.options` is an empty list the `if clarif.options:` guard evaluates `False`; the hint list block is skipped. The instruction line `"*Type your answer in the box below.*"` still renders below the question.

This covers two realistic sources of an empty list:

- **Layer 2 (`EmbeddingAmbiguityGate`) Signal B path** where `triggered_by=None` and options are derived from embedding neighbors — in theory always non-empty, but a defensive empty-list is possible if no neighbors exceed the threshold.
- **Future pipeline extensions** that intentionally emit open-ended clarifications without enumerated options.

No special error state is needed. The user sees the question and the instruction, and free-text input works identically. This is the same graceful behavior the chat already provides today for Layer 2 append-mode clarifications.

---

## 5. Tests

### 5a. Tests that require NO changes

| Test file | Why it passes unchanged |
|---|---|
| `tests/test_conversation.py` | Tests feed option text as a raw `user_input` string directly to `run_with_session()` / `compute_canonical_query()`. No Streamlit buttons are simulated anywhere. The B6 lambda fix, substitute/append logic, and max-turns cap are unaffected. |
| `tests/test_sufficiency_gate.py` | Gate logic is pipeline-only; no UI dependency. |
| `tests/test_ambiguity_coverage.py` | Same — feeds strings, inspects `ClarificationOutput`/`NLPOutput`. |
| `tests/test_snomed_strategies.py` | SNOMED strategy; no UI. |
| `tests/test_negation.py` | Negation annotator; no UI. |
| `tests/test_llm_provider.py` | Provider mock; no UI. |
| `tests/batch_eval.py` | Multi-turn cases use the `>>>` separator format to supply option text as the next input string. No button simulation. |
| `qa_testing/test_agent.py` | Calls `pipeline.run_with_session()` directly. Does not import or invoke Streamlit. |

### 5b. New tests needed

**Decision E: No new automated tests are required.** The change removes Streamlit widget state mutations (`pending_input`, `st.button`, `st.rerun`) and replaces them with a pure markdown render. There is no branching logic to cover: `if clarif.options` is a trivial guard already in the existing code. The empty-options edge case (section 4) is covered by the guard evaluating `False` — a one-liner with no side effects.

A manual smoke test is sufficient:

1. Submit a query that triggers Layer 1 clarification (e.g., "cancer in Boston phase 3"). Verify: question renders in bold, options render as a bulleted list, instruction line appears, `st.chat_input` is present and accepts free-text.
2. Submit a query that triggers Layer 2 clarification (e.g., an ambiguous single anatomy term). Same verification.
3. Type one of the listed options verbatim into `st.chat_input` and submit. Verify the pipeline runs the substitution path correctly.
4. Type a free-text answer not matching any option. Verify the pipeline runs the append path correctly.
5. Verify clicking "New Search" resets the conversation (unchanged behavior — sidebar button not touched).

---

## 6. Security / HIPAA Notes

**Decision F: No new attack vector is introduced.**

The security posture is unchanged or marginally improved:

- Free-text via `st.chat_input` was already accepted during clarification turns. `conversation_terminal` evaluated `False` for clarification turns (lines 360–365), so `st.chat_input` was already rendered and active. The preprocessor stack (`Preprocessor.process()` → `assert_safe(canonical)`) already ran on every chat-input submission. Removing buttons does not bypass or alter this path.
- The `pending_input` mechanism (write session state → rerun → consume) was an additional code path that fed the same option text into `_run_pipeline_turn_and_capture`. Removing it removes a session-state mutation point, which is a net reduction in attack surface, not an increase.
- `_safe()` / `html.escape()` continues to wrap all user-derived and LLM-derived display strings. The hint list applies `_safe()` to each option individually before composing the markdown string. The `unsafe_allow_html=True` flag on the `st.markdown` call is necessary for the escaped characters to render correctly (e.g., `&amp;`, `&lt;`) and is consistent with the existing usage on line 237.
- HIPAA log hygiene is unaffected. No new log statements are added. The `pending_input` pop at lines 346–349 emitted no log lines, so its removal changes nothing in the log record.
- Rate limiting logic is untouched. Clarifications still count toward the 5/60s and 30/session caps because the user now submits via `st.chat_input` which routes through the same `_run_pipeline_turn_and_capture` function.

**Reviewer checkpoint:** Confirm that `_safe()` at the option-list render site wraps individual option strings before string interpolation into the markdown, not after. The spec as written (`f"- {_safe(opt)}"`) satisfies this.

---

## 7. Out of Scope

The following are explicitly NOT changing in this work item:

- **Pipeline** (`src/pipeline.py`) — `run_with_session()`, parallel execution, all gate logic, turn appending.
- **SufficiencyGate / AmbiguousTermsRegistry / EmbeddingAmbiguityGate** — trigger detection, option derivation, canonical-query merge.
- **`ClarificationOutput` schema** (`src/assembler.py`) — `question`, `options`, `canonical_query`, `turn_number`, `max_turns`, `metadata` fields unchanged.
- **`canonical_query` merge** (`src/conversation.py:compute_canonical_query`) — substitute-or-append logic, B6 lambda fix. Free-text answers already route through this correctly.
- **Log lines** — no new log statements; existing log statements unchanged.
- **Rate limiting** — session-state counters and timestamps untouched.
- **`_render_nlp_output()`** — search-result rendering unchanged.
- **`render_sidebar()`** — sidebar content, New Search button, system status unchanged.
- **`_run_pipeline_turn_and_capture()`** — pipeline invocation, error handling, rerun logic unchanged.
- **`conversation_terminal` evaluation** (lines 360–365) — clarification turns already evaluated `False`; no change needed.
- **`data/ambiguous_terms.json`** — trigger/option definitions unchanged.
- **`requirements.txt`** — no new dependencies.
- **All test files** — no edits (see section 5a).
- **`context.md`** — update the `app.py` description line in the architecture section after implementation to remove the button reference. That is a documentation-only follow-up, not part of this work item.

---

## Summary of Decisions

| ID | Decision | Rationale |
|---|---|---|
| A | Delete `pending_input` consumer block (lines 345–349) | Dead code after buttons removed; avoids silent no-op on future stale state |
| B | Delete `st.columns` / `st.button` loop (lines 238–249) | Core change; replaces button widgets with passive hint list |
| C | Replace with plain bulleted `st.markdown` (no `_safe`, no `unsafe_allow_html`) + static instruction line | Options pre-escaped in assembler; drops latent double-escape bug and unneeded HTML flag |
| C.2 | Drop `_safe()` and `unsafe_allow_html=True` from question render (line 237) | Same rationale as C |
| D | Empty options: skip hint list, keep instruction line | Graceful; Layer 2 open-ended clarifications already work without listed options |
| E | No new automated tests | Pure UI simplification; branching logic reduced, not added |
| F | No new security concerns | Free-text path already existed and was already guarded by full preprocessor stack; defense-in-depth improved by dropping `unsafe_allow_html=True` |
