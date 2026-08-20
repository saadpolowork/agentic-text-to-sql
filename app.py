"""
Agentic Text-to-SQL — live demo.

Watch a 3-stage agentic pipeline reason its way from a natural-language question
to executed SQL: exploration → hypothesis review → 5 candidate answers.

Run locally:  streamlit run app.py   (set GROQ_API_KEY in env or .streamlit/secrets.toml)
"""

import glob
import json
import os

import streamlit as st

from pipeline.runner import run_pipeline
from pipeline.llm import LLMError, RateLimitedError, STAGE1_MODEL, STAGE3_MODEL

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "cached_traces")

DB_LABELS = {
    "superhero": "🦸 Superheroes (750 heroes, powers, publishers)",
    "california_schools": "🏫 California Schools (17k schools, SAT scores, funding)",
}

st.set_page_config(page_title="Agentic Text-to-SQL", page_icon="🔍", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    key = key or os.environ.get("GROQ_API_KEY", "")
    return st.session_state.get("user_api_key", "") or key


@st.cache_data
def load_samples() -> dict:
    with open(os.path.join(APP_DIR, "data", "sample_questions.json")) as f:
        return json.load(f)


@st.cache_data
def load_cached_traces() -> list:
    traces = []
    for path in sorted(glob.glob(os.path.join(CACHE_DIR, "*.json"))):
        try:
            with open(path) as f:
                traces.append(json.load(f))
        except Exception:
            pass
    return traces


def result_table(columns, rows):
    if not rows:
        st.caption("(no rows returned)")
        return
    st.dataframe(
        {c: [r[i] if i < len(r) else None for r in rows] for i, c in enumerate(columns)},
        width='stretch', height=min(38 * (len(rows) + 1), 300),
    )


def render_queries(queries):
    ok = sum(1 for q in queries if q["status"] == "success")
    st.caption(f"{len(queries)} queries executed — {ok} succeeded, {len(queries) - ok} failed")
    for q in queries:
        icon = "✅" if q["status"] == "success" else "❌"
        with st.expander(f"{icon} Q{q['n']} — {q['comment'] or '(no comment)'}"):
            st.code(q["sql"], language="sql")
            if q["status"] == "success":
                result_table(q.get("columns", []), q.get("head", []))
                if q.get("row_count"):
                    st.caption(f"{q['row_count']} row(s) fetched")
            else:
                st.error(q.get("error", "unknown error"))


def render_trace(trace, live_placeholders=None):
    """Render a complete trace (used for cached replays and after live runs)."""
    s = trace.get("stages", {})

    if "stage1" in s:
        st.subheader("Stage 1 — Explorer 🕵️")
        st.caption(f"Model: `{s['stage1']['model']}` — writes 15+ exploratory queries "
                   "before attempting any answer: DISTINCT probes, join verification, "
                   "null counts, scope checks.")
        with st.expander("💭 Phase 1 — Assumptions (written before seeing any data)"):
            st.text(s["stage1"].get("phase1", ""))
        with st.expander("📋 Phase 2 — Schema review"):
            st.text(s["stage1"].get("phase2", ""))
        st.markdown("**Exploratory queries & live results:**")
        render_queries(s["stage1"].get("queries", []))
        with st.expander("🔍 View exact prompt sent to the model"):
            st.text(s["stage1"]["system_prompt"][:4000] + "\n…")
            st.text("---- USER PROMPT ----")
            st.text(s["stage1"]["user_prompt"][:4000])

    if "stage2" in s:
        st.subheader("Stage 2 — Reviewer 🧐")
        st.caption(f"Model: `{s['stage2']['model']}` — audits every Stage 1 assumption "
                   "against real query results, tags them [CONFIRMED] / [CONTRADICTED] / "
                   "[OPEN-VALUE], then writes targeted confirmation queries.")
        with st.expander("🏷️ Phase 4 — Assumption verdicts", expanded=False):
            st.text(s["stage2"].get("phase4", ""))
        with st.expander("🎯 Phase 4b — Winning hypothesis", expanded=True):
            st.text(s["stage2"].get("phase4b", ""))
        st.markdown("**Confirmation queries & live results:**")
        render_queries(s["stage2"].get("queries", []))

    if "stage3" in s:
        st.subheader("Stage 3 — Director 🎬")
        st.caption(f"Model: `{s['stage3']['model']}` — sees findings only (no SQL), writes "
                   "5 candidate answers under genuinely different interpretations (C1–C5 sweep), "
                   "each executed against the database.")
        chosen = s["stage3"].get("chosen")
        for c in s["stage3"].get("candidates", []):
            is_chosen = c["n"] == chosen
            badge = " ⭐ CHOSEN ANSWER" if is_chosen else ""
            gold = ""
            if c.get("matches_gold") is True:
                gold = " · ✅ matches gold answer"
            elif c.get("matches_gold") is False:
                gold = " · ✖ differs from gold"
            icon = "🟢" if c["status"] == "success" else "🔴"
            with st.expander(
                f"{icon} Candidate {c['n']}{badge} — {c['comment'] or '(no comment)'}{gold}",
                expanded=is_chosen,
            ):
                st.code(c["sql"], language="sql")
                if c["status"] == "success":
                    result_table(c.get("columns", []), c.get("rows", []))
                    extra = " (truncated)" if c.get("truncated") else ""
                    st.caption(f"{c.get('row_count', 0)} row(s){extra}")
                else:
                    st.error(c.get("error", "unknown error"))
        if trace.get("gold_sql"):
            with st.expander("🥇 Official BIRD gold SQL for this question"):
                st.code(trace["gold_sql"], language="sql")
                gr = s["stage3"].get("gold_result")
                if gr:
                    result_table(gr["columns"], gr["rows"])

    if trace.get("error"):
        st.warning(trace["error"])
    if trace.get("elapsed_s"):
        st.caption(f"⏱️ Full pipeline: {trace['elapsed_s']}s")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🔍 Agentic Text-to-SQL")
st.markdown(
    "**Ask a database a question in plain English — and watch a 3-stage agentic pipeline "
    "prove its way to the answer.** No schema guessing: the pipeline explores the real "
    "database, verifies every assumption with SQL, then writes five candidate answers. "
    "This prompt stack scored **59.7% execution accuracy** on BIRD-SQL (from a 30.9% "
    "single-shot baseline with the same class of open models). "
    "[Read the research ↗](https://github.com/saadpolowork)"
)

with st.sidebar:
    st.header("⚙️ Settings")
    db_id = st.radio("Database", list(DB_LABELS), format_func=lambda k: DB_LABELS[k])
    st.divider()
    st.markdown(f"**Models (via Groq):**\n- Stages 1–2: `{STAGE1_MODEL}`\n- Stage 3: `{STAGE3_MODEL}`")
    st.divider()
    st.text_input("Your Groq API key (optional)", type="password", key="user_api_key",
                  help="The demo has a shared free-tier key which may hit rate limits under "
                       "load. Paste your own free key from console.groq.com to skip the queue. "
                       "Your key is used only for your requests and never stored.")
    st.divider()
    st.markdown("Built by **Saad Tariq** · 12 research trials on BIRD-SQL · "
                "3-stage pipeline: Explorer → Reviewer → Director")

tab_live, tab_gallery, tab_about = st.tabs(
    ["🚀 Ask live", "📚 Example gallery (instant)", "ℹ️ How it works"])

# ---------------------------------------------------------------------------
# Live tab
# ---------------------------------------------------------------------------
with tab_live:
    samples = load_samples()
    st.markdown("**Try a benchmark question** (gold answer known — the pipeline is scored "
                "against it) **or ask your own:**")

    if "question_text" not in st.session_state:
        st.session_state.question_text = ""

    cols = st.columns(2)
    for i, s_q in enumerate(samples.get(db_id, [])):
        if cols[i % 2].button(s_q["question"], key=f"sample_{db_id}_{i}",
                              width='stretch'):
            st.session_state.question_text = s_q["question"]
            st.session_state.selected_sample = s_q

    question = st.text_area("Your question", key="question_text",
                            placeholder="e.g. Which publisher has the tallest superhero?")
    run = st.button("▶️ Run the pipeline", type="primary")

    if run and question.strip():
        api_key = get_api_key()
        if not api_key:
            st.error("No API key available. Paste a free Groq API key in the sidebar "
                     "(console.groq.com) — or browse the Example gallery, which needs no key.")
        else:
            sample = st.session_state.get("selected_sample")
            evidence, gold_sql = "", ""
            if sample and sample["question"] == question.strip():
                evidence = sample.get("evidence", "")
                gold_sql = sample.get("gold_sql", "")

            status = st.status("Running the 3-stage pipeline… (typically 1–3 minutes)",
                               expanded=True)

            def on_event(ev):
                k = ev["kind"]
                if k == "stage_start":
                    status.write(f"**Stage {ev['stage']}** — {ev['label']}…")
                elif k == "llm_done" and "n_queries" in ev:
                    status.write(f"→ model wrote {ev['n_queries']} queries")
                elif k == "llm_done" and "n_candidates" in ev:
                    status.write(f"→ model wrote {ev['n_candidates']} candidate answers")
                elif k == "exec_done":
                    status.write(f"→ executed {ev['n_queries']} queries "
                                 f"({ev['n_success']} succeeded)")
                elif k == "candidate_done":
                    g = ""
                    if ev.get("matches_gold") is True:
                        g = " · matches gold ✅"
                    status.write(f"→ candidate C{ev['n']}: {ev['status']}"
                                 f" ({ev.get('row_count', 0)} rows){g}")

            try:
                trace = run_pipeline(db_id=db_id, question=question.strip(),
                                     api_key=api_key, evidence=evidence,
                                     gold_sql=gold_sql, on_event=on_event)
                status.update(label=f"Done in {trace.get('elapsed_s', '?')}s ✅",
                              state="complete", expanded=False)
                render_trace(trace)
            except RateLimitedError:
                status.update(label="Rate limited", state="error")
                st.warning("The shared free-tier key is rate-limited right now. "
                           "Paste your own free Groq key in the sidebar, or explore the "
                           "**Example gallery** tab — it replays real runs instantly.")
            except LLMError as e:
                status.update(label="LLM error", state="error")
                st.error(f"LLM call failed: {e}")
    elif run:
        st.info("Type or pick a question first.")

# ---------------------------------------------------------------------------
# Gallery tab
# ---------------------------------------------------------------------------
with tab_gallery:
    traces = [t for t in load_cached_traces() if t.get("db_id") == db_id]
    if not traces:
        st.info("No cached examples for this database yet.")
    else:
        labels = [f"{t['question']}" for t in traces]
        pick = st.selectbox("Pick a pre-run example (instant replay of a real run):", labels)
        trace = traces[labels.index(pick)]
        st.divider()
        render_trace(trace)

# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
### Why an *agentic* pipeline?

Single-shot text-to-SQL — showing an LLM the schema and asking for SQL — plateaus hard.
The dominant failure is **schema hallucination**: the model filters on values that don't
exist, joins columns that don't connect, and guesses at what opaque columns mean.

This pipeline replaces guessing with **proof**:

| Stage | Role | What it does |
|---|---|---|
| 1 — Explorer | junior analyst | States assumptions, then writes 15+ exploratory queries: `SELECT DISTINCT` probes on every filter value, join-key verification, null counts, scope checks |
| 2 — Reviewer | same analyst, next day | Audits each assumption against real results: `[CONFIRMED]` / `[CONTRADICTED]` / `[OPEN-VALUE]`, then writes targeted confirmation queries |
| 3 — Director | senior reviewer | Sees **findings only** (no SQL), writes 5 candidates spanning genuinely different interpretations — the C1–C5 sweep |

### Results (BIRD-SQL benchmark)

| Configuration | Execution accuracy |
|---|---|
| Schema-only single shot (Llama 70B) | 30.9% |
| + evidence hints (single-shot ceiling) | 48.8% |
| **This 3-stage agentic pipeline** | **59.7%** |

All with small open models (Qwen3-32B + Llama-3.3-70B) on free-tier inference — the gain
comes from the *method*, not model scale.

Built through 12 documented research trials with per-trial failure analysis.
Source & full write-up: **github.com/saadpolowork/agentic-text-to-sql**
""")
