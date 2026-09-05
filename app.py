from __future__ import annotations
"""
Agentic Text-to-SQL — worked-examples showcase.

A gallery of complete, recorded runs. For every question it shows five answers
side by side — a one-shot model call with no hint / with a hint, the 3-stage
agentic pipeline run with no hint / with a hint, and the human reference (gold)
query — then lets you step through everything the pipeline saw, asked, and
produced. The pipeline is graded pass@5 (see app for details), with its
auto-picked single answer shown alongside so that gap stays visible.

No live model calls, no API key, nothing to configure.

Run locally:  streamlit run app.py
Regenerate the gallery:  python scripts/generate_cache.py   (needs an LLM endpoint)
"""

import glob
import json
import os

import streamlit as st

from pipeline.schema import load_tables_index, build_profiled_schema

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "cached_traces")

DB_LABELS = {
    "superhero": "🦸 Superheroes (750 heroes, powers, publishers)",
    "california_schools": "🏫 California Schools (17k schools, SAT scores, funding)",
}

st.set_page_config(page_title="Agentic Text-to-SQL", page_icon="🔍", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


@st.cache_data
def load_tables() -> dict:
    return load_tables_index()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def result_table(columns, rows):
    if not rows:
        st.caption("(no rows returned)")
        return
    st.dataframe(
        {c: [r[i] if i < len(r) else None for r in rows] for i, c in enumerate(columns)},
        width="stretch", height=min(38 * (len(rows) + 1), 300),
    )


def gold_badge(rec) -> str:
    if rec.get("matches_gold") is True:
        return "✅ same rows as the reference answer"
    if rec.get("matches_gold") is False:
        return "✖ different rows from the reference answer"
    if rec.get("status") != "success":
        return "⚠️ query failed to run"
    return "— not compared"


def render_answer(title: str, rec: dict, blurb: str, expanded: bool = False):
    """One answer card: SQL + result + how it compares to gold."""
    if not rec:
        return
    head = f"{title} — {gold_badge(rec)}"
    with st.expander(head, expanded=expanded):
        st.caption(blurb)
        if rec.get("sql"):
            st.code(rec["sql"], language="sql")
        else:
            st.caption("(no SQL produced)")
        if rec.get("status") == "success":
            result_table(rec.get("columns", []), rec.get("rows", []))
            extra = " (truncated)" if rec.get("truncated") else ""
            st.caption(f"{rec.get('row_count', 0)} row(s){extra}")
        elif rec.get("error"):
            st.error(rec["error"])
        if rec.get("system_prompt"):
            with st.popover("🔍 exact prompt sent to the model"):
                st.markdown("**System prompt**")
                st.code(rec["system_prompt"], language="text")
                st.markdown("**User prompt**")
                st.code(rec["user_prompt"], language="text")


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


def render_prompt(stage: dict, note: str = ""):
    with st.expander("🔍 View the exact prompt sent to the model"):
        if note:
            st.caption(note)
        st.markdown("**System prompt** — role, method and rules (identical for every question):")
        st.code(stage.get("system_prompt", ""), language="text")
        st.markdown("**User prompt** — the per-question context the pipeline assembled:")
        st.code(stage.get("user_prompt", ""), language="text")


def chosen_from(stages: dict) -> dict | None:
    s3 = (stages or {}).get("stage3", {})
    chosen = s3.get("chosen")
    return next((c for c in s3.get("candidates", []) if c["n"] == chosen), None)


def candidates_from(stages: dict) -> list:
    return (stages or {}).get("stage3", {}).get("candidates", [])


def in_pool(stages: dict) -> bool:
    """The pipeline's real claim: is the correct answer ANYWHERE among C1-C5?
    (Not whether the auto-picked candidate happens to be it — see render_pipeline_pool.)"""
    return any(c.get("matches_gold") for c in candidates_from(stages))


def chosen_candidate(trace) -> dict | None:
    """The with-hint pipeline's auto-picked candidate (trace['stages'])."""
    return chosen_from(trace.get("stages", {}))


def chosen_noev(trace) -> dict | None:
    """The no-hint pipeline's auto-picked candidate (trace['stages_noev'])."""
    return chosen_from(trace.get("stages_noev", {}))


def render_pipeline_pool(title: str, stages: dict, blurb: str, expanded: bool = False):
    """A pipeline run's answer, graded as pass@5: correct if it's ANYWHERE in C1-C5 —
    not only if the auto-picked candidate happens to be it. Shows the whole pool so
    that tension is visible rather than hidden."""
    cands = candidates_from(stages)
    if not cands:
        return
    chosen_n = (stages or {}).get("stage3", {}).get("chosen")
    chosen_rec = chosen_from(stages)
    pool_hit = in_pool(stages)
    chosen_hit = bool(chosen_rec and chosen_rec.get("matches_gold"))

    if pool_hit:
        verdict = "✅ correct answer found among the 5 candidates"
    elif any(c.get("status") == "success" for c in cands):
        verdict = "✖ none of the 5 candidates matched"
    else:
        verdict = "⚠️ candidates failed to execute"
    with st.expander(f"{title} — {verdict}", expanded=expanded):
        st.caption(blurb)
        if pool_hit and not chosen_hit:
            st.warning(f"The pipeline's auto-picked answer (C{chosen_n}) was actually **wrong** "
                       "— the right one was in the pool, just not the one it committed to.")
        elif pool_hit and chosen_hit:
            st.success(f"The pipeline auto-picked C{chosen_n} — and that pick was correct.")
        for c in sorted(cands, key=lambda c: c["n"]):
            hit = c.get("matches_gold")
            badge = "✅ matches gold" if hit else ("✖ differs from gold" if c["status"] == "success" else "🔴 failed")
            star = " ⭐ auto-picked" if c["n"] == chosen_n else ""
            st.markdown(f"**C{c['n']}**{star} — {c['comment'] or '(no comment)'} · {badge}")
            st.code(c["sql"], language="sql")
            if c["status"] == "success":
                st.caption(f"{c.get('row_count', 0)} row(s)"
                           + (" (truncated)" if c.get("truncated") else ""))
            else:
                st.caption(f"error: {c.get('error', 'unknown')}")


def gold_record(trace) -> dict:
    gr = trace.get("stages", {}).get("stage3", {}).get("gold_result") or {}
    return {
        "sql": trace.get("gold_sql", ""),
        "status": "success" if gr else "error",
        "columns": gr.get("columns", []),
        "rows": gr.get("rows", []),
        "row_count": gr.get("row_count", 0),
    }


def render_pipeline_trace(s: dict):
    """The full stage-by-stage trace for one pipeline run (a `stages` dict)."""
    s = s or {}

    if "stage1" in s:
        st.subheader("Stage 1 — Explorer 🕵️")
        st.caption(f"Model: `{s['stage1']['model']}` — states its assumptions with the schema "
                   "face-down, then writes 15+ exploratory queries: `SELECT DISTINCT` probes on "
                   "every filter value, join-key verification, null counts, scope checks.")
        with st.expander("💭 Phase 1 — Assumptions (written before seeing any data)"):
            st.text(s["stage1"].get("phase1", ""))
        with st.expander("📋 Phase 2 — Schema review"):
            st.text(s["stage1"].get("phase2", ""))
        st.markdown("**Exploratory queries & the results they returned:**")
        render_queries(s["stage1"].get("queries", []))
        render_prompt(s["stage1"])

    if "stage2" in s:
        st.subheader("Stage 2 — Reviewer 🧐")
        st.caption(f"Model: `{s['stage2']['model']}` — the same analyst the next day, auditing "
                   "every Stage 1 assumption against the real query results: `[CONFIRMED]` / "
                   "`[CONTRADICTED]` / `[OPEN-VALUE]`, then writing targeted confirmation queries.")
        with st.expander("🏷️ Phase 4 — Assumption verdicts", expanded=False):
            st.text(s["stage2"].get("phase4", ""))
        with st.expander("🎯 Phase 4b — Winning hypothesis", expanded=True):
            st.text(s["stage2"].get("phase4b", ""))
        st.markdown("**Confirmation queries & the results they returned:**")
        render_queries(s["stage2"].get("queries", []))
        render_prompt(s["stage2"],
                      note="The user prompt now carries Stage 1's assumptions, its schema notes, "
                           "and the full text + results of every exploratory query.")

    if "stage3" in s:
        st.subheader("Stage 3 — Director 🎬")
        st.caption(f"Model: `{s['stage3']['model']}` — sees **findings only**: no SQL, no schema. "
                   "From that it writes 5 candidate answers spanning genuinely different "
                   "interpretations of the question (the C1–C5 sweep), each executed against the DB.")
        chosen = s["stage3"].get("chosen")
        for c in s["stage3"].get("candidates", []):
            is_chosen = c["n"] == chosen
            badge = " ⭐ CHOSEN" if is_chosen else ""
            gold = ""
            if c.get("matches_gold") is True:
                gold = " · ✅ matches reference"
            elif c.get("matches_gold") is False:
                gold = " · ✖ differs from reference"
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
        render_prompt(s["stage3"],
                      note="Note what is **absent**: the Director never sees a table schema or a "
                           "single line of the analyst's SQL — only prose findings and one result "
                           "row per check. That forces it to reason about meaning, not syntax.")

    if s.get("error"):
        st.warning(s["error"])


def tally(traces: list) -> dict:
    """Pipeline metric is 'in pool' (pass@5) — see in_pool(). The auto-picked ('chosen')
    accuracy is tracked alongside to make the pool-vs-single-pick gap visible, not hidden."""
    out = {"pipe_ev_pool": 0, "pipe_noev_pool": 0, "pipe_ev_chosen": 0, "pipe_noev_chosen": 0,
           "ss_ev": 0, "ss_none": 0, "n": len(traces)}
    for t in traces:
        out["pipe_ev_pool"] += in_pool(t.get("stages", {}))
        out["pipe_noev_pool"] += in_pool(t.get("stages_noev", {}))
        cc, cn = chosen_candidate(t), chosen_noev(t)
        out["pipe_ev_chosen"] += bool(cc and cc.get("matches_gold"))
        out["pipe_noev_chosen"] += bool(cn and cn.get("matches_gold"))
        ss = t.get("single_shot", {})
        out["ss_ev"] += bool(ss.get("with_evidence", {}).get("matches_gold"))
        out["ss_none"] += bool(ss.get("no_evidence", {}).get("matches_gold"))
    return out


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🔍 Agentic Text-to-SQL")
st.markdown(
    "**A text-to-SQL pipeline that refuses to guess.** The usual approach — show a model the "
    "schema, ask for a query — falls over on *schema hallucination*: filtering on values that "
    "don't exist, joining columns that don't connect, guessing what an opaque column means. "
    "This pipeline instead **explores the real database first** — 15+ probe queries per question, "
    "checking every assumption against actual data — and only then writes the answer, as five "
    "competing candidates.\n\n"
    "This page is a **gallery of recorded runs**. Every question shows five answers to compare — "
    "a one-shot model call **with no hint** and **with a hint**, the **3-stage pipeline** run "
    "**with no hint** and **with a hint**, and the **human reference query** — all executed "
    "against the live database. The question it's built to answer: *does the pipeline, by "
    "exploring the database, reach on its own the answer it reaches when handed the hint?*\n\n"
    "The pipeline is graded **as a pool, not a single guess**: a question counts as solved if the "
    "correct answer appears *anywhere* among its 5 candidates — the same standard code-generation "
    "evals use (\"pass@5\"). That's a deliberate, disclosed choice, not a way to dodge being wrong: "
    "the pipeline still auto-picks one candidate to hand back as *the* answer, and that single pick "
    "is shown too, so you can see exactly how often having the right answer in the pool and *picking* "
    "it are two different things. Then you can open the full pipeline trace: every stage, every "
    "query, the verbatim prompts."
)

traces = load_cached_traces()
pipe_models = sorted({m for t in traces for m in t.get("models", {}).values()})
ss_models = sorted({t.get("single_shot_model", "") for t in traces if t.get("single_shot_model")})
all_models = sorted(set(pipe_models) | set(ss_models))

with st.sidebar:
    st.header("⚙️ Settings")
    db_id = st.radio("Database", list(DB_LABELS), format_func=lambda k: DB_LABELS[k])
    st.divider()
    st.markdown("**3-stage pipeline**\n\n"
                "1. Explorer 🕵️ — probe the DB\n"
                "2. Reviewer 🧐 — audit every assumption\n"
                "3. Director 🎬 — 5 candidate answers")
    if all_models:
        st.caption("All runs used: " + ", ".join(f"`{m}`" for m in all_models)
                   + " — one local open model, same for the pipeline and the one-shot baseline.")
    st.divider()
    st.markdown("Built by **Saad Tariq** · "
                "[source ↗](https://github.com/saadpolowork/agentic-text-to-sql)")

tab_gallery, tab_schema, tab_about = st.tabs(
    ["📚 Worked examples", "🗄️ Schema", "ℹ️ How it works & why"])

# ---------------------------------------------------------------------------
# Gallery tab
# ---------------------------------------------------------------------------
with tab_gallery:
    db_traces = [t for t in traces if t.get("db_id") == db_id]
    if not db_traces:
        st.info("No recorded examples for this database yet.")
    else:
        db_name = DB_LABELS[db_id].split("(")[0].strip()
        t_ = tally(db_traces)
        n = t_["n"]
        st.markdown(f"**{n} recorded questions** on {db_name} (all moderate / challenging). "
                    "Same-rows-as-reference count across this set:")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("One-shot · no hint", f"{t_['ss_none']} / {n}")
        c2.metric("One-shot · with hint", f"{t_['ss_ev']} / {n}")
        c3.metric("Pipeline · no hint", f"{t_['pipe_noev_pool']} / {n}",
                  help="Correct answer found anywhere among the pipeline's 5 candidates.")
        c4.metric("Pipeline · with hint", f"{t_['pipe_ev_pool']} / {n}",
                  help="Correct answer found anywhere among the pipeline's 5 candidates.")
        st.caption(
            f"Pipeline columns count a question solved if the right answer is **anywhere in its "
            f"pool of 5 candidates**. If it had to commit to one, its auto-picked answer alone is "
            f"only right {t_['pipe_noev_chosen']}/{n} (no hint) and {t_['pipe_ev_chosen']}/{n} "
            f"(with hint) — generating a pool of interpretations finds the right one more often "
            f"than any single pick does. The comparison to watch is **one-shot no-hint vs. "
            f"pipeline no-hint** — same input, whether exploration recovers what the hint gives."
        )
        st.divider()

        labels = []
        for t in db_traces:
            mark = "✅" if in_pool(t.get("stages_noev", {})) else "•"
            labels.append(f"{mark}  {t['question']}")
        pick = st.selectbox("Pick a question:", labels)
        trace = db_traces[labels.index(pick)]

        st.markdown(f"### {trace['question']}")
        meta = []
        if trace.get("difficulty"):
            meta.append(f"difficulty: **{trace['difficulty']}**")
        if trace.get("evidence"):
            meta.append(f"hint supplied: *{trace['evidence']}*")
        if meta:
            st.caption(" · ".join(meta))

        st.markdown("#### Five answers, head to head")
        ss = trace.get("single_shot", {})
        ssm = trace.get("single_shot_model", "the model")
        render_answer("① One-shot · no hint", ss.get("no_evidence"),
                      f"`{ssm}` given only the schema and the question — one call, no exploration.")
        render_answer("② One-shot · with hint", ss.get("with_evidence"),
                      f"`{ssm}` given the schema, the question, **and** the natural-language hint — "
                      "still one call, still no exploration.")
        render_pipeline_pool("③ Pipeline · no hint", trace.get("stages_noev", {}),
                      "The 3-stage pipeline on the **same input as ①** — schema + question, no "
                      "hint. Everything it knows about values and joins, it learned by querying "
                      "the database. Graded pass@5 — see the whole pool below.", expanded=True)
        render_pipeline_pool("④ Pipeline · with hint", trace.get("stages", {}),
                      "The 3-stage pipeline also handed the natural-language hint — an upper "
                      "bound for what the hint buys on top of exploration.")
        render_answer("⑤ Human reference (gold) query", gold_record(trace),
                      "The hand-written reference query — the yardstick the other four are "
                      "checked against (same rows, order-insensitive).", expanded=True)

        st.divider()
        show_noev = st.toggle("🔬 Full trace — pipeline · no hint", value=False)
        if show_noev:
            render_pipeline_trace(trace.get("stages_noev", {}))
        show_ev = st.toggle("🔬 Full trace — pipeline · with hint", value=False)
        if show_ev:
            render_pipeline_trace(trace.get("stages", {}))

# ---------------------------------------------------------------------------
# Schema tab
# ---------------------------------------------------------------------------
with tab_schema:
    tables_index = load_tables()
    meta = tables_index.get(db_id)
    if not meta:
        st.info("No schema metadata for this database.")
    else:
        tbls = meta["table_names_original"]
        st.markdown(f"### {DB_LABELS[db_id].split('(')[0].strip()} — {len(tbls)} tables")
        st.markdown("`" + "`  ·  `".join(tbls) + "`")
        st.markdown(
            "Below is the schema text **exactly as it is fed to the model** — for both the "
            "one-shot baseline and Stages 1–2 of the pipeline: table names, column names, types, "
            "and primary / foreign keys. Note what is *not* here: no row counts, no min/max, no "
            "sample values, no column descriptions. Discovering all of that from the real data is "
            "Stage 1's entire job."
        )
        st.caption("Legend:  `[PK]` primary key   ·   `[FK → table.column]` foreign key reference")
        st.code(build_profiled_schema(meta), language="text")

# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
### Why build this?

Single-shot text-to-SQL — showing a model the schema and asking for a query — plateaus hard. The
dominant error is **schema hallucination**: the model filters on values that don't exist
(`WHERE status = 'Active'` when the column stores `'A'`), joins columns that don't connect, or
guesses at what an opaque column means.

You can't prompt that away by asking the model to "be careful". The fix is to **ground every
assumption in an actual query against the real database** before writing the answer. That is what
this pipeline does — and the gallery lets you see, question by question, how much it buys you over
a one-shot call with the same model.

### The three stages

| Stage | Role | What it does |
|---|---|---|
| **1 — Explorer** 🕵️ | junior analyst, first day | States every assumption with the schema face-down, then writes 15+ exploratory queries: `SELECT DISTINCT` probes on every filter value, join-key verification, null counts, scope proofs. Every query runs against the live DB. |
| **2 — Reviewer** 🧐 | same analyst, next day | Audits each assumption against the real results — `[CONFIRMED]` / `[CONTRADICTED]` / `[OPEN-VALUE]` / `[LIKE-VS-ENUM]` — then writes targeted confirmation queries to close every remaining gap. |
| **3 — Director** 🎬 | senior reviewer | Sees **findings only**: no SQL, no schema. Writes 5 candidates (C1–C5) spanning genuinely different interpretations of the question, each executed against the DB. Withholding the schema is deliberate — it forces reasoning about *meaning*, not syntax. |

### Why five candidates?

The Director writes C1–C5 under a *proof-by-cases* principle: five structurally different readings
of the question, so if the correct interpretation exists at all, at least one candidate should
capture it. The pipeline then also **auto-picks one** — the first candidate that runs clean and
returns rows — to hand back as *the* answer, the way a deployed system would have to.

### How the pipeline is graded — pool, not a single guess

The gallery scores the pipeline **pass@5**: a question counts as solved if the correct answer
appears *anywhere* among its 5 candidates, the same standard used to evaluate code-generation
models sampling multiple attempts. That's a disclosed choice, not a way to dodge being wrong —
every worked example also shows the pipeline's **auto-picked single answer**, and on this gallery
it undercounts the pool by a real margin (no hint: 23/50 auto-picked vs. **27/50** in the pool;
with hint: 31/50 vs. **37/50**). Generating five genuinely different interpretations and checking
each against the database finds the right one more often than committing to any single guess does
— exactly the failure mode you'd expect from a model that's usually *close*, not usually *wrong*.
A better selector is the obvious next lever on this pipeline; the pool score is what the exploration
method itself is capable of.

### What the gallery compares

For every question, five queries run against the live SQLite database and are checked for
returning the same rows (order-insensitive) as the human reference:

1. **One-shot · no hint** — the model gets the schema and the question. One call.
2. **One-shot · with hint** — same, plus a one-line natural-language hint about the intended
   columns / filters.
3. **Pipeline · no hint** — the 3-stage pipeline on the *same input as (1)*, graded pass@5.
   Anything it knows about actual values, encodings and join keys, it discovered by querying
   the database.
4. **Pipeline · with hint** — the pipeline also handed the hint, graded pass@5 — an upper bound
   for what the hint adds on top of exploration.
5. **Human reference (gold) query** — the hand-written yardstick.

The comparison the project is really about is **(1) vs (3)** — identical inputs, and whether
the pipeline's exploration recovers on its own what the human hint in (2) would have supplied.
It's a small hand-picked set of moderate/challenging questions per database, not a formal
benchmark.

### Where exploration can't substitute for the hint

Splitting the two databases apart makes the answer clear. On **superhero**, most hints just name
a value (`"blue eyes" → colour = 'Blue'`) — exactly what a `DISTINCT` probe recovers, and the
no-hint pipeline lands close to the with-hint one (20/25 vs. 22/25 pool). On
**california_schools**, most hints are **arithmetic definitions**
(`eligible free rate = Free Meal Count (K-12) / Enrollment (K-12)`) or **business-code lookups**
(`DOC = 52` means "Elementary School District") — no query reveals that a rate is a ratio of two
specific columns, or what an internal code stands for. There the gap barely closes (7/25 vs. 15/25
pool). Across every no-hint miss in the gallery, roughly 60% trace directly to a formula in the
withheld hint — the rest split between other undiscoverable domain conventions and a handful of
plain misses on grounding the pipeline should have caught. **Exploration replaces guessing at
values and joins. It can't invent a business formula or a code's meaning that exists only in
someone's head.**

### Engineering notes baked into the code

- **Unnumbered-C1 recovery** — models reliably emit the first candidate as raw SQL without its
  `1. --` prefix; a parser that misses it silently zeroes your C1 results.
- **Think-block stripping** for reasoning models.
- **CTE ban in Stage 3** — cut the syntax-error rate sharply.
- **Conditional `LIMIT`** — "always `LIMIT`" truncates list-type answers.
- **Contrapositive filter test** — models love adding `status = 'Active'` filters nobody asked for.
- **Execution deadlines via a progress handler** — SQLite's connection timeout only bounds lock
  waits, not runaway queries.

Databases and questions are from the [BIRD-SQL](https://bird-bench.github.io/) dataset
(CC BY-SA 4.0), included here for demonstration. Source:
**[github.com/saadpolowork/agentic-text-to-sql](https://github.com/saadpolowork/agentic-text-to-sql)**
""")
