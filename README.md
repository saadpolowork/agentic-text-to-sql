# 🔍 Agentic Text-to-SQL

**Ask a database a question in plain English — and watch a 3-stage agentic pipeline prove its way to the answer.**

Most text-to-SQL systems show an LLM the schema and hope. This one doesn't guess: it **explores the real database first**, verifies every assumption with actual SQL, and only then writes the answer — five candidate answers, in fact, each executed live.

🚀 **[Open the showcase](https://huggingface.co/spaces/)** *(link goes live after deploy — see below)* — a gallery of recorded runs. Every question shows five answers side by side: a one-shot model call **no hint** / **with hint**, the 3-stage pipeline run **no hint** / **with hint**, and the **human reference (gold) query** — all executed against the live database. Then step through the full pipeline trace: every stage, every query, the verbatim prompts. No API key, nothing to configure.

![pipeline architecture](assets/architecture.svg)

## How it works

The pipeline treats SQL generation as **hypothesis testing**, not autocomplete:

| Stage | Role | What it does |
|---|---|---|
| **1 — Explorer** 🕵️ | junior analyst | States assumptions *before* seeing data, then writes 15+ exploratory queries: `SELECT DISTINCT` probes on every filter value, join-key verification, null-count checks, scope proofs |
| **2 — Reviewer** 🧐 | same analyst, next day | Audits every assumption against real query results — `[CONFIRMED]`, `[CONTRADICTED]`, `[OPEN-VALUE]`, `[LIKE-VS-ENUM]` — then writes targeted confirmation queries to close every gap |
| **3 — Director** 🎬 | senior reviewer | Sees **findings only** (no SQL, no schema), writes 5 candidates spanning genuinely different interpretations of the question — the C1–C5 sweep — each executed against SQLite |

The design directly attacks the dominant failure mode in text-to-SQL: **schema hallucination** — filtering on values that don't exist, joining columns that don't connect, guessing what an opaque column means. Grounding every filter value in a `DISTINCT` probe against the real data is what suppresses it.

### Why five candidates?

The Director writes C1–C5 under a *proof-by-cases* principle: five structurally different readings of the question, so if the correct interpretation exists, at least one candidate captures it. The pipeline also auto-picks one — the first that runs clean and returns rows — to hand back as *the* answer, the way a deployed system would have to.

## What the gallery compares — and what it found

For every question, five queries run against the live SQLite database and are checked for returning the same rows (order-insensitive) as the human reference:

1. **One-shot, no hint** — the model gets the schema and the question. One call.
2. **One-shot, with hint** — same, plus a one-line natural-language hint about the intended columns / filters.
3. **Pipeline, no hint** — the 3-stage pipeline on the *same input as (1)*.
4. **Pipeline, with hint** — the pipeline also handed the hint — an upper bound for what the hint adds on top of exploration.
5. **Human reference (gold) query** — the hand-written yardstick.

The pipeline (3 and 4) is graded **pass@5**: solved if the right answer appears *anywhere* in its 5 candidates, not only if its auto-picked answer happens to be it — the same standard used to grade code-generation models that sample multiple attempts. Both numbers are shown side by side in the app so the gap between "in the pool" and "the single answer you'd actually get back" is visible, not hidden.

On 50 hand-picked moderate/challenging BIRD-dev questions across both databases: one-shot scored 20/50 (no hint) and 32/50 (with hint); the pipeline scored 27/50 (no hint, pass@5) and 37/50 (with hint, pass@5) — its auto-picked single answer alone was right 23/50 and 31/50. Split by database, the story is sharp: on **superhero**, whose hints mostly name a value (`"blue eyes" → colour = 'Blue'`), the no-hint pipeline (20/25) nearly matches the with-hint one (22/25) — exploration recovers what the hint would have given. On **california_schools**, whose hints are mostly arithmetic definitions (`rate = Free Meal Count / Enrollment`) or business-code lookups (`DOC = 52` → "Elementary School District"), it barely moves (7/25 vs. 15/25) — no query reveals a formula or a code's meaning. It's a small hand-picked set, not a formal benchmark, but the shape holds: exploration replaces guessing at values and joins; it can't invent domain knowledge that exists only in someone's head.

Some hard-won engineering lessons are baked into the code:

- **Unnumbered-C1 recovery** — LLMs reliably emit the first candidate as raw SQL without its `1. --` prefix; a parser that misses it silently zeroes your C1 results
- **Think-block stripping** for reasoning models
- **CTE ban in Stage 3** — cut the syntax-error rate sharply
- **Conditional LIMIT** — "always LIMIT" truncates list-type answers
- **Contrapositive filter test** — models love adding `status = 'Active'` filters nobody asked for
- **Execution deadlines via progress handler** — SQLite's connection timeout only bounds lock waits, not runaway queries

## Run it locally

```bash
git clone https://github.com/saadpolowork/agentic-text-to-sql
cd agentic-text-to-sql
pip install -r requirements.txt
streamlit run app.py
```

That's it — the app is a **gallery of recorded runs**. No API key, no LLM endpoint, no network calls. Two databases are bundled: **superhero** (750 heroes, powers, publishers) and **california_schools** (17k schools, SAT scores, funding), each with 25 questions and their hand-written reference queries.

### Regenerating the gallery (needs an LLM endpoint)

The recorded traces in `cached_traces/` are produced by actually running the pipeline **and** the one-shot baseline. To regenerate them you need an OpenAI-compatible endpoint. Any works via env vars — e.g. a local LM Studio server:

```bash
export LLM_BASE_URL=http://localhost:1234/v1
export LLM_STAGE1_MODEL=google/gemma-4-31b
export LLM_STAGE2_MODEL=google/gemma-4-31b
export LLM_STAGE3_MODEL=google/gemma-4-31b
export LLM_SINGLE_SHOT_MODEL=google/gemma-4-31b

python scripts/generate_cache.py                     # all questions, both databases
python scripts/generate_cache.py --db superhero --limit 3
```

For a hosted endpoint, set `GROQ_API_KEY` (and leave `LLM_BASE_URL` unset for Groq).

## Deploy your own (free)

The showcase is a plain Streamlit app serving pre-recorded JSON traces — no secrets, no backend, no per-visitor cost. It runs on Hugging Face Spaces (free CPU tier):

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) → SDK: **Streamlit**
2. Push this repo to it (or use "Import from GitHub")
3. Done — the Space builds and serves the app at a permanent public URL

(Streamlit Community Cloud works the same way — point it at the GitHub repo, no config.)

## Repository layout

```
app.py                     Streamlit UI — 4-way comparison gallery + schema browser
pipeline/
  prompts.py               The 3-stage prompt stack (S1 Explorer / S2 Reviewer / S3 Director)
  runner.py                3-stage orchestration, trace assembly
  single_shot.py           The one-shot baseline the pipeline is compared against
  parsing.py               Query/candidate extraction (with the C1-recovery fix)
  executor.py              Safe read-only SQLite execution (deadline, LIMIT injection)
  schema.py                tables.json → annotated schema text
  llm.py                   OpenAI-compatible inference w/ 429 backoff (used by generate_cache.py)
scripts/generate_cache.py  Run every question through the pipeline + baseline → cached_traces/
data/                      2 bundled databases + 25 curated questions each
cached_traces/             Recorded runs served by the gallery
```

## Data & license

Code: MIT. Bundled databases and questions are from the [BIRD-SQL](https://bird-bench.github.io/) dataset (CC BY-SA 4.0) — © the BIRD authors, included here for demonstration.

---

Built by **Saad Tariq**. If you're working on text-to-SQL grounding or the reliability of agentic pipelines, I'm happy to talk.
