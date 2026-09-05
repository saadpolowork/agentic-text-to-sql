from __future__ import annotations
"""
generate_cache.py — Pre-run the pipeline AND the single-shot baseline on every
curated question, saving full traces to cached_traces/ for the app gallery.

Each trace gets:
  stages.stage1 / stage2 / stage3   — the 3-stage agentic pipeline
  single_shot.no_evidence           — one LLM call, schema only
  single_shot.with_evidence         — one LLM call, schema + BIRD evidence hint
  gold_result                       — rows returned by the official gold SQL

Usage:
    LLM_BASE_URL=http://localhost:1234/v1 \
    LLM_STAGE1_MODEL=google/gemma-4-31b LLM_STAGE2_MODEL=google/gemma-4-31b \
    LLM_STAGE3_MODEL=google/gemma-4-31b \
    python scripts/generate_cache.py [--db superhero] [--limit N] [--only i,j] [--force]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.runner import run_pipeline  # noqa: E402
from pipeline.llm import RateLimitedError  # noqa: E402
from pipeline.single_shot import run_single_shot  # noqa: E402
from pipeline.schema import (  # noqa: E402
    load_tables_index, build_profiled_schema, db_path_for,
)
from pipeline.executor import execute_candidate  # noqa: E402

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(APP_DIR, "cached_traces")

SINGLE_SHOT_MODEL = os.environ.get(
    "LLM_SINGLE_SHOT_MODEL", os.environ.get("LLM_STAGE3_MODEL", "google/gemma-4-31b"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="Only this database")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="Comma-separated sample indices")
    ap.add_argument("--force", action="store_true", help="Regenerate existing traces")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    if not api_key:
        if base_url and "groq.com" not in base_url:
            api_key = "local"  # local backends (LM Studio, Ollama) ignore the key
        else:
            sys.exit("Set GROQ_API_KEY (or point LLM_BASE_URL at a local endpoint).")

    with open(os.path.join(APP_DIR, "data", "sample_questions.json")) as f:
        samples = json.load(f)

    tables_index = load_tables_index()
    os.makedirs(CACHE_DIR, exist_ok=True)
    only = [int(x) for x in args.only.split(",")] if args.only else None

    for db_id, questions in samples.items():
        if args.db and db_id != args.db:
            continue
        schema_text = build_profiled_schema(tables_index[db_id])
        db_path = db_path_for(db_id)

        for i, s in enumerate(questions):
            if only is not None and i not in only:
                continue
            if args.limit is not None and i >= args.limit:
                break
            out_path = os.path.join(CACHE_DIR, f"{db_id}_{i:02d}.json")
            if os.path.exists(out_path) and not args.force:
                print(f"[skip] {out_path} exists")
                continue

            q, evidence, gold_sql = s["question"], s.get("evidence", ""), s.get("gold_sql", "")
            print(f"\n[run ] {db_id} #{i} ({s.get('difficulty', '?')}): {q[:90]}")

            # --- gold rows (for single-shot comparison) ---
            gold_exec = execute_candidate(db_path, gold_sql) if gold_sql else {"status": "error"}
            gold_ok = gold_exec.get("status") == "success"
            gold_rows = gold_exec.get("rows") if gold_ok else None

            # --- 3-stage pipeline ---
            t0 = time.time()
            try:
                trace = run_pipeline(
                    db_id=db_id, question=q, api_key=api_key,
                    evidence=evidence, gold_sql=gold_sql,
                    on_event=lambda ev: print("   ", ev["kind"],
                                              {k: v for k, v in ev.items() if k != "kind"}),
                )
            except RateLimitedError as e:
                print(f"    RATE LIMITED — sleeping {e.retry_after:.0f}s then retrying once")
                time.sleep(e.retry_after + 2)
                trace = run_pipeline(db_id=db_id, question=q, api_key=api_key,
                                     evidence=evidence, gold_sql=gold_sql)
            pipe_s = time.time() - t0

            # --- single-shot baseline, both variants ---
            print("    single-shot (no evidence)…")
            ss_none = run_single_shot(db_path, q, schema_text, evidence,
                                      SINGLE_SHOT_MODEL, api_key, include_evidence=False,
                                      gold_rows=gold_rows, gold_ok=gold_ok)
            print("    single-shot (with evidence)…")
            ss_ev = run_single_shot(db_path, q, schema_text, evidence,
                                    SINGLE_SHOT_MODEL, api_key, include_evidence=True,
                                    gold_rows=gold_rows, gold_ok=gold_ok)

            trace["difficulty"] = s.get("difficulty", "")
            trace["single_shot_model"] = SINGLE_SHOT_MODEL
            trace["single_shot"] = {"no_evidence": ss_none, "with_evidence": ss_ev}

            with open(out_path, "w") as f:
                json.dump(trace, f, ensure_ascii=False, indent=1)

            s3 = trace.get("stages", {}).get("stage3", {})
            chosen = s3.get("chosen")
            chosen_rec = next((c for c in s3.get("candidates", []) if c["n"] == chosen), None)
            pipe_hit = bool(chosen_rec and chosen_rec.get("matches_gold"))
            print(f"[done] pipeline {pipe_s:.0f}s → C{chosen} "
                  f"{'✅' if pipe_hit else '✖'} | single-shot "
                  f"{'✅' if ss_none.get('matches_gold') else '✖'}(no-ev) "
                  f"{'✅' if ss_ev.get('matches_gold') else '✖'}(ev) → {out_path}")


if __name__ == "__main__":
    main()
