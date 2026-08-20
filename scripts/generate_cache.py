"""
generate_cache.py — Pre-run the pipeline on the curated sample questions and save
full traces to cached_traces/ for the app's instant Example gallery.

Usage:
    GROQ_API_KEY=... python scripts/generate_cache.py [--db superhero] [--limit N] [--only i,j]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.runner import run_pipeline  # noqa: E402
from pipeline.llm import RateLimitedError  # noqa: E402

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(APP_DIR, "cached_traces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="Only this database")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="Comma-separated sample indices")
    ap.add_argument("--force", action="store_true", help="Regenerate existing traces")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        sys.exit("Set GROQ_API_KEY in the environment.")

    with open(os.path.join(APP_DIR, "data", "sample_questions.json")) as f:
        samples = json.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    only = [int(x) for x in args.only.split(",")] if args.only else None

    for db_id, questions in samples.items():
        if args.db and db_id != args.db:
            continue
        for i, s in enumerate(questions):
            if only is not None and i not in only:
                continue
            if args.limit is not None and i >= args.limit:
                break
            out_path = os.path.join(CACHE_DIR, f"{db_id}_{i:02d}.json")
            if os.path.exists(out_path) and not args.force:
                print(f"[skip] {out_path} exists")
                continue
            print(f"[run ] {db_id} #{i}: {s['question'][:80]}")
            t0 = time.time()
            try:
                trace = run_pipeline(
                    db_id=db_id, question=s["question"], api_key=api_key,
                    evidence=s.get("evidence", ""), gold_sql=s.get("gold_sql", ""),
                    on_event=lambda ev: print("   ", ev["kind"],
                                              {k: v for k, v in ev.items() if k != "kind"}),
                )
            except RateLimitedError as e:
                print(f"    RATE LIMITED — sleeping {e.retry_after:.0f}s then retrying once")
                time.sleep(e.retry_after + 2)
                trace = run_pipeline(
                    db_id=db_id, question=s["question"], api_key=api_key,
                    evidence=s.get("evidence", ""), gold_sql=s.get("gold_sql", ""))
            with open(out_path, "w") as f:
                json.dump(trace, f, ensure_ascii=False, indent=1)
            chosen = trace.get("stages", {}).get("stage3", {}).get("chosen")
            cands = trace.get("stages", {}).get("stage3", {}).get("candidates", [])
            gold_hits = [c["n"] for c in cands if c.get("matches_gold")]
            print(f"[done] {time.time() - t0:.0f}s — chosen C{chosen}, "
                  f"gold matches: {gold_hits or 'none'} -> {out_path}")


if __name__ == "__main__":
    main()
