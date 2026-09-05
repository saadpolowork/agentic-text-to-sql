from __future__ import annotations
"""
add_noevidence_pipeline.py — For every trace already in cached_traces/, run the
3-stage pipeline AGAIN with the evidence hint withheld, and merge the result in
as trace["stages_noev"].

This answers: does the pipeline reach the same answer on its own — by exploring
the database — that it reaches when handed the BIRD natural-language hint?

Existing trace["stages"] (the *with-hint* pipeline) and trace["single_shot"] are
left untouched. Idempotent: skips traces that already have "stages_noev".

Usage:
    LLM_BASE_URL=http://localhost:1234/v1 \
    LLM_STAGE1_MODEL=google/gemma-4-31b LLM_STAGE2_MODEL=google/gemma-4-31b \
    LLM_STAGE3_MODEL=google/gemma-4-31b LLM_REQUEST_TIMEOUT=1800 \
    python scripts/add_noevidence_pipeline.py [--db superhero] [--force]
"""

import argparse
import glob
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
    ap.add_argument("--db", default=None)
    ap.add_argument("--force", action="store_true", help="Re-run even if stages_noev exists")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    if not api_key:
        api_key = "local" if (base_url and "groq.com" not in base_url) else ""
    if not api_key:
        sys.exit("Set GROQ_API_KEY (or point LLM_BASE_URL at a local endpoint).")

    paths = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    for path in paths:
        with open(path) as f:
            trace = json.load(f)
        db_id = trace.get("db_id", "")
        if args.db and db_id != args.db:
            continue
        if trace.get("stages_noev") and not args.force:
            print(f"[skip] {os.path.basename(path)} already has stages_noev")
            continue

        q = trace["question"]
        gold_sql = trace.get("gold_sql", "")
        print(f"\n[run ] {os.path.basename(path)} — NO evidence — {q[:80]}")
        t0 = time.time()
        try:
            noev = run_pipeline(db_id=db_id, question=q, api_key=api_key,
                                evidence="", gold_sql=gold_sql,
                                on_event=lambda ev: print("   ", ev["kind"],
                                    {k: v for k, v in ev.items() if k != "kind"}))
        except RateLimitedError as e:
            print(f"    RATE LIMITED — sleeping {e.retry_after:.0f}s")
            time.sleep(e.retry_after + 2)
            noev = run_pipeline(db_id=db_id, question=q, api_key=api_key,
                                evidence="", gold_sql=gold_sql)

        trace["stages_noev"] = noev.get("stages", {})
        trace["noev_elapsed_s"] = noev.get("elapsed_s")
        trace["noev_error"] = noev.get("error")

        with open(path, "w") as f:
            json.dump(trace, f, ensure_ascii=False, indent=1)

        s3 = trace["stages_noev"].get("stage3", {})
        cc = next((c for c in s3.get("candidates", []) if c["n"] == s3.get("chosen")), None)
        s3_ev = trace.get("stages", {}).get("stage3", {})
        cc_ev = next((c for c in s3_ev.get("candidates", []) if c["n"] == s3_ev.get("chosen")), None)
        print(f"[done] {time.time() - t0:.0f}s — no-ev pipeline C{s3.get('chosen')} "
              f"{'✅' if (cc and cc.get('matches_gold')) else '✖'}  "
              f"(with-ev was {'✅' if (cc_ev and cc_ev.get('matches_gold')) else '✖'})")


if __name__ == "__main__":
    main()
