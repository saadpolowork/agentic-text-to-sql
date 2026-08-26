from __future__ import annotations
"""
runner.py — Orchestrates the 3-stage agentic pipeline for a single question.

Stage 1  Explorer  : LLM writes 15+ exploratory queries  → executed against SQLite
Stage 2  Reviewer  : LLM reviews results, tags assumptions, writes confirmation queries → executed
Stage 3  Director  : LLM sees findings-only context, writes candidates C1–C5 → executed

Emits progress events via an on_event callback so a UI can render the trace live.
Returns a complete trace dict (also the cached-gallery format).
"""

import time

from .llm import call_llm, STAGE1_MODEL, STAGE2_MODEL, STAGE3_MODEL
from .executor import run_parsed_queries, execute_candidate, rows_match
from .parsing import (
    strip_think_blocks, parse_queries, parse_candidates,
    extract_stage1_phases, extract_stage2_phases,
)
from .prompts import (
    STAGE1_SYSTEM, STAGE2_SYSTEM, STAGE3_SYSTEM,
    build_stage1_user, build_stage2_user, build_stage3_user,
)
from .schema import load_tables_index, build_profiled_schema, db_path_for


def format_execution_results(queries: list, max_rows: int = 3) -> str:
    """Stage 2 context: full SQL + results (research format)."""
    lines = []
    for q in queries:
        lines.append(f"{q.get('n', '?')}. -- {q.get('comment', '').strip()}")
        lines.append(f"   {q.get('sql', '').strip()}")
        if q.get("status") == "success":
            cols = q.get("columns", [])
            head = q.get("head", [])
            row_count = q.get("row_count", 0)
            if cols:
                lines.append(f"   Columns: {', '.join(cols)}")
            if head:
                for row in head[:max_rows]:
                    lines.append(f"   {row}")
                if row_count and row_count > max_rows:
                    lines.append(f"   ... ({row_count} rows total)")
            else:
                lines.append("   (no rows returned)")
        else:
            lines.append(f"   ERROR: {q.get('error', 'unknown error')}")
        lines.append("")
    return "\n".join(lines)


def format_findings(queries: list) -> str:
    """Stage 3 context: findings-only — what was checked + 1 result row, no SQL."""
    lines = []
    for q in queries:
        lines.append(f"{q.get('n', '?')}. Checked: {q.get('comment', '').strip()}")
        if q.get("status") == "success":
            cols = q.get("columns", [])
            head = q.get("head", [])
            if head and cols:
                row = head[0]
                pairs = [f"{cols[i]}: {row[i]}" for i in range(min(len(cols), len(row)))]
                lines.append(f"   Result: {' | '.join(pairs)}")
                row_count = q.get("row_count", 0)
                if row_count and row_count > 1:
                    lines.append(f"   ({row_count} rows total)")
            else:
                lines.append("   Result: (no rows returned)")
        else:
            lines.append(f"   Result: ERROR — {q.get('error', 'unknown')}")
        lines.append("")
    return "\n".join(lines)


def _emit(on_event, kind: str, **payload):
    if on_event:
        on_event({"kind": kind, **payload})


def run_pipeline(db_id: str, question: str, api_key: str,
                 evidence: str = "", gold_sql: str = "",
                 on_event=None, db_root: str | None = None,
                 tables_path: str | None = None) -> dict:
    """Run the full 3-stage pipeline for one question. Returns the trace dict."""
    t0 = time.time()
    tables_index = load_tables_index(tables_path)
    if db_id not in tables_index:
        raise ValueError(f"Unknown database: {db_id}")
    schema_text = build_profiled_schema(tables_index[db_id])
    db_path = db_path_for(db_id, db_root)

    trace = {
        "db_id": db_id,
        "question": question,
        "evidence": evidence,
        "gold_sql": gold_sql,
        "models": {"stage1": STAGE1_MODEL, "stage2": STAGE2_MODEL, "stage3": STAGE3_MODEL},
        "stages": {},
    }

    # ---------------- Stage 1 — Explorer ----------------
    _emit(on_event, "stage_start", stage=1, label="Explorer — writing exploratory queries")
    s1_user = build_stage1_user(db_id, schema_text, question, evidence)
    s1_raw = call_llm(STAGE1_SYSTEM, s1_user, STAGE1_MODEL, api_key)
    s1_clean = strip_think_blocks(s1_raw)
    phase1, phase2 = extract_stage1_phases(s1_clean)
    s1_parsed = parse_queries(s1_clean)
    _emit(on_event, "llm_done", stage=1, n_queries=len(s1_parsed))

    _emit(on_event, "exec_start", stage=1, n_queries=len(s1_parsed))
    s1_results = run_parsed_queries(db_path, s1_parsed)
    s1_ok = sum(1 for r in s1_results if r["status"] == "success")
    trace["stages"]["stage1"] = {
        "model": STAGE1_MODEL,
        "system_prompt": STAGE1_SYSTEM,
        "user_prompt": s1_user,
        "response": s1_clean,
        "phase1": phase1,
        "phase2": phase2,
        "queries": s1_results,
        "n_queries": len(s1_results),
        "n_success": s1_ok,
    }
    _emit(on_event, "exec_done", stage=1, n_queries=len(s1_results), n_success=s1_ok)

    if len(s1_parsed) < 5:
        trace["error"] = ("Stage 1 produced fewer than 5 exploratory queries — "
                          "the integrity gate rejects this record. Try re-running.")
        trace["elapsed_s"] = round(time.time() - t0, 1)
        return trace

    # ---------------- Stage 2 — Reviewer ----------------
    _emit(on_event, "stage_start", stage=2, label="Reviewer — checking hypotheses against results")
    s2_user = build_stage2_user(
        db_id, question, evidence, schema_text,
        phase1, phase2, format_execution_results(s1_results),
    )
    s2_raw = call_llm(STAGE2_SYSTEM, s2_user, STAGE2_MODEL, api_key)
    s2_clean = strip_think_blocks(s2_raw)
    phase4, phase4b = extract_stage2_phases(s2_clean)
    s2_parsed = parse_queries(s2_clean)
    _emit(on_event, "llm_done", stage=2, n_queries=len(s2_parsed))

    _emit(on_event, "exec_start", stage=2, n_queries=len(s2_parsed))
    s2_results = run_parsed_queries(db_path, s2_parsed)
    s2_ok = sum(1 for r in s2_results if r["status"] == "success")
    trace["stages"]["stage2"] = {
        "model": STAGE2_MODEL,
        "system_prompt": STAGE2_SYSTEM,
        "user_prompt": s2_user,
        "response": s2_clean,
        "phase4": phase4,
        "phase4b": phase4b,
        "queries": s2_results,
        "n_queries": len(s2_results),
        "n_success": s2_ok,
    }
    _emit(on_event, "exec_done", stage=2, n_queries=len(s2_results), n_success=s2_ok)

    if len(s2_parsed) < 5:
        trace["error"] = ("Stage 2 produced fewer than 5 confirmation queries — "
                          "the integrity gate rejects this record. Try re-running.")
        trace["elapsed_s"] = round(time.time() - t0, 1)
        return trace

    # ---------------- Stage 3 — Director ----------------
    _emit(on_event, "stage_start", stage=3, label="Director — writing 5 candidate answers")
    s3_user = build_stage3_user(
        db_id, question, evidence,
        format_findings(s1_results), phase4, phase4b, format_findings(s2_results),
    )
    s3_raw = call_llm(STAGE3_SYSTEM, s3_user, STAGE3_MODEL, api_key)
    s3_clean = strip_think_blocks(s3_raw)
    candidates = parse_candidates(s3_clean)
    _emit(on_event, "llm_done", stage=3, n_candidates=len(candidates))

    # Gold rows (only for curated sample questions where gold SQL is known)
    gold_exec = None
    if gold_sql:
        gold_exec = execute_candidate(db_path, gold_sql)

    cand_records = []
    chosen = None
    for c in candidates:
        _emit(on_event, "candidate_start", n=c["n"])
        res = execute_candidate(db_path, c["sql"])
        rec = {
            "n": c["n"],
            "comment": c["comment"],
            "sql": c["sql"],
            "status": res["status"],
            "columns": res["columns"],
            "rows": res["rows"][:50],           # keep traces light
            "row_count": res["row_count"],
            "truncated": res["truncated"],
            "error": res["error"],
            "matches_gold": None,
        }
        if gold_exec and gold_exec["status"] == "success" and res["status"] == "success":
            rec["matches_gold"] = rows_match(res["rows"], gold_exec["rows"])
        cand_records.append(rec)
        _emit(on_event, "candidate_done", n=c["n"], status=res["status"],
              row_count=res["row_count"], matches_gold=rec["matches_gold"])

    # Chosen answer: first candidate that ran clean with rows; else first clean one
    for rec in cand_records:
        if rec["status"] == "success" and rec["row_count"] > 0:
            chosen = rec["n"]
            break
    if chosen is None:
        for rec in cand_records:
            if rec["status"] == "success":
                chosen = rec["n"]
                break

    trace["stages"]["stage3"] = {
        "model": STAGE3_MODEL,
        "system_prompt": STAGE3_SYSTEM,
        "user_prompt": s3_user,
        "response": s3_clean,
        "candidates": cand_records,
        "chosen": chosen,
        "gold_result": ({"columns": gold_exec["columns"], "rows": gold_exec["rows"][:50],
                         "row_count": gold_exec["row_count"]}
                        if gold_exec and gold_exec["status"] == "success" else None),
    }
    trace["elapsed_s"] = round(time.time() - t0, 1)
    _emit(on_event, "done", chosen=chosen, elapsed_s=trace["elapsed_s"])
    return trace
