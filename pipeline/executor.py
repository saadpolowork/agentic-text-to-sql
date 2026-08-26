from __future__ import annotations
"""
executor.py — Safe SQLite execution for exploratory queries and candidates.

Safety model (this app exposes SQL execution to LLM output, so it is strict):
  - PRAGMA query_only = ON  (connection cannot write)
  - Only statements starting with SELECT are executed
  - A progress-handler deadline aborts any statement exceeding the timeout
    (the fix from Trial 12: connection timeout only bounds lock waits, not execution)
  - Automatic LIMIT injection for exploratory queries without one
"""

import os
import re
import sqlite3
import time

HEAD_ROWS = 5
EXPLORE_ROW_CAP = 20          # rows fetched per exploratory query
QUERY_TIMEOUT_S = 10          # per-statement execution deadline
CANDIDATE_TIMEOUT_S = 30
MAX_RESULT_ROWS = 500         # cap for candidate result sets shown/compared


def _connect_readonly(db_path: str, timeout_s: float) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA query_only = ON")
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    return conn


def _is_select(sql: str) -> bool:
    return bool(re.match(r'\s*SELECT\b', sql, re.IGNORECASE)) or \
           bool(re.match(r'\s*WITH\b', sql, re.IGNORECASE))


def execute_exploratory(db_path: str, sql: str) -> dict:
    """Execute one exploratory query; return the research-format result record."""
    result = {"status": "success", "head": [], "columns": [],
              "error": None, "row_count": None}
    if not os.path.exists(db_path):
        result["status"] = "error"
        result["error"] = f"Database file not found: {os.path.basename(db_path)}"
        return result
    if not _is_select(sql):
        result["status"] = "error"
        result["error"] = "Only SELECT statements are executed."
        return result

    try:
        conn = _connect_readonly(db_path, QUERY_TIMEOUT_S)
        sql_safe = sql.rstrip("; \n")
        if not re.search(r'\bLIMIT\b', sql_safe, re.IGNORECASE):
            sql_safe = f"{sql_safe} LIMIT {EXPLORE_ROW_CAP}"
        cur = conn.execute(sql_safe)
        all_rows = cur.fetchmany(EXPLORE_ROW_CAP)
        result["columns"] = [d[0] for d in cur.description] if cur.description else []
        result["head"] = [list(row) for row in all_rows[:HEAD_ROWS]]
        result["row_count"] = len(all_rows)
        conn.close()
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    return result


def run_parsed_queries(db_path: str, queries_parsed: list) -> list:
    """Execute every parsed query (splitting multi-statement items on ';')."""
    query_results = []
    for q in queries_parsed:
        stmts = [s.strip() for s in q["sql"].split(";") if s.strip()]
        for si, stmt in enumerate(stmts):
            exec_result = execute_exploratory(db_path, stmt)
            label = f"{q['n']}" if len(stmts) == 1 else f"{q['n']}.{si + 1}"
            query_results.append({
                "n": label,
                "comment": q["comment"] if si == 0 else "",
                "sql": stmt,
                "status": exec_result["status"],
                "columns": exec_result["columns"],
                "head": exec_result["head"],
                "row_count": exec_result["row_count"],
                "error": exec_result["error"],
            })
    return query_results


def execute_candidate(db_path: str, sql: str) -> dict:
    """Execute a candidate query fully (capped); return rows for display/comparison."""
    out = {"status": "success", "columns": [], "rows": [],
           "row_count": 0, "truncated": False, "error": None}
    if not _is_select(sql):
        out["status"] = "error"
        out["error"] = "Only SELECT statements are executed."
        return out
    try:
        conn = _connect_readonly(db_path, CANDIDATE_TIMEOUT_S)
        cur = conn.execute(sql.rstrip("; \n"))
        rows = cur.fetchmany(MAX_RESULT_ROWS + 1)
        out["columns"] = [d[0] for d in cur.description] if cur.description else []
        if len(rows) > MAX_RESULT_ROWS:
            out["truncated"] = True
            rows = rows[:MAX_RESULT_ROWS]
        out["rows"] = [list(r) for r in rows]
        out["row_count"] = len(rows)
        conn.close()
    except Exception as e:
        out["status"] = "error"
        out["error"] = str(e)
    return out


def normalize_rows(rows: list) -> list:
    """Order-insensitive, type-insensitive normalization (research comparison rule)."""
    return sorted([tuple(str(v) for v in row) for row in rows])


def rows_match(rows_a: list, rows_b: list) -> bool:
    return normalize_rows(rows_a) == normalize_rows(rows_b)
