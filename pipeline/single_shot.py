from __future__ import annotations
"""
single_shot.py — The baseline the pipeline is compared against.

One LLM call: a generic "write the SQL" prompt with the schema (and optionally the
BIRD evidence hint). No exploration, no verification. This is what most text-to-SQL
setups do, and what the 3-stage pipeline is trying to beat.
"""

import re

from .llm import call_llm
from .parsing import strip_think_blocks
from .executor import execute_candidate, rows_match

SINGLE_SHOT_SYSTEM = (
    "You are an expert data analyst who writes SQLite queries. "
    "Given a database schema and a question, write ONE SQLite SELECT query that answers it. "
    "Use only tables and columns that appear in the schema. "
    "Output the query and nothing else — no explanation, no markdown fences."
)


def _extract_sql(text: str) -> str:
    """Pull a single SQL statement out of a model response."""
    text = strip_think_blocks(text or "")
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    m = re.search(r"((?:WITH|SELECT)\b.*)", text, re.DOTALL | re.IGNORECASE)
    sql = (m.group(1) if m else text).strip()
    # first statement only
    sql = sql.split(";")[0].strip()
    return sql


def build_single_shot_user(question: str, schema_text: str, evidence: str,
                           include_evidence: bool) -> str:
    lines = ["Schema:", "", schema_text, "", f"Question: {question}"]
    if include_evidence and evidence.strip():
        lines.append(f"Hint: {evidence.strip()}")
    lines += ["", "Write the SQLite query:"]
    return "\n".join(lines)


def run_single_shot(db_path: str, question: str, schema_text: str, evidence: str,
                    model: str, api_key: str, include_evidence: bool,
                    gold_rows: list | None = None, gold_ok: bool = False) -> dict:
    """One-call baseline. Returns a record shaped like a Stage 3 candidate."""
    user = build_single_shot_user(question, schema_text, evidence, include_evidence)
    rec = {
        "include_evidence": include_evidence,
        "system_prompt": SINGLE_SHOT_SYSTEM,
        "user_prompt": user,
        "sql": "",
        "status": "error",
        "columns": [], "rows": [], "row_count": 0, "truncated": False,
        "error": None,
        "matches_gold": None,
    }
    try:
        raw = call_llm(SINGLE_SHOT_SYSTEM, user, model, api_key)
    except Exception as e:  # noqa: BLE001 — surface any LLM failure as a record
        rec["error"] = f"LLM call failed: {e}"
        return rec

    sql = _extract_sql(raw)
    rec["sql"] = sql
    if not sql:
        rec["error"] = "Model produced no SQL."
        return rec

    res = execute_candidate(db_path, sql)
    rec.update({
        "status": res["status"],
        "columns": res["columns"],
        "rows": res["rows"][:50],
        "row_count": res["row_count"],
        "truncated": res["truncated"],
        "error": res["error"],
    })
    if gold_ok and res["status"] == "success" and gold_rows is not None:
        rec["matches_gold"] = rows_match(res["rows"], gold_rows)
    return rec
