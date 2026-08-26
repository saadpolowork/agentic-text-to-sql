from __future__ import annotations
"""
parsing.py — Extraction of queries, candidates, and phase sections from LLM output.

Carries the hard-won parser fixes from the research trials:
  - think-block stripping (reasoning models)
  - unnumbered-C1 recovery (LLMs often emit candidate 1 as raw SQL with no "1. --" prefix)
  - comment-bleed guard (take the LAST SELECT before the first numbered item)
"""

import re


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def extract_phase(text: str, phase_header: str, next_header: str | None) -> str:
    pattern = rf"{re.escape(phase_header)}\s*(.*?)"
    if next_header:
        pattern += rf"(?={re.escape(next_header)})"
    else:
        pattern += r"$"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_stage1_phases(clean_text: str) -> tuple[str, str]:
    """Return (phase1, phase2) text from a Stage 1 response."""
    phase1 = (
        extract_phase(clean_text, "## PHASE 1 — TWO SEMANTIC HYPOTHESES", "## PHASE 2")
        or extract_phase(clean_text, "## PHASE 1 — THREE HYPOTHESES", "## PHASE 2")
        or extract_phase(clean_text, "## PHASE 1 — INITIAL QUESTIONS & ASSUMPTIONS", "## PHASE 2")
        or extract_phase(clean_text, "## PHASE 1", "## PHASE 2")
    )
    phase2 = (
        extract_phase(clean_text, "## PHASE 2 — SCHEMA DEEP DIVE", "## PHASE 3")
        or extract_phase(clean_text, "## PHASE 2 — SCHEMA REVIEW", "## PHASE 3")
        or extract_phase(clean_text, "## PHASE 2", "## PHASE 3")
    )
    if not phase1 and not phase2:
        phase3_start = clean_text.find("## PHASE 3")
        if phase3_start > 0:
            phase1 = clean_text[:phase3_start].strip()
            phase2 = "(Not separately extracted — see Phase 1 above)"
        else:
            phase1 = clean_text[:2000]
            phase2 = "(Not available)"
    return phase1, phase2


def extract_stage2_phases(clean_text: str) -> tuple[str, str]:
    """Return (phase4, phase4b) text from a Stage 2 response."""
    phase4 = (
        extract_phase(clean_text, "## PHASE 4 — HYPOTHESIS VERDICT", "## PHASE 4b")
        or extract_phase(clean_text, "## PHASE 4 — REVISED ASSUMPTIONS", "## PHASE 4b")
        or extract_phase(clean_text, "## PHASE 4", "## PHASE 5")
    )
    phase4b = (
        extract_phase(clean_text, "## PHASE 4b — WINNING HYPOTHESIS", "## PHASE 5")
        or extract_phase(clean_text, "## PHASE 4b — HYPOTHESIS REVIEW", "## PHASE 5")
        or "(Not available)"
    )
    return phase4, phase4b


def parse_queries(text: str) -> list:
    """Extract numbered exploratory SQL queries from a Stage 1/2 response."""
    if not text or not text.strip():
        return []

    text = strip_think_blocks(text)

    # Isolate the query section (Phase 3 or Phase 5)
    phase_match = re.search(r'##\s*PHASE\s*[35][^\n]*\n', text, re.IGNORECASE)
    body = text[phase_match.end():] if phase_match else text

    queries = []

    # Strategy 1: ```sql blocks
    sql_blocks = list(re.finditer(r'```sql\s*(.*?)```', body, flags=re.DOTALL | re.IGNORECASE))
    if sql_blocks:
        for i, m in enumerate(sql_blocks, start=1):
            sql = m.group(1).strip()
            comment = ""
            inline = re.match(r'--\s*(.+?)(?:\n|$)', sql)
            if inline:
                comment = inline.group(1).strip()
                sql = sql[inline.end():].strip()
            if not comment:
                before = body[:m.start()]
                prev_line = before.rstrip().split('\n')[-1].strip()
                cm = re.match(r'--\s*(.+)', prev_line)
                if cm:
                    comment = cm.group(1).strip()
            if sql:
                queries.append({"n": i, "comment": comment, "sql": sql})
        return queries

    # Strategy 2: numbered items, plain SQL
    parts = re.split(r'\n(?=\d+[\.\)])', body)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        num_match = re.match(r'^(\d+)[\.]\s*', part)
        if not num_match:
            continue
        n = int(num_match.group(1))
        rest = part[num_match.end():]

        comment = ""
        comment_match = re.match(r'--\s*(.+?)(?:\n|$)', rest)
        if comment_match:
            comment = comment_match.group(1).strip()
            rest = rest[comment_match.end():]

        sql_match = re.search(
            r'((?:SELECT|INSERT|UPDATE|DELETE|PRAGMA|WITH\s+\w+\s+AS\s*\()\b.*)',
            rest, flags=re.DOTALL | re.IGNORECASE
        )
        if sql_match:
            sql = sql_match.group(1).strip()
            sql = re.split(r'\n\d+[\.\)]', sql)[0].strip()
            sql = re.sub(r'```\s*$', '', sql).strip()
            if sql and re.search(r'\b(SELECT|FROM|WHERE|WITH)\b', sql, re.IGNORECASE):
                queries.append({"n": n, "comment": comment, "sql": sql})

    return queries


def parse_candidates(text: str) -> list:
    """Extract the 5 candidate SQL queries from a Stage 3 response."""
    if not text or not text.strip():
        return []

    text = strip_think_blocks(text)

    section_match = re.search(r'##\s*CANDIDATE SQL QUERIES[^\n]*\n', text, re.IGNORECASE)
    body = text[section_match.end():] if section_match else text

    candidates = []

    # Strategy 1: ```sql blocks
    sql_blocks = list(re.finditer(r'```sql\s*(.*?)```', body, flags=re.DOTALL | re.IGNORECASE))
    if sql_blocks:
        for i, m in enumerate(sql_blocks, start=1):
            sql = m.group(1).strip()
            comment = ""
            inline = re.match(r'--\s*(.+?)(?:\n|$)', sql)
            if inline:
                comment = inline.group(1).strip()
                sql = sql[inline.end():].strip()
            if not comment:
                before = body[:m.start()]
                prev = before.rstrip().split('\n')[-1].strip()
                cm = re.match(r'--\s*(.+)', prev)
                if cm:
                    comment = cm.group(1).strip()
            if sql:
                candidates.append({"n": i, "comment": comment, "sql": sql})
        return candidates

    # Strategy 2: numbered items
    parts = re.split(r'\n(?=\d+[\.\)])', body)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        num_match = re.match(r'^(\d+)[\.]\s*', part)
        if not num_match:
            continue
        n = int(num_match.group(1))
        rest = part[num_match.end():]

        comment = ""
        cm = re.match(r'--\s*(.+?)(?:\n|$)', rest)
        if cm:
            comment = cm.group(1).strip()
            rest = rest[cm.end():]

        sql_match = re.search(
            r'((?:SELECT|WITH\s+\w+\s+AS\s*\()\b.*)',
            rest, flags=re.DOTALL | re.IGNORECASE
        )
        if sql_match:
            sql = sql_match.group(1).strip()
            sql = re.split(r'\n\d+[\.\)]', sql)[0].strip()
            sql = re.sub(r'```\s*$', '', sql).strip()
            if sql and re.search(r'\b(SELECT|FROM|WHERE|WITH)\b', sql, re.IGNORECASE):
                candidates.append({"n": n, "comment": comment, "sql": sql})

    # C1 recovery: if C1 missing but there's SQL before the first numbered item
    if candidates and not any(c["n"] == 1 for c in candidates):
        first_num = re.search(r'\n\d+[\.\)]', body)
        if first_num:
            pre = body[:first_num.start()].strip()
            # Take the LAST SELECT block to avoid comment-bleed
            pre_lines = pre.split('\n')
            last_select_idx = None
            for i, line in enumerate(pre_lines):
                if re.match(r'\s*SELECT\b', line, re.IGNORECASE):
                    last_select_idx = i
            if last_select_idx is not None:
                sql = '\n'.join(pre_lines[last_select_idx:])
                sql = re.sub(r'```\s*$', '', sql).strip()
                if re.search(r'\b(FROM|WHERE|JOIN)\b', sql, re.IGNORECASE):
                    candidates.insert(0, {"n": 1, "comment": "", "sql": sql})

    return candidates
