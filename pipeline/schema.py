"""
schema.py — Profiled schema builder (BIRD tables.json format → annotated text schema).
"""

import json
import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_tables_index(path: str | None = None) -> dict:
    path = path or os.path.join(_DATA_DIR, "tables.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {entry["db_id"]: entry for entry in raw}


def db_path_for(db_id: str, db_root: str | None = None) -> str:
    root = db_root or os.path.join(_DATA_DIR, "databases")
    return os.path.join(root, db_id, f"{db_id}.sqlite")


def build_profiled_schema(meta: dict) -> str:
    """Table names, column names, types, PK/FK annotations — exact research format."""
    table_names = meta["table_names_original"]
    col_names = meta["column_names_original"]
    col_types = meta["column_types"]
    primary_keys = meta.get("primary_keys", [])
    foreign_keys = meta.get("foreign_keys", [])

    pk_set = set()
    for pk in primary_keys:
        if isinstance(pk, list):
            for idx in pk:
                pk_set.add(idx)
        else:
            pk_set.add(pk)

    fk_map = {}
    for fk_col_idx, ref_col_idx in foreign_keys:
        ref_tbl_idx, ref_col_name = col_names[ref_col_idx]
        ref_tbl_name = table_names[ref_tbl_idx] if ref_tbl_idx >= 0 else "?"
        fk_map[fk_col_idx] = (ref_tbl_name, ref_col_name)

    by_table = {i: [] for i in range(len(table_names))}
    for col_idx, (tbl_idx, col_name) in enumerate(col_names):
        if tbl_idx != -1:
            by_table[tbl_idx].append((col_idx, col_name, col_types[col_idx]))

    lines = []
    for tbl_idx, tbl_name in enumerate(table_names):
        lines.append(f"TABLE: {tbl_name}")
        for col_idx, col_name, col_type in by_table[tbl_idx]:
            annotations = []
            if col_idx in pk_set:
                annotations.append("[PK]")
            if col_idx in fk_map:
                ref_tbl, ref_col = fk_map[col_idx]
                annotations.append(f"[FK → {ref_tbl}.{ref_col}]")
            ann_str = "    " + " ".join(annotations) if annotations else ""
            lines.append(f"  {col_name:<32} {col_type:<10}{ann_str}")
        lines.append("")
    return "\n".join(lines)
