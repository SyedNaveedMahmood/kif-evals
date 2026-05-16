#!/usr/bin/env python3
"""Convert KIF Module-A prompts.jsonl into LUNAR's JSON format.

Input KIF record example:
  {"prompt": "What is X's birthplace?", "expected": "Y", "subject": "X", ...}

Output LUNAR record example:
  {"edge": "X", "subject": "X", "question": "What is X's birthplace?", "answer": "Y"}

The edge is a sanitized subject identifier because LUNAR forgets by `edge`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sanitize_edge(subject: str) -> str:
    x = re.sub(r"[^A-Za-z0-9]+", "_", subject.strip())
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "unknown_subject"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kif-prompts-jsonl", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--mapping-out", required=True, type=Path)
    ap.add_argument("--drop-control", action="store_true", help="Drop KIF records with is_control=true.")
    ap.add_argument("--drop-misleading", action="store_true", help="Drop KIF records with is_misleading=true.")
    ap.add_argument("--max-per-subject", type=int, default=0, help="Optional cap per subject; 0 = no cap.")
    args = ap.parse_args()

    rows = read_jsonl(args.kif_prompts_jsonl)
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    subject_to_edge: dict[str, str] = {}
    edge_to_subject: dict[str, str] = {}
    skipped = Counter()

    for rec in rows:
        subject = str(rec.get("subject") or rec.get("author") or "").strip()
        question = str(rec.get("prompt") or rec.get("question") or "").strip()
        if not subject or not question:
            skipped["missing_subject_or_question"] += 1
            continue
        if args.drop_control and bool(rec.get("is_control", False)):
            skipped["control"] += 1
            continue
        if args.drop_misleading and bool(rec.get("is_misleading", False)):
            skipped["misleading"] += 1
            continue
        if args.max_per_subject > 0 and counts[subject] >= args.max_per_subject:
            skipped["max_per_subject"] += 1
            continue

        edge = sanitize_edge(subject)
        # Resolve rare collisions deterministically.
        if edge in edge_to_subject and edge_to_subject[edge] != subject:
            edge = f"{edge}_{abs(hash(subject)) % 100000}"
        subject_to_edge[subject] = edge
        edge_to_subject[edge] = subject

        answer = rec.get("expected") or rec.get("answer") or rec.get("object") or subject
        if isinstance(answer, list):
            answer = answer[0] if answer else subject
        answer = str(answer).strip() or subject

        out.append({
            "edge": edge,
            "subject": subject,
            "question": question,
            "answer": answer,
            "kif_id": rec.get("id"),
            "category": rec.get("category"),
            "predicate": rec.get("predicate"),
            "is_misleading": bool(rec.get("is_misleading", False)),
            "is_control": bool(rec.get("is_control", False)),
        })
        counts[subject] += 1

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.mapping_out.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    args.mapping_out.write_text(json.dumps({
        "subject_to_edge": subject_to_edge,
        "edge_to_subject": edge_to_subject,
        "counts_per_subject": dict(counts),
        "num_records": len(out),
        "skipped": dict(skipped),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "out_json": str(args.out_json),
        "mapping_out": str(args.mapping_out),
        "num_records": len(out),
        "num_subjects": len(subject_to_edge),
        "skipped": dict(skipped),
    }, indent=2))


if __name__ == "__main__":
    main()
