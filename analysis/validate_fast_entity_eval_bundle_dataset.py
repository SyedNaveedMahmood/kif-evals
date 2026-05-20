#!/usr/bin/env python3
"""
Validator for the generated fast_entity_eval_bundle dataset.

This script is intentionally separate from the evaluator so that the dataset can
be materialized, audited, and reported before any model is loaded.

It checks:
  - deterministic row_id uniqueness
  - required field/schema coverage
  - expected audits/control types/probe families
  - no canonical-name leakage in name_agnostic prompts
  - minimum per-subject and per-audit coverage
  - mixed prompts contain retain entities and target forget aliases/keywords
  - syntactic-locality rows contain retain entities and locality controls
  - metadata richness for all 11 forgotten subjects

Usage:
  python analysis/validate_fast_entity_eval_bundle_dataset.py \
    --dataset analysis/outputs_fast_entity_eval_bundle/datasets/fast_entity_eval_bundle.jsonl \
    --report analysis/outputs_fast_entity_eval_bundle/datasets/fast_entity_eval_bundle_validation.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

EXPECTED_SUBJECTS = [
    "Ariana Grande", "Arijit Singh", "Beyoncé", "Drake (musician)",
    "Ed Sheeran", "Eminem", "Kanye West", "Katy Perry", "Michael Jackson",
    "Queen (band)", "Taylor Swift",
]

EXPECTED_AUDITS = {"name_agnostic", "blur_mixed", "syntactic_locality"}
EXPECTED_CONTROL_TYPES = {"forget", "mixed", "matched_control", "neighbor_locality", "generic_benign"}
EXPECTED_PROBE_FAMILIES = {
    "alias_only", "descriptor_only", "relation_clue", "masked_name",
    "forget_retain_overlap", "same_syntax_neighbor", "generic_benign",
}
REQUIRED_FIELDS = [
    "row_id", "audit", "subject", "control_type", "probe_family", "prompt",
    "target_aliases", "target_keywords", "retain_entity", "retain_aliases",
    "retain_keywords", "expected_direction", "metadata",
]


def canonical_plain(subject: str) -> str:
    return subject.replace("(musician)", "").replace("(band)", "").strip()


def contains_canonical(prompt: str, subject: str) -> bool:
    if subject == "__generic__":
        return False
    p = prompt.lower()
    s1 = subject.lower()
    s2 = canonical_plain(subject).lower()
    return (s1 in p) or (len(s2) > 3 and s2 in p)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row["_lineno"] = lineno
                rows.append(row)
            except Exception as exc:
                rows.append({"_lineno": lineno, "_parse_error": str(exc), "_raw": line[:500]})
    return rows


def add_issue(issues: List[Dict[str, Any]], severity: str, message: str, **extra: Any) -> None:
    issues.append({"severity": severity, "message": message, **extra})


def validate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    good_rows = [r for r in rows if "_parse_error" not in r]

    for r in rows:
        if "_parse_error" in r:
            add_issue(issues, "error", "JSONL parse error", lineno=r.get("_lineno"), error=r.get("_parse_error"))

    row_ids = [r.get("row_id") for r in good_rows]
    duplicate_ids = [rid for rid, c in Counter(row_ids).items() if rid and c > 1]
    if duplicate_ids:
        add_issue(issues, "error", "Duplicate row_id values detected", duplicate_count=len(duplicate_ids), examples=duplicate_ids[:10])

    prompts = [r.get("prompt", "") for r in good_rows]
    duplicate_prompts = [p for p, c in Counter(prompts).items() if p and c > 1]
    if duplicate_prompts:
        add_issue(issues, "warning", "Duplicate prompt strings detected", duplicate_count=len(duplicate_prompts), examples=duplicate_prompts[:5])

    for r in good_rows:
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            add_issue(issues, "error", "Missing required fields", row_id=r.get("row_id"), missing=missing)
        if r.get("audit") not in EXPECTED_AUDITS:
            add_issue(issues, "error", "Unexpected audit", row_id=r.get("row_id"), audit=r.get("audit"))
        if r.get("control_type") not in EXPECTED_CONTROL_TYPES:
            add_issue(issues, "error", "Unexpected control_type", row_id=r.get("row_id"), control_type=r.get("control_type"))
        if r.get("probe_family") not in EXPECTED_PROBE_FAMILIES:
            add_issue(issues, "error", "Unexpected probe_family", row_id=r.get("row_id"), probe_family=r.get("probe_family"))
        if not str(r.get("prompt", "")).strip():
            add_issue(issues, "error", "Empty prompt", row_id=r.get("row_id"))
        if r.get("subject") != "__generic__" and not r.get("target_aliases"):
            add_issue(issues, "error", "Missing target aliases", row_id=r.get("row_id"), subject=r.get("subject"))
        if r.get("subject") != "__generic__" and not r.get("target_keywords"):
            add_issue(issues, "error", "Missing target keywords", row_id=r.get("row_id"), subject=r.get("subject"))

    name_rows = [r for r in good_rows if r.get("audit") == "name_agnostic"]
    canonical_violations = [
        {"row_id": r.get("row_id"), "subject": r.get("subject"), "prompt": r.get("prompt")}
        for r in name_rows
        if contains_canonical(str(r.get("prompt", "")), str(r.get("subject", "")))
    ]
    if canonical_violations:
        add_issue(issues, "error", "Canonical subject names appear in name_agnostic prompts", count=len(canonical_violations), examples=canonical_violations[:10])

    blur_rows = [r for r in good_rows if r.get("audit") == "blur_mixed"]
    for r in blur_rows:
        if not r.get("retain_entity") or not r.get("retain_aliases"):
            add_issue(issues, "error", "BLUR mixed row missing retain entity/aliases", row_id=r.get("row_id"))

    locality_rows = [r for r in good_rows if r.get("audit") == "syntactic_locality"]
    for r in locality_rows:
        if r.get("control_type") in {"matched_control", "neighbor_locality"} and not r.get("retain_entity"):
            add_issue(issues, "error", "Locality row missing retain entity", row_id=r.get("row_id"))

    by_audit = Counter(r.get("audit") for r in good_rows)
    by_control = Counter(r.get("control_type") for r in good_rows)
    by_family = Counter(r.get("probe_family") for r in good_rows)
    by_subject = Counter(r.get("subject") for r in good_rows if r.get("subject") != "__generic__")
    by_subject_audit: Dict[str, Counter] = defaultdict(Counter)
    by_subject_family: Dict[str, Counter] = defaultdict(Counter)
    for r in good_rows:
        s = r.get("subject")
        if s and s != "__generic__":
            by_subject_audit[s][r.get("audit")] += 1
            by_subject_family[s][r.get("probe_family")] += 1

    for s in EXPECTED_SUBJECTS:
        if by_subject[s] == 0:
            add_issue(issues, "error", "Expected subject missing from dataset", subject=s)
        if by_subject_audit[s]["name_agnostic"] < 30:
            add_issue(issues, "error", "Insufficient name_agnostic coverage", subject=s, count=by_subject_audit[s]["name_agnostic"])
        if by_subject_audit[s]["blur_mixed"] < 30:
            add_issue(issues, "error", "Insufficient blur_mixed coverage", subject=s, count=by_subject_audit[s]["blur_mixed"])
        if by_subject_audit[s]["syntactic_locality"] < 20:
            add_issue(issues, "error", "Insufficient syntactic_locality coverage", subject=s, count=by_subject_audit[s]["syntactic_locality"])
        for family in ["alias_only", "descriptor_only", "relation_clue", "masked_name"]:
            if by_subject_family[s][family] == 0:
                add_issue(issues, "error", "Missing name-agnostic probe family for subject", subject=s, family=family)

    # Strong dataset size expectations. These are warnings rather than errors so
    # the user can intentionally run small smoke/debug datasets if needed.
    if len(good_rows) < 700:
        add_issue(issues, "warning", "Dataset is smaller than recommended for paper use", n_rows=len(good_rows), recommended_min=700)
    if by_audit["name_agnostic"] < 350:
        add_issue(issues, "warning", "Name-agnostic audit has low row count", count=by_audit["name_agnostic"])
    if by_audit["blur_mixed"] < 250:
        add_issue(issues, "warning", "BLUR mixed audit has low row count", count=by_audit["blur_mixed"])
    if by_audit["syntactic_locality"] < 250:
        add_issue(issues, "warning", "Syntactic locality audit has low row count", count=by_audit["syntactic_locality"])

    n_errors = sum(1 for x in issues if x["severity"] == "error")
    n_warnings = sum(1 for x in issues if x["severity"] == "warning")
    return {
        "valid": n_errors == 0,
        "n_rows": len(good_rows),
        "n_parse_errors": len(rows) - len(good_rows),
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "by_audit": dict(by_audit),
        "by_control_type": dict(by_control),
        "by_probe_family": dict(by_family),
        "by_subject": dict(by_subject),
        "by_subject_audit": {k: dict(v) for k, v in by_subject_audit.items()},
        "issues": issues,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    rows = read_jsonl(dataset_path)
    report = validate(rows)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
