#!/usr/bin/env python3
"""
Prepare TOFU forget10 data for method training and FQ/MU evaluation.

Outputs a local, stable JSONL layout under framework/outputs/tofu by default:
  data/forget10.jsonl
  data/retain90.jsonl
  data/real_authors.jsonl          if available in locuslab/TOFU
  data/world_facts.jsonl           if available in locuslab/TOFU
  data/forget10_paraphrased.jsonl  if available
  data/forget10_perturbed.jsonl    if available
  method_inputs/full_forget10_retain90.jsonl
  subjects/forget10_subjects.txt

The method input JSONL uses the same {subject,prompt,response} schema as the
existing KIF method plugins, so LUNAR/ReGLU/OPT-OUT/SimNPO can reuse their
loaders on TOFU without changing their core code.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _norm_subject(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _extract_subject(row: Dict[str, Any]) -> str:
    for key in ("subject", "author", "name", "person", "target_subject"):
        if row.get(key):
            return str(row[key]).strip()
    q = str(row.get("question") or row.get("prompt") or "")
    # TOFU questions are generated so the full author name appears in the question.
    # Common patterns include "What ... <Name> ...?". If no explicit author field is
    # present, use the first title-like bigram/trigram as a conservative fallback.
    m = re.search(r"(?:about|for|by|of|does|did|is|was|has|had|will|would)\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3})", q)
    if m:
        return m.group(1).strip()
    caps = re.findall(r"\b[A-Z][A-Za-z'\-]+\b", q)
    if len(caps) >= 2:
        return " ".join(caps[:2])
    return "__unknown_author__"


def _prompt(row: Dict[str, Any]) -> str:
    for key in ("question", "prompt", "query", "instruction"):
        if row.get(key):
            return str(row[key]).strip()
    return ""


def _answer(row: Dict[str, Any]) -> str:
    for key in ("answer", "response", "completion", "output", "target"):
        if row.get(key):
            return str(row[key]).strip()
    return ""


def _paraphrased_answer(row: Dict[str, Any]) -> str:
    for key in ("paraphrased_answer", "paraphrase", "paraphrased", "answer_paraphrased"):
        if row.get(key):
            return str(row[key]).strip()
    return _answer(row)


def _perturbed_answers(row: Dict[str, Any]) -> List[str]:
    for key in ("perturbed_answer", "perturbed_answers", "perturbation", "perturbations", "wrong_answers", "incorrect_answers"):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            out = [str(x).strip() for x in val if str(x).strip()]
            if out:
                return out
        if isinstance(val, str) and val.strip():
            # Some dataset variants store a JSON-like list string; try JSON first.
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    out = [str(x).strip() for x in parsed if str(x).strip()]
                    if out:
                        return out
            except Exception:
                pass
            return [val.strip()]
    return []


def _to_basic_row(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    p = _prompt(row)
    a = _answer(row)
    if not p or not a:
        return None
    return {"subject": _extract_subject(row), "prompt": p, "response": a}


def _to_eval_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    p = _prompt(row)
    a = _answer(row)
    if not p or not a:
        return None
    return {
        "subject": _extract_subject(row),
        "prompt": p,
        "answer": a,
        "paraphrased_answer": _paraphrased_answer(row),
        "perturbed_answers": _perturbed_answers(row),
        "raw": row,
    }


def _load_hf_split(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    return [dict(x) for x in ds]


def _try_load(dataset_name: str, split_names: Iterable[str]) -> Optional[List[Dict[str, Any]]]:
    for split in split_names:
        try:
            rows = _load_hf_split(dataset_name, split)
            print(f"[OK] loaded {dataset_name}:{split} rows={len(rows)}")
            return rows
        except Exception as exc:
            print(f"[skip] {dataset_name}:{split}: {exc}")
    return None


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"[write] {path} rows={n}")
    return n


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _ensure_perturbations(eval_rows: List[Dict[str, Any]], donor_rows: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    # Faithful path: use dataset-provided perturbations. Fallback only if the HF split lacks them.
    donor_answers = [str(r.get("answer") or r.get("response") or "").strip() for r in donor_rows]
    donor_answers = [a for a in donor_answers if a]
    for i, row in enumerate(eval_rows):
        if row.get("perturbed_answers"):
            continue
        own = row.get("answer", "")
        wrong = [a for a in donor_answers if a and a != own]
        if not wrong:
            wrong = ["I do not know the answer."]
        start = (i * k) % len(wrong)
        vals = [wrong[(start + j) % len(wrong)] for j in range(min(k, len(wrong)))]
        row["perturbed_answers"] = vals
        row["perturbation_source"] = "fallback_other_tofu_answers"
    return eval_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locuslab/TOFU")
    ap.add_argument("--out_dir", default="framework/outputs/tofu")
    ap.add_argument("--split", default="forget10", choices=["forget01", "forget05", "forget10"])
    args = ap.parse_args()

    out = Path(args.out_dir)
    data_dir = out / "data"
    method_dir = out / "method_inputs"
    subj_dir = out / "subjects"
    data_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)
    subj_dir.mkdir(parents=True, exist_ok=True)

    retain_split = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}[args.split]

    forget_raw = _try_load(args.dataset, [args.split])
    retain_raw = _try_load(args.dataset, [retain_split])
    if forget_raw is None or retain_raw is None:
        raise SystemExit(f"Could not load required TOFU splits: {args.split}, {retain_split}")

    # Optional evaluation splits have slightly different names across TOFU/OpenUnlearning versions.
    real_raw = _try_load(args.dataset, ["real_authors", "real_author", "real_authors_perturbed", "real_authors_qa"])
    world_raw = _try_load(args.dataset, ["world_facts", "world_fact", "world_facts_perturbed", "world_facts_qa"])

    forget_eval = [_to_eval_row(r) for r in forget_raw]
    retain_eval = [_to_eval_row(r) for r in retain_raw]
    forget_eval = [r for r in forget_eval if r is not None]
    retain_eval = [r for r in retain_eval if r is not None]
    forget_eval = _ensure_perturbations(forget_eval, retain_eval, k=5)
    retain_eval = _ensure_perturbations(retain_eval, forget_eval, k=5)

    forget_basic = [{"subject": r["subject"], "prompt": r["prompt"], "response": r["answer"]} for r in forget_eval]
    retain_basic = [{"subject": r["subject"], "prompt": r["prompt"], "response": r["answer"]} for r in retain_eval]

    _write_jsonl(data_dir / f"{args.split}.jsonl", forget_eval)
    _write_jsonl(data_dir / f"{retain_split}.jsonl", retain_eval)
    _write_jsonl(method_dir / f"full_{args.split}_{retain_split}.jsonl", forget_basic + retain_basic)
    _write_jsonl(method_dir / f"{args.split}_only.jsonl", forget_basic)
    _write_jsonl(method_dir / f"{retain_split}_only.jsonl", retain_basic)

    if real_raw is not None:
        real_eval = [_to_eval_row(r) for r in real_raw]
        real_eval = [r for r in real_eval if r is not None]
        real_eval = _ensure_perturbations(real_eval, retain_eval + forget_eval, k=5)
        _write_jsonl(data_dir / "real_authors.jsonl", real_eval)
    else:
        print("[warn] real_authors split not found; MU will use retain/world available metrics only unless provided later.")

    if world_raw is not None:
        world_eval = [_to_eval_row(r) for r in world_raw]
        world_eval = [r for r in world_eval if r is not None]
        world_eval = _ensure_perturbations(world_eval, retain_eval + forget_eval, k=5)
        _write_jsonl(data_dir / "world_facts.jsonl", world_eval)
    else:
        print("[warn] world_facts split not found; MU will use retain/real available metrics only unless provided later.")

    subjects = sorted({r["subject"] for r in forget_basic})
    (subj_dir / f"{args.split}_subjects.txt").write_text("\n".join(subjects) + "\n", encoding="utf-8")
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "retain_split": retain_split,
        "forget_rows": len(forget_basic),
        "retain_rows": len(retain_basic),
        "forget_subjects": subjects,
        "forget_subject_count": len(subjects),
        "forget_counts": dict(Counter(r["subject"] for r in forget_basic)),
        "method_input": str(method_dir / f"full_{args.split}_{retain_split}.jsonl"),
        "note": "If the HF split lacks provided perturbations, prepare_tofu falls back to other TOFU answers and marks perturbation_source.",
    }
    (out / f"prepare_summary_{args.split}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
