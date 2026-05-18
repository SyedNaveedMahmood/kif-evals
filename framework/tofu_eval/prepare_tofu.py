#!/usr/bin/env python3
"""
Prepare TOFU forget10 data with OpenUnlearning/TOFU-faithful split semantics.

Important correction:
OpenUnlearning does NOT pair retain90 (3600 rows) with retain_perturbed row-by-row.
For evaluation, retain probability/ROUGE/truth-ratio are computed on the official
`retain_perturbed` config itself, which has 400 rows. The large retain90 split is
still useful for method training/retain regularization, so we save both:

  data/retain90.jsonl          -> official retain_perturbed eval rows, 400 rows
  data/retain90_train.jsonl    -> full retain90 training rows, 3600 rows

No fabricated perturbations are allowed. If an official perturbed config lacks
paraphrased_answer or perturbed_answer, this script fails loudly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


QUESTION_KEYS = ("question", "prompt", "query", "instruction")
ANSWER_KEYS = ("answer", "response", "completion", "output", "target")
PARA_KEYS = ("paraphrased_answer", "paraphrase", "paraphrased", "answer_paraphrased")
PERT_KEYS = ("perturbed_answer", "perturbed_answers", "perturbation", "perturbations", "wrong_answers", "incorrect_answers")
SUBJECT_KEYS = ("subject", "author", "name", "person", "target_subject")


def _first(row: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _question(row: Dict[str, Any]) -> str:
    return _first(row, QUESTION_KEYS)


def _answer(row: Dict[str, Any]) -> str:
    return _first(row, ANSWER_KEYS)


def _paraphrased(row: Dict[str, Any]) -> str:
    return _first(row, PARA_KEYS)


def _perturbed(row: Dict[str, Any]) -> List[str]:
    for key in PERT_KEYS:
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            out = [str(x).strip() for x in val if str(x).strip()]
            if out:
                return out
        if isinstance(val, str) and val.strip():
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


def _subject(row: Dict[str, Any], prefix: str, idx: int) -> str:
    explicit = _first(row, SUBJECT_KEYS)
    if explicit:
        return explicit
    # TOFU has 20 QA pairs per synthetic author. Public rows often omit an
    # explicit subject field, so deterministic 20-row grouping preserves author
    # boundaries for method-level forgetting.
    return f"{prefix}_author_{idx // 20:04d}"


def _load_config(dataset: str, name: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset, name=name, split="train")
    rows = [dict(x) for x in ds]
    if not rows:
        raise RuntimeError(f"Loaded empty TOFU config: {dataset}:{name}")
    print(f"[OK] loaded {dataset} name={name!r} split='train' rows={len(rows)} keys={sorted(rows[0].keys())}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"[write] {path} rows={n}")
    return n


def _standard_rows(rows: List[Dict[str, Any]], subject_prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        q = _question(row)
        ans = _answer(row)
        if not q or not ans:
            raise RuntimeError(f"Missing question/answer at {subject_prefix} row {i}: keys={row.keys()}")
        out.append(
            {
                "subject": _subject(row, subject_prefix, i),
                "prompt": q,
                "answer": ans,
                "response": ans,
                "tofu_index": i,
                "tofu_subject_id": _subject(row, subject_prefix, i),
                "source": subject_prefix,
            }
        )
    return out


def _merge_standard_and_perturbed(
    standard_rows: List[Dict[str, Any]],
    perturbed_rows: List[Dict[str, Any]],
    subject_prefix: str,
) -> List[Dict[str, Any]]:
    if len(standard_rows) != len(perturbed_rows):
        raise RuntimeError(
            f"Standard/perturbed row-count mismatch for {subject_prefix}: "
            f"standard={len(standard_rows)} perturbed={len(perturbed_rows)}. "
            "This is expected only for retain90 vs retain_perturbed; do not call "
            "this helper for that case."
        )

    merged: List[Dict[str, Any]] = []
    for i, std in enumerate(standard_rows):
        pert = perturbed_rows[i]
        q = _question(std) or _question(pert)
        ans = _answer(std) or _answer(pert)
        para = _paraphrased(pert)
        perts = _perturbed(pert)

        if not q or not ans:
            raise RuntimeError(f"Missing question/answer at {subject_prefix} row {i}: std_keys={std.keys()} pert_keys={pert.keys()}")
        if not para:
            raise RuntimeError(f"Missing paraphrased_answer at {subject_prefix} row {i}; official *_perturbed config is required.")
        if not perts:
            raise RuntimeError(f"Missing perturbed_answer at {subject_prefix} row {i}; official *_perturbed config is required.")

        q_pert = _question(pert)
        if q_pert and q_pert.strip() != q.strip():
            print(f"[warn] question mismatch at {subject_prefix} row {i}; using standard question")

        merged.append(
            {
                "subject": _subject(std, subject_prefix, i),
                "prompt": q,
                "answer": ans,
                "response": ans,
                "paraphrased_answer": para,
                "perturbed_answers": perts,
                "tofu_index": i,
                "tofu_subject_id": _subject(std, subject_prefix, i),
                "source": subject_prefix,
            }
        )
    return merged


def _from_perturbed_only(rows: List[Dict[str, Any]], subject_prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        q = _question(row)
        ans = _answer(row)
        para = _paraphrased(row)
        perts = _perturbed(row)
        if not q or not ans:
            raise RuntimeError(f"Missing question/answer at {subject_prefix} row {i}: keys={row.keys()}")
        if not para:
            raise RuntimeError(f"Missing paraphrased_answer at {subject_prefix} row {i}; official *_perturbed config is required.")
        if not perts:
            raise RuntimeError(f"Missing perturbed_answer at {subject_prefix} row {i}; official *_perturbed config is required.")
        out.append(
            {
                "subject": _subject(row, subject_prefix, i),
                "prompt": q,
                "answer": ans,
                "response": ans,
                "paraphrased_answer": para,
                "perturbed_answers": perts,
                "tofu_index": i,
                "tofu_subject_id": _subject(row, subject_prefix, i),
                "source": subject_prefix,
            }
        )
    return out


def _method_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"subject": r["subject"], "prompt": r["prompt"], "response": r["answer"]} for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locuslab/TOFU")
    ap.add_argument("--out_dir", default="framework/outputs/tofu")
    ap.add_argument("--split", default="forget10", choices=["forget01", "forget05", "forget10"])
    args = ap.parse_args()

    retain_split = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}[args.split]
    holdout_split = {"forget01": "holdout01", "forget05": "holdout05", "forget10": "holdout10"}[args.split]

    out = Path(args.out_dir)
    data_dir = out / "data"
    method_dir = out / "method_inputs"
    subj_dir = out / "subjects"
    data_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)
    subj_dir.mkdir(parents=True, exist_ok=True)

    # OpenUnlearning-faithful HF configs.
    forget_std = _load_config(args.dataset, args.split)
    retain_std = _load_config(args.dataset, retain_split)
    forget_pert = _load_config(args.dataset, f"{args.split}_perturbed")
    retain_pert = _load_config(args.dataset, "retain_perturbed")
    real_pert = _load_config(args.dataset, "real_authors_perturbed")
    world_pert = _load_config(args.dataset, "world_facts_perturbed")
    holdout_std = _load_config(args.dataset, holdout_split)

    # Eval datasets.
    forget_rows = _merge_standard_and_perturbed(forget_std, forget_pert, args.split)
    # Correct OpenUnlearning behavior: retain eval uses retain_perturbed directly
    # rather than row-matching against full retain90.
    retain_eval_rows = _from_perturbed_only(retain_pert, "retain_perturbed")
    real_rows = _from_perturbed_only(real_pert, "real_authors")
    world_rows = _from_perturbed_only(world_pert, "world_facts")
    holdout_rows = _standard_rows(holdout_std, holdout_split)

    # Training/method datasets.
    retain_train_rows = _standard_rows(retain_std, retain_split)

    _write_jsonl(data_dir / f"{args.split}.jsonl", forget_rows)
    _write_jsonl(data_dir / f"{retain_split}.jsonl", retain_eval_rows)
    _write_jsonl(data_dir / f"{retain_split}_train.jsonl", retain_train_rows)
    _write_jsonl(data_dir / "real_authors.jsonl", real_rows)
    _write_jsonl(data_dir / "world_facts.jsonl", world_rows)
    _write_jsonl(data_dir / f"{holdout_split}.jsonl", holdout_rows)

    # Methods train on forget10 + full retain90. Evaluator uses data/*.jsonl above.
    full_method = _method_rows(forget_rows) + _method_rows(retain_train_rows)
    _write_jsonl(method_dir / f"full_{args.split}_{retain_split}.jsonl", full_method)
    _write_jsonl(method_dir / f"{args.split}_only.jsonl", _method_rows(forget_rows))
    _write_jsonl(method_dir / f"{retain_split}_only.jsonl", _method_rows(retain_train_rows))

    forget_subjects = sorted({r["subject"] for r in forget_rows})
    (subj_dir / f"{args.split}_subjects.txt").write_text("\n".join(forget_subjects) + "\n", encoding="utf-8")

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "retain_split": retain_split,
        "holdout_split": holdout_split,
        "faithfulness": "strict_open_unlearning_tofu_configs_no_fallback_perturbations",
        "hf_configs": {
            "forget_train_eval": args.split,
            "forget_perturbed_eval": f"{args.split}_perturbed",
            "retain_train": retain_split,
            "retain_eval": "retain_perturbed",
            "real_authors_eval": "real_authors_perturbed",
            "world_facts_eval": "world_facts_perturbed",
        },
        "forget_rows": len(forget_rows),
        "retain_eval_rows": len(retain_eval_rows),
        "retain_train_rows": len(retain_train_rows),
        "real_authors_rows": len(real_rows),
        "world_facts_rows": len(world_rows),
        "forget_subject_count": len(forget_subjects),
        "forget_counts": dict(Counter(r["subject"] for r in forget_rows)),
        "method_input": str(method_dir / f"full_{args.split}_{retain_split}.jsonl"),
        "subjects_file": str(subj_dir / f"{args.split}_subjects.txt"),
        "status": "ok",
    }
    (out / f"prepare_summary_{args.split}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
