#!/usr/bin/env python3
"""Evaluate No-Capsule LoRA ablation on adversarial forget-recovery.

This uses the same 1,474-row adversarial recovery dataset and metrics as the
main adversarial_forget_recovery_eval.py, but evaluates only the no_capsule
adapter and writes a compact comparison summary against existing PRE and KIF
checkpoint rows when available.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from fast_entity_eval_bundle import (  # type: ignore
    PREFERRED_FORGET_SUBJECTS,
    append_jsonl,
    completed_ids,
    eval_one_row,
    load_model,
    load_tokenizer,
    parse_subjects,
    read_jsonl,
    write_json,
)
from adversarial_forget_recovery_eval import build_adversarial_dataset, aggregate  # type: ignore


def log(msg: str) -> None:
    print(f"[NO-CAPSULE-EVAL] {msg}", flush=True)


def summarize(rows: Sequence[Dict[str, Any]], keyword_threshold: float, mass_threshold: float) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    by_family: Dict[str, Any] = {}
    for fam in sorted({str(r.get("probe_family")) for r in ok}):
        by_family[fam] = aggregate([r for r in ok if str(r.get("probe_family")) == fam], keyword_threshold, mass_threshold)
    return {"overall": aggregate(rows, keyword_threshold, mass_threshold), "by_probe_family": by_family}


def paper_key(summaries: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "overall_recovery": {
            m: {
                "target_alias_hit": summaries[m]["overall"].get("target_alias_hit"),
                "target_keyword_hit_rate": summaries[m]["overall"].get("target_keyword_hit_rate"),
                "target_mass": summaries[m]["overall"].get("target_mass"),
                "recovery_success": summaries[m]["overall"].get("recovery_success"),
                "refusal_like": summaries[m]["overall"].get("refusal_like"),
            } for m in summaries
        },
        "by_attack_family_recovery_success": {
            fam: {m: summaries[m]["by_probe_family"].get(fam, {}).get("recovery_success") for m in summaries}
            for fam in sorted({fam for m in summaries for fam in summaries[m].get("by_probe_family", {})})
        },
    }


def write_comparison(out_dir: Path, main_advrec_dir: Path, dataset_rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    ckpt = out_dir / "checkpoints"
    rows_nc = read_jsonl(ckpt / "eval_rows_no_capsule.jsonl")
    rows_pre = read_jsonl(main_advrec_dir / "checkpoints" / "eval_rows_pre.jsonl")
    rows_kif = read_jsonl(main_advrec_dir / "checkpoints" / "eval_rows_kif.jsonl")
    summaries = {"no_capsule": summarize(rows_nc, args.recovery_keyword_threshold, args.recovery_mass_threshold)}
    if rows_pre:
        summaries["pre"] = summarize(rows_pre, args.recovery_keyword_threshold, args.recovery_mass_threshold)
    if rows_kif:
        summaries["kif"] = summarize(rows_kif, args.recovery_keyword_threshold, args.recovery_mass_threshold)
    dataset_ids = {r["row_id"] for r in dataset_rows}
    completed = {
        "no_capsule": {r.get("row_id") for r in rows_nc if r.get("row_id")},
        "pre": {r.get("row_id") for r in rows_pre if r.get("row_id")},
        "kif": {r.get("row_id") for r in rows_kif if r.get("row_id")},
    }
    summary = {
        "metadata": {
            "model_dir": args.model_dir,
            "no_capsule_adapter_path": args.no_capsule_adapter_path,
            "main_advrec_dir": str(main_advrec_dir),
            "args": vars(args),
        },
        "completion": {
            m: {"eval_completed": len(completed[m]), "eval_total": len(dataset_ids), "eval_remaining": len(dataset_ids - completed[m])}
            for m in completed if (m == "no_capsule" or completed[m])
        },
        "same_row_ids_pre_kif_no_capsule": bool(completed["pre"] and completed["kif"] and completed["pre"] == completed["kif"] == completed["no_capsule"]),
        "paper_key_results": paper_key(summaries),
    }
    write_json(out_dir / "no_capsule_ablation_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--no_capsule_adapter_path", required=True)
    ap.add_argument("--prompts_jsonl", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--out_dir", default="analysis/outputs_no_capsule_lora_ablation_eval")
    ap.add_argument("--main_advrec_dir", default="analysis/outputs_adversarial_forget_recovery")
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "8bit", "bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_eval_rows", type=int, default=2000)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--el_steps", type=int, default=8)
    ap.add_argument("--max_keywords", type=int, default=12)
    ap.add_argument("--recovery_keyword_threshold", type=float, default=0.15)
    ap.add_argument("--recovery_mass_threshold", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    ckpt = out_dir / "checkpoints"
    dataset_dir = out_dir / "datasets"
    ckpt.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    subjects = parse_subjects(Path(args.prompts_jsonl), args.max_subjects) or PREFERRED_FORGET_SUBJECTS[:args.max_subjects]
    dataset_rows = build_adversarial_dataset(subjects, dataset_dir / "adversarial_forget_recovery.jsonl", rebuild=False)
    eval_path = ckpt / "eval_rows_no_capsule.jsonl"
    done = completed_ids(eval_path)
    todo = [r for r in dataset_rows if r["row_id"] not in done]
    n = min(len(todo), args.max_eval_rows)
    log(f"No-capsule eval progress before run: {len(done)}/{len(dataset_rows)}; this run={n}")

    if n > 0:
        tok = load_tokenizer(args.model_dir)
        model = load_model(args.no_capsule_adapter_path, args.model_dir, args.device, args.load_mode)
        try:
            for i, r in enumerate(todo[:n], 1):
                rec = eval_one_row(model, tok, r, "no_capsule", args.device, args.max_new_tokens, args.el_steps, args.max_keywords)
                append_jsonl(eval_path, rec)
                if i % 25 == 0:
                    log(f"evaluated {i}/{n} rows this run")
        finally:
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = write_comparison(out_dir, Path(args.main_advrec_dir), dataset_rows, args)
    log(json.dumps({
        "completion": summary.get("completion"),
        "same_row_ids_pre_kif_no_capsule": summary.get("same_row_ids_pre_kif_no_capsule"),
        "paper_key_results": summary.get("paper_key_results"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
