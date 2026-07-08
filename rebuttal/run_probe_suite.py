#!/usr/bin/env python3
"""Convenience launcher for ERUF rebuttal probe scripts.

This does not hide experiment details; it only expands the common path arguments
used across the rebuttal runbook and executes one or more scripts in this folder.
Use --dry_run before launching GPU-heavy runs.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent

TASK_TO_SCRIPT = {
    "e2-separability": "signature_separability_collapse_v2.py",
    "e2-erasure": "representation_erasure_suite.py",
    "e3-locality": "cross_domain_locality_probe.py",
    "e3-specificity": "subject_specificity_robustness_suite_fixed.py",
    "e4-hidden": "hidden_space_selectivity_eval.py",
    "e7-el10": "el10_token_audit.py",
    "e8-eval-no-capsule": "evaluate_no_capsule_ablation.py",
}

DEFAULT_OUT = {
    "e2-separability": "outputs/e2_signature_separability",
    "e2-erasure": "outputs/e2_representation_erasure",
    "e3-locality": "outputs/e3_cross_domain_locality",
    "e3-specificity": "outputs/e3_subject_specificity",
    "e4-hidden": "outputs/e4_hidden_space_selectivity",
    "e7-el10": "outputs/e7_el10_token_audit",
    "e8-eval-no-capsule": "outputs/e8_no_capsule_eval",
}


def add_if(cmd: List[str], flag: str, value: str | None) -> None:
    if value:
        cmd.extend([flag, value])


def build_cmd(task: str, args: argparse.Namespace) -> List[str]:
    script = str(ROOT / TASK_TO_SCRIPT[task])
    cmd = [sys.executable, script]
    out_dir = args.out_dir or DEFAULT_OUT[task]

    if task in {"e2-separability", "e2-erasure", "e3-specificity"}:
        add_if(cmd, "--model_dir", args.base_model)
        add_if(cmd, "--kif_adapter_path", args.kif_adapter)
        add_if(cmd, "--baseline_model_dir", args.baseline_model)
        add_if(cmd, "--baseline_prefer", args.baseline_prefer)
        add_if(cmd, "--capsules_dir", args.capsules_dir)
        add_if(cmd, "--prompts_jsonl", args.prompts_jsonl)
        add_if(cmd, "--out_dir", out_dir)
        cmd.extend(["--max_subjects", str(args.max_subjects), "--seed", str(args.seed)])
        if args.use_4bit:
            cmd.append("--use_4bit")
    elif task == "e3-locality":
        add_if(cmd, "--model_dir", args.base_model)
        add_if(cmd, "--kif_adapter_path", args.kif_adapter)
        add_if(cmd, "--baseline_model_dir", args.baseline_model)
        add_if(cmd, "--baseline_prefer", args.baseline_prefer)
        add_if(cmd, "--out_dir", out_dir)
        cmd.extend(["--models", args.models])
        cmd.extend(["--max_eval_rows_per_model", str(args.max_eval_rows_per_model), "--seed", str(args.seed)])
        cmd.extend(["--load_mode", args.load_mode])
    elif task == "e4-hidden":
        add_if(cmd, "--base_model_dir", args.base_model)
        add_if(cmd, "--model_dir", args.baseline_model)
        add_if(cmd, "--model_label", args.model_label)
        add_if(cmd, "--prompts_jsonl", args.prompts_jsonl)
        add_if(cmd, "--out_dir", out_dir)
        cmd.extend(["--max_subjects", str(args.max_subjects), "--layer", str(args.layer), "--load_mode", args.load_mode])
    elif task == "e7-el10":
        add_if(cmd, "--model_dir", args.model_dir or args.base_model)
        add_if(cmd, "--capsules_dir", args.capsules_dir)
        add_if(cmd, "--prompts_jsonl", args.prompts_jsonl)
        add_if(cmd, "--out_dir", out_dir)
        cmd.extend(["--max_subjects", str(args.max_subjects), "--el_steps", str(args.el_steps), "--max_keywords", str(args.max_keywords)])
    elif task == "e8-eval-no-capsule":
        add_if(cmd, "--model_dir", args.base_model)
        add_if(cmd, "--no_capsule_adapter_path", args.no_capsule_adapter)
        add_if(cmd, "--prompts_jsonl", args.prompts_jsonl)
        add_if(cmd, "--out_dir", out_dir)
        cmd.extend(["--max_subjects", str(args.max_subjects), "--load_mode", args.load_mode, "--seed", str(args.seed)])
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one or more ERUF rebuttal probe scripts.")
    ap.add_argument("tasks", nargs="+", choices=sorted(TASK_TO_SCRIPT.keys()) + ["all-probes"])
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--kif_adapter", default="")
    ap.add_argument("--baseline_model", default="")
    ap.add_argument("--baseline_prefer", default="optout")
    ap.add_argument("--model_label", default="Baseline")
    ap.add_argument("--no_capsule_adapter", default="")
    ap.add_argument("--capsules_dir", default="outputs/capsules")
    ap.add_argument("--prompts_jsonl", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--models", default="pre,kif,baseline")
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--max_eval_rows_per_model", type=int, default=120)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "8bit", "bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--el_steps", type=int, default=32)
    ap.add_argument("--max_keywords", type=int, default=10)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    tasks = list(args.tasks)
    if "all-probes" in tasks:
        tasks = ["e2-separability", "e2-erasure", "e3-locality", "e3-specificity", "e4-hidden", "e7-el10"]

    for task in tasks:
        cmd = build_cmd(task, args)
        print("\n$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
