#!/usr/bin/env python3
"""One-command apple-to-apple experiment:

1. Convert KIF custom prompts into LUNAR format.
2. Run LUNAR forgetting on selected KIF subjects using meta-llama/Llama-3.1-8B.
3. Run patched KIF Module 8 dual-metric evaluation on the saved LUNAR model.

This script assumes the archive layout:
  ./LUNAR-main
  ./KIF-main
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def sanitize_edge(subject: str) -> str:
    x = re.sub(r"[^A-Za-z0-9]+", "_", subject.strip())
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "unknown_subject"


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--kif-prompts-jsonl", type=Path, required=True)
    ap.add_argument("--kif-capsules-dir", type=Path, required=True)
    ap.add_argument("--subjects", required=True, help="Comma-separated original KIF subject names, e.g. 'Taylor Swift,Ariana Grande'.")
    ap.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--data-name", default="kif_entities")
    ap.add_argument("--layer", default="22", help="Single layer or comma-separated layers, e.g. 22 or 20,22,24.")
    ap.add_argument("--coeff", default="2.0", help="Single coeff or comma-separated coeffs aligned with layers.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--use-harmful", action="store_true", help="Use harmful Dref instead of unverified Dref.")
    ap.add_argument("--run-kif-eval", action="store_true", help="Run KIF Module 8 after LUNAR saving.")
    ap.add_argument("--max-per-subject", type=int, default=0)
    ap.add_argument("--drop-control", action="store_true")
    ap.add_argument("--drop-misleading", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    lunar_dir = root / "LUNAR-main"
    kif_dir = root / "KIF-main"
    if not lunar_dir.exists():
        raise FileNotFoundError(f"Missing LUNAR dir: {lunar_dir}")
    if not kif_dir.exists():
        raise FileNotFoundError(f"Missing KIF dir: {kif_dir}")

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not subjects:
        raise ValueError("No subjects supplied.")
    forget_edges = [sanitize_edge(s) for s in subjects]

    dataset_path = lunar_dir / "dataset" / "unlearning" / f"{args.data_name}.json"
    mapping_path = lunar_dir / "run_results" / "kif_entity_mapping.json"

    convert_cmd = [
        sys.executable,
        "scripts/convert_kif_prompts_to_lunar.py",
        "--kif-prompts-jsonl", str(args.kif_prompts_jsonl.resolve()),
        "--out-json", str(dataset_path),
        "--mapping-out", str(mapping_path),
    ]
    if args.max_per_subject > 0:
        convert_cmd += ["--max-per-subject", str(args.max_per_subject)]
    if args.drop_control:
        convert_cmd.append("--drop-control")
    if args.drop_misleading:
        convert_cmd.append("--drop-misleading")
    run(convert_cmd, cwd=lunar_dir)

    # Validate requested subjects are present in the converted mapping.
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    missing = [s for s in subjects if s not in mapping["subject_to_edge"]]
    if missing:
        raise ValueError(f"Requested subjects are absent from converted KIF dataset: {missing}")
    forget_edges = [mapping["subject_to_edge"][s] for s in subjects]

    layers = [int(x.strip()) for x in args.layer.split(",") if x.strip()]
    coeffs = [float(x.strip()) for x in args.coeff.split(",") if x.strip()]
    if len(coeffs) == 1 and len(layers) > 1:
        coeffs = coeffs * len(layers)
    if len(layers) != len(coeffs):
        raise ValueError(f"layers and coeffs must align, got {layers} and {coeffs}")

    edge_slug = "__".join(forget_edges)
    save_model_path = lunar_dir / "models_lunar" / args.data_name / "llama31-8b-base" / edge_slug
    save_path = lunar_dir / "run_results" / "completions" / "llama31-8b-base" / "lunar_kif_dualmetric" / edge_slug

    hydra_list_edges = "[" + ",".join(forget_edges) + "]"
    hydra_list_layers = "[" + ",".join(str(x) for x in layers) + "]"
    hydra_list_coeffs = "[" + ",".join((f"+{x}" if x >= 0 else str(x)) for x in coeffs) + "]"

    lunar_cmd = [
        sys.executable,
        "run_lunar.py",
        "model_family=llama31-8b-base",
        f"base_model_path={args.model_id}",
        f"model_path={args.model_id}",
        f"data_name={args.data_name}",
        f"forget_edge={hydra_list_edges}",
        f"layer_modified={hydra_list_layers}",
        f"coeff_list={hydra_list_coeffs}",
        f"num_epochs={args.epochs}",
        f"lr={args.lr}",
        f"use_harmful={'true' if args.use_harmful else 'false'}",
        f"use_unverified={'false' if args.use_harmful else 'true'}",
        "save_unlearned_model=true",
        f"save_unlearned_model_path={save_model_path}",
        "skip_native_eval=true",
        "if_eval_factual=false",
        f"save_path={save_path}",
    ]
    run(lunar_cmd, cwd=lunar_dir)

    if args.run_kif_eval:
        out_dir = root / "outputs" / "eval_lunar_dualmetric" / edge_slug
        env = os.environ.copy()
        env.update({
            "MODEL_DIR": args.model_id,
            "MERGED_MODEL_DIR": str(save_model_path),
            "CAPSULES_DIR": str(args.kif_capsules_dir.resolve()),
            "PROMPTS_JSONL": str(args.kif_prompts_jsonl.resolve()),
            "OUT_DIR": str(out_dir),
            "EVAL_SUBJECTS_JSON": json.dumps(subjects),
            "MAX_SUBJECTS": str(len(subjects)),
            "USE_4BIT": env.get("USE_4BIT", "true"),
        })
        run([sys.executable, "-m", "src.llama20.modules.module8_lunar_apples"], cwd=kif_dir, env=env)
        print(f"\nKIF dual-metric eval written to: {out_dir}")

    print("\nDONE")
    print(json.dumps({
        "subjects": subjects,
        "forget_edges": forget_edges,
        "lunar_dataset": str(dataset_path),
        "lunar_model": str(save_model_path),
        "mapping": str(mapping_path),
    }, indent=2))


if __name__ == "__main__":
    main()
