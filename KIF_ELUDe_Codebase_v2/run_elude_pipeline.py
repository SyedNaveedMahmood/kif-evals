#!/usr/bin/env python3
"""Run KIF on ELUDe as an external entity-level benchmark.

Default flow:
  Module 0  -> load/save Llama-3.1-8B-Instruct into outputs/model
  ELUDe     -> prepare prompts/splits
  Module B  -> activation probing
  Module C  -> signature mining
  Module D  -> capsule forging
  Module E  -> ELUDe-prompt sentinel harvesting
  Module 7  -> representation-aware UPU LoRA distillation
  Module 8E -> ELUDe QA benchmark evaluation

For iterative runs, use --skip-module0 if outputs/model already exists.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llama20.modules import module0, module_b, module_c, module_d, module_e, module7, module_elude, module8e


def _parse_targets(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()] or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the KIF -> ELUDe external benchmark pipeline.")
    parser.add_argument("--targets", type=str, default="", help="Comma-separated ELUDe target names. If omitted, first N alphabetically are used.")
    parser.add_argument("--max-targets", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-module0", action="store_true", help="Skip model download/save if outputs/model already exists.")
    parser.add_argument("--skip-b", action="store_true")
    parser.add_argument("--skip-c", action="store_true")
    parser.add_argument("--skip-d", action="store_true")
    parser.add_argument("--skip-e", action="store_true")
    parser.add_argument("--skip-7", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Only run Module 8E. Provide --adapter-path or KIF_ADAPTER_PATH.")
    parser.add_argument("--adapter-path", type=str, default="", help="Existing LoRA adapter path for eval-only or re-eval.")
    args = parser.parse_args()

    os.environ["ELUDE_SEED"] = str(args.seed)
    os.environ["ELUDE_TRAIN_RATIO"] = str(args.train_ratio)
    if args.targets:
        os.environ["ELUDE_TARGETS"] = args.targets
    else:
        os.environ["ELUDE_MAX_TARGETS"] = str(args.max_targets)
    if args.adapter_path:
        os.environ["KIF_ADAPTER_PATH"] = args.adapter_path

    if args.eval_only:
        print("==> Module 8E: ELUDe external evaluation")
        module8e.run_module8_elude(adapter_path=args.adapter_path or None)
        return 0

    if not args.skip_module0:
        print("==> Module 0: Llama-3.1-8B-Instruct setup")
        model, tok = module0.run_module0()
        if model is None or tok is None:
            print("Module 0 failed; aborting.")
            return 1
    else:
        print("==> Skipping Module 0")

    print("==> Module ELUDe: prepare external benchmark data")
    module_elude.run_module_elude(
        targets=_parse_targets(args.targets),
        max_targets=args.max_targets,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    if not args.skip_b:
        print("==> Module B: activation probing")
        module_b.run_module_b()
    if not args.skip_c:
        print("==> Module C: signature mining")
        module_c.run_module_c()
    if not args.skip_d:
        print("==> Module D: capsule forging")
        module_d.run_module_d()
    if not args.skip_e:
        print("==> Module E: ELUDe-prompt sentinel harvesting")
        module_e.run_module_e_final()

    adapter_path = args.adapter_path or None
    if not args.skip_7:
        print("==> Module 7: ELUDe-compatible UPU LoRA distillation")
        adapter_path = module7.run_module7_repaware()
        if not adapter_path:
            print("Module 7 did not return an adapter path; aborting before eval.")
            return 2

    print("==> Module 8E: ELUDe external evaluation")
    module8e.run_module8_elude(adapter_path=adapter_path)
    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
