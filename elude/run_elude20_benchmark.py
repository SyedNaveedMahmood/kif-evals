#!/usr/bin/env python3
"""Run the publication-style KIF-on-ELUDe benchmark.

Protocol:
  - Model: meta-llama/Meta-Llama-3.1-8B-Instruct saved by Module 0 to outputs/model
  - Targets: all 20 public ELUDe target entities, discovered from forget_qa
  - Setting: 20 independent single-target runs, one target per adapter
  - Forget set: full ELUDe forget_qa/train for that target, used for KIF unlearning
    and forget evaluation, matching the public Opt-Out data loader
  - Retain set: ELUDe retain_qa/train for retention anchors, retain_qa/test for RQ
  - World set: Alpaca-GPT4 train anchors and test evaluation when data files exist
  - Metrics: ELUDe FQ/RQ-style QA metrics, independent of KIF SMR/EL10

This script intentionally runs each target as an isolated subprocess and archives
its outputs before cleaning stage artifacts, preventing accidental multi-target
adapter training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

ROOT = Path(__file__).resolve().parent


def _safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch in {" ", "/", "\\", ":"}:
            out.append("_")
    return "".join(out).strip("_") or "target"


def _load_targets() -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("6rightjade/ELUDe", "forget_qa", split="train")
    return sorted({str(x).strip() for x in ds["entity"] if str(x).strip()})


def _stage_paths() -> List[Path]:
    return [
        ROOT / "outputs" / "datasets",
        ROOT / "outputs" / "elude",
        ROOT / "outputs" / "activations",
        ROOT / "outputs" / "capsules",
        ROOT / "outputs" / "sentinel",
        ROOT / "outputs" / "global_adapters",
        ROOT / "outputs" / "eval_elude",
    ]


def _clean_stage_outputs() -> None:
    for p in _stage_paths():
        if p.exists():
            shutil.rmtree(p)


def _archive_run(target: str, seed: int, out_root: Path) -> Path:
    run_dir = out_root / f"{_safe_name(target)}__seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for p in _stage_paths():
        if p.exists():
            dst = run_dir / p.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(p, dst)
    (run_dir / "target.txt").write_text(target + "\n", encoding="utf-8")
    return run_dir


def _read_summary(run_dir: Path) -> Dict[str, object]:
    p = run_dir / "eval_elude" / "elude_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _aggregate(out_root: Path, seed: int) -> None:
    rows = []
    for run_dir in sorted(out_root.glob(f"*__seed{seed}")):
        summary = _read_summary(run_dir)
        if not summary:
            continue
        target = (run_dir / "target.txt").read_text(encoding="utf-8").strip()
        row = {"target": target, "seed": seed, "run_dir": str(run_dir)}
        for k, v in summary.items():
            if isinstance(v, (int, float, str)):
                row[k] = v
        rows.append(row)

    if not rows:
        return

    csv_path = out_root / "elude20_per_target_metrics.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    numeric_keys = sorted(k for k in fieldnames if k not in {"target", "run_dir", "adapter_path", "model_dir", "elude_dir", "world_eval_file", "aggregate_note"})
    agg = {"n_runs": len(rows), "seed": seed, "protocol": "20 independent single-target KIF-on-ELUDe runs"}
    for k in numeric_keys:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except Exception:
                pass
        if vals:
            agg[k + "/mean"] = mean(vals)
            agg[k + "/std_population"] = pstdev(vals) if len(vals) > 1 else 0.0
    (out_root / "elude20_aggregate_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all 20 ELUDe targets as single-target KIF runs.")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-root", type=str, default="outputs/elude20_benchmark")
    ap.add_argument("--skip-module0", action="store_true", help="Assume outputs/model already exists.")
    ap.add_argument("--targets-file", type=str, default="", help="Optional newline-separated target list. Default: all 20 from HF.")
    ap.add_argument("--resume", action="store_true", help="Skip target if archived elude_summary.json already exists.")
    ap.add_argument("--extra-args", type=str, default="", help="Extra args passed verbatim to run_elude_pipeline.py, e.g. '--skip-b'.")
    args = ap.parse_args()

    out_root = ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.targets_file:
        targets = [x.strip() for x in Path(args.targets_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        targets = _load_targets()
    if len(targets) != 20:
        print(f"WARNING: expected 20 ELUDe targets, found {len(targets)}", file=sys.stderr)
    (out_root / "elude_targets_used.txt").write_text("\n".join(targets) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("ELUDE_EXACT_PROTOCOL", "1")
    env.setdefault("ELUDE_RETAIN_EVAL_SPLIT", "test")
    env.setdefault("ELUDE_USE_RETAIN_ANCHORS", "1")
    env.setdefault("ELUDE_USE_WORLD_ANCHORS", "1")
    env.setdefault("ELUDE_WORLD_TRAIN_FILE", "data/alpaca_gpt4_data_train.json")
    env.setdefault("ELUDE_WORLD_EVAL_FILE", "data/alpaca_gpt4_data_test.json")

    if not args.skip_module0:
        print("==> Running Module 0 once")
        subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0,'src'); from llama20.modules import module0; module0.run_module0()"], cwd=ROOT, env=env, check=True)

    extra = args.extra_args.split() if args.extra_args.strip() else []
    for idx, target in enumerate(targets, start=1):
        run_dir = out_root / f"{_safe_name(target)}__seed{args.seed}"
        if args.resume and (run_dir / "eval_elude" / "elude_summary.json").exists():
            print(f"[{idx}/{len(targets)}] Skipping completed: {target}")
            continue
        print(f"\n[{idx}/{len(targets)}] Running target: {target}")
        _clean_stage_outputs()
        cmd = [sys.executable, "run_elude_pipeline.py", "--targets", target, "--seed", str(args.seed), "--skip-module0"] + extra
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        archived = _archive_run(target, args.seed, out_root)
        print(f"Archived: {archived}")
        _clean_stage_outputs()
        _aggregate(out_root, args.seed)

    _aggregate(out_root, args.seed)
    print(f"\nDone. Results: {out_root}")
    print(f"Aggregate: {out_root / 'elude20_aggregate_summary.json'}")
    print(f"Per-target CSV: {out_root / 'elude20_per_target_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
