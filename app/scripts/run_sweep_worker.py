#!/usr/bin/env python3
"""Run one full KIF sweep worker: Module E -> Module 7 -> optional Module 8.

One worker is bound to one visible GPU by the Slurm script via CUDA_VISIBLE_DEVICES.
The worker writes only inside its own run_root, so parallel workers do not collide.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def read_adapter_path(module7_dir: Path) -> str:
    p = module7_dir / "last_adapter_path.json"
    if not p.exists():
        raise FileNotFoundError(f"Module 7 did not create {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    adapter_path = data.get("adapter_path")
    if not adapter_path:
        raise RuntimeError(f"No adapter_path inside {p}")
    return str(adapter_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one tau/k/seed sweep worker.")
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--z_tau", type=float, required=True)
    parser.add_argument("--soft_gate_k", type=float, required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--capsules_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--remap_json", default=None)
    parser.add_argument("--run_root", required=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--min_subject_pairs", type=int, default=800)
    parser.add_argument("--min_anchor_pairs", type=int, default=600)
    parser.add_argument("--variants_per_subject", type=int, default=60)
    parser.add_argument("--harvest_variants_per_subject", type=int, default=50)

    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--skip_module_e", action="store_true")
    parser.add_argument("--run_eval", action="store_true", help="Run Module 8 immediately after Module 7.")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    module_e_dir = run_root / "module_e"
    module7_dir = run_root / "module7"
    module8_dir = run_root / "module8"
    for d in [module_e_dir, module7_dir, module8_dir]:
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_meta = vars(args) | {
        "run_root": str(run_root),
        "module_e_dir": str(module_e_dir),
        "module7_dir": str(module7_dir),
        "module8_dir": str(module8_dir),
        "visible_cuda_devices": env.get("CUDA_VISIBLE_DEVICES"),
    }
    (run_root / "worker_run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    if not args.skip_module_e:
        cmd_e = [
            sys.executable, str(ROOT / "scripts" / "run_module_e_one.py"),
            "--model_dir", args.model_dir,
            "--capsules_dir", args.capsules_dir,
            "--dataset_dir", args.dataset_dir,
            "--out_dir", str(module_e_dir),
            "--seed", str(args.seed),
            "--z_tau", str(args.z_tau),
            "--soft_gate_k", str(args.soft_gate_k),
            "--harvest_variants_per_subject", str(args.harvest_variants_per_subject),
        ]
        if args.remap_json:
            cmd_e += ["--remap_json", args.remap_json]
        if args.no_4bit:
            cmd_e += ["--no_4bit"]
        run(cmd_e, env)

    interactions = module_e_dir / "interactions.jsonl"
    if not interactions.exists():
        raise FileNotFoundError(f"Missing interactions log: {interactions}")

    cmd7 = [
        sys.executable, "-m", "llama20.modules.module7",
        "--model_dir", args.model_dir,
        "--capsules_dir", args.capsules_dir,
        "--interactions_log", str(interactions),
        "--out_dir", str(module7_dir),
        "--adapter_name_prefix", f"{args.config_name}_seed{args.seed}",
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--min_subject_pairs", str(args.min_subject_pairs),
        "--min_anchor_pairs", str(args.min_anchor_pairs),
        "--variants_per_subject", str(args.variants_per_subject),
    ]
    if args.max_steps is not None:
        cmd7 += ["--max_steps", str(args.max_steps)]
    if args.no_4bit:
        cmd7 += ["--no_4bit"]
    run(cmd7, env)

    adapter_path = read_adapter_path(module7_dir)

    if args.run_eval:
        # Module 8 already supports env/path overrides through run_module8_clean.
        eval_code = (
            "from llama20.modules.module8 import run_module8_clean; "
            "run_module8_clean(model_dir=%r, adapter_path=%r, out_dir=%r, capsules_dir=%r, prompts_jsonl=%r)"
            % (args.model_dir, adapter_path, str(module8_dir), args.capsules_dir, str(Path(args.dataset_dir) / "prompts.jsonl"))
        )
        run([sys.executable, "-c", eval_code], env)

    result = {
        **run_meta,
        "interactions_log": str(interactions),
        "adapter_path": adapter_path,
        "module7_last_adapter_json": str(module7_dir / "last_adapter_path.json"),
        "eval_dir": str(module8_dir) if args.run_eval else None,
    }
    (run_root / "worker_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
