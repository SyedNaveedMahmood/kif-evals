#!/usr/bin/env python3
"""Run one Qwen-3B baseline through the KIF framework with paper-style configs.

This is a local/WSL-friendly launcher. It keeps the method hyperparameters aligned
with the existing main Slurm scripts, but points the base model to Qwen 3B and
uses the Adele-retain augmented prompt file required by retain-aware baselines.

Run one method at a time, for example:

  python -u framework/scripts/run_qwen3b_baseline_local.py \
    --method lunar \
    --model_dir Qwen/Qwen2.5-3B-Instruct \
    --prompts_jsonl framework/outputs/qwen3b/prompts_with_adele_retain.jsonl \
    --output_root framework/outputs/qwen3b_baselines
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

# Import after sys.path setup.
import methods  # noqa: F401  triggers registry imports
import methods.lunar as lunar
import methods.reglu as reglu
import methods.simnpo as simnpo
import methods.optout as optout
from orchestrate import run_pipeline

QWEN_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
SUBJECTS = [
    "Ariana Grande",
    "Arijit Singh",
    "Beyoncé",
    "Drake (musician)",
    "Ed Sheeran",
    "Eminem",
    "Kanye West",
    "Katy Perry",
    "Michael Jackson",
    "Queen (band)",
    "Taylor Swift",
]


def install_qwen_templates(model_family: str) -> None:
    # LUNAR/ReGLU already have Qwen2.5-7B-Instruct; add explicit 3B aliases.
    for d in [lunar._CHAT_TEMPLATES, reglu._CHAT_TEMPLATES, simnpo.CHAT_TEMPLATES, optout.CHAT_TEMPLATES]:
        d[model_family] = QWEN_TEMPLATE
        d["Qwen2.5-3B-Instruct"] = QWEN_TEMPLATE
        d["Qwen2.5-7B-Instruct"] = QWEN_TEMPLATE
        d["qwen2.5-instruct"] = QWEN_TEMPLATE


def method_config(method: str, model_family: str) -> Dict[str, Any]:
    m = method.lower()
    if m == "lunar":
        return {
            "model_family": model_family,
            "auto_select_layer": True,
            "use_closed_form": True,
            "lambda_reg": 1e-3,
            "n_forget": 128,
            "n_retain": 128,
            "n_ref": 64,
            "max_per_subject": 50,
            "batch_size": 8,
            "seed": 42,
        }
    if m == "reglu":
        # Mirrors framework/scripts/reglu_main_run.slurm.
        return {
            "model_family": model_family,
            "variant": "ihl",
            "lora_targets": "all",
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "rila_beta": 0.5,
            "rila_samples_per_split": 128,
            "rila_cov_shrink": 0.0001,
            "rol_lambda": 0.5,
            "rol_rank": 128,
            "retain_gamma": 1.0,
            "n_forget": 132,
            "n_retain": 128,
            "max_per_subject": 12,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "num_epochs": 5,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "max_length": 256,
            "torch_dtype": "bfloat16",
            "save_merged_model": True,
            "seed": 17,
        }
    if m == "simnpo":
        # Mirrors framework/scripts/simnpo_main_run.slurm.
        return {
            "model_family": model_family,
            "beta": 2.5,
            "gamma": 0.0,
            "npo_coeff": 0.1375,
            "grad_diff_coeff": 1.0,
            "lr": 0.00001,
            "num_epochs": 10,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "weight_decay": 0.01,
            "max_seq_len": 500,
            "use_retain_loss": True,
            "optimizer_name": "paged_adamw_32bit",
            "warmup_steps": "steps_per_epoch",
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "use_gradient_checkpointing": True,
            "n_forget": 132,
            "n_retain": 132,
            "max_per_subject": 12,
            "trainable_modules": "all",
            "device_map_auto": True,
            "save_model": True,
            "seed": 42,
            "logging_steps": 1,
        }
    if m == "optout":
        # Mirrors framework/methods/optout.py defaults and OPT-OUT main setup.
        return {
            "method": "npo+rt+wd+ot",
            "model_family": model_family,
            "dpo_beta": 0.1,
            "reg_lambda": 0.1,
            "learning_rate": 0.00001,
            "warmup_ratio": 0.0,
            "weight_decay": 0.01,
            "num_epochs": 3,
            "max_grad_norm": 1.0,
            "max_length": 512,
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "use_gradient_checkpointing": True,
            "alternate_updates": True,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "n_forget": 132,
            "n_retain": 132,
            "n_world": 132,
            "max_per_subject": 12,
            "swd_n_projections": 100,
            "swd_p": 2,
            "seed": 42,
            "device_map_auto": False,
            "trainable_modules": "all",
            "swd_max_tensors": None,
            "swd_max_rows": None,
            "save_merged_model": True,
            "logging_steps": 1,
        }
    raise ValueError(f"Unsupported method={method!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["lunar", "reglu", "simnpo", "optout"])
    ap.add_argument("--model_dir", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    ap.add_argument("--model_family", default="Qwen2.5-3B-Instruct")
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--capsules_dir", default="framework/outputs/empty_capsules")
    ap.add_argument("--output_root", default="framework/outputs/qwen3b_baselines")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--print_config", action="store_true")
    args = ap.parse_args()

    install_qwen_templates(args.model_family)
    cfg = method_config(args.method, args.model_family)
    if args.print_config:
        print(json.dumps(cfg, indent=2), flush=True)

    out_dir = Path(args.output_root) / args.method
    eval_dir = out_dir / "eval"
    Path(args.capsules_dir).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"method={args.method}", flush=True)
    print(f"model_dir={args.model_dir}", flush=True)
    print(f"model_family={args.model_family}", flush=True)
    print(f"prompts_jsonl={args.prompts_jsonl}", flush=True)
    print(f"output_dir={out_dir}", flush=True)
    print(f"skip_eval={args.skip_eval}", flush=True)
    print("=" * 80, flush=True)

    summary = run_pipeline(
        method_name=args.method,
        model_dir=args.model_dir,
        prompts_jsonl=args.prompts_jsonl,
        capsules_dir=args.capsules_dir,
        output_dir=str(out_dir),
        subjects=SUBJECTS,
        config_overrides=cfg,
        skip_eval=bool(args.skip_eval),
        eval_out_dir=str(eval_dir),
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
