#!/usr/bin/env python3
"""Run one Qwen-3B baseline through the KIF framework with paper-style configs.

This is a local/WSL-friendly launcher. It keeps the method hyperparameters aligned
with the existing main Slurm scripts, but points the base model to Qwen 3B and
uses the Adele-retain augmented prompt file required by retain-aware baselines.

For ReGLU, the launcher optionally patches only the RILA eigensolver backend:
the paper-style hyperparameters remain unchanged, but the large eigensolves are
moved off CUDA so a 16 GB consumer GPU does not OOM during RILA initialization.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

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


def log(msg: str) -> None:
    print(f"[QWEN3B-LOCAL] {msg}", flush=True)


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


def _top_eigenvectors_cpu(mat: torch.Tensor, k: int, label: str, full_eigh_dim: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top-k eigenpairs of a symmetric matrix on CPU.

    This preserves ReGLU's hyperparameters but avoids CUDA OOM during RILA for
    very wide Qwen MLP matrices. For small matrices we use full eigh. For large
    matrices we use LOBPCG to avoid materializing the full eigenvector matrix.
    """
    mat = mat.detach().float().cpu().contiguous()
    n = int(mat.shape[0])
    k = min(int(k), max(1, n - 1))
    t0 = time.time()
    log(f"ReGLU CPU eigensolve {label}: n={n}, k={k}")
    if n <= full_eigh_dim:
        vals, vecs = torch.linalg.eigh(mat)
        vals, vecs = vals[-k:].contiguous(), vecs[:, -k:].contiguous()
    else:
        try:
            x0 = torch.randn(n, k, dtype=mat.dtype)
            vals, vecs = torch.lobpcg(mat, k=k, X=x0, largest=True, method="ortho", niter=80, tol=1e-4)
            order = torch.argsort(vals)
            vals, vecs = vals[order].contiguous(), vecs[:, order].contiguous()
        except Exception as exc:
            log(f"LOBPCG failed for {label}: {exc}; falling back to full CPU eigh. This may be slow.")
            vals, vecs = torch.linalg.eigh(mat)
            vals, vecs = vals[-k:].contiguous(), vecs[:, -k:].contiguous()
    log(f"ReGLU CPU eigensolve done {label}: {time.time() - t0:.1f}s")
    return vals, vecs


def install_reglu_cpu_rila_patch() -> None:
    """Patch only the ReGLU RILA eigensolver backend for local 16 GB GPUs."""

    def _apply_rila_initialization_cpu(
        model,
        tokenizer,
        forget_rows: List[Dict[str, str]],
        retain_rows: List[Dict[str, str]],
        cfg,
        target_modules: Dict[str, torch.nn.Module],
        device: torch.device,
        output_dir: Path,
    ) -> Dict[str, torch.Tensor]:
        n = int(cfg.rila_samples_per_split)
        forget_sample = reglu._repeat_to_len(forget_rows, n)
        retain_sample = reglu._repeat_to_len(retain_rows, n)
        h_forget = reglu._collect_representations_for_modules(
            model, tokenizer, forget_sample, cfg.model_family, cfg.max_length, cfg.batch_size, target_modules, device, "forget/RILA"
        )
        h_retain = reglu._collect_representations_for_modules(
            model, tokenizer, retain_sample, cfg.model_family, cfg.max_length, cfg.batch_size, target_modules, device, "retain/RILA"
        )

        rol_bases: Dict[str, torch.Tensor] = {}
        cache_layers: Dict[str, Dict[str, torch.Tensor]] = {}
        rank = int(cfg.lora_r)
        beta = float(cfg.rila_beta)
        eps = float(cfg.rila_cov_shrink)

        for mi, (name, module) in enumerate(target_modules.items(), 1):
            if name not in h_forget or name not in h_retain:
                log(f"ReGLU RILA missing activations for {name}; skipping")
                continue
            t_mod = time.time()
            log(f"ReGLU RILA module {mi}/{len(target_modules)}: {name}")
            hf = h_forget[name].float().cpu()
            hr = h_retain[name].float().cpu()
            nf, d = hf.shape
            nr, _ = hr.shape

            cf = (hf.T @ hf) / max(1, nf)
            cr = (hr.T @ hr) / max(1, nr)
            eye = torch.eye(d, dtype=torch.float32)
            cf = cf + eps * eye
            cr = cr + eps * eye
            delta = (1.0 - beta) * cf - beta * cr

            top_evals, q_delta = _top_eigenvectors_cpu(delta, rank, f"delta::{name}")
            k_basis = min(int(cfg.rol_rank), d)
            _, q_retain = _top_eigenvectors_cpu(cr, k_basis, f"retain::{name}")

            w0 = module.base_layer.weight.detach().float().cpu()
            if q_delta.shape[0] != w0.shape[0]:
                log(f"ReGLU RILA shape mismatch for {name}: Q={tuple(q_delta.shape)} W={tuple(w0.shape)}; skipping")
                continue

            a_init = q_delta.T @ w0
            b_init = q_delta
            scaling = float(getattr(module, "scaling", {}).get("default", float(cfg.lora_alpha) / float(cfg.lora_r)))
            w_res = w0 - scaling * (b_init @ a_init)
            dtype = module.base_layer.weight.dtype
            mod_device = module.base_layer.weight.device
            with torch.no_grad():
                module.base_layer.weight.copy_(w_res.to(dtype=dtype, device=mod_device))
                module.lora_A["default"].weight.copy_(a_init.to(dtype=dtype, device=mod_device))
                module.lora_B["default"].weight.copy_(b_init.to(dtype=dtype, device=mod_device))

            rol_bases[name] = q_retain.float().cpu()
            cache_layers[name] = {
                "A": a_init.float().cpu(),
                "B": b_init.float().cpu(),
                "Qr_retain": q_retain.float().cpu(),
                "top_eigenvalues": top_evals.float().cpu(),
                "eigensolver_backend": "cpu_lobpcg_or_eigh",
            }
            log(f"ReGLU RILA initialized {name}: B={tuple(b_init.shape)} A={tuple(a_init.shape)} in {time.time() - t_mod:.1f}s")

            del hf, hr, cf, cr, eye, delta, q_delta, q_retain, w0, a_init, b_init, w_res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cache_path = output_dir / "reglu_rila_cache.pt"
        torch.save({"config": reglu.asdict(cfg), "layers": cache_layers, "rila_backend": "cpu_lobpcg_or_eigh"}, cache_path)
        log(f"ReGLU RILA cache saved -> {cache_path}")
        return rol_bases

    reglu._apply_rila_initialization = _apply_rila_initialization_cpu
    log("Installed ReGLU CPU/LOBPCG RILA patch for local Qwen-3B run")


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
    ap.add_argument("--reglu_cpu_rila", action="store_true", help="Offload ReGLU RILA eigensolves to CPU/LOBPCG to avoid CUDA OOM on 16 GB GPUs.")
    args = ap.parse_args()

    install_qwen_templates(args.model_family)
    if args.method == "reglu" and args.reglu_cpu_rila:
        install_reglu_cpu_rila_patch()

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
    print(f"reglu_cpu_rila={args.reglu_cpu_rila}", flush=True)
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
