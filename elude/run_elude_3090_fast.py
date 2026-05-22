#!/usr/bin/env python3
"""Optimized 3090 runner for KIF on ELUDe.

This file keeps the original ELUDe pipeline intact and adds a separate fast path
for local 24GB-GPU experimentation.  It does not change the core algorithms;
instead it narrows expensive search/plotting/evaluation knobs through explicit
environment variables and direct config construction.

Recommended smoke command:
  python -u elude/run_elude_3090_fast.py --max-targets 1 --seed 17

Recommended larger local command:
  python -u elude/run_elude_3090_fast.py --skip-module0 --max-targets 5 --seed 17
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llama20.modules import module0, module_b, module_c, module_d, module_e, module7, module_elude, module8e


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if raw == "":
        return default
    return raw not in {"0", "false", "False", "no", "NO"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _layers_env(name: str, default: List[int]) -> List[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out)) or list(default)


def _parse_targets(raw: str) -> Optional[List[str]]:
    xs = [x.strip() for x in raw.split(",") if x.strip()]
    return xs or None


def _set_default_env() -> None:
    # Model/runtime defaults for a single RTX 3090 24GB.
    os.environ.setdefault("KIF_USE_4BIT", "1")
    os.environ.setdefault("KIF_HF_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Module B fast defaults. Original pipeline remains available separately.
    os.environ.setdefault("KIF_PROBE_LAYERS", "11,12,13,14")
    os.environ.setdefault("KIF_PROBE_BATCH_SIZE", "16")
    os.environ.setdefault("KIF_PROBE_MAX_LENGTH", "128")
    os.environ.setdefault("KIF_PROBE_CAPTURE_SCOPE", "last_token")
    os.environ.setdefault("KIF_PROBE_COMPRESSION_LEVEL", "1")
    os.environ.setdefault("KIF_PROBE_SKIP_EXISTING", "1")

    # Module 8E local eval safety. Set to 0 for full eval.
    os.environ.setdefault("ELUDE_EVAL_MAX_NEW_TOKENS", "96")


def _maybe_write_capped_jsonl(src: str, dst: str, max_rows_per_subject: int) -> str:
    if max_rows_per_subject <= 0:
        return src
    src_path = Path(src)
    if not src_path.exists():
        return src
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = defaultdict(int)
    kept = 0
    with src_path.open("r", encoding="utf-8") as fin, dst_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            subj = str(rec.get("subject") or rec.get("entity") or "__unknown__")
            if counts[subj] >= max_rows_per_subject:
                continue
            counts[subj] += 1
            kept += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[fast] capped {src} -> {dst} rows={kept} max_rows_per_subject={max_rows_per_subject}")
    return str(dst_path)


def run_module_c_fast(disable_plots: bool = True):
    """Run Module C using explicit 3090-friendly config."""
    if disable_plots:
        module_c.CausalTracer.generate_pca_visualization = lambda self, pos_features, neg_features, subject, layer: None
        module_c.CausalTracer.plot_signature_distributions = lambda self, subject, subject_results: None
        print("[fast] Module C plots disabled for speed")

    layers = _layers_env("KIF_C_LAYERS", _layers_env("KIF_PROBE_LAYERS", [11, 12, 13, 14]))
    cfg = module_c.SignatureMiningConfig(
        rome_hparams=module_c.ROMEHyperParams(
            layers=layers,
            layer_selection="all",
            target_module="mlp",
            significance_threshold=_float_env("KIF_C_SIGNIFICANCE_THRESHOLD", 1.5),
        ),
        top_k_directions=_int_env("KIF_C_TOP_K_DIRECTIONS", 1),
        min_prompts_per_subject=_int_env("KIF_C_MIN_PROMPTS_PER_SUBJECT", 2),
        use_semantic_negatives=_bool_env("KIF_C_USE_SEMANTIC_NEGATIVES", True),
        min_controls_per_subject=_int_env("KIF_C_MIN_CONTROLS_PER_SUBJECT", 1),
        allow_synthetic_fallback=_bool_env("KIF_C_ALLOW_SYNTHETIC_FALLBACK", True),
        enable_oversampling=_bool_env("KIF_C_ENABLE_OVERSAMPLING", True),
        oversample_strategy=os.getenv("KIF_C_OVERSAMPLE_STRATEGY", "median"),
        oversample_separately=True,
        preserve_original_ratio=False,
        activation_strategy=os.getenv("KIF_C_ACTIVATION_STRATEGY", "mean_token"),
        standardize_dims=True,
        device="cuda" if module_c.torch.cuda.is_available() else "cpu",
        use_half_precision=_bool_env("KIF_C_USE_HALF_PRECISION", False),
        enable_memory_cleanup=True,
        cleanup_frequency=_int_env("KIF_C_CLEANUP_FREQUENCY", 2),
        n_bootstrap_samples=_int_env("KIF_C_BOOTSTRAP_SAMPLES", 20),
        random_state=_int_env("KIF_SEED", 17),
    )
    extractor = module_c.SignatureExtractor(cfg)
    signatures = extractor.extract_all_signatures()
    extractor.save_signature_index(signatures)
    extractor.create_summary_report()
    return signatures


def run_module_e_fast():
    cfg = module_e.EConfig()
    cfg.seed = _int_env("KIF_SEED", 17)
    cfg.use_4bit = _bool_env("KIF_USE_4BIT", True)
    cfg.gen_max_new_tokens = _int_env("KIF_E_GEN_MAX_NEW_TOKENS", 64)
    cfg.refusal_calibration_prompts = _int_env("KIF_E_REFUSAL_CALIBRATION_PROMPTS", 8)
    cfg.harvest_variants_per_subject = _int_env("KIF_E_HARVEST_VARIANTS_PER_SUBJECT", 20)

    cap = _int_env("KIF_E_MAX_ELUDE_HARVEST_ROWS_PER_SUBJECT", 0)
    if cap > 0:
        capped = _maybe_write_capped_jsonl(
            cfg.elude_upu_file,
            "outputs/elude/elude_upu_train.fast.jsonl",
            cap,
        )
        cfg.elude_upu_file = capped
        os.environ["ELUDE_UPU_FILE"] = capped

    sent = module_e.Sentinel(cfg)
    probes: List[str] = []
    for rec in module_e._read_elude_upu_prompts(cfg.elude_upu_file):
        if str(rec.get("subject")) in sent.capsules:
            probes.append(str(rec.get("prompt")))
    for s in sent.capsules.keys():
        probes += [
            f"Tell me about {s}.",
            f"Give two facts about {s}.",
            f"Write two sentences describing {s}.",
            f"Provide a concise overview of {s}.",
        ]
    if probes:
        sent.calibrate_z(list(dict.fromkeys(probes))[: max(8, cfg.refusal_calibration_prompts * max(1, len(sent.capsules)))])
    sent.harvest_interactions(variants_per_subject=cfg.harvest_variants_per_subject)
    return sent


def run_module7_fast():
    cfg = module7.M7QwenConfig()
    cfg.seed = _int_env("KIF_SEED", 17)
    cfg.use_4bit = _bool_env("KIF_USE_4BIT", True)
    cfg.min_subject_pairs = _int_env("KIF_M7_MIN_SUBJECT_PAIRS", 240)
    cfg.min_anchor_pairs = _int_env("KIF_M7_MIN_ANCHOR_PAIRS", 240)
    cfg.variants_per_subject = _int_env("KIF_M7_VARIANTS_PER_SUBJECT", 24)
    cfg.gen_max_new_tokens = _int_env("KIF_M7_GEN_MAX_NEW_TOKENS", 64)
    cfg.max_seq_len = _int_env("KIF_M7_MAX_SEQ_LEN", 256)
    cfg.epochs = _int_env("KIF_M7_EPOCHS", 3)
    cfg.batch_size = _int_env("KIF_M7_BATCH_SIZE", 1)
    cfg.grad_accum = _int_env("KIF_M7_GRAD_ACCUM", 8)
    cfg.lr = _float_env("KIF_M7_LR", 5e-6)
    max_steps = _int_env("KIF_M7_MAX_STEPS", 0)
    cfg.max_steps = max_steps if max_steps > 0 else None
    cfg.sig_suppression_strength = _float_env("KIF_M7_SIG_STRENGTH", 1.0)

    trainer = module7.RepAwareUPUTrainer(cfg)
    return trainer.train()


def main() -> int:
    _set_default_env()
    parser = argparse.ArgumentParser(description="Optimized 3090 runner for KIF -> ELUDe.")
    parser.add_argument("--targets", type=str, default="", help="Comma-separated ELUDe targets.")
    parser.add_argument("--max-targets", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-module0", action="store_true")
    parser.add_argument("--skip-b", action="store_true")
    parser.add_argument("--skip-c", action="store_true")
    parser.add_argument("--skip-d", action="store_true")
    parser.add_argument("--skip-e", action="store_true")
    parser.add_argument("--skip-7", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--full-c-plots", action="store_true", help="Keep Module C PCA/distribution plots.")
    args = parser.parse_args()

    os.environ["KIF_SEED"] = str(args.seed)
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
        print("==> Module 0: model setup")
        model, tok = module0.run_module0()
        if model is None or tok is None:
            print("Module 0 failed; aborting.")
            return 1
    else:
        print("==> Skipping Module 0")

    print("==> Module ELUDe: prepare data")
    module_elude.run_module_elude(
        targets=_parse_targets(args.targets),
        max_targets=args.max_targets,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    if not args.skip_b:
        print("==> Module B: activation probing, 3090 fast config")
        module_b.run_module_b()
    if not args.skip_c:
        print("==> Module C: signature mining, 3090 fast config")
        run_module_c_fast(disable_plots=not args.full_c_plots)
    if not args.skip_d:
        print("==> Module D: capsule forging")
        module_d.run_module_d()
    if not args.skip_e:
        print("==> Module E: sentinel harvesting, 3090 fast config")
        run_module_e_fast()

    adapter_path = args.adapter_path or None
    if not args.skip_7:
        print("==> Module 7: UPU LoRA distillation, 3090 fast config")
        adapter_path = run_module7_fast()
        if not adapter_path:
            print("Module 7 did not return an adapter path; aborting before eval.")
            return 2

    print("==> Module 8E: ELUDe external evaluation")
    module8e.run_module8_elude(adapter_path=adapter_path)
    print("Optimized 3090 ELUDe pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
