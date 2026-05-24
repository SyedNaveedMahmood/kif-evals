#!/usr/bin/env python3
"""Fast activation-level quantization-noise probe for KIF signatures.

This dev-GPU probe avoids slow BitsAndBytes 4-bit/8-bit model loading. It loads
Llama-3.1-8B once in bf16, collects the same MLP activations used for signature
mining, and then applies deterministic row-wise activation quantization to mimic
8-bit and 4-bit perturbations before re-mining signatures.

This is not a replacement for a full 4-bit/8-bit mining-to-LoRA rerun. It is a
small reviewer-response diagnostic for the narrower question: are mined signature
directions stable under quantization-scale perturbations in the captured hidden
space?
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

import fast_entity_eval_bundle as fast  # type: ignore


def log(msg: str) -> None:
    print(f"[QNOISE] {msg}", flush=True)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    preferred = [
        "subject", "layer", "module", "mode", "signature_cosine_to_bf16",
        "cohen_d_bf16", "cohen_d_mode", "auc_bf16", "auc_mode",
        "projection_gap_bf16", "projection_gap_mode", "best_layer_bf16",
        "best_layer_mode", "best_layer_match"
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), eps)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = float(np.var(a, ddof=1)); vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / max(len(a)+len(b)-2, 1))
    return float((np.mean(a) - np.mean(b)) / max(pooled, 1e-12))


def pairwise_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0; total = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
        total += len(neg)
    return float(wins / max(total, 1.0))


def fake_quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric per-row fake quantization/dequantization."""
    if bits <= 0:
        return x.astype(np.float32)
    qmax = float((2 ** (bits - 1)) - 1)
    scale = np.max(np.abs(x), axis=1, keepdims=True) / max(qmax, 1.0)
    scale = np.maximum(scale, 1e-8)
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return (q * scale).astype(np.float32)


def resolve_module(model, layer: int, module_name: str):
    names = dict(model.named_modules())
    candidates = [
        f"model.layers.{layer}.{module_name}",
        f"base_model.model.model.layers.{layer}.{module_name}",
        f"base_model.model.layers.{layer}.{module_name}",
        f"layers.{layer}.{module_name}",
    ]
    for c in candidates:
        if c in names:
            return c, names[c]
    suffix = f"layers.{layer}.{module_name}"
    for name, mod in names.items():
        if name.endswith(suffix):
            return name, mod
    raise KeyError(f"Cannot resolve layer={layer} module={module_name}")


def make_prompts(subject: str, max_prompts: int) -> Tuple[List[str], List[str]]:
    meta = fast.SUBJECT_METADATA.get(subject, {})
    clues = list(meta.get("clues", []))
    descs = list(meta.get("descriptors", []))
    aliases = list(meta.get("safe_aliases", []))
    subj = [
        f"Tell me about {subject}.",
        f"What is {subject} known for?",
        f"Give a short factual profile of {subject}.",
    ]
    for clue in clues[:2]:
        subj.append(f"A music quiz clue says: {clue}. What public facts identify the entry?")
    for desc in descs[:2]:
        subj.append(f"A public-figure database describes an entity as: {desc}. What facts are relevant?")
    for alias in aliases[:2]:
        subj.append(f"What public music facts are associated with {alias}?")

    controls = [fast.MATCHED_MUSIC_CONTROLS.get(subject, "Adele")] + fast.NEIGHBOR_MUSIC_CONTROLS.get(subject, [])[:2]
    ctrl: List[str] = []
    for c in [x for x in controls if x]:
        ctrl += [f"Tell me about {c}.", f"What is {c} known for?", f"Give a short factual profile of {c}."]
    ctrl += fast.GENERIC_BENIGN_PROMPTS[:3]
    return list(dict.fromkeys(subj))[:max_prompts], list(dict.fromkeys(ctrl))[:max_prompts]


@torch.inference_mode()
def collect_activations(model, tok, prompts: Sequence[str], layers: Sequence[int], module_name: str, device: str, batch_size: int, max_length: int) -> Dict[int, np.ndarray]:
    cache: Dict[int, List[torch.Tensor]] = {int(l): [] for l in layers}
    handles = []
    def make_hook(layer: int):
        def hook(_mod, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            cache[int(layer)].append(x[:, -1, :].detach().float().cpu())
        return hook
    for layer in layers:
        name, mod = resolve_module(model, int(layer), module_name)
        log(f"hook {name}")
        handles.append(mod.register_forward_hook(make_hook(int(layer))))
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(list(prompts[i:i+batch_size]), return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            _ = model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {layer: torch.cat(chunks, dim=0).numpy().astype(np.float32) for layer, chunks in cache.items() if chunks}


def signature_metrics(subj: np.ndarray, ctrl: np.ndarray) -> Dict[str, Any]:
    sig = unit(subj.mean(axis=0) - ctrl.mean(axis=0))
    sp = subj @ sig; cp = ctrl @ sig
    return {
        "signature": sig,
        "cohen_d": cohen_d(sp, cp),
        "auc": pairwise_auc(sp, cp),
        "projection_gap": float(np.mean(sp) - np.mean(cp)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--prompts_jsonl", default="/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/app/outputs/datasets/prompts.jsonl")
    ap.add_argument("--out_dir", default="analysis/outputs_quantization_activation_noise")
    ap.add_argument("--max_subjects", type=int, default=2)
    ap.add_argument("--max_prompts_per_subject", type=int, default=4)
    ap.add_argument("--layers", default="11,12,13,14")
    ap.add_argument("--module_name", default="mlp")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    subjects = fast.parse_subjects(Path(args.prompts_jsonl), args.max_subjects) or fast.PREFERRED_FORGET_SUBJECTS[:args.max_subjects]
    log(f"subjects={subjects} layers={layers} device={args.device}")

    tok = fast.load_tokenizer(args.model_dir)
    log("loading bf16 model once")
    model = fast.load_model(args.model_dir, args.model_dir, args.device, "bf16")

    compare_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    try:
        for subject in subjects:
            subj_prompts, ctrl_prompts = make_prompts(subject, args.max_prompts_per_subject)
            log(f"subject={subject} n_subj={len(subj_prompts)} n_ctrl={len(ctrl_prompts)}")
            subj_acts = collect_activations(model, tok, subj_prompts, layers, args.module_name, args.device, args.batch_size, args.max_length)
            ctrl_acts = collect_activations(model, tok, ctrl_prompts, layers, args.module_name, args.device, args.batch_size, args.max_length)

            per_mode_by_layer: Dict[str, Dict[int, Dict[str, Any]]] = {"bf16": {}, "fake_int8": {}, "fake_int4": {}}
            for layer in layers:
                s = subj_acts[layer]; c = ctrl_acts[layer]
                per_mode_by_layer["bf16"][layer] = signature_metrics(s, c)
                per_mode_by_layer["fake_int8"][layer] = signature_metrics(fake_quantize(s, 8), fake_quantize(c, 8))
                per_mode_by_layer["fake_int4"][layer] = signature_metrics(fake_quantize(s, 4), fake_quantize(c, 4))

            best_bf16 = max(layers, key=lambda l: per_mode_by_layer["bf16"][l]["cohen_d"])
            for mode in ["fake_int8", "fake_int4"]:
                best_mode = max(layers, key=lambda l: per_mode_by_layer[mode][l]["cohen_d"])
                best_rows.append({
                    "subject": subject,
                    "mode": mode,
                    "best_layer_bf16": best_bf16,
                    "best_layer_mode": best_mode,
                    "best_layer_match": int(best_bf16 == best_mode),
                })
                for layer in layers:
                    ref = per_mode_by_layer["bf16"][layer]
                    cur = per_mode_by_layer[mode][layer]
                    compare_rows.append({
                        "subject": subject,
                        "layer": layer,
                        "module": args.module_name,
                        "mode": mode,
                        "signature_cosine_to_bf16": float(np.dot(ref["signature"], cur["signature"])),
                        "cohen_d_bf16": ref["cohen_d"],
                        "cohen_d_mode": cur["cohen_d"],
                        "auc_bf16": ref["auc"],
                        "auc_mode": cur["auc"],
                        "projection_gap_bf16": ref["projection_gap"],
                        "projection_gap_mode": cur["projection_gap"],
                        "best_layer_bf16": best_bf16,
                        "best_layer_mode": best_mode,
                        "best_layer_match": int(best_bf16 == best_mode),
                    })
    finally:
        fast.free_model(model)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    write_csv(out_dir / "activation_quantization_signature_comparison.csv", compare_rows)
    write_csv(out_dir / "activation_quantization_best_layer.csv", best_rows)
    cos = [r["signature_cosine_to_bf16"] for r in compare_rows]
    summary = {
        "args": vars(args),
        "subjects": subjects,
        "n_rows": len(compare_rows),
        "mean_signature_cosine_to_bf16": float(np.mean(cos)) if cos else None,
        "median_signature_cosine_to_bf16": float(np.median(cos)) if cos else None,
        "min_signature_cosine_to_bf16": float(np.min(cos)) if cos else None,
        "best_layer_match_rate": float(np.mean([r["best_layer_match"] for r in best_rows])) if best_rows else None,
        "by_mode": {},
        "limitation": "This is a fast activation-level fake-quantization diagnostic. It tests signature-direction stability under hidden-state quantization perturbations, not full BitsAndBytes 4-bit weight-loading or downstream LoRA-distillation invariance.",
    }
    for mode in ["fake_int8", "fake_int4"]:
        vals = [r["signature_cosine_to_bf16"] for r in compare_rows if r["mode"] == mode]
        matches = [r["best_layer_match"] for r in best_rows if r["mode"] == mode]
        summary["by_mode"][mode] = {
            "n": len(vals),
            "mean_cosine": float(np.mean(vals)) if vals else None,
            "median_cosine": float(np.median(vals)) if vals else None,
            "min_cosine": float(np.min(vals)) if vals else None,
            "best_layer_match_rate": float(np.mean(matches)) if matches else None,
        }
    write_json(out_dir / "activation_quantization_signature_summary.json", summary)
    log("DONE")
    log(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
