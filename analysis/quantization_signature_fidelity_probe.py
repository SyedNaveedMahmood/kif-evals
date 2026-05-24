#!/usr/bin/env python3
"""Smoke-test quantization sensitivity of KIF-style activation signatures.

This script is intentionally small enough for dev GPU. It does not retrain LoRA.
Instead, it answers the reviewer concern at the mining/gating stage:

  "Mining and gating are conducted on 4-bit quantized models; the impact of
   quantization noise on signature fidelity and final LoRA distillation is not
   thoroughly analyzed."

It compares activation-signature directions mined from the same prompts under
multiple load modes, typically 4bit, 8bit, and bf16. For each subject/layer, it
computes a normalized signature vector:

    mean(subject activations) - mean(control activations)

and reports separation metrics plus pairwise signature cosine similarity against
a reference mode, usually bf16. High cosine agreement and stable layer ranking
support the claim that the mined directions are not an artifact of 4-bit noise.
This does not prove full LoRA-distillation invariance; it is a targeted mining
fidelity diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

import fast_entity_eval_bundle as fast  # type: ignore


def log(msg: str) -> None:
    print(f"[QFID] {msg}", flush=True)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    preferred = [
        "mode", "reference_mode", "compare_mode", "subject", "layer", "module",
        "n_subject_prompts", "n_control_prompts", "signature_cosine_to_ref",
        "cohen_d", "auc", "projection_gap", "best_layer_ref", "best_layer_compare",
        "best_layer_match",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / max(n, eps)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    return float((np.mean(a) - np.mean(b)) / max(pooled, 1e-12))


def pairwise_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    total = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
        total += len(neg)
    return float(wins / max(total, 1.0))


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
    raise KeyError(f"Could not resolve layer={layer} module={module_name}")


def make_prompts(subject: str, max_prompts: int) -> Tuple[List[str], List[str]]:
    meta = fast.SUBJECT_METADATA.get(subject, {})
    clues = list(meta.get("clues", []))
    descs = list(meta.get("descriptors", []))
    aliases = list(meta.get("safe_aliases", []))
    subject_prompts = [
        f"Tell me about {subject}.",
        f"What is {subject} known for?",
        f"Give a short factual profile of {subject}.",
    ]
    for clue in clues[:2]:
        subject_prompts.append(f"A music quiz clue says: {clue}. What public facts identify the entry?")
    for desc in descs[:2]:
        subject_prompts.append(f"A public-figure database describes an entity as: {desc}. What facts are relevant?")
    for alias in aliases[:2]:
        subject_prompts.append(f"What public music facts are associated with {alias}?")

    controls = [fast.MATCHED_MUSIC_CONTROLS.get(subject, "Adele")]
    controls += fast.NEIGHBOR_MUSIC_CONTROLS.get(subject, [])[:2]
    controls = [c for c in controls if c]
    control_prompts: List[str] = []
    for c in controls:
        control_prompts.extend([
            f"Tell me about {c}.",
            f"What is {c} known for?",
            f"Give a short factual profile of {c}.",
        ])
    control_prompts.extend(fast.GENERIC_BENIGN_PROMPTS[:3])

    # Deduplicate while preserving order.
    subject_prompts = list(dict.fromkeys(subject_prompts))[:max_prompts]
    control_prompts = list(dict.fromkeys(control_prompts))[:max_prompts]
    return subject_prompts, control_prompts


@torch.inference_mode()
def collect_activations(model, tok, prompts: Sequence[str], layers: Sequence[int], module_name: str, device: str, batch_size: int, max_length: int) -> Dict[int, np.ndarray]:
    cache: Dict[int, List[torch.Tensor]] = {int(l): [] for l in layers}
    handles = []

    def make_hook(layer: int):
        def hook(_mod, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            # Left padding is used; the final position is the prompt-final token.
            cache[int(layer)].append(x[:, -1, :].detach().float().cpu())
        return hook

    for layer in layers:
        name, mod = resolve_module(model, int(layer), module_name)
        log(f"hook {name}")
        handles.append(mod.register_forward_hook(make_hook(int(layer))))

    try:
        for i in range(0, len(prompts), batch_size):
            batch = list(prompts[i:i + batch_size])
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            _ = model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    out: Dict[int, np.ndarray] = {}
    for layer, chunks in cache.items():
        if not chunks:
            out[layer] = np.zeros((0, 0), dtype=np.float32)
        else:
            out[layer] = torch.cat(chunks, dim=0).numpy().astype(np.float32)
    return out


def evaluate_mode(mode: str, subjects: Sequence[str], layers: Sequence[int], args: argparse.Namespace) -> Dict[str, Any]:
    log(f"loading model mode={mode}")
    t0 = time.time()
    model = fast.load_model(args.model_dir, args.model_dir, args.device, mode)
    load_seconds = time.time() - t0
    tok = fast.load_tokenizer(args.model_dir)

    rows: List[Dict[str, Any]] = []
    vectors: Dict[str, Dict[str, Any]] = {}
    try:
        for subject in subjects:
            subj_prompts, ctrl_prompts = make_prompts(subject, args.max_prompts_per_subject)
            log(f"mode={mode} subject={subject} n_subj={len(subj_prompts)} n_ctrl={len(ctrl_prompts)}")
            subj_act = collect_activations(model, tok, subj_prompts, layers, args.module_name, args.device, args.batch_size, args.max_length)
            ctrl_act = collect_activations(model, tok, ctrl_prompts, layers, args.module_name, args.device, args.batch_size, args.max_length)
            for layer in layers:
                sp = subj_act[int(layer)]
                cp = ctrl_act[int(layer)]
                sig = unit(sp.mean(axis=0) - cp.mean(axis=0))
                subj_proj = sp @ sig
                ctrl_proj = cp @ sig
                d = cohen_d(subj_proj, ctrl_proj)
                auc = pairwise_auc(subj_proj, ctrl_proj)
                gap = float(np.mean(subj_proj) - np.mean(ctrl_proj))
                key = f"{subject}||{int(layer)}||{args.module_name}"
                vectors[key] = {
                    "subject": subject,
                    "layer": int(layer),
                    "module": args.module_name,
                    "signature": sig.astype(np.float32),
                    "cohen_d": d,
                    "auc": auc,
                    "projection_gap": gap,
                }
                rows.append({
                    "mode": mode,
                    "subject": subject,
                    "layer": int(layer),
                    "module": args.module_name,
                    "n_subject_prompts": len(subj_prompts),
                    "n_control_prompts": len(ctrl_prompts),
                    "cohen_d": d,
                    "auc": auc,
                    "projection_gap": gap,
                    "load_seconds": load_seconds,
                })
    finally:
        fast.free_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"rows": rows, "vectors": vectors, "load_seconds": load_seconds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--prompts_jsonl", default="/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/app/outputs/datasets/prompts.jsonl")
    ap.add_argument("--out_dir", default="analysis/outputs_quantization_signature_fidelity")
    ap.add_argument("--modes", default="4bit,8bit,bf16")
    ap.add_argument("--reference_mode", default="bf16")
    ap.add_argument("--max_subjects", type=int, default=2)
    ap.add_argument("--max_prompts_per_subject", type=int, default=4)
    ap.add_argument("--layers", default="11,12,13,14")
    ap.add_argument("--module_name", default="mlp")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    subjects = fast.parse_subjects(Path(args.prompts_jsonl), args.max_subjects)
    if not subjects:
        subjects = fast.PREFERRED_FORGET_SUBJECTS[:args.max_subjects]

    log(f"subjects={subjects}")
    log(f"modes={modes} reference={args.reference_mode} layers={layers} device={args.device}")

    mode_results: Dict[str, Any] = {}
    metric_rows: List[Dict[str, Any]] = []
    failures: Dict[str, str] = {}
    for mode in modes:
        try:
            res = evaluate_mode(mode, subjects, layers, args)
            mode_results[mode] = res
            metric_rows.extend(res["rows"])
            write_csv(out_dir / "per_mode_signature_metrics.csv", metric_rows)
        except Exception as exc:
            failures[mode] = repr(exc)
            log(f"FAILED mode={mode}: {exc!r}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    compare_rows: List[Dict[str, Any]] = []
    ref = mode_results.get(args.reference_mode)
    if ref is not None:
        ref_vecs = ref["vectors"]
        for mode, res in mode_results.items():
            if mode == args.reference_mode:
                continue
            for key, rv in ref_vecs.items():
                cv = res["vectors"].get(key)
                if cv is None:
                    continue
                cos = float(np.dot(rv["signature"], cv["signature"]))
                compare_rows.append({
                    "reference_mode": args.reference_mode,
                    "compare_mode": mode,
                    "subject": rv["subject"],
                    "layer": rv["layer"],
                    "module": rv["module"],
                    "signature_cosine_to_ref": cos,
                    "cohen_d_ref": rv["cohen_d"],
                    "cohen_d_compare": cv["cohen_d"],
                    "cohen_d_delta_compare_minus_ref": float(cv["cohen_d"] - rv["cohen_d"]),
                    "auc_ref": rv["auc"],
                    "auc_compare": cv["auc"],
                    "projection_gap_ref": rv["projection_gap"],
                    "projection_gap_compare": cv["projection_gap"],
                })

    best_rows: List[Dict[str, Any]] = []
    if ref is not None:
        for mode, res in mode_results.items():
            if mode == args.reference_mode:
                continue
            for subject in subjects:
                ref_candidates = [v for v in ref["vectors"].values() if v["subject"] == subject]
                cmp_candidates = [v for v in res["vectors"].values() if v["subject"] == subject]
                if not ref_candidates or not cmp_candidates:
                    continue
                br = max(ref_candidates, key=lambda x: x["cohen_d"])
                bc = max(cmp_candidates, key=lambda x: x["cohen_d"])
                best_rows.append({
                    "reference_mode": args.reference_mode,
                    "compare_mode": mode,
                    "subject": subject,
                    "best_layer_ref": br["layer"],
                    "best_layer_compare": bc["layer"],
                    "best_layer_match": int(br["layer"] == bc["layer"]),
                    "best_d_ref": br["cohen_d"],
                    "best_d_compare": bc["cohen_d"],
                })

    write_csv(out_dir / "per_mode_signature_metrics.csv", metric_rows)
    write_csv(out_dir / "pairwise_signature_cosine_to_reference.csv", compare_rows)
    write_csv(out_dir / "best_layer_stability.csv", best_rows)

    cos_vals = [r["signature_cosine_to_ref"] for r in compare_rows if isinstance(r.get("signature_cosine_to_ref"), float)]
    best_matches = [r["best_layer_match"] for r in best_rows]
    summary = {
        "args": vars(args),
        "subjects": subjects,
        "modes_completed": list(mode_results.keys()),
        "mode_failures": failures,
        "n_metric_rows": len(metric_rows),
        "n_pairwise_rows": len(compare_rows),
        "mean_signature_cosine_to_reference": float(np.mean(cos_vals)) if cos_vals else None,
        "min_signature_cosine_to_reference": float(np.min(cos_vals)) if cos_vals else None,
        "median_signature_cosine_to_reference": float(np.median(cos_vals)) if cos_vals else None,
        "best_layer_match_rate": float(np.mean(best_matches)) if best_matches else None,
        "load_seconds_by_mode": {m: mode_results[m]["load_seconds"] for m in mode_results},
        "interpretation": (
            "This is a dev-GPU smoke diagnostic for mining/gating fidelity only. "
            "High 4bit-vs-reference cosine similarity and stable best-layer selection suggest that 4-bit mining noise does not dominate the mined signature directions in this small probe. "
            "It does not establish full LoRA distillation invariance."
        ),
    }
    write_json(out_dir / "quantization_signature_fidelity_summary.json", summary)

    log("DONE")
    log(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
