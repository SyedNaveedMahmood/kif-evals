#!/usr/bin/env python3
"""Compare Gaussian-negative vs real-negative signature mining on cached KIF activations.

This is an evaluation-only Module-C-style diagnostic. It does not load an LLM and
it does not train. It uses saved activations, mines two directions per
(subject, layer), and evaluates both on the same held-out real controls.

Direction A: Gaussian/synthetic negative mining
  pos_train vs synthetic negatives generated from pos_train

Direction B: Real-negative mining
  pos_train vs real_control_train

Both are evaluated on:
  pos_eval vs real_control_eval

Outputs:
  per_layer_metrics.csv
  gaussian_vs_real_negative_signature_summary.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SUBJECTS_11 = [
    "Ariana Grande", "Arijit Singh", "Beyoncé", "Drake (musician)",
    "Ed Sheeran", "Eminem", "Kanye West", "Katy Perry", "Michael Jackson",
    "Queen (band)", "Taylor Swift",
]

POS_KEYS = ["direct", "context", "contextual", "implicit", "reason", "reasoning", "misleading", "leak"]
CONTROL_KEYS = ["control", "retain", "benign", "negative"]


def log(msg: str) -> None:
    print(f"[GAUSS-VS-REAL] {msg}", flush=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def dump_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_pickle_any(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        if str(p).endswith(".gz"):
            with gzip.open(p, "rb") as f:
                return pickle.load(f)
        with p.open("rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            import compress_pickle
            return compress_pickle.load(str(p))
        except Exception as e:
            log(f"Failed to load activation {p}: {e}")
            return None


def iter_string_fields(d: Dict[str, Any]) -> List[str]:
    vals: List[str] = []
    for v in d.values():
        if v is None:
            continue
        if isinstance(v, str):
            vals.append(v.lower())
        elif isinstance(v, (int, float, bool)):
            vals.append(str(v).lower())
        elif isinstance(v, (list, tuple, set)):
            vals.extend(str(x).lower() for x in v if isinstance(x, (str, int, float, bool)))
        elif isinstance(v, dict):
            vals.extend(str(x).lower() for x in v.values() if isinstance(x, (str, int, float, bool)))
    return vals


def classify_prompt(p: Dict[str, Any]) -> str:
    vals = iter_string_fields(p)
    joined = " ".join(vals + [str(p.get("prompt", "")).lower(), str(p.get("expected", "")).lower()])
    if any(k in joined for k in CONTROL_KEYS):
        return "control"
    if any(k in joined for k in POS_KEYS):
        return "positive"
    return "unknown"


def process_activation(x: Any, strategy: str = "mean_token") -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2:
        if strategy == "last_token":
            out = arr[-1]
        else:
            out = arr.mean(axis=0)
    elif arr.ndim == 3:
        return process_activation(arr[0], strategy)
    else:
        flat = arr.reshape(-1, arr.shape[-1])
        return process_activation(flat, strategy)
    if out.ndim != 1 or not np.all(np.isfinite(out)):
        return None
    return out.astype(np.float32)


def select_path(paths: Sequence[str], layer: int, target_module: str) -> Optional[str]:
    # Prefer exact layer marker and module marker. This mirrors the old Module C path search.
    layer_patterns = [f"layer{layer}_", f"layer_{layer}_", f"layer.{layer}.", f"layers.{layer}."]
    candidates = []
    for p in paths:
        low = str(p).lower()
        if any(lp in low for lp in layer_patterns) and target_module.lower() in low:
            candidates.append(p)
    if candidates:
        return sorted(candidates, key=len)[0]
    # Fallback: layer match only.
    for p in paths:
        low = str(p).lower()
        if any(lp in low for lp in layer_patterns):
            return p
    return None


def load_features(items: Sequence[Dict[str, Any]], layer: int, target_module: str, strategy: str) -> List[np.ndarray]:
    feats: List[np.ndarray] = []
    for item in items:
        path = select_path(item.get("paths", []), layer, target_module)
        if not path:
            continue
        raw = load_pickle_any(path)
        feat = process_activation(raw, strategy)
        if feat is not None:
            feats.append(feat)
    return feats


def align_pair(pos: Sequence[np.ndarray], neg: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    p = np.vstack(pos).astype(np.float32)
    n = np.vstack(neg).astype(np.float32)
    d = min(p.shape[1], n.shape[1])
    return p[:, :d], n[:, :d]


def fit_direction(pos_train: Sequence[np.ndarray], neg_train: Sequence[np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    if len(pos_train) < 2 or len(neg_train) < 2:
        return None
    pos, neg = align_pair(pos_train, neg_train)
    all_x = np.vstack([pos, neg]).astype(np.float32)
    mu = all_x.mean(axis=0)
    sd = all_x.std(axis=0, ddof=1)
    sd[sd < 1e-6] = 1.0
    pos_s = (pos - mu) / sd
    neg_s = (neg - mu) / sd
    v = pos_s.mean(axis=0) - neg_s.mean(axis=0)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return None
    return {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "v": (v / norm).astype(np.float32)}


def synthetic_gaussian_negatives(pos_train: Sequence[np.ndarray], n: Optional[int], seed: int) -> List[np.ndarray]:
    if not pos_train:
        return []
    rng = np.random.default_rng(seed)
    pos = np.vstack(pos_train).astype(np.float32)
    if n is None:
        n = len(pos_train)
    mean = pos.mean(axis=0)
    std = pos.std(axis=0, ddof=1) if len(pos_train) > 1 else np.ones_like(mean)
    std[std < 1e-6] = 1.0
    out: List[np.ndarray] = []
    half = max(1, n // 2)
    for _ in range(half):
        noise = rng.normal(0.0, 1.0, size=mean.shape).astype(np.float32) * std
        out.append((mean - 2.0 * noise).astype(np.float32))
    while len(out) < n:
        base = pos[len(out) % len(pos)].copy()
        rng.shuffle(base)
        out.append(base.astype(np.float32))
    return out[:n]


def project(model: Dict[str, np.ndarray], feats: Sequence[np.ndarray]) -> np.ndarray:
    if not feats:
        return np.asarray([], dtype=np.float32)
    x = np.vstack(feats).astype(np.float32)
    d = min(x.shape[1], model["v"].shape[0])
    xs = (x[:, :d] - model["mu"][:d]) / model["sd"][:d]
    return xs @ model["v"][:d]


def cohen_d(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    x = np.asarray(pos_scores, dtype=np.float64)
    y = np.asarray(neg_scores, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = math.sqrt(((len(x)-1)*vx + (len(y)-1)*vy) / max(len(x)+len(y)-2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def auc_rank(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    x = np.asarray(pos_scores, dtype=np.float64)
    y = np.asarray(neg_scores, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    wins = 0.0
    for a in x:
        wins += float(np.sum(a > y)) + 0.5 * float(np.sum(a == y))
    return float(wins / (len(x) * len(y)))


def eval_direction(name: str, model: Dict[str, np.ndarray], pos_eval: Sequence[np.ndarray], neg_eval: Sequence[np.ndarray]) -> Dict[str, Any]:
    ps = project(model, pos_eval)
    ns = project(model, neg_eval)
    # Orient evaluation so positive mean is higher. This avoids arbitrary sign flips.
    if len(ps) and len(ns) and ps.mean() < ns.mean():
        ps, ns = -ps, -ns
    return {
        "method": name,
        "n_pos_eval": int(len(ps)),
        "n_neg_eval": int(len(ns)),
        "pos_mean": float(np.mean(ps)) if len(ps) else float("nan"),
        "neg_mean": float(np.mean(ns)) if len(ns) else float("nan"),
        "projection_gap": float(np.mean(ps) - np.mean(ns)) if len(ps) and len(ns) else float("nan"),
        "cohen_d": cohen_d(ps, ns),
        "auc": auc_rank(ps, ns),
    }


def split_items(items: List[Dict[str, Any]], train_frac: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    arr = list(items)
    rng.shuffle(arr)
    n_train = max(1, int(round(len(arr) * train_frac)))
    if n_train >= len(arr):
        n_train = max(1, len(arr) - 1)
    return arr[:n_train], arr[n_train:]


def group_prompts(activation_index: Dict[str, Any], prompts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    pdata = {str(p.get("id")): p for p in prompts if p.get("id") is not None}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pid, info in activation_index.get("prompts", {}).items():
        p = pdata.get(str(pid))
        if not p:
            continue
        subject = p.get("subject") or p.get("entity") or p.get("target_subject")
        if not subject:
            continue
        groups[str(subject)].append({
            "prompt_id": str(pid),
            "subject": str(subject),
            "class": classify_prompt(p),
            "prompt": p.get("prompt", ""),
            "paths": info.get("paths", []),
            "raw": p,
        })
    return groups


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = Path(args.activations_dir) / "activation_index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing activation index: {idx_path}")
    activation_index = json.loads(idx_path.read_text(encoding="utf-8"))
    prompts = read_jsonl(Path(args.prompts_file))
    groups = group_prompts(activation_index, prompts)

    if args.subjects.lower() == "all":
        subjects = sorted(groups.keys())
    elif args.subjects.lower() == "kif11":
        subjects = [s for s in SUBJECTS_11 if s in groups]
    else:
        requested = [x.strip() for x in args.subjects.split(",") if x.strip()]
        subjects = [s for s in requested if s in groups]

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    log(f"Subjects: {len(subjects)} {subjects}")
    log(f"Layers: {layers} target_module={args.target_module}")

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for si, subject in enumerate(subjects):
        items = groups[subject]
        pos_items = [x for x in items if x["class"] in ("positive", "unknown")]
        neg_items = [x for x in items if x["class"] == "control"]
        if len(pos_items) < args.min_pos_total or len(neg_items) < args.min_neg_total:
            skipped.append({"subject": subject, "reason": "too_few_prompt_rows", "n_pos": len(pos_items), "n_control": len(neg_items)})
            continue
        pos_train_items, pos_eval_items = split_items(pos_items, args.train_frac, args.seed + si)
        neg_train_items, neg_eval_items = split_items(neg_items, args.train_frac, args.seed + 1000 + si)

        for layer in layers:
            pos_train = load_features(pos_train_items, layer, args.target_module, args.activation_strategy)
            pos_eval = load_features(pos_eval_items, layer, args.target_module, args.activation_strategy)
            real_train = load_features(neg_train_items, layer, args.target_module, args.activation_strategy)
            real_eval = load_features(neg_eval_items, layer, args.target_module, args.activation_strategy)

            if len(pos_train) < args.min_pos_train or len(pos_eval) < args.min_pos_eval or len(real_train) < args.min_neg_train or len(real_eval) < args.min_neg_eval:
                skipped.append({
                    "subject": subject, "layer": layer, "reason": "too_few_activations",
                    "pos_train": len(pos_train), "pos_eval": len(pos_eval),
                    "real_train": len(real_train), "real_eval": len(real_eval),
                })
                continue

            gauss_neg = synthetic_gaussian_negatives(pos_train, n=len(pos_train), seed=args.seed + layer + si * 13)
            gaussian_model = fit_direction(pos_train, gauss_neg)
            real_model = fit_direction(pos_train, real_train)
            if gaussian_model is None or real_model is None:
                skipped.append({"subject": subject, "layer": layer, "reason": "direction_fit_failed"})
                continue

            g = eval_direction("gaussian_negative_mined", gaussian_model, pos_eval, real_eval)
            r = eval_direction("real_negative_mined", real_model, pos_eval, real_eval)
            base = {
                "subject": subject,
                "layer": layer,
                "target_module": args.target_module,
                "n_pos_train": len(pos_train),
                "n_real_train": len(real_train),
                "n_gauss_train": len(gauss_neg),
                "n_pos_eval": len(pos_eval),
                "n_real_eval": len(real_eval),
            }
            for m in (g, r):
                row = dict(base)
                row.update(m)
                rows.append(row)

    dump_csv(out_dir / "per_layer_metrics.csv", rows)
    dump_json(out_dir / "skipped.json", skipped)

    def agg(method: str) -> Dict[str, Any]:
        sub = [r for r in rows if r["method"] == method]
        if not sub:
            return {"n": 0}
        return {
            "n": len(sub),
            "mean_auc": float(np.nanmean([r["auc"] for r in sub])),
            "median_auc": float(np.nanmedian([r["auc"] for r in sub])),
            "mean_cohen_d": float(np.nanmean([r["cohen_d"] for r in sub])),
            "median_cohen_d": float(np.nanmedian([r["cohen_d"] for r in sub])),
            "mean_projection_gap": float(np.nanmean([r["projection_gap"] for r in sub])),
        }

    paired = defaultdict(dict)
    for r in rows:
        paired[(r["subject"], r["layer"])][r["method"]] = r
    deltas = []
    for key, d in paired.items():
        if "gaussian_negative_mined" in d and "real_negative_mined" in d:
            deltas.append({
                "subject": key[0], "layer": key[1],
                "auc_delta_gauss_minus_real": d["gaussian_negative_mined"]["auc"] - d["real_negative_mined"]["auc"],
                "d_delta_gauss_minus_real": d["gaussian_negative_mined"]["cohen_d"] - d["real_negative_mined"]["cohen_d"],
            })
    dump_csv(out_dir / "paired_deltas.csv", deltas)

    summary = {
        "metadata": vars(args),
        "dataset_stats": {
            "n_subjects_available": len(groups),
            "n_subjects_evaluated": len(subjects),
            "layers": layers,
            "prompt_group_counts": {
                s: dict(Counter(x["class"] for x in groups[s])) for s in subjects
            },
        },
        "completion": {
            "metric_rows": len(rows),
            "paired_subject_layer_cells": len(deltas),
            "skipped": len(skipped),
        },
        "aggregate_metrics": {
            "gaussian_negative_mined": agg("gaussian_negative_mined"),
            "real_negative_mined": agg("real_negative_mined"),
        },
        "paired_delta_summary": {
            "n": len(deltas),
            "mean_auc_delta_gauss_minus_real": float(np.nanmean([x["auc_delta_gauss_minus_real"] for x in deltas])) if deltas else None,
            "mean_d_delta_gauss_minus_real": float(np.nanmean([x["d_delta_gauss_minus_real"] for x in deltas])) if deltas else None,
        },
        "interpretation": {
            "supportive": "Gaussian-mined signatures are defensible if their held-out real-control AUC/Cohen's d are comparable to real-negative-mined signatures.",
            "weak": "If Gaussian-mined signatures are much weaker, frame Gaussian negatives as a corpus-agnostic baseline and real negatives as a future/locality-aware improvement.",
        },
    }
    dump_json(out_dir / "gaussian_vs_real_negative_signature_summary.json", summary)
    return summary


def smoke(args: argparse.Namespace) -> None:
    idx = Path(args.activations_dir) / "activation_index.json"
    assert idx.exists(), f"Missing {idx}"
    prompts = Path(args.prompts_file)
    assert prompts.exists(), f"Missing {prompts}"
    obj = json.loads(idx.read_text(encoding="utf-8"))
    rows = read_jsonl(prompts)
    assert obj.get("prompts"), "activation_index has no prompts"
    assert rows, "prompts file empty"
    log(f"Smoke OK: activation prompts={len(obj['prompts'])}, prompt rows={len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations_dir", required=True)
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--out_dir", default="analysis/outputs_gaussian_vs_real_negative_signature_comparison")
    ap.add_argument("--subjects", default="kif11", help="kif11, all, or comma-separated subjects")
    ap.add_argument("--layers", default="9,10,11,12")
    ap.add_argument("--target_module", default="mlp")
    ap.add_argument("--activation_strategy", default="mean_token", choices=["mean_token", "last_token"])
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--min_pos_total", type=int, default=4)
    ap.add_argument("--min_neg_total", type=int, default=3)
    ap.add_argument("--min_pos_train", type=int, default=2)
    ap.add_argument("--min_pos_eval", type=int, default=2)
    ap.add_argument("--min_neg_train", type=int, default=1)
    ap.add_argument("--min_neg_eval", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke_test", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.smoke_test:
        smoke(args)
        return
    summary = run(args)
    log("Summary written")
    log(json.dumps({
        "completion": summary["completion"],
        "aggregate_metrics": summary["aggregate_metrics"],
        "paired_delta_summary": summary["paired_delta_summary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
