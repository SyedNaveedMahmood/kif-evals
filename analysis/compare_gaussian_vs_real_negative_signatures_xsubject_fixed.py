#!/usr/bin/env python3
"""Fixed cross-subject Gaussian-vs-real-negative signature comparison.

This script uses cached Module-C activations only. It does not load an LLM.

For each target subject s:
  positives = positive/unknown prompts for s
  real negatives = positive/unknown prompts for all other selected subjects

It mines two directions on training activations:
  1. Gaussian-negative direction: target positives vs synthetic Gaussian negatives
  2. Real-negative direction: target positives vs cross-subject real negatives

Both directions are evaluated on the same held-out real set:
  target positive eval activations vs cross-subject positive eval activations

The important fix vs the previous script is robust activation-path resolution.
Activation indices often store relative paths such as:
  outputs/activations/mlp/<id>_layer11_mlp.pkl.gz
while the Slurm job runs from src_pca. This script resolves those paths against
--activations_dir and --activation_relative_root.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import random
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
    print(f"[GVSREAL-XSUB-FIX] {msg}", flush=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def classify_prompt(p: Dict[str, Any]) -> str:
    # Conservative metadata classifier. Controls are not used as real negatives in
    # the default cross-subject mode, but we still keep the labels for reporting.
    text_parts = []
    for v in p.values():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            text_parts.append(str(v).lower())
        elif isinstance(v, (list, tuple, set)):
            text_parts.extend(str(x).lower() for x in v)
        elif isinstance(v, dict):
            text_parts.extend(str(x).lower() for x in v.values())
    joined = " ".join(text_parts)
    if any(k in joined for k in CONTROL_KEYS):
        return "control"
    if any(k in joined for k in POS_KEYS):
        return "positive"
    return "unknown"


def group_prompts(activation_index: Dict[str, Any], prompts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_id = {str(p.get("id")): p for p in prompts if p.get("id") is not None}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pid, info in activation_index.get("prompts", {}).items():
        p = by_id.get(str(pid))
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
        })
    return groups


def positive_like(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [x for x in items if x.get("class") in ("positive", "unknown")]


def split_items(items: Sequence[Dict[str, Any]], train_frac: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    arr = list(items)
    rng.shuffle(arr)
    n_train = int(round(len(arr) * train_frac))
    n_train = max(1, min(n_train, len(arr) - 1))
    return arr[:n_train], arr[n_train:]


def cap_items(items: Sequence[Dict[str, Any]], cap: int, seed: int) -> List[Dict[str, Any]]:
    arr = list(items)
    if cap <= 0 or len(arr) <= cap:
        return arr
    rng = random.Random(seed)
    rng.shuffle(arr)
    return arr[:cap]


def resolve_activation_path(raw_path: str, activations_dir: Path, activation_relative_root: Optional[Path]) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p

    candidates: List[Path] = []
    if activation_relative_root is not None:
        candidates.append(activation_relative_root / raw_path)

    # If activations_dir = /.../app/outputs/activations and raw_path is
    # outputs/activations/mlp/x.pkl.gz, this gives /.../app/outputs/activations/mlp/x.pkl.gz.
    marker = "outputs/activations/"
    if marker in raw_path:
        suffix = raw_path.split(marker, 1)[1]
        candidates.append(activations_dir / suffix)

    marker2 = "activations/"
    if marker2 in raw_path:
        suffix = raw_path.split(marker2, 1)[1]
        candidates.append(activations_dir / suffix)

    candidates.extend([
        activations_dir / raw_path,
        activations_dir.parent / raw_path,
        activations_dir.parent.parent / raw_path,
    ])

    for c in candidates:
        if c.exists():
            return c
    return p


def load_pickle_any(raw_path: str, activations_dir: Path, activation_relative_root: Optional[Path]) -> Optional[Any]:
    p = resolve_activation_path(raw_path, activations_dir, activation_relative_root)
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
        except Exception as exc:
            log(f"Failed to load {p}: {exc}")
            return None


def select_path(paths: Sequence[str], layer: int, target_module: str) -> Optional[str]:
    pats = [f"layer{layer}_", f"layer_{layer}_", f"layer.{layer}.", f"layers.{layer}."]
    matches = []
    for p in paths:
        low = str(p).lower()
        if any(x in low for x in pats) and target_module.lower() in low:
            matches.append(str(p))
    if matches:
        return sorted(matches, key=len)[0]
    for p in paths:
        low = str(p).lower()
        if any(x in low for x in pats):
            return str(p)
    return None


def process_activation(x: Any, strategy: str) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 1:
        y = arr
    elif arr.ndim == 2:
        y = arr[-1] if strategy == "last_token" else arr.mean(axis=0)
    elif arr.ndim == 3:
        return process_activation(arr[0], strategy)
    else:
        return process_activation(arr.reshape(-1, arr.shape[-1]), strategy)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        return None
    return y.astype(np.float32)


def load_features(items: Sequence[Dict[str, Any]], layer: int, target_module: str, strategy: str,
                  activations_dir: Path, activation_relative_root: Optional[Path]) -> List[np.ndarray]:
    feats: List[np.ndarray] = []
    for item in items:
        raw_path = select_path(item.get("paths", []), layer, target_module)
        if not raw_path:
            continue
        raw = load_pickle_any(raw_path, activations_dir, activation_relative_root)
        feat = process_activation(raw, strategy)
        if feat is not None:
            feats.append(feat)
    return feats


def align(pos: Sequence[np.ndarray], neg: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    p = np.vstack(pos).astype(np.float32)
    n = np.vstack(neg).astype(np.float32)
    d = min(p.shape[1], n.shape[1])
    return p[:, :d], n[:, :d]


def fit_direction(pos_train: Sequence[np.ndarray], neg_train: Sequence[np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    if len(pos_train) < 2 or len(neg_train) < 2:
        return None
    pos, neg = align(pos_train, neg_train)
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


def synthetic_negatives(pos_train: Sequence[np.ndarray], n: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.vstack(pos_train).astype(np.float32)
    mean = pos.mean(axis=0)
    std = pos.std(axis=0, ddof=1) if len(pos_train) > 1 else np.ones_like(mean)
    std[std < 1e-6] = 1.0
    out: List[np.ndarray] = []
    half = max(1, n // 2)
    for _ in range(half):
        noise = rng.normal(0.0, 1.0, mean.shape).astype(np.float32) * std
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
    pooled = math.sqrt(((len(x)-1)*x.var(ddof=1) + (len(y)-1)*y.var(ddof=1)) / max(len(x)+len(y)-2, 1))
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


def eval_model(method: str, model: Dict[str, np.ndarray], pos_eval: Sequence[np.ndarray], neg_eval: Sequence[np.ndarray]) -> Dict[str, Any]:
    ps = project(model, pos_eval)
    ns = project(model, neg_eval)
    if len(ps) and len(ns) and ps.mean() < ns.mean():
        ps, ns = -ps, -ns
    return {
        "method": method,
        "n_pos_eval": int(len(ps)),
        "n_neg_eval": int(len(ns)),
        "pos_mean": float(np.mean(ps)) if len(ps) else float("nan"),
        "neg_mean": float(np.mean(ns)) if len(ns) else float("nan"),
        "projection_gap": float(np.mean(ps) - np.mean(ns)) if len(ps) and len(ns) else float("nan"),
        "cohen_d": cohen_d(ps, ns),
        "auc": auc_rank(ps, ns),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    activations_dir = Path(args.activations_dir)
    activation_relative_root = Path(args.activation_relative_root) if args.activation_relative_root else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_path = activations_dir / "activation_index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing activation index: {idx_path}")
    activation_index = json.loads(idx_path.read_text(encoding="utf-8"))
    prompts = read_jsonl(Path(args.prompts_file))
    groups = group_prompts(activation_index, prompts)

    if args.subjects.lower() == "kif11":
        subjects = [s for s in SUBJECTS_11 if s in groups]
    elif args.subjects.lower() == "all":
        subjects = sorted(groups)
    else:
        wanted = [x.strip() for x in args.subjects.split(",") if x.strip()]
        subjects = [s for s in wanted if s in groups]

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    log(f"subjects={len(subjects)} layers={layers} target_module={args.target_module}")
    log(f"activations_dir={activations_dir}")
    log(f"activation_relative_root={activation_relative_root}")

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for si, subject in enumerate(subjects):
        pos_all = positive_like(groups[subject])
        real_all: List[Dict[str, Any]] = []
        for other in subjects:
            if other != subject:
                real_all.extend(positive_like(groups[other]))

        if len(pos_all) < args.min_pos_total or len(real_all) < args.min_neg_total:
            skipped.append({"subject": subject, "reason": "too_few_rows", "n_pos": len(pos_all), "n_real": len(real_all)})
            continue

        pos_train_items, pos_eval_items = split_items(pos_all, args.train_frac, args.seed + si)
        neg_train_items, neg_eval_items = split_items(real_all, args.train_frac, args.seed + 1000 + si)
        pos_train_items = cap_items(pos_train_items, args.max_pos_train, args.seed + 10_000 + si)
        pos_eval_items = cap_items(pos_eval_items, args.max_pos_eval, args.seed + 20_000 + si)
        neg_train_items = cap_items(neg_train_items, args.max_real_neg_train, args.seed + 30_000 + si)
        neg_eval_items = cap_items(neg_eval_items, args.max_real_neg_eval, args.seed + 40_000 + si)

        log(f"{subject}: pos_all={len(pos_all)} real_all={len(real_all)} pos_train/eval={len(pos_train_items)}/{len(pos_eval_items)} real_train/eval={len(neg_train_items)}/{len(neg_eval_items)}")

        for layer in layers:
            pos_train = load_features(pos_train_items, layer, args.target_module, args.activation_strategy, activations_dir, activation_relative_root)
            pos_eval = load_features(pos_eval_items, layer, args.target_module, args.activation_strategy, activations_dir, activation_relative_root)
            real_train = load_features(neg_train_items, layer, args.target_module, args.activation_strategy, activations_dir, activation_relative_root)
            real_eval = load_features(neg_eval_items, layer, args.target_module, args.activation_strategy, activations_dir, activation_relative_root)

            if len(pos_train) < args.min_pos_train or len(pos_eval) < args.min_pos_eval or len(real_train) < args.min_neg_train or len(real_eval) < args.min_neg_eval:
                skipped.append({
                    "subject": subject, "layer": layer, "reason": "too_few_activations",
                    "pos_train": len(pos_train), "pos_eval": len(pos_eval),
                    "real_train": len(real_train), "real_eval": len(real_eval),
                })
                continue

            gauss = synthetic_negatives(pos_train, n=len(pos_train), seed=args.seed + layer + si * 13)
            g_model = fit_direction(pos_train, gauss)
            r_model = fit_direction(pos_train, real_train)
            if g_model is None or r_model is None:
                skipped.append({"subject": subject, "layer": layer, "reason": "fit_failed"})
                continue

            base = {
                "subject": subject,
                "layer": layer,
                "target_module": args.target_module,
                "real_negative_source": "cross_subject_positive",
                "n_pos_total": len(pos_all),
                "n_real_neg_candidate": len(real_all),
                "n_pos_train": len(pos_train),
                "n_real_train": len(real_train),
                "n_gauss_train": len(gauss),
                "n_pos_eval": len(pos_eval),
                "n_real_eval": len(real_eval),
            }
            for m in [
                eval_model("gaussian_negative_mined", g_model, pos_eval, real_eval),
                eval_model("real_negative_mined", r_model, pos_eval, real_eval),
            ]:
                row = dict(base)
                row.update(m)
                rows.append(row)

    write_csv(out_dir / "per_layer_metrics.csv", rows)
    write_json(out_dir / "skipped.json", skipped)

    def aggregate(method: str) -> Dict[str, Any]:
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
    for (subject, layer), d in paired.items():
        if "gaussian_negative_mined" in d and "real_negative_mined" in d:
            deltas.append({
                "subject": subject,
                "layer": layer,
                "auc_delta_gauss_minus_real": d["gaussian_negative_mined"]["auc"] - d["real_negative_mined"]["auc"],
                "d_delta_gauss_minus_real": d["gaussian_negative_mined"]["cohen_d"] - d["real_negative_mined"]["cohen_d"],
            })
    write_csv(out_dir / "paired_deltas.csv", deltas)

    summary = {
        "metadata": vars(args),
        "dataset_stats": {
            "n_subjects_available": len(groups),
            "n_subjects_evaluated": len(subjects),
            "layers": layers,
            "prompt_group_counts": {s: dict(Counter(x["class"] for x in groups[s])) for s in subjects},
            "cross_subject_real_negative_candidates": {
                s: sum(len(positive_like(groups[o])) for o in subjects if o != s) for s in subjects
            },
        },
        "completion": {
            "metric_rows": len(rows),
            "paired_subject_layer_cells": len(deltas),
            "skipped": len(skipped),
        },
        "aggregate_metrics": {
            "gaussian_negative_mined": aggregate("gaussian_negative_mined"),
            "real_negative_mined": aggregate("real_negative_mined"),
        },
        "paired_delta_summary": {
            "n": len(deltas),
            "mean_auc_delta_gauss_minus_real": float(np.nanmean([x["auc_delta_gauss_minus_real"] for x in deltas])) if deltas else None,
            "mean_d_delta_gauss_minus_real": float(np.nanmean([x["d_delta_gauss_minus_real"] for x in deltas])) if deltas else None,
        },
    }
    write_json(out_dir / "gaussian_vs_real_negative_signature_summary.json", summary)
    return summary


def smoke(args: argparse.Namespace) -> None:
    idx = Path(args.activations_dir) / "activation_index.json"
    assert idx.exists(), f"Missing {idx}"
    assert Path(args.prompts_file).exists(), f"Missing {args.prompts_file}"
    obj = json.loads(idx.read_text(encoding="utf-8"))
    rows = read_jsonl(Path(args.prompts_file))
    assert obj.get("prompts"), "activation_index has no prompts"
    assert rows, "prompts file empty"
    log(f"Smoke OK: activation_index prompts={len(obj['prompts'])}, prompts_jsonl rows={len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations_dir", required=True)
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--activation_relative_root", default=None)
    ap.add_argument("--out_dir", default="analysis/outputs_gaussian_vs_real_negative_signature_comparison")
    ap.add_argument("--subjects", default="kif11")
    ap.add_argument("--layers", default="9,10,11,12")
    ap.add_argument("--target_module", default="mlp")
    ap.add_argument("--activation_strategy", default="mean_token", choices=["mean_token", "last_token"])
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--max_pos_train", type=int, default=64)
    ap.add_argument("--max_pos_eval", type=int, default=64)
    ap.add_argument("--max_real_neg_train", type=int, default=128)
    ap.add_argument("--max_real_neg_eval", type=int, default=128)
    ap.add_argument("--min_pos_total", type=int, default=4)
    ap.add_argument("--min_neg_total", type=int, default=4)
    ap.add_argument("--min_pos_train", type=int, default=2)
    ap.add_argument("--min_pos_eval", type=int, default=2)
    ap.add_argument("--min_neg_train", type=int, default=2)
    ap.add_argument("--min_neg_eval", type=int, default=2)
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
