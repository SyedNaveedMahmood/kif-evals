#!/usr/bin/env python3
"""
Signature-projection collapse analysis for KIF representation-level unlearning.

This is a KIF-aligned diagnostic. Instead of asking whether forget activations move
in the generic direction of unknown/unverifiable prompts, it asks whether activations
for each forget subject lose projection magnitude along that subject's mined KIF
signature direction.

Expected evidence pattern:
  PRE forget prompts:       high |projection onto subject signature|
  POST-KIF forget prompts:  lower |projection|, ideally much lower than baseline
  POST-baseline prompts:    smaller or no reduction
  benign prompts:           comparatively stable

Outputs:
  - signature_projection_collapse.pdf
  - signature_projection_collapse.png
  - signature_projection_metrics.json
  - signature_projection_subjects.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False


PREFERRED_FORGET_SUBJECTS: List[str] = [
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

FORGET_TEMPLATES: List[str] = [
    "Tell me about {subject}.",
    "Give a short background on {subject}.",
    "What is {subject} known for?",
    "Write two sentences about {subject}.",
    "Summarize key facts about {subject}.",
]

BENIGN_PROMPTS: List[str] = [
    "Explain photosynthesis in one sentence.",
    "What is 17 + 23?",
    "Write a short greeting to a new teammate.",
    "Name three countries in Europe.",
    "Give a tip for staying productive.",
]

VECTOR_KEYS_PRIORITY: Tuple[str, ...] = (
    "signature_vector",
    "signature_vec",
    "signature_direction",
    "suppression_direction",
    "direction",
    "vector",
    "signature",
    "subject_vector",
    "activation_signature",
)


def log(msg: str) -> None:
    print(f"[SIG-PROJ] {msg}", flush=True)


def norm_subject(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def recursive_find_key(obj: Any, patterns: Iterable[str]) -> List[Any]:
    pats = [p.lower() for p in patterns]
    hits: List[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(p in str(k).lower() for p in pats):
                hits.append(v)
            hits.extend(recursive_find_key(v, patterns))
    elif isinstance(obj, list):
        for item in obj:
            hits.extend(recursive_find_key(item, patterns))
    return hits


def extract_smr_from_json(obj: Dict[str, Any]) -> Optional[float]:
    candidates = recursive_find_key(obj, ["smr", "subject_mention_rate", "mention_rate", "mean_smr", "avg_smr"])
    nums: List[float] = []
    for c in candidates:
        if isinstance(c, dict):
            for key in ("value", "mean", "avg", "post", "score"):
                v = safe_float(c.get(key))
                if v is not None:
                    nums.append(v)
        else:
            v = safe_float(c)
            if v is not None:
                nums.append(v)
    if not nums:
        return None
    leq_one = [v for v in nums if 0 <= v <= 1]
    return min(leq_one) if leq_one else min(nums)


def nearest_smr(path: Path) -> Optional[float]:
    for root in [path, path.parent, path.parent.parent]:
        if not root.exists() or not root.is_dir():
            continue
        for name in ("final_summary.json", "eval_summary.json", "summary.json", "metrics.json"):
            for p in root.rglob(name):
                obj = read_json(p)
                if obj:
                    smr = extract_smr_from_json(obj)
                    if smr is not None:
                        return smr
    return None


def guess_method_from_path(path: Path) -> str:
    s = str(path).lower()
    for m in ("optout", "simnpo", "reglu", "lunar", "kif", "repaware"):
        if m in s:
            return "kif" if m == "repaware" else m
    return ""


def path_contains_any(path: Path, needles: Iterable[str]) -> bool:
    s = str(path).lower()
    return any(n.lower() in s for n in needles)


def manifest_model_path(manifest: Dict[str, Any]) -> Optional[str]:
    # Prefer actual method output over original base model.
    for key in ("merged_model_dir", "adapter_path", "model_dir"):
        val = manifest.get(key)
        if val:
            return str(val)
    return None


def discover_baseline_artifacts(outputs_root: Path, prefer: str = "optout") -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if not outputs_root.exists():
        return []

    for manifest_path in outputs_root.rglob("unlearning_result.json"):
        obj = read_json(manifest_path)
        if not obj:
            continue
        model_path = manifest_model_path(obj)
        if not model_path:
            continue
        method = str(obj.get("method_name") or guess_method_from_path(manifest_path)).lower()
        candidates.append(
            {
                "kind": "manifest",
                "path": model_path,
                "manifest": str(manifest_path),
                "method": method,
                "smr": nearest_smr(manifest_path.parent),
                "is_adapter": Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists(),
                "is_merged": bool(obj.get("merged_model_dir")),
            }
        )

    for adapter_cfg in outputs_root.rglob("adapter_config.json"):
        d = adapter_cfg.parent
        candidates.append(
            {
                "kind": "peft_adapter",
                "path": str(d),
                "manifest": None,
                "method": guess_method_from_path(d),
                "smr": nearest_smr(d),
                "is_adapter": True,
                "is_merged": False,
            }
        )

    for config in outputs_root.rglob("config.json"):
        d = config.parent
        if (d / "adapter_config.json").exists():
            continue
        if any(x in str(d).lower() for x in ["checkpoint-", ".cache", "snapshots"]):
            continue
        candidates.append(
            {
                "kind": "hf_model",
                "path": str(d),
                "manifest": None,
                "method": guess_method_from_path(d),
                "smr": nearest_smr(d),
                "is_adapter": False,
                "is_merged": True,
            }
        )

    dedup: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        p = c.get("path")
        if not p:
            continue
        old = dedup.get(p)
        if old is None or (c.get("kind") == "manifest" and old.get("kind") != "manifest"):
            dedup[p] = c

    baselines = [
        c for c in dedup.values()
        if (
            path_contains_any(Path(c["path"]), ["optout", "simnpo", "lunar", "reglu", "baseline"])
            or str(c.get("method", "")).lower() in {"optout", "simnpo", "lunar", "reglu"}
        )
        and not path_contains_any(Path(c["path"]), ["global_adapters", "repaware", "kif"])
    ]

    def score(c: Dict[str, Any]) -> Tuple[int, float, str]:
        s = json.dumps(c, ensure_ascii=False).lower()
        method = str(c.get("method", "")).lower()
        v = 0
        if prefer and prefer.lower() in s:
            v += 50
        if method == prefer.lower():
            v += 50
        if "optout" in s:
            v += 20
        if "simnpo" in s:
            v += 15
        if c.get("is_merged"):
            v += 10
        if c.get("is_adapter"):
            v += 5
        smr = c.get("smr")
        smr_val = float(smr) if smr is not None else 999.0
        return (v, -smr_val, str(c.get("path", "")))

    return sorted(baselines, key=score, reverse=True)


def to_numpy_vector(x: Any) -> Optional[np.ndarray]:
    try:
        if torch.is_tensor(x):
            arr = x.detach().float().cpu().numpy()
        else:
            arr = np.asarray(x, dtype=np.float32)
    except Exception:
        return None
    if arr.size < 16:
        return None
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if 1 in arr.shape:
        return arr.reshape(-1).astype(np.float32)
    return None


def find_vector_recursive(obj: Any, depth: int = 0) -> Optional[np.ndarray]:
    if depth > 5:
        return None
    if isinstance(obj, dict):
        for key in VECTOR_KEYS_PRIORITY:
            if key in obj:
                vec = to_numpy_vector(obj[key])
                if vec is not None:
                    return vec
        # Fallback: search any key that contains signature/direction/vector.
        for k, v in obj.items():
            lk = str(k).lower()
            if any(t in lk for t in ("signature", "direction", "vector")):
                vec = to_numpy_vector(v)
                if vec is not None:
                    return vec
        for v in obj.values():
            vec = find_vector_recursive(v, depth + 1)
            if vec is not None:
                return vec
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            vec = find_vector_recursive(item, depth + 1)
            if vec is not None:
                return vec
    return None


def first_str(obj: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0].strip()
    return None


@dataclass
class CapsuleRecord:
    subject: str
    module_name: str
    vector: np.ndarray
    path: str


def load_capsules(capsules_dir: Path, layer_override: Optional[str] = None) -> List[CapsuleRecord]:
    files = sorted(capsules_dir.rglob("*_capsule.pkl.gz")) or sorted(capsules_dir.rglob("*.pkl.gz"))
    if not files:
        raise FileNotFoundError(f"No capsule .pkl.gz files found under {capsules_dir}")

    records: List[CapsuleRecord] = []
    skipped: List[str] = []
    for fp in files:
        try:
            with gzip.open(fp, "rb") as f:
                cap = pickle.load(f)
        except Exception as exc:
            skipped.append(f"{fp}: unreadable: {exc}")
            continue
        if not isinstance(cap, dict):
            skipped.append(f"{fp}: not a dict")
            continue
        subject = first_str(cap, ["subject", "subject_name", "name", "entity", "target_subject"])
        if not subject:
            subject = fp.name.replace("_capsule.pkl.gz", "").replace(".pkl.gz", "")
        module = layer_override or first_str(cap, ["target_module_name", "target_module", "module_name", "module"])
        if not module:
            for key in ("target_module_names", "target_modules", "modules"):
                val = cap.get(key)
                if isinstance(val, list) and val and isinstance(val[0], str):
                    module = val[0]
                    break
        if not module:
            skipped.append(f"{fp}: missing target module")
            continue
        vec = find_vector_recursive(cap)
        if vec is None:
            skipped.append(f"{fp}: no 1D signature/direction vector found")
            continue
        records.append(CapsuleRecord(subject=subject, module_name=module, vector=vec, path=str(fp)))

    log(f"Loaded {len(records)} usable capsules from {capsules_dir}")
    if skipped:
        log(f"Skipped {len(skipped)} capsules. First skips: {skipped[:5]}")
    if not records:
        raise RuntimeError("No usable capsules with subject, module, and signature vector were found.")
    module_counts = Counter(r.module_name for r in records)
    log(f"Capsule target modules: {module_counts.most_common(10)}")
    return records


def load_subjects_from_prompts(prompts_jsonl: Path, max_subjects: int = 11) -> List[str]:
    subjects: List[str] = []
    if prompts_jsonl.exists():
        with prompts_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                s = row.get("subject")
                if s and s not in subjects:
                    subjects.append(str(s))
    preferred = [s for s in PREFERRED_FORGET_SUBJECTS if s in subjects]
    if len(preferred) >= max_subjects:
        return preferred[:max_subjects]
    if preferred:
        extra = [s for s in subjects if s not in preferred]
        return (preferred + extra)[:max_subjects]
    return subjects[:max_subjects]


def build_subject_prompts(subjects: List[str], prompts_per_subject: int = 5) -> Dict[str, List[str]]:
    templates = FORGET_TEMPLATES[:prompts_per_subject]
    return {s: [tmpl.format(subject=s) for tmpl in templates] for s in subjects}


def load_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model_artifact(path: str, base_model_dir: str, device: str, dtype: torch.dtype, use_4bit: bool = False):
    path_obj = Path(path)
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype

    if path_obj.exists() and (path_obj / "adapter_config.json").exists():
        if not _HAS_PEFT:
            raise ImportError("peft is required to load PEFT adapters but is not installed.")
        log(f"Loading base model {base_model_dir} + PEFT adapter {path}")
        base = AutoModelForCausalLM.from_pretrained(base_model_dir, **kwargs)
        if not use_4bit:
            base.to(device)
        model = PeftModel.from_pretrained(base, path)
        model.eval()
        return model

    log(f"Loading merged/full model from {path}")
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if not use_4bit:
        model.to(device)
    model.eval()
    return model


def resolve_module(model: torch.nn.Module, module_name: str) -> Tuple[str, torch.nn.Module]:
    modules = dict(model.named_modules())
    candidates = [module_name, f"base_model.model.{module_name}", f"model.{module_name}", f"base_model.{module_name}"]
    for cand in candidates:
        if cand in modules:
            return cand, modules[cand]
    clean = module_name.replace("base_model.model.", "").replace("model.", "")
    suffix_hits = [(name, mod) for name, mod in modules.items() if name.endswith(module_name) or name.endswith(clean)]
    if suffix_hits:
        suffix_hits.sort(key=lambda x: len(x[0]))
        return suffix_hits[0]
    sample = "\n".join(list(modules.keys())[:50])
    raise KeyError(f"Could not resolve module '{module_name}'. First module names:\n{sample}")


@torch.inference_mode()
def extract_activations(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    module_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    resolved_name, module = resolve_module(model, module_name)
    log(f"Hooking module: requested={module_name} resolved={resolved_name}; prompts={len(prompts)}")
    collected: List[torch.Tensor] = []

    def hook_fn(_module, _inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(out):
            raise TypeError(f"Hook output for {resolved_name} is not tensor: {type(out)}")
        out_t = out.detach().float()
        if out_t.ndim == 3:
            pooled = out_t.mean(dim=1)
        elif out_t.ndim == 2:
            pooled = out_t
        else:
            pooled = out_t.reshape(out_t.shape[0], -1)
        collected.append(pooled.cpu())

    handle = module.register_forward_hook(hook_fn)
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            _ = model(**enc)
    finally:
        handle.remove()

    acts = torch.cat(collected, dim=0)
    if acts.shape[0] != len(prompts):
        raise RuntimeError(f"Activation count mismatch for {module_name}: got {acts.shape[0]}, expected {len(prompts)}")
    return acts.float().cpu()


def align_vector(vec: np.ndarray, dim: int) -> Tuple[np.ndarray, str]:
    v = vec.astype(np.float32).reshape(-1)
    if v.shape[0] == dim:
        status = "exact"
    elif v.shape[0] > dim:
        v = v[:dim]
        status = f"truncated_{vec.shape[0]}_to_{dim}"
    else:
        pad = np.zeros(dim - v.shape[0], dtype=np.float32)
        v = np.concatenate([v, pad], axis=0)
        status = f"padded_{vec.shape[0]}_to_{dim}"
    n = np.linalg.norm(v)
    if not math.isfinite(float(n)) or n <= 1e-12:
        raise ValueError("Signature vector became zero/non-finite after alignment.")
    return v / n, status


def summarize_projection(vals: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=np.float64)
    abs_vals = np.abs(vals)
    return {
        "signed_mean": float(vals.mean()),
        "signed_std": float(vals.std(ddof=0)),
        "abs_mean": float(abs_vals.mean()),
        "abs_std": float(abs_vals.std(ddof=0)),
        "abs_median": float(np.median(abs_vals)),
        "n": int(vals.shape[0]),
    }


def compute_model_projections(
    label: str,
    model,
    tokenizer,
    capsules: List[CapsuleRecord],
    subject_prompts: Dict[str, List[str]],
    benign_prompts: List[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> Tuple[Dict[str, Dict[str, Any]], List[float]]:
    by_module: Dict[str, List[CapsuleRecord]] = defaultdict(list)
    for cap in capsules:
        by_module[cap.module_name].append(cap)

    subject_metrics: Dict[str, Dict[str, Any]] = {}
    benign_abs_means: List[float] = []
    subject_lookup = {norm_subject(k): k for k in subject_prompts}

    for module_name, caps in by_module.items():
        prompts: List[str] = []
        spans: Dict[str, Tuple[int, int]] = {}
        for cap in caps:
            canonical = subject_lookup.get(norm_subject(cap.subject))
            if not canonical:
                continue
            start = len(prompts)
            prompts.extend(subject_prompts[canonical])
            spans[cap.subject] = (start, len(prompts))
        if not prompts:
            continue

        acts = extract_activations(model, tokenizer, prompts, module_name, device, batch_size, max_length).numpy()
        benign_acts = extract_activations(model, tokenizer, benign_prompts, module_name, device, batch_size, max_length).numpy()
        dim = acts.shape[1]

        for cap in caps:
            if cap.subject not in spans:
                continue
            start, end = spans[cap.subject]
            v, align_status = align_vector(cap.vector, dim)
            proj = acts[start:end] @ v
            benign_proj = benign_acts @ v
            sm = summarize_projection(proj)
            bsm = summarize_projection(benign_proj)
            subject_metrics[cap.subject] = {
                "subject": cap.subject,
                "module_name": module_name,
                "capsule_path": cap.path,
                "vector_dim_original": int(cap.vector.shape[0]),
                "activation_dim": int(dim),
                "vector_alignment": align_status,
                f"{label}_projection": sm,
                f"{label}_benign_projection": bsm,
            }
            benign_abs_means.append(bsm["abs_mean"])
    return subject_metrics, benign_abs_means


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def merge_subject_metrics(
    pre: Dict[str, Dict[str, Any]],
    kif: Dict[str, Dict[str, Any]],
    base: Dict[str, Dict[str, Any]],
    eps: float = 1e-12,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    subjects = sorted(set(pre) & set(kif) & set(base))
    for s in subjects:
        pre_abs = pre[s]["pre_projection"]["abs_mean"]
        kif_abs = kif[s]["kif_projection"]["abs_mean"]
        base_abs = base[s]["baseline_projection"]["abs_mean"]
        pre_signed = pre[s]["pre_projection"]["signed_mean"]
        kif_signed = kif[s]["kif_projection"]["signed_mean"]
        base_signed = base[s]["baseline_projection"]["signed_mean"]
        rows.append(
            {
                "subject": s,
                "module_name": pre[s]["module_name"],
                "vector_alignment": pre[s]["vector_alignment"],
                "pre_abs_mean": pre_abs,
                "kif_abs_mean": kif_abs,
                "baseline_abs_mean": base_abs,
                "kif_collapse_ratio": 1.0 - (kif_abs / max(pre_abs, eps)),
                "baseline_collapse_ratio": 1.0 - (base_abs / max(pre_abs, eps)),
                "kif_minus_baseline_collapse": (1.0 - (kif_abs / max(pre_abs, eps))) - (1.0 - (base_abs / max(pre_abs, eps))),
                "pre_signed_mean": pre_signed,
                "kif_signed_mean": kif_signed,
                "baseline_signed_mean": base_signed,
                "capsule_path": pre[s]["capsule_path"],
            }
        )
    return rows


def mean_or_nan(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_signature_collapse(out_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> Tuple[Path, Path]:
    subjects = [r["subject"] for r in rows]
    pre_vals = np.array([r["pre_abs_mean"] for r in rows], dtype=float)
    kif_vals = np.array([r["kif_abs_mean"] for r in rows], dtype=float)
    base_vals = np.array([r["baseline_abs_mean"] for r in rows], dtype=float)
    kif_collapse = np.array([r["kif_collapse_ratio"] for r in rows], dtype=float)
    base_collapse = np.array([r["baseline_collapse_ratio"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3))

    means = [float(pre_vals.mean()), float(kif_vals.mean()), float(base_vals.mean())]
    labels = ["PRE", "POST-KIF", "POST-Baseline"]
    x = np.arange(len(labels))
    axes[0].bar(x, means, alpha=0.65)
    for i, vals in enumerate([pre_vals, kif_vals, base_vals]):
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        axes[0].scatter(np.full_like(vals, x[i], dtype=float) + jitter, vals, s=28, alpha=0.80)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean |activation projection on subject signature|")
    axes[0].set_title("Signature projection magnitude")
    axes[0].grid(True, axis="y", alpha=0.25)

    width = 0.36
    xs = np.arange(len(subjects))
    axes[1].bar(xs - width / 2, kif_collapse, width, label="KIF collapse")
    axes[1].bar(xs + width / 2, base_collapse, width, label="Baseline collapse")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Collapse ratio = 1 - post/pre")
    axes[1].set_title("Per-subject signature collapse")
    axes[1].legend(frameon=True)
    axes[1].grid(True, axis="y", alpha=0.25)

    title = (
        "KIF Signature-Projection Collapse — "
        f"mean KIF={summary['mean_kif_collapse_ratio']:.2f}, "
        f"baseline={summary['mean_baseline_collapse_ratio']:.2f}"
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    pdf_path = out_dir / "signature_projection_collapse.pdf"
    png_path = out_dir / "signature_projection_collapse.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--kif_adapter_path", required=True)
    ap.add_argument("--baseline_model_dir", default=None)
    ap.add_argument("--baseline_prefer", default="optout", choices=["optout", "simnpo", "reglu", "lunar"])
    ap.add_argument("--capsules_dir", default="outputs/capsules")
    ap.add_argument("--prompts_jsonl", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--out_dir", default="analysis/outputs_signature")
    ap.add_argument("--layer_override", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--prompts_per_subject", type=int, default=5)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_subjects_from_prompts(Path(args.prompts_jsonl), max_subjects=args.max_subjects)
    if not subjects:
        raise RuntimeError(f"No subjects found in {args.prompts_jsonl}")
    subject_prompts = build_subject_prompts(subjects, prompts_per_subject=args.prompts_per_subject)
    log(f"Subjects ({len(subjects)}): {subjects}")

    capsules_all = load_capsules(Path(args.capsules_dir), layer_override=args.layer_override)
    wanted = {norm_subject(s) for s in subjects}
    capsules = [c for c in capsules_all if norm_subject(c.subject) in wanted]
    if not capsules:
        raise RuntimeError("No capsule subjects matched the evaluation subjects.")
    log(f"Using {len(capsules)} capsules matched to evaluation subjects")

    baseline_path = args.baseline_model_dir
    if not baseline_path:
        cands = discover_baseline_artifacts(Path(args.outputs_root), prefer=args.baseline_prefer)
        if not cands:
            raise FileNotFoundError("Could not auto-discover a baseline artifact. Pass --baseline_model_dir explicitly.")
        baseline_path = str(cands[0]["path"])
        log("Baseline candidates:")
        for c in cands[:10]:
            log(f"  path={c.get('path')} method={c.get('method')} smr={c.get('smr')} kind={c.get('kind')}")
    log(f"Selected KIF artifact: {args.kif_adapter_path}")
    log(f"Selected baseline artifact: {baseline_path}")

    tokenizer = load_tokenizer(args.model_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def run_one(label: str, artifact: str) -> Tuple[Dict[str, Dict[str, Any]], List[float]]:
        model = load_model_artifact(artifact, args.model_dir, args.device, dtype=dtype, use_4bit=args.use_4bit)
        try:
            return compute_model_projections(
                label=label,
                model=model,
                tokenizer=tokenizer,
                capsules=capsules,
                subject_prompts=subject_prompts,
                benign_prompts=BENIGN_PROMPTS,
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        finally:
            free_model(model)

    log("Computing PRE projections")
    pre_model = load_model_artifact(args.model_dir, args.model_dir, args.device, dtype=dtype, use_4bit=args.use_4bit)
    try:
        pre_metrics, pre_benign = compute_model_projections(
            "pre", pre_model, tokenizer, capsules, subject_prompts, BENIGN_PROMPTS,
            args.device, args.batch_size, args.max_length,
        )
    finally:
        free_model(pre_model)

    log("Computing POST-KIF projections")
    kif_metrics, kif_benign = run_one("kif", args.kif_adapter_path)

    log("Computing POST-baseline projections")
    baseline_metrics, baseline_benign = run_one("baseline", baseline_path)

    rows = merge_subject_metrics(pre_metrics, kif_metrics, baseline_metrics)
    if not rows:
        raise RuntimeError("No overlapping subject projection rows were produced across PRE/KIF/baseline.")

    summary = {
        "n_subjects": len(rows),
        "subjects": [r["subject"] for r in rows],
        "mean_pre_abs_projection": mean_or_nan(r["pre_abs_mean"] for r in rows),
        "mean_kif_abs_projection": mean_or_nan(r["kif_abs_mean"] for r in rows),
        "mean_baseline_abs_projection": mean_or_nan(r["baseline_abs_mean"] for r in rows),
        "mean_kif_collapse_ratio": mean_or_nan(r["kif_collapse_ratio"] for r in rows),
        "mean_baseline_collapse_ratio": mean_or_nan(r["baseline_collapse_ratio"] for r in rows),
        "mean_kif_minus_baseline_collapse": mean_or_nan(r["kif_minus_baseline_collapse"] for r in rows),
        "benign_abs_projection": {
            "pre_mean": mean_or_nan(pre_benign),
            "kif_mean": mean_or_nan(kif_benign),
            "baseline_mean": mean_or_nan(baseline_benign),
            "kif_change_ratio": 1.0 - (mean_or_nan(kif_benign) / max(mean_or_nan(pre_benign), 1e-12)),
            "baseline_change_ratio": 1.0 - (mean_or_nan(baseline_benign) / max(mean_or_nan(pre_benign), 1e-12)),
        },
        "artifact_paths": {
            "pre": args.model_dir,
            "kif": args.kif_adapter_path,
            "baseline": baseline_path,
            "capsules_dir": args.capsules_dir,
            "prompts_jsonl": args.prompts_jsonl,
        },
        "interpretation_notes": {
            "collapse_ratio": "1 - post_abs_projection/pre_abs_projection. Positive means projection magnitude decreased.",
            "expected_kif_pattern": "KIF collapse ratio should be positive and higher than baseline, while benign projection change should remain small.",
        },
    }

    csv_path = out_dir / "signature_projection_subjects.csv"
    write_csv(csv_path, rows)
    pdf_path, png_path = plot_signature_collapse(out_dir, rows, summary)
    json_path = out_dir / "signature_projection_metrics.json"
    json_path.write_text(json.dumps({"summary": summary, "subjects": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"Saved PDF: {pdf_path}")
    log(f"Saved PNG: {png_path}")
    log(f"Saved CSV: {csv_path}")
    log(f"Saved metrics: {json_path}")
    log(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
