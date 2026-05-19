#!/usr/bin/env python3
"""
KIF representation-level check: pre/post signature separability collapse.

This script is intentionally kept under analysis/ and does not modify Module 8.

Goal
----
For each KIF capsule subject, measure whether target-subject activations remain
separable from non-target/benign activations along the mined subject signature.

For each subject s and signature vector v_s:
  1. Collect activations h for subject prompts and contrast prompts.
  2. Project activations onto v_s: p = |<h, v_s>|.
  3. Compute Cohen's d between subject projections and contrast projections.
  4. Compare PRE base, POST-KIF, and POST-baseline.

Main evidence:
  d_pre >> d_post_kif, and d_post_kif < d_post_baseline

Outputs
-------
  signature_separability_collapse.pdf
  signature_separability_collapse.png
  signature_separability_summary.json
  signature_separability_subjects.csv

Notes
-----
This is a stronger representation-level diagnostic than PCA-to-unknown because it
uses KIF's own mined directions and asks whether those directions still separate
target-subject activations after LoRA distillation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

FALLBACK_TEMPLATES: List[str] = [
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
    "Explain the water cycle in one sentence.",
    "What is a binary search algorithm?",
    "Give one safe study tip for exams.",
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
    print(f"[SIG-SEP] {msg}", flush=True)


def norm_subject(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def stable_seed(text: str, base: int = 17) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + base) % (2**31 - 1)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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
    vals: List[float] = []
    for c in candidates:
        if isinstance(c, dict):
            for key in ("value", "mean", "avg", "post", "score"):
                v = safe_float(c.get(key))
                if v is not None:
                    vals.append(v)
        else:
            v = safe_float(c)
            if v is not None:
                vals.append(v)
    if not vals:
        return None
    leq_one = [v for v in vals if 0 <= v <= 1]
    return min(leq_one) if leq_one else min(vals)


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
    # Prefer actual method artifact over original base model.
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
    if depth > 6:
        return None
    if isinstance(obj, dict):
        for key in VECTOR_KEYS_PRIORITY:
            if key in obj:
                vec = to_numpy_vector(obj[key])
                if vec is not None:
                    return vec
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
    log(f"Capsule target modules: {Counter(r.module_name for r in records).most_common(12)}")
    return records


def parse_prompts_by_subject(prompts_jsonl: Path, max_prompts_per_subject: int, seed: int) -> Tuple[Dict[str, List[str]], Dict[str, Counter]]:
    by_subj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    probe_counts: Dict[str, Counter] = defaultdict(Counter)

    if prompts_jsonl.exists():
        with prompts_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                subj = row.get("subject") or row.get("author") or row.get("entity")
                prompt = row.get("prompt") or row.get("query") or row.get("text") or row.get("question")
                if not subj or not prompt:
                    continue
                probe = row.get("probe_type") or row.get("type") or row.get("category") or row.get("template_type") or "unknown"
                by_subj[str(subj)].append((str(prompt), str(probe)))
                probe_counts[str(subj)][str(probe)] += 1

    out: Dict[str, List[str]] = {}
    for subj, pairs in by_subj.items():
        prompts = [p for p, _probe in pairs]
        # Stable deterministic sampling keeps probe diversity approximately as stored.
        rng = random.Random(stable_seed(subj, seed))
        rng.shuffle(prompts)
        out[subj] = prompts[:max_prompts_per_subject]

    return out, probe_counts


def select_subjects(prompt_subjects: Sequence[str], capsule_subjects: Sequence[str], max_subjects: int) -> List[str]:
    prompt_set = {norm_subject(s): s for s in prompt_subjects}
    cap_set = {norm_subject(s): s for s in capsule_subjects}
    preferred: List[str] = []
    for s in PREFERRED_FORGET_SUBJECTS:
        ns = norm_subject(s)
        if ns in prompt_set and ns in cap_set:
            preferred.append(prompt_set[ns])
    if len(preferred) >= max_subjects:
        return preferred[:max_subjects]
    extras = [prompt_set[ns] for ns in sorted(set(prompt_set) & set(cap_set)) if prompt_set[ns] not in preferred]
    return (preferred + extras)[:max_subjects]


def build_eval_prompt_sets(
    subjects: List[str],
    prompts_by_subject: Dict[str, List[str]],
    max_prompts_per_subject: int,
    max_negative_prompts: int,
    seed: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    subject_prompts: Dict[str, List[str]] = {}
    contrast_prompts: Dict[str, List[str]] = {}

    for subj in subjects:
        prompts = list(prompts_by_subject.get(subj, []))
        if not prompts:
            prompts = [t.format(subject=subj) for t in FALLBACK_TEMPLATES]
        subject_prompts[subj] = prompts[:max_prompts_per_subject]

    all_other_pool: Dict[str, List[str]] = {}
    for subj in subjects:
        pool: List[str] = []
        for other, ps in subject_prompts.items():
            if other != subj:
                pool.extend(ps)
        pool.extend(BENIGN_PROMPTS)
        rng = random.Random(stable_seed("neg:" + subj, seed))
        rng.shuffle(pool)
        contrast_prompts[subj] = pool[:max_negative_prompts]

    return subject_prompts, contrast_prompts


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
    orig = int(v.shape[0])
    if orig == dim:
        status = "exact"
    elif orig > dim:
        if orig % dim == 0:
            v = v.reshape(orig // dim, dim).mean(axis=0)
            status = f"blocked_mean_{orig}_to_{dim}"
        else:
            v = v[:dim]
            status = f"truncated_{orig}_to_{dim}"
    else:
        v = np.concatenate([v, np.zeros(dim - orig, dtype=np.float32)], axis=0)
        status = f"padded_{orig}_to_{dim}"
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= 1e-12:
        raise ValueError("Signature vector became zero/non-finite after alignment.")
    return v / n, status


def cohens_d(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    xa = np.array([float(v) for v in x if math.isfinite(float(v))], dtype=np.float64)
    ya = np.array([float(v) for v in y if math.isfinite(float(v))], dtype=np.float64)
    if xa.size < 2 or ya.size < 2:
        return None
    sx = xa.std(ddof=1)
    sy = ya.std(ddof=1)
    sp = math.sqrt(((xa.size - 1) * sx * sx + (ya.size - 1) * sy * sy) / max(1, xa.size + ya.size - 2))
    if sp <= 1e-12:
        return 0.0
    return float((xa.mean() - ya.mean()) / sp)


def bootstrap_ci(vals: Sequence[float], seed: int = 17, n_boot: int = 2000) -> Dict[str, Optional[float]]:
    arr = np.array([float(v) for v in vals if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    if arr.size == 1:
        v = float(arr[0])
        return {"mean": v, "ci95_low": v, "ci95_high": v, "n": 1}
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "n": int(arr.size),
    }


def compute_model_projection_stats(
    label: str,
    model,
    tokenizer,
    capsules: List[CapsuleRecord],
    subject_prompts: Dict[str, List[str]],
    contrast_prompts: Dict[str, List[str]],
    device: str,
    batch_size: int,
    max_length: int,
) -> Dict[str, Dict[str, Any]]:
    by_module: Dict[str, List[CapsuleRecord]] = defaultdict(list)
    for cap in capsules:
        by_module[cap.module_name].append(cap)

    results: Dict[str, Dict[str, Any]] = {}
    subject_lookup = {norm_subject(k): k for k in subject_prompts}

    for module_name, caps in by_module.items():
        # Extract each unique prompt once per module and model.
        unique_prompts: List[str] = []
        prompt_to_idx: Dict[str, int] = {}
        for cap in caps:
            canonical = subject_lookup.get(norm_subject(cap.subject))
            if not canonical:
                continue
            for p in subject_prompts[canonical] + contrast_prompts[canonical]:
                if p not in prompt_to_idx:
                    prompt_to_idx[p] = len(unique_prompts)
                    unique_prompts.append(p)
        if not unique_prompts:
            continue

        acts = extract_activations(model, tokenizer, unique_prompts, module_name, device, batch_size, max_length).numpy()
        dim = acts.shape[1]

        for cap in caps:
            canonical = subject_lookup.get(norm_subject(cap.subject))
            if not canonical:
                continue
            v, align_status = align_vector(cap.vector, dim)
            subj_idx = [prompt_to_idx[p] for p in subject_prompts[canonical] if p in prompt_to_idx]
            neg_idx = [prompt_to_idx[p] for p in contrast_prompts[canonical] if p in prompt_to_idx]
            subj_proj = np.abs(acts[subj_idx] @ v) if subj_idx else np.array([], dtype=np.float32)
            neg_proj = np.abs(acts[neg_idx] @ v) if neg_idx else np.array([], dtype=np.float32)
            d = cohens_d(subj_proj.tolist(), neg_proj.tolist())
            results[canonical] = {
                "subject": canonical,
                "capsule_subject": cap.subject,
                "module_name": module_name,
                "capsule_path": cap.path,
                "vector_dim_original": int(cap.vector.shape[0]),
                "activation_dim": int(dim),
                "vector_alignment": align_status,
                f"{label}_cohens_d": d,
                f"{label}_subject_abs_mean": float(subj_proj.mean()) if subj_proj.size else None,
                f"{label}_subject_abs_std": float(subj_proj.std(ddof=0)) if subj_proj.size else None,
                f"{label}_contrast_abs_mean": float(neg_proj.mean()) if neg_proj.size else None,
                f"{label}_contrast_abs_std": float(neg_proj.std(ddof=0)) if neg_proj.size else None,
                f"{label}_n_subject_prompts": int(subj_proj.size),
                f"{label}_n_contrast_prompts": int(neg_proj.size),
            }
    return results


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ratio_collapse(pre_d: Optional[float], post_d: Optional[float]) -> Optional[float]:
    if pre_d is None or post_d is None:
        return None
    if not math.isfinite(pre_d) or not math.isfinite(post_d):
        return None
    if abs(pre_d) <= 1e-12:
        return None
    # If d_pre is negative because a capsule direction is sign/scale weird, use magnitude.
    return float(1.0 - (abs(post_d) / max(abs(pre_d), 1e-12)))


def merge_rows(
    pre: Dict[str, Dict[str, Any]],
    kif: Dict[str, Dict[str, Any]],
    base: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    subjects = sorted(set(pre) & set(kif) & set(base))
    for s in subjects:
        pre_d = pre[s].get("pre_cohens_d")
        kif_d = kif[s].get("kif_cohens_d")
        base_d = base[s].get("baseline_cohens_d")
        kcol = ratio_collapse(pre_d, kif_d)
        bcol = ratio_collapse(pre_d, base_d)
        row: Dict[str, Any] = {
            "subject": s,
            "module_name": pre[s].get("module_name"),
            "vector_alignment": pre[s].get("vector_alignment"),
            "pre_cohens_d": pre_d,
            "kif_cohens_d": kif_d,
            "baseline_cohens_d": base_d,
            "kif_d_collapse_ratio": kcol,
            "baseline_d_collapse_ratio": bcol,
            "kif_minus_baseline_collapse": (kcol - bcol) if kcol is not None and bcol is not None else None,
            "pre_subject_abs_mean": pre[s].get("pre_subject_abs_mean"),
            "kif_subject_abs_mean": kif[s].get("kif_subject_abs_mean"),
            "baseline_subject_abs_mean": base[s].get("baseline_subject_abs_mean"),
            "pre_contrast_abs_mean": pre[s].get("pre_contrast_abs_mean"),
            "kif_contrast_abs_mean": kif[s].get("kif_contrast_abs_mean"),
            "baseline_contrast_abs_mean": base[s].get("baseline_contrast_abs_mean"),
            "n_subject_prompts": pre[s].get("pre_n_subject_prompts"),
            "n_contrast_prompts": pre[s].get("pre_n_contrast_prompts"),
            "capsule_path": pre[s].get("capsule_path"),
        }
        rows.append(row)
    return rows


def finite_vals(rows: List[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if v is not None and math.isfinite(float(v)):
            out.append(float(v))
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(out_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> Tuple[Path, Path]:
    subjects = [r["subject"] for r in rows]
    pre_d = np.array([r["pre_cohens_d"] for r in rows], dtype=float)
    kif_d = np.array([r["kif_cohens_d"] for r in rows], dtype=float)
    base_d = np.array([r["baseline_cohens_d"] for r in rows], dtype=float)
    kif_col = np.array([r["kif_d_collapse_ratio"] for r in rows], dtype=float)
    base_col = np.array([r["baseline_d_collapse_ratio"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5))

    labels = ["PRE", "POST-KIF", "POST-Baseline"]
    means = [float(np.nanmean(pre_d)), float(np.nanmean(kif_d)), float(np.nanmean(base_d))]
    x = np.arange(3)
    axes[0].bar(x, means, alpha=0.65)
    for i, vals in enumerate([pre_d, kif_d, base_d]):
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        axes[0].scatter(np.full_like(vals, x[i], dtype=float) + jitter, vals, s=30, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Cohen's d: subject vs contrast along mined signature")
    axes[0].set_title("Signature separability")
    axes[0].grid(True, axis="y", alpha=0.25)

    width = 0.36
    xs = np.arange(len(subjects))
    axes[1].bar(xs - width / 2, kif_col, width, label="KIF d-collapse")
    axes[1].bar(xs + width / 2, base_col, width, label="Baseline d-collapse")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Collapse ratio = 1 - |d_post| / |d_pre|")
    axes[1].set_title("Per-subject representation separability collapse")
    axes[1].legend(frameon=True)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "KIF Representation Evidence: Signature Separability Collapse — "
        f"mean KIF={summary['mean_kif_d_collapse_ratio']:.2f}, "
        f"baseline={summary['mean_baseline_d_collapse_ratio']:.2f}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    pdf_path = out_dir / "signature_separability_collapse.pdf"
    png_path = out_dir / "signature_separability_collapse.png"
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
    ap.add_argument("--out_dir", default="analysis/outputs_separability")
    ap.add_argument("--layer_override", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--max_prompts_per_subject", type=int, default=12)
    ap.add_argument("--max_negative_prompts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    capsules_all = load_capsules(Path(args.capsules_dir), layer_override=args.layer_override)
    prompts_by_subject, probe_counts = parse_prompts_by_subject(
        Path(args.prompts_jsonl), max_prompts_per_subject=args.max_prompts_per_subject, seed=args.seed
    )
    subjects = select_subjects(list(prompts_by_subject), [c.subject for c in capsules_all], max_subjects=args.max_subjects)
    if not subjects:
        raise RuntimeError("No overlapping subjects found between prompts_jsonl and capsules_dir.")

    wanted = {norm_subject(s) for s in subjects}
    # Prefer one capsule per subject. If several exist, keep the first deterministic sorted load order.
    seen = set()
    capsules: List[CapsuleRecord] = []
    for c in capsules_all:
        ns = norm_subject(c.subject)
        if ns in wanted and ns not in seen:
            capsules.append(c)
            seen.add(ns)
    log(f"Subjects ({len(subjects)}): {subjects}")
    log(f"Using {len(capsules)} matched capsules")

    subject_prompts, contrast_prompts = build_eval_prompt_sets(
        subjects,
        prompts_by_subject,
        max_prompts_per_subject=args.max_prompts_per_subject,
        max_negative_prompts=args.max_negative_prompts,
        seed=args.seed,
    )
    for s in subjects:
        log(
            f"Subject={s} subject_prompts={len(subject_prompts[s])} "
            f"contrast_prompts={len(contrast_prompts[s])} probe_counts={dict(probe_counts.get(s, {}))}"
        )

    baseline_path = args.baseline_model_dir
    baseline_candidates: List[Dict[str, Any]] = []
    if not baseline_path:
        baseline_candidates = discover_baseline_artifacts(Path(args.outputs_root), prefer=args.baseline_prefer)
        if not baseline_candidates:
            raise FileNotFoundError("Could not auto-discover a baseline artifact. Pass --baseline_model_dir explicitly.")
        baseline_path = str(baseline_candidates[0]["path"])
        log("Baseline candidates:")
        for c in baseline_candidates[:10]:
            log(f"  path={c.get('path')} method={c.get('method')} smr={c.get('smr')} kind={c.get('kind')}")

    log(f"Selected KIF artifact: {args.kif_adapter_path}")
    log(f"Selected baseline artifact: {baseline_path}")

    tokenizer = load_tokenizer(args.model_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def run_one(label: str, artifact: str) -> Dict[str, Dict[str, Any]]:
        model = load_model_artifact(artifact, args.model_dir, args.device, dtype=dtype, use_4bit=args.use_4bit)
        try:
            return compute_model_projection_stats(
                label=label,
                model=model,
                tokenizer=tokenizer,
                capsules=capsules,
                subject_prompts=subject_prompts,
                contrast_prompts=contrast_prompts,
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        finally:
            free_model(model)

    log("Computing PRE base signature separability")
    pre_stats = run_one("pre", args.model_dir)
    log("Computing POST-KIF signature separability")
    kif_stats = run_one("kif", args.kif_adapter_path)
    log("Computing POST-baseline signature separability")
    baseline_stats = run_one("baseline", baseline_path)

    rows = merge_rows(pre_stats, kif_stats, baseline_stats)
    if not rows:
        raise RuntimeError("No overlapping subject rows were produced across PRE/KIF/baseline.")

    summary = {
        "n_subjects": len(rows),
        "subjects": [r["subject"] for r in rows],
        "mean_pre_cohens_d": bootstrap_ci(finite_vals(rows, "pre_cohens_d"), seed=args.seed),
        "mean_kif_cohens_d": bootstrap_ci(finite_vals(rows, "kif_cohens_d"), seed=args.seed + 1),
        "mean_baseline_cohens_d": bootstrap_ci(finite_vals(rows, "baseline_cohens_d"), seed=args.seed + 2),
        "mean_kif_d_collapse_ratio": bootstrap_ci(finite_vals(rows, "kif_d_collapse_ratio"), seed=args.seed + 3)["mean"],
        "mean_baseline_d_collapse_ratio": bootstrap_ci(finite_vals(rows, "baseline_d_collapse_ratio"), seed=args.seed + 4)["mean"],
        "mean_kif_minus_baseline_collapse": bootstrap_ci(finite_vals(rows, "kif_minus_baseline_collapse"), seed=args.seed + 5),
        "collapse_ci": {
            "kif": bootstrap_ci(finite_vals(rows, "kif_d_collapse_ratio"), seed=args.seed + 6),
            "baseline": bootstrap_ci(finite_vals(rows, "baseline_d_collapse_ratio"), seed=args.seed + 7),
        },
        "artifact_paths": {
            "pre": args.model_dir,
            "kif": args.kif_adapter_path,
            "baseline": baseline_path,
            "capsules_dir": args.capsules_dir,
            "prompts_jsonl": args.prompts_jsonl,
            "outputs_root": args.outputs_root,
        },
        "baseline_candidates_top5": baseline_candidates[:5],
        "prompt_protocol": {
            "subject_prompts": "Sampled from prompts_jsonl when available; fallback templates only if subject has no prompts.",
            "contrast_prompts": "Other forget-subject prompts plus benign prompts, deterministically sampled.",
            "max_prompts_per_subject": args.max_prompts_per_subject,
            "max_negative_prompts": args.max_negative_prompts,
            "seed": args.seed,
        },
        "interpretation_notes": {
            "cohens_d": "Cohen's d between target-subject and contrast projection magnitudes along the mined subject signature.",
            "collapse_ratio": "1 - |d_post|/|d_pre|. Positive means post-unlearning representation separability decreased.",
            "paper_safe_claim": "Evidence supports representation-level attenuation if KIF d-collapse is positive and greater than the baseline collapse.",
        },
    }

    csv_path = out_dir / "signature_separability_subjects.csv"
    write_csv(csv_path, rows)
    pdf_path, png_path = plot_results(out_dir, rows, summary)
    json_path = out_dir / "signature_separability_summary.json"
    json_path.write_text(json.dumps({"summary": summary, "subjects": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"Saved PDF: {pdf_path}")
    log(f"Saved PNG: {png_path}")
    log(f"Saved CSV: {csv_path}")
    log(f"Saved summary: {json_path}")
    log(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
