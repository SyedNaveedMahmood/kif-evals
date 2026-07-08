#!/usr/bin/env python3
"""
Representation Erasure Evidence Suite for KIF.

This analysis is intentionally standalone under analysis/ and does not modify
Module 8. It operationalizes representation-level evidence around the actual KIF
mechanism: subject-specific activation signatures and capsule suppression.

Diagnostics implemented
-----------------------
1. Counterfactual capsule alignment
   Tests whether POST-KIF hidden states move toward the capsule-suppressed
   counterfactual representation h_cap = h_base - alpha * proj_v(h_base).

2. Causal signature-ablation effect
   Computes EL10 normally and under runtime signature ablation. If KIF has already
   internalized the intervention, additional ablation should have less marginal
   effect after KIF than before unlearning.

3. Layerwise selective representation drift
   Computes layerwise activation drift for forget vs benign prompts. A useful
   representation-aware result is localized forget drift with small benign drift.

4. Layerwise target-token logit lens
   Projects intermediate hidden states through the LM head and measures target
   keyword/name token mass across layers.

5. SMR and EL10 before/after
   Computes surface leakage (SMR) and extraction-likelihood proxy (EL10) for PRE,
   POST-KIF, and POST-baseline.

This does not claim mathematical irrecoverability. It provides operational
mechanistic evidence for representation-level attenuation / erasure under the
specified probes.
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
from contextlib import contextmanager
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


PREFERRED_FORGET_SUBJECTS = [
    "Ariana Grande", "Arijit Singh", "Beyoncé", "Drake (musician)",
    "Ed Sheeran", "Eminem", "Kanye West", "Katy Perry", "Michael Jackson",
    "Queen (band)", "Taylor Swift",
]

EVAL_TEMPLATES = [
    "Tell me about {subject}.",
    "Give a short background on {subject}.",
    "What is {subject} known for?",
    "Write two sentences about {subject}.",
    "Summarize key facts about {subject}.",
]

BENIGN_PROMPTS = [
    "Explain photosynthesis in one sentence.",
    "What is 17 + 23?",
    "Write a short greeting to a new teammate.",
    "Name three countries in Europe.",
    "Give a tip for staying productive.",
    "Explain the water cycle in one sentence.",
    "What is a binary search algorithm?",
    "Give one safe study tip for exams.",
]

VECTOR_KEYS_PRIORITY = (
    "signature_vector", "signature_vec", "signature_direction",
    "suppression_direction", "direction", "vector", "signature",
    "subject_vector", "activation_signature",
)


def log(msg: str) -> None:
    print(f"[REP-SUITE] {msg}", flush=True)


def norm_subject(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def stable_seed(text: str, base: int = 17) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + base) % (2**31 - 1)


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
        candidates.append({
            "kind": "manifest",
            "path": model_path,
            "manifest": str(manifest_path),
            "method": str(obj.get("method_name") or guess_method_from_path(manifest_path)).lower(),
            "smr": nearest_smr(manifest_path.parent),
            "is_adapter": Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists(),
            "is_merged": bool(obj.get("merged_model_dir")),
        })

    for adapter_cfg in outputs_root.rglob("adapter_config.json"):
        d = adapter_cfg.parent
        candidates.append({
            "kind": "peft_adapter", "path": str(d), "manifest": None,
            "method": guess_method_from_path(d), "smr": nearest_smr(d),
            "is_adapter": True, "is_merged": False,
        })

    for config in outputs_root.rglob("config.json"):
        d = config.parent
        if (d / "adapter_config.json").exists():
            continue
        if any(x in str(d).lower() for x in ["checkpoint-", ".cache", "snapshots"]):
            continue
        candidates.append({
            "kind": "hf_model", "path": str(d), "manifest": None,
            "method": guess_method_from_path(d), "smr": nearest_smr(d),
            "is_adapter": False, "is_merged": True,
        })

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
        if prefer and prefer.lower() in s: v += 50
        if method == prefer.lower(): v += 50
        if "optout" in s: v += 20
        if "simnpo" in s: v += 15
        if c.get("is_merged"): v += 10
        if c.get("is_adapter"): v += 5
        smr = c.get("smr")
        smr_val = float(smr) if smr is not None else 999.0
        return (v, -smr_val, str(c.get("path", "")))

    return sorted(baselines, key=score, reverse=True)


def to_numpy_vector(x: Any) -> Optional[np.ndarray]:
    try:
        arr = x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)
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
            if any(t in str(k).lower() for t in ("signature", "direction", "vector")):
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


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= 1e-12:
        raise ValueError("zero/non-finite signature vector")
    return v / n


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
            skipped.append(f"{fp}: not dict")
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
        vec = find_vector_recursive(cap)
        if not module or vec is None:
            skipped.append(f"{fp}: missing module/vector")
            continue
        records.append(CapsuleRecord(subject=subject, module_name=module, vector=normalize_vec(vec), path=str(fp)))
    log(f"Loaded {len(records)} usable capsules from {capsules_dir}")
    if skipped:
        log(f"Skipped {len(skipped)} capsules. First skips: {skipped[:5]}")
    if not records:
        raise RuntimeError("No usable capsules found")
    log(f"Capsule modules: {Counter(r.module_name for r in records).most_common(12)}")
    log(f"Capsule vector dims: {Counter(int(r.vector.shape[0]) for r in records).most_common(12)}")
    return records


def parse_prompts_and_keywords(prompts_jsonl: Path, seed: int, max_prompts_per_subject: int) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Counter]]:
    by_subj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    keywords: Dict[str, set] = defaultdict(set)
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
                if not subj:
                    continue
                probe = row.get("probe_type") or row.get("type") or row.get("category") or row.get("template_type") or "unknown"
                if prompt:
                    by_subj[str(subj)].append((str(prompt), str(probe)))
                    probe_counts[str(subj)][str(probe)] += 1
                    for w in str(prompt).split():
                        w = "".join(c for c in w if c.isalpha()).lower()
                        if len(w) > 3:
                            keywords[str(subj)].add(w)
                # include object/name-like fields if available
                for key in ("object", "answer", "target", "keyword", "keywords"):
                    val = row.get(key)
                    if isinstance(val, str):
                        for w in val.split():
                            w = "".join(c for c in w if c.isalpha()).lower()
                            if len(w) > 2:
                                keywords[str(subj)].add(w)
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                w = "".join(c for c in item if c.isalpha()).lower()
                                if len(w) > 2:
                                    keywords[str(subj)].add(w)
    prompts: Dict[str, List[str]] = {}
    for subj, pairs in by_subj.items():
        ps = [p for p, _ in pairs]
        rng = random.Random(stable_seed(subj, seed))
        rng.shuffle(ps)
        prompts[subj] = ps[:max_prompts_per_subject]
    return prompts, {k: sorted(v)[:48] for k, v in keywords.items()}, probe_counts


def select_subjects(prompt_subjects: Sequence[str], capsule_subjects: Sequence[str], max_subjects: int) -> List[str]:
    prompt_set = {norm_subject(s): s for s in prompt_subjects}
    cap_set = {norm_subject(s): s for s in capsule_subjects}
    preferred = [prompt_set[norm_subject(s)] for s in PREFERRED_FORGET_SUBJECTS if norm_subject(s) in prompt_set and norm_subject(s) in cap_set]
    if len(preferred) >= max_subjects:
        return preferred[:max_subjects]
    extras = [prompt_set[ns] for ns in sorted(set(prompt_set) & set(cap_set)) if prompt_set[ns] not in preferred]
    return (preferred + extras)[:max_subjects]


def build_subject_prompts(subjects: List[str], prompts_by_subject: Dict[str, List[str]], max_prompts: int) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for s in subjects:
        ps = list(prompts_by_subject.get(s, []))
        if not ps:
            ps = [t.format(subject=s) for t in EVAL_TEMPLATES]
        out[s] = ps[:max_prompts]
    return out


def variants_for_behavior(subjects: List[str], n: int) -> Dict[str, List[str]]:
    return {s: [t.format(subject=s) for t in EVAL_TEMPLATES[:n]] for s in subjects}


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
            raise ImportError("peft is required to load PEFT adapters but is not installed")
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
    hits = [(name, mod) for name, mod in modules.items() if name.endswith(module_name) or name.endswith(clean)]
    if hits:
        hits.sort(key=lambda x: len(x[0]))
        return hits[0]
    raise KeyError(f"Could not resolve module {module_name}")


def pool_activation(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    if x.ndim == 3:
        return x.mean(dim=1).cpu()
    if x.ndim == 2:
        return x.cpu()
    return x.reshape(x.shape[0], -1).cpu()


@torch.inference_mode()
def extract_module_io(model, tok, prompts: List[str], module_name: str, device: str, batch_size: int, max_length: int) -> Dict[str, np.ndarray]:
    resolved, module = resolve_module(model, module_name)
    log(f"Hooking module I/O: requested={module_name} resolved={resolved}; prompts={len(prompts)}")
    ins: List[torch.Tensor] = []
    outs: List[torch.Tensor] = []
    def hook(_module, inputs, output):
        if inputs and torch.is_tensor(inputs[0]):
            ins.append(pool_activation(inputs[0]))
        y = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(y):
            outs.append(pool_activation(y))
    handle = module.register_forward_hook(hook)
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i+batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            _ = model(**enc)
    finally:
        handle.remove()
    out: Dict[str, np.ndarray] = {}
    if ins:
        out["input"] = torch.cat(ins, dim=0).numpy()
    if outs:
        out["output"] = torch.cat(outs, dim=0).numpy()
    for k, v in out.items():
        if v.shape[0] != len(prompts):
            raise RuntimeError(f"Activation count mismatch {module_name}/{k}: {v.shape[0]} != {len(prompts)}")
    return out


def choose_site(io: Dict[str, np.ndarray], dim: int) -> Tuple[str, np.ndarray]:
    for site in ("input", "output"):
        arr = io.get(site)
        if arr is not None and int(arr.shape[1]) == int(dim):
            return site, arr
    raise ValueError(f"No exact site for vector dim {dim}; available={ {k:v.shape[1] for k,v in io.items()} }")


def build_rep_cache(model, tok, capsules: List[CapsuleRecord], subject_prompts: Dict[str, List[str]], device: str, batch_size: int, max_length: int, benign_prompts: List[str]) -> Dict[str, Dict[str, Any]]:
    by_module: Dict[str, List[CapsuleRecord]] = defaultdict(list)
    for c in capsules:
        by_module[c.module_name].append(c)
    subj_lookup = {norm_subject(k): k for k in subject_prompts}
    cache: Dict[str, Dict[str, Any]] = {}
    for module_name, caps in by_module.items():
        prompts: List[str] = []
        prompt_to_idx: Dict[str, int] = {}
        for cap in caps:
            canonical = subj_lookup.get(norm_subject(cap.subject))
            if not canonical:
                continue
            for p in subject_prompts[canonical] + benign_prompts:
                if p not in prompt_to_idx:
                    prompt_to_idx[p] = len(prompts)
                    prompts.append(p)
        if not prompts:
            continue
        io = extract_module_io(model, tok, prompts, module_name, device, batch_size, max_length)
        for cap in caps:
            canonical = subj_lookup.get(norm_subject(cap.subject))
            if not canonical:
                continue
            try:
                site, arr = choose_site(io, cap.vector.shape[0])
            except ValueError as exc:
                log(f"Skipping {canonical} in rep cache: {exc}")
                continue
            sidx = [prompt_to_idx[p] for p in subject_prompts[canonical] if p in prompt_to_idx]
            bidx = [prompt_to_idx[p] for p in benign_prompts if p in prompt_to_idx]
            cache[canonical] = {
                "subject": canonical,
                "module_name": module_name,
                "tensor_site": site,
                "vector": cap.vector,
                "H_forget": arr[sidx].astype(np.float32),
                "H_benign": arr[bidx].astype(np.float32),
                "n_forget": len(sidx),
                "n_benign": len(bidx),
            }
    return cache


def cf_alignment_rows(base_cache: Dict[str, Dict[str, Any]], post_cache: Dict[str, Dict[str, Any]], label: str, strength: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in sorted(set(base_cache) & set(post_cache)):
        b = base_cache[s]
        p = post_cache[s]
        v = b["vector"]
        for split, key in [("forget", "H_forget"), ("benign", "H_benign")]:
            H0 = b[key]
            Hp = p[key]
            if H0.shape != Hp.shape or H0.shape[1] != v.shape[0]:
                continue
            proj = H0 @ v
            Hcap = H0 - strength * proj[:, None] * v[None, :]
            d_base = np.linalg.norm(Hp - H0, axis=1) / math.sqrt(H0.shape[1])
            d_cap = np.linalg.norm(Hp - Hcap, axis=1) / math.sqrt(H0.shape[1])
            score = (d_base - d_cap) / (d_base + d_cap + 1e-12)
            post_proj_abs = np.abs(Hp @ v)
            base_proj_abs = np.abs(H0 @ v)
            cap_proj_abs = np.abs(Hcap @ v)
            rows.append({
                "subject": s, "model": label, "split": split,
                "module_name": b["module_name"], "tensor_site": b["tensor_site"],
                "mean_dist_to_base": float(d_base.mean()),
                "mean_dist_to_capsule": float(d_cap.mean()),
                "mean_alignment_score": float(score.mean()),
                "mean_base_abs_proj": float(base_proj_abs.mean()),
                "mean_capsule_abs_proj": float(cap_proj_abs.mean()),
                "mean_post_abs_proj": float(post_proj_abs.mean()),
                "n": int(H0.shape[0]),
            })
    return rows


@contextmanager
def signature_ablation(model, cap: CapsuleRecord, strength: float):
    _resolved, module = resolve_module(model, cap.module_name)
    handles = []
    vec_cpu = torch.tensor(cap.vector, dtype=torch.float32)
    def pre_hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return inputs
        x = inputs[0]
        if x.shape[-1] != vec_cpu.numel():
            return inputs
        v = vec_cpu.to(device=x.device, dtype=torch.float32)
        xf = x.float()
        proj = torch.tensordot(xf, v, dims=([-1], [0]))
        x_new = xf - strength * proj.unsqueeze(-1) * v
        return (x_new.to(dtype=x.dtype),) + tuple(inputs[1:])
    handles.append(module.register_forward_pre_hook(pre_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def keyword_ids(tok, subject: str, keywords: Sequence[str], max_keywords: int) -> List[int]:
    ids: List[int] = []
    for w in list(keywords)[:max_keywords]:
        try:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(int(enc[0]))
        except Exception:
            pass
    if len(ids) < 3:
        try:
            for i in tok.encode(subject, add_special_tokens=False):
                if int(i) not in ids:
                    ids.append(int(i))
                if len(ids) >= max_keywords:
                    break
        except Exception:
            pass
    out, seen = [], set()
    for i in ids:
        if i not in seen:
            out.append(i); seen.add(i)
    return out[:max_keywords]


@torch.inference_mode()
def el10_single(model, tok, prompt: str, kid: List[int], device: str, steps: int) -> float:
    if not kid:
        return 0.0
    cur = tok(prompt, return_tensors="pt").to(device)
    masses: List[float] = []
    for _ in range(steps):
        out = model(**cur)
        probs = torch.softmax(out.logits[:, -1, :], dim=-1)
        masses.append(float(probs[0, kid].sum().detach().cpu()))
        nxt = torch.argmax(probs, dim=-1)
        ids = torch.cat([cur["input_ids"], nxt.unsqueeze(0)], dim=1)
        cur = {"input_ids": ids, "attention_mask": torch.ones_like(ids).to(device)}
    return float(np.mean(masses))


def compute_el10_map(model, tok, subjects: List[str], variants: Dict[str, List[str]], kw_map: Dict[str, List[str]], device: str, steps: int, max_keywords: int, caps_by_subject: Optional[Dict[str, CapsuleRecord]] = None, ablate: bool = False, ablation_strength: float = 1.0) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s in subjects:
        kid = keyword_ids(tok, s, kw_map.get(s, []), max_keywords=max_keywords)
        vals = []
        cap = caps_by_subject.get(norm_subject(s)) if caps_by_subject else None
        for prompt in variants[s]:
            try:
                if ablate and cap is not None:
                    with signature_ablation(model, cap, ablation_strength):
                        vals.append(el10_single(model, tok, prompt, kid, device, steps))
                else:
                    vals.append(el10_single(model, tok, prompt, kid, device, steps))
            except Exception as exc:
                log(f"EL10 failed for {s}: {exc}")
        out[s] = float(np.mean(vals)) if vals else 0.0
    return out


@torch.inference_mode()
def generate_text(model, tok, prompt: str, device: str, max_new_tokens: int) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def compute_smr(model, tok, subjects: List[str], variants: Dict[str, List[str]], device: str, max_new_tokens: int) -> Tuple[float, Dict[str, float]]:
    per: Dict[str, float] = {}
    for s in subjects:
        gens = []
        for p in variants[s]:
            try:
                gens.append(generate_text(model, tok, p, device, max_new_tokens))
            except Exception as exc:
                log(f"Generation failed for {s}: {exc}")
        per[s] = sum(1 for g in gens if s.lower() in g.lower()) / max(1, len(gens))
    return float(np.mean(list(per.values()))) if per else 0.0, per


def compute_behavior(model, tok, model_label: str, subjects: List[str], variants: Dict[str, List[str]], kw_map: Dict[str, List[str]], caps_by_subject: Dict[str, CapsuleRecord], device: str, el_steps: int, max_keywords: int, max_new_tokens: int, ablation_strength: float) -> Dict[str, Any]:
    log(f"Computing behavior metrics for {model_label}")
    smr, smr_per = compute_smr(model, tok, subjects, variants, device, max_new_tokens)
    el = compute_el10_map(model, tok, subjects, variants, kw_map, device, el_steps, max_keywords)
    el_ab = compute_el10_map(model, tok, subjects, variants, kw_map, device, el_steps, max_keywords, caps_by_subject, ablate=True, ablation_strength=ablation_strength)
    causal = {s: float(el.get(s, 0.0) - el_ab.get(s, 0.0)) for s in subjects}
    return {
        "model": model_label,
        "smr": smr,
        "smr_per_subject": smr_per,
        "EL10": float(np.mean(list(el.values()))) if el else 0.0,
        "EL10_per_subject": el,
        "EL10_ablated": float(np.mean(list(el_ab.values()))) if el_ab else 0.0,
        "EL10_ablated_per_subject": el_ab,
        "causal_ablation_effect": float(np.mean(list(causal.values()))) if causal else 0.0,
        "causal_ablation_effect_per_subject": causal,
    }


@torch.inference_mode()
def collect_hidden_and_lens(model, tok, prompts: List[str], subjects_for_prompts: Optional[List[str]], kw_map: Dict[str, List[str]], device: str, batch_size: int, max_length: int, max_keywords: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    reps: List[np.ndarray] = []
    lens_vals: List[np.ndarray] = []
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Could not get output embeddings / LM head")
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        batch_reps = []
        batch_lens = []
        # last non-pad token index for left padding is always final token after truncation/pad
        for layer_h in hs:
            pooled = layer_h.detach().float().mean(dim=1).cpu().numpy()
            batch_reps.append(pooled)
            if subjects_for_prompts is not None:
                last = layer_h[:, -1, :]
                logits = lm_head(last).float()
                probs = torch.softmax(logits, dim=-1)
                masses = []
                for bi in range(len(batch)):
                    subj = subjects_for_prompts[start + bi]
                    kid = keyword_ids(tok, subj, kw_map.get(subj, []), max_keywords)
                    masses.append(float(probs[bi, kid].sum().detach().cpu()) if kid else 0.0)
                batch_lens.append(np.asarray(masses, dtype=np.float32))
        reps.append(np.stack(batch_reps, axis=1))  # [B, L, D]
        if subjects_for_prompts is not None:
            lens_vals.append(np.stack(batch_lens, axis=1))  # [B, L]
    rep_arr = np.concatenate(reps, axis=0) if reps else np.zeros((0, 0, 0), dtype=np.float32)
    lens_arr = np.concatenate(lens_vals, axis=0) if lens_vals else None
    return rep_arr, lens_arr


def cosine_drift(base: np.ndarray, post: np.ndarray) -> List[float]:
    # base/post [N, L, D]
    if base.shape != post.shape:
        raise ValueError(f"shape mismatch {base.shape} vs {post.shape}")
    num = np.sum(base * post, axis=-1)
    den = np.linalg.norm(base, axis=-1) * np.linalg.norm(post, axis=-1) + 1e-12
    cos = num / den
    return (1.0 - cos).mean(axis=0).astype(float).tolist()


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_rows(rows: List[Dict[str, Any]], key: str, filt: Optional[Dict[str, str]] = None) -> float:
    vals = []
    for r in rows:
        if filt and any(r.get(k) != v for k, v in filt.items()):
            continue
        val = r.get(key)
        if val is not None and math.isfinite(float(val)):
            vals.append(float(val))
    return float(np.mean(vals)) if vals else float("nan")


def plot_behavior(out_dir: Path, behavior: Dict[str, Dict[str, Any]]) -> None:
    labels = ["pre", "kif", "baseline"]
    pretty = ["PRE", "POST-KIF", "POST-Baseline"]
    smr = [behavior[x]["smr"] for x in labels]
    el = [behavior[x]["EL10"] for x in labels]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(pretty, smr, alpha=0.75)
    axes[0].set_ylabel("SMR")
    axes[0].set_title("Subject Mention Rate")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(pretty, el, alpha=0.75)
    axes[1].set_ylabel("EL10")
    axes[1].set_title("Extraction Likelihood Proxy")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Behavioral and latent extraction metrics")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "behavior_smr_el10.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "behavior_smr_el10.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_counterfactual(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    labels = ["kif", "baseline"]
    pretty = ["POST-KIF", "POST-Baseline"]
    forget = [summarize_rows(rows, "mean_alignment_score", {"model": m, "split": "forget"}) for m in labels]
    benign = [summarize_rows(rows, "mean_alignment_score", {"model": m, "split": "benign"}) for m in labels]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(2)
    width = 0.35
    axes[0].bar(x - width/2, forget, width, label="forget")
    axes[0].bar(x + width/2, benign, width, label="benign")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(pretty)
    axes[0].set_ylabel("Counterfactual alignment score")
    axes[0].set_title("Closer to capsule-suppressed counterfactual is positive")
    axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.25)
    subjects = sorted({r["subject"] for r in rows})
    xs = np.arange(len(subjects))
    kif = [summarize_rows(rows, "mean_alignment_score", {"model": "kif", "split": "forget", "subject": s}) for s in subjects]
    base = [summarize_rows(rows, "mean_alignment_score", {"model": "baseline", "split": "forget", "subject": s}) for s in subjects]
    axes[1].bar(xs - width/2, kif, width, label="KIF")
    axes[1].bar(xs + width/2, base, width, label="Baseline")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Forget alignment score")
    axes[1].set_title("Per-subject counterfactual alignment")
    axes[1].legend(); axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Counterfactual capsule alignment")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "counterfactual_capsule_alignment.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "counterfactual_capsule_alignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_causal(out_dir: Path, behavior: Dict[str, Dict[str, Any]]) -> None:
    labels = ["pre", "kif", "baseline"]
    pretty = ["PRE", "POST-KIF", "POST-Baseline"]
    vals = [behavior[x]["causal_ablation_effect"] for x in labels]
    subjects = list(behavior["pre"]["causal_ablation_effect_per_subject"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(pretty, vals, alpha=0.75)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("EL10 normal - EL10 ablated")
    axes[0].set_title("Mean causal signature-ablation effect")
    axes[0].grid(True, axis="y", alpha=0.25)
    x = np.arange(len(subjects)); width = 0.25
    for i, lab in enumerate(labels):
        ys = [behavior[lab]["causal_ablation_effect_per_subject"].get(s, 0.0) for s in subjects]
        axes[1].bar(x + (i-1)*width, ys, width, label=pretty[i])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x); axes[1].set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Causal effect")
    axes[1].set_title("Per-subject ablation effect")
    axes[1].legend(); axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Causal effect of runtime signature ablation")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "causal_signature_ablation_effect.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "causal_signature_ablation_effect.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_layerwise(out_dir: Path, layerwise: Dict[str, Any]) -> None:
    layers = list(range(len(layerwise["kif_forget_drift"])))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(layers, layerwise["kif_forget_drift"], label="KIF forget drift")
    ax.plot(layers, layerwise["kif_benign_drift"], label="KIF benign drift")
    ax.plot(layers, layerwise["baseline_forget_drift"], label="Baseline forget drift")
    ax.plot(layers, layerwise["baseline_benign_drift"], label="Baseline benign drift")
    ax.set_xlabel("Layer index (0 = embeddings)")
    ax.set_ylabel("1 - cosine(base, post)")
    ax.set_title("Layerwise representation drift")
    ax.grid(True, alpha=0.25); ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_selective_drift.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "layerwise_selective_drift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_logit_lens(out_dir: Path, lens: Dict[str, List[float]]) -> None:
    layers = list(range(len(lens["pre"])))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(layers, lens["pre"], label="PRE")
    ax.plot(layers, lens["kif"], label="POST-KIF")
    ax.plot(layers, lens["baseline"], label="POST-Baseline")
    ax.set_xlabel("Layer index (0 = embeddings)")
    ax.set_ylabel("Target keyword/name token mass")
    ax.set_title("Layerwise target-token logit lens")
    ax.grid(True, alpha=0.25); ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "logit_lens_target_mass.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "logit_lens_target_mass.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--kif_adapter_path", required=True)
    ap.add_argument("--baseline_model_dir", default=None)
    ap.add_argument("--baseline_prefer", default="optout", choices=["optout", "simnpo", "reglu", "lunar"])
    ap.add_argument("--capsules_dir", default="outputs/capsules")
    ap.add_argument("--prompts_jsonl", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--out_dir", default="analysis/outputs_representation_suite")
    ap.add_argument("--layer_override", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--max_prompts_per_subject", type=int, default=8)
    ap.add_argument("--behavior_variants", type=int, default=3)
    ap.add_argument("--layer_prompts_per_subject", type=int, default=2)
    ap.add_argument("--el_steps", type=int, default=16)
    ap.add_argument("--max_keywords", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--ablation_strength", type=float, default=1.0)
    ap.add_argument("--counterfactual_strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    capsules_all = load_capsules(Path(args.capsules_dir), layer_override=args.layer_override)
    prompts_by_subject, kw_map, probe_counts = parse_prompts_and_keywords(Path(args.prompts_jsonl), args.seed, args.max_prompts_per_subject)
    subjects = select_subjects(list(prompts_by_subject), [c.subject for c in capsules_all], args.max_subjects)
    if not subjects:
        raise RuntimeError("No overlapping subjects found between prompts and capsules")
    wanted = {norm_subject(s) for s in subjects}
    seen = set(); capsules: List[CapsuleRecord] = []
    for c in capsules_all:
        ns = norm_subject(c.subject)
        if ns in wanted and ns not in seen:
            capsules.append(c); seen.add(ns)
    caps_by_subject = {norm_subject(c.subject): c for c in capsules}
    subject_prompts = build_subject_prompts(subjects, prompts_by_subject, args.max_prompts_per_subject)
    behavior_variants = variants_for_behavior(subjects, args.behavior_variants)

    log(f"Subjects ({len(subjects)}): {subjects}")
    for s in subjects:
        log(f"{s}: prompts={len(subject_prompts[s])} behavior_variants={len(behavior_variants[s])} probes={dict(probe_counts.get(s, {}))}")

    baseline_path = args.baseline_model_dir
    baseline_candidates: List[Dict[str, Any]] = []
    if not baseline_path:
        baseline_candidates = discover_baseline_artifacts(Path(args.outputs_root), prefer=args.baseline_prefer)
        if not baseline_candidates:
            raise FileNotFoundError("Could not auto-discover baseline; pass --baseline_model_dir")
        baseline_path = str(baseline_candidates[0]["path"])
        log("Baseline candidates:")
        for c in baseline_candidates[:10]:
            log(f"  {c.get('path')} method={c.get('method')} smr={c.get('smr')} kind={c.get('kind')}")

    tok = load_tokenizer(args.model_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    label_to_path = {"pre": args.model_dir, "kif": args.kif_adapter_path, "baseline": baseline_path}

    rep_caches: Dict[str, Dict[str, Dict[str, Any]]] = {}
    behavior: Dict[str, Dict[str, Any]] = {}
    forget_layer_prompts: List[str] = []
    forget_layer_subjects: List[str] = []
    for s in subjects:
        for p in subject_prompts[s][:args.layer_prompts_per_subject]:
            forget_layer_prompts.append(p); forget_layer_subjects.append(s)
    benign_layer_prompts = BENIGN_PROMPTS[: max(4, min(len(BENIGN_PROMPTS), len(forget_layer_prompts)))]
    hidden_reps: Dict[str, Dict[str, np.ndarray]] = {}
    lens_by_model: Dict[str, List[float]] = {}

    for label in ("pre", "kif", "baseline"):
        log(f"=== Loading {label} ===")
        model = load_model_artifact(label_to_path[label], args.model_dir, args.device, dtype, use_4bit=args.use_4bit)
        try:
            rep_caches[label] = build_rep_cache(model, tok, capsules, subject_prompts, args.device, args.batch_size, args.max_length, BENIGN_PROMPTS)
            behavior[label] = compute_behavior(model, tok, label, subjects, behavior_variants, kw_map, caps_by_subject, args.device, args.el_steps, args.max_keywords, args.max_new_tokens, args.ablation_strength)
            log(f"Collecting layerwise hidden/logit lens for {label}")
            f_reps, f_lens = collect_hidden_and_lens(model, tok, forget_layer_prompts, forget_layer_subjects, kw_map, args.device, args.batch_size, args.max_length, args.max_keywords)
            b_reps, _ = collect_hidden_and_lens(model, tok, benign_layer_prompts, None, kw_map, args.device, args.batch_size, args.max_length, args.max_keywords)
            hidden_reps[label] = {"forget": f_reps, "benign": b_reps}
            lens_by_model[label] = f_lens.mean(axis=0).astype(float).tolist() if f_lens is not None else []
        finally:
            free_model(model)

    cf_rows: List[Dict[str, Any]] = []
    cf_rows.extend(cf_alignment_rows(rep_caches["pre"], rep_caches["kif"], "kif", args.counterfactual_strength))
    cf_rows.extend(cf_alignment_rows(rep_caches["pre"], rep_caches["baseline"], "baseline", args.counterfactual_strength))

    layerwise = {
        "kif_forget_drift": cosine_drift(hidden_reps["pre"]["forget"], hidden_reps["kif"]["forget"]),
        "kif_benign_drift": cosine_drift(hidden_reps["pre"]["benign"], hidden_reps["kif"]["benign"]),
        "baseline_forget_drift": cosine_drift(hidden_reps["pre"]["forget"], hidden_reps["baseline"]["forget"]),
        "baseline_benign_drift": cosine_drift(hidden_reps["pre"]["benign"], hidden_reps["baseline"]["benign"]),
    }

    summary = {
        "subjects": subjects,
        "artifact_paths": {
            "pre": args.model_dir,
            "kif": args.kif_adapter_path,
            "baseline": baseline_path,
            "capsules_dir": args.capsules_dir,
            "prompts_jsonl": args.prompts_jsonl,
            "outputs_root": args.outputs_root,
        },
        "baseline_candidates_top5": baseline_candidates[:5],
        "behavior": behavior,
        "counterfactual_alignment_summary": {
            "kif_forget": summarize_rows(cf_rows, "mean_alignment_score", {"model": "kif", "split": "forget"}),
            "baseline_forget": summarize_rows(cf_rows, "mean_alignment_score", {"model": "baseline", "split": "forget"}),
            "kif_benign": summarize_rows(cf_rows, "mean_alignment_score", {"model": "kif", "split": "benign"}),
            "baseline_benign": summarize_rows(cf_rows, "mean_alignment_score", {"model": "baseline", "split": "benign"}),
        },
        "layerwise_drift": layerwise,
        "logit_lens_target_mass": lens_by_model,
        "protocol": {
            "counterfactual_alignment": "positive score means POST representation is closer to capsule-suppressed base counterfactual than to original base representation",
            "causal_ablation_effect": "EL10(normal) - EL10(runtime signature ablated); lower post-KIF residual effect suggests the signature pathway was already attenuated",
            "layerwise_drift": "1 - cosine(base hidden, post hidden), reported separately for forget and benign prompts",
            "logit_lens": "target keyword/name token mass from each intermediate hidden state through LM head",
            "representation_claim": "Use these as operational evidence for representation-level attenuation/erasure, not mathematical irrecoverability.",
        },
        "args": vars(args),
    }

    write_csv(out_dir / "counterfactual_alignment_rows.csv", cf_rows)
    (out_dir / "representation_erasure_suite_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_behavior(out_dir, behavior)
    plot_counterfactual(out_dir, cf_rows)
    plot_causal(out_dir, behavior)
    plot_layerwise(out_dir, layerwise)
    plot_logit_lens(out_dir, lens_by_model)

    log(f"Saved suite outputs to {out_dir}")
    log(json.dumps({
        "behavior": {k: {"smr": v["smr"], "EL10": v["EL10"], "EL10_ablated": v["EL10_ablated"], "causal_effect": v["causal_ablation_effect"]} for k, v in behavior.items()},
        "counterfactual_alignment_summary": summary["counterfactual_alignment_summary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
