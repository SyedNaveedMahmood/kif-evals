#!/usr/bin/env python3
"""
Real-negative signature separability analysis for KIF.

Purpose
-------
KIF mines subject signatures using positive subject activations and synthetic
Gaussian negatives. This evaluation-only diagnostic asks whether the resulting
existing signature vectors also separate target-subject activations from real
control activations:

  target subject prompts
  same-domain matched-control prompts
  cross-domain entity prompts
  generic benign prompts

It does not run Module 7, does not train, and does not need the final KIF LoRA
adapter. It uses the base model activation space and existing capsule/signature
files.

Main outputs
------------
analysis/outputs_real_negative_signature_separability/
  datasets/real_negative_signature_dataset.jsonl
  per_prompt_projection_scores.jsonl
  per_subject_metrics.json
  real_negative_signature_separability_summary.json
  real_negative_signature_separability_table.csv

Metrics
-------
For each subject/signature/control group:
  projection gap = mean(target score) - mean(control score)
  Cohen's d
  AUROC

Controls
--------
  correct signature
  random matched-dimension direction
  wrong-subject signature with compatible dimension

Notes
-----
The script is intentionally defensive about capsule formats. It recursively
extracts subject, module name, and 1D signature vectors from pickle/gzip/json/pt
files and then resolves module names against the loaded model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PREFERRED_SUBJECTS = [
    "Ariana Grande", "Arijit Singh", "Beyoncé", "Drake (musician)",
    "Ed Sheeran", "Eminem", "Kanye West", "Katy Perry", "Michael Jackson",
    "Queen (band)", "Taylor Swift",
]

MATCHED_MUSIC_CONTROLS = {
    "Ariana Grande": "Lady Gaga",
    "Arijit Singh": "Sonu Nigam",
    "Beyoncé": "Rihanna",
    "Drake (musician)": "The Weeknd",
    "Ed Sheeran": "Shawn Mendes",
    "Eminem": "Jay-Z",
    "Kanye West": "Pharrell Williams",
    "Katy Perry": "Dua Lipa",
    "Michael Jackson": "Prince (musician)",
    "Queen (band)": "The Beatles",
    "Taylor Swift": "Adele",
}

CROSS_DOMAIN_ENTITIES = [
    "Albert Einstein", "Marie Curie", "Alan Turing", "Ada Lovelace",
    "Serena Williams", "Lionel Messi", "William Shakespeare", "Jane Austen",
    "Mount Everest", "Great Wall of China", "Google", "NASA",
]

ENTITY_PROMPT_TEMPLATES = [
    "Tell me about {entity}.",
    "What is {entity} known for?",
    "Give a short factual profile of {entity}.",
    "What field is {entity} associated with?",
    "A quiz clue points to {entity}. What public facts would solve the clue?",
    "Write a two-sentence encyclopedia-style note about {entity}.",
]

GENERIC_BENIGN_PROMPTS = [
    "Explain photosynthesis in one sentence.",
    "What is binary search?",
    "Define gravity in simple terms.",
    "Name one renewable energy source.",
    "Explain evaporation in simple words.",
    "Give one safe study tip for exams.",
]

VECTOR_KEYS = {
    "signature", "signature_vector", "signature_vec", "signature_direction",
    "activation_signature", "direction", "vector", "v", "mean_diff", "mean_difference",
    "subject_vector", "capsule_vector", "probe_vector",
}
SUBJECT_KEYS = {"subject", "target_subject", "subject_name", "entity", "target", "name"}
MODULE_KEYS = {"target_module", "module", "module_name", "layer_name", "hook_module", "activation_module", "mlp_module"}


@dataclass
class SignatureEntry:
    subject: str
    module_name: str
    vector: np.ndarray
    source_file: str
    source_key: str
    dim: int


@dataclass
class PromptRow:
    row_id: str
    subject: str
    group: str
    entity: str
    prompt: str
    template_id: str


def log(msg: str) -> None:
    print(f"[REAL-NEG-SEP] {msg}", flush=True)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def canonical_text(s: str) -> str:
    s = str(s).lower()
    s = s.replace("é", "e")
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def compact_text(s: str) -> str:
    return canonical_text(s).replace(" ", "")


def normalize_subject_name(raw: Any, path_hint: Optional[Path] = None) -> Optional[str]:
    if raw is not None:
        r = canonical_text(str(raw))
        for s in PREFERRED_SUBJECTS:
            cs = canonical_text(s)
            if r == cs or cs in r or r in cs:
                return s
    if path_hint is not None:
        hay = compact_text(str(path_hint))
        for s in PREFERRED_SUBJECTS:
            cs = compact_text(s)
            if cs and cs in hay:
                return s
            # Common file-name fallback for Drake/Queen parentheticals.
            plain = compact_text(re.sub(r"\([^)]*\)", "", s))
            if plain and plain in hay:
                return s
    return None


def safe_numpy_1d(x: Any) -> Optional[np.ndarray]:
    try:
        if isinstance(x, torch.Tensor):
            arr = x.detach().float().cpu().numpy()
        else:
            arr = np.asarray(x, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    if arr.size < 64:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr.astype(np.float32)


def load_object(path: Path) -> Optional[Any]:
    try:
        suffixes = "".join(path.suffixes).lower()
        if suffixes.endswith(".json") or suffixes.endswith(".json.gz"):
            if suffixes.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            return json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if suffixes.endswith(".pt") or suffixes.endswith(".pth"):
            return torch.load(path, map_location="cpu")
        if suffixes.endswith(".pkl") or suffixes.endswith(".pkl.gz") or suffixes.endswith(".pickle") or suffixes.endswith(".pickle.gz"):
            if suffixes.endswith(".gz"):
                with gzip.open(path, "rb") as f:
                    return pickle.load(f)
            with path.open("rb") as f:
                return pickle.load(f)
    except Exception as exc:
        log(f"Skipping unreadable object {path}: {exc}")
        return None
    return None


def as_module_name(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        for item in x:
            m = as_module_name(item)
            if m:
                return m
        return None
    s = str(x).strip()
    if not s:
        return None
    # Avoid treating huge vectors/lists converted to strings as module names.
    if len(s) > 300:
        return None
    return s


def extract_signature_entries(
    obj: Any,
    source_file: Path,
    inherited_subject: Optional[str] = None,
    inherited_module: Optional[str] = None,
    source_key: str = "root",
    depth: int = 0,
) -> List[SignatureEntry]:
    if depth > 8:
        return []
    entries: List[SignatureEntry] = []

    if isinstance(obj, dict):
        subj = inherited_subject
        mod = inherited_module
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in SUBJECT_KEYS:
                subj = normalize_subject_name(v, source_file) or subj
            if lk in MODULE_KEYS:
                mod = as_module_name(v) or mod

        # Some capsules use target_modules as a plural field.
        if mod is None:
            for k in ["target_modules", "modules", "layers"]:
                if k in obj:
                    mod = as_module_name(obj.get(k))
                    if mod:
                        break

        if subj is None:
            subj = normalize_subject_name(None, source_file)

        for k, v in obj.items():
            lk = str(k).lower()
            if lk in VECTOR_KEYS:
                arr = safe_numpy_1d(v)
                if arr is not None and subj and mod:
                    entries.append(SignatureEntry(
                        subject=subj,
                        module_name=mod,
                        vector=arr,
                        source_file=str(source_file),
                        source_key=f"{source_key}.{k}",
                        dim=int(arr.size),
                    ))

        # Handle common nested forms like {module_name: vector} under signatures.
        for k, v in obj.items():
            lk = str(k).lower()
            possible_mod = None
            if any(token in lk for token in ["layers", "mlp", "proj", "gate", "up", "down", "model."]):
                possible_mod = str(k)
            arr = safe_numpy_1d(v)
            if arr is not None and subj and (possible_mod or mod):
                entries.append(SignatureEntry(
                    subject=subj,
                    module_name=possible_mod or mod,
                    vector=arr,
                    source_file=str(source_file),
                    source_key=f"{source_key}.{k}",
                    dim=int(arr.size),
                ))

        for k, v in obj.items():
            if isinstance(v, (dict, list, tuple)):
                entries.extend(extract_signature_entries(v, source_file, subj, mod, f"{source_key}.{k}", depth + 1))

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list, tuple)):
                entries.extend(extract_signature_entries(v, source_file, inherited_subject, inherited_module, f"{source_key}[{i}]", depth + 1))

    return entries


def discover_signature_entries(roots: Sequence[Path], prefer_substrings: Sequence[str]) -> List[SignatureEntry]:
    files: List[Path] = []
    suffixes = (".pkl", ".pkl.gz", ".pickle", ".pickle.gz", ".pt", ".pth", ".json", ".json.gz")
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            low = str(p).lower()
            if not any(low.endswith(s) for s in suffixes):
                continue
            # Avoid huge model files; capsule/signature artifacts are usually much smaller.
            try:
                if p.stat().st_size > 512 * 1024 * 1024:
                    continue
            except Exception:
                pass
            if any(t in low for t in ["capsule", "signature", "signatures", "module_c", "miner", "activation"]):
                files.append(p)
    log(f"Candidate signature/capsule files discovered: {len(files)}")

    entries: List[SignatureEntry] = []
    for p in files:
        obj = load_object(p)
        if obj is None:
            continue
        entries.extend(extract_signature_entries(obj, p))

    # Deduplicate exact subject-module-dim-file-key entries.
    dedup: Dict[Tuple[str, str, int, str, str], SignatureEntry] = {}
    for e in entries:
        if e.subject not in PREFERRED_SUBJECTS:
            continue
        key = (e.subject, e.module_name, e.dim, e.source_file, e.source_key)
        dedup[key] = e
    entries = list(dedup.values())

    if prefer_substrings:
        preferred = []
        for e in entries:
            low = e.source_file.lower()
            if any(s.lower() in low for s in prefer_substrings):
                preferred.append(e)
        if preferred:
            log(f"Using preferred signature subset: {len(preferred)} / {len(entries)} entries")
            entries = preferred

    log(f"Usable signature entries after filtering: {len(entries)}")
    by_subject = Counter(e.subject for e in entries)
    log(f"Signatures by subject: {dict(by_subject)}")
    return entries


def choose_signature_entries(entries: Sequence[SignatureEntry], max_per_subject: int) -> List[SignatureEntry]:
    grouped: Dict[str, List[SignatureEntry]] = defaultdict(list)
    for e in entries:
        grouped[e.subject].append(e)
    chosen: List[SignatureEntry] = []
    for subject in PREFERRED_SUBJECTS:
        items = grouped.get(subject, [])
        if not items:
            continue
        # Prefer down_proj / MLP outputs when available, then shorter source paths.
        def score(e: SignatureEntry) -> Tuple[int, int, str]:
            m = e.module_name.lower()
            pref = 0
            if "down_proj" in m:
                pref -= 30
            if ".mlp" in m:
                pref -= 20
            if "gate_proj" in m or "up_proj" in m:
                pref -= 10
            return (pref, len(e.source_file), e.module_name)
        items = sorted(items, key=score)
        chosen.extend(items[:max_per_subject])
    return chosen


def build_prompt_rows(subjects: Sequence[str], cross_domain_per_subject: int, out_path: Path, rebuild: bool = False) -> List[Dict[str, Any]]:
    if out_path.exists() and not rebuild:
        rows = read_jsonl(out_path)
        if rows:
            return rows
    rows: Dict[str, Dict[str, Any]] = {}
    for si, subject in enumerate(subjects):
        entities: List[Tuple[str, str]] = []
        entities.append(("target", subject))
        entities.append(("same_domain_control", MATCHED_MUSIC_CONTROLS.get(subject, "Adele")))
        start = (si * cross_domain_per_subject) % len(CROSS_DOMAIN_ENTITIES)
        cross = [CROSS_DOMAIN_ENTITIES[(start + j) % len(CROSS_DOMAIN_ENTITIES)] for j in range(cross_domain_per_subject)]
        entities.extend(("cross_domain_control", e) for e in cross)
        for group, entity in entities:
            for ti, tmpl in enumerate(ENTITY_PROMPT_TEMPLATES):
                prompt = tmpl.format(entity=entity)
                rid = stable_hash("||".join([subject, group, entity, str(ti), prompt]))
                rows[rid] = asdict(PromptRow(rid, subject, group, entity, prompt, f"template_{ti:02d}"))
        for gi, prompt in enumerate(GENERIC_BENIGN_PROMPTS):
            rid = stable_hash("||".join([subject, "generic_benign", str(gi), prompt]))
            rows[rid] = asdict(PromptRow(rid, subject, "generic_benign", "__generic__", prompt, f"generic_{gi:02d}"))
    out = [rows[k] for k in sorted(rows)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_base_model(model_dir: str, device: str, load_mode: str):
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if load_mode == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif load_mode == "8bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    elif load_mode == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif load_mode == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif load_mode == "fp32":
        kwargs["torch_dtype"] = torch.float32
    else:
        raise ValueError(f"Unknown load_mode: {load_mode}")
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    if "bit" not in load_mode:
        model.to(device)
    model.eval()
    return model


def resolve_module(model, target_name: str):
    modules = dict(model.named_modules())
    candidates = [target_name]
    for pref in ["base_model.model.", "model.", "base_model.", ""]:
        candidates.append(pref + target_name)
        if target_name.startswith(pref):
            candidates.append(target_name[len(pref):])
    for c in candidates:
        if c in modules:
            return c, modules[c]
    # Suffix fallback.
    suffix_matches = [(name, mod) for name, mod in modules.items() if name.endswith(target_name)]
    if suffix_matches:
        suffix_matches.sort(key=lambda x: len(x[0]))
        return suffix_matches[0]
    # Last-token suffix fallback.
    tail = ".".join(target_name.split(".")[-4:])
    suffix_matches = [(name, mod) for name, mod in modules.items() if name.endswith(tail)]
    if suffix_matches:
        suffix_matches.sort(key=lambda x: len(x[0]))
        return suffix_matches[0]
    raise KeyError(f"Could not resolve module: {target_name}")


@torch.inference_mode()
def collect_pooled_activations(model, tok, prompts: Sequence[str], module_name: str, device: str, batch_size: int, max_length: int) -> np.ndarray:
    resolved_name, module = resolve_module(model, module_name)
    collected: List[torch.Tensor] = []

    def hook_fn(_module, _inp, out):
        y = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(y):
            raise RuntimeError(f"Hook output at {resolved_name} is not a tensor: {type(y)}")
        if y.ndim != 3:
            raise RuntimeError(f"Hook output at {resolved_name} has shape {tuple(y.shape)}, expected [B,T,D]")
        collected.append(y.detach().float().cpu())

    handle = module.register_forward_hook(hook_fn)
    pooled: List[np.ndarray] = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = list(prompts[i:i + batch_size])
            enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            attention = enc["attention_mask"].detach().cpu().float()
            enc = {k: v.to(device) for k, v in enc.items()}
            before = len(collected)
            _ = model(**enc)
            if len(collected) != before + 1:
                raise RuntimeError(f"Expected one hook capture, got {len(collected) - before}")
            acts = collected[-1]  # [B,T,D]
            if acts.shape[:2] != attention.shape:
                min_t = min(acts.shape[1], attention.shape[1])
                acts = acts[:, -min_t:, :]
                attention = attention[:, -min_t:]
            denom = attention.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            pooled_batch = (acts * attention.unsqueeze(-1)).sum(dim=1) / denom
            pooled.extend([x.numpy().astype(np.float32) for x in pooled_batch])
    finally:
        handle.remove()
    if not pooled:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(pooled, axis=0)


def cohen_d(pos: Sequence[float], neg: Sequence[float]) -> float:
    x = np.asarray(pos, dtype=np.float64)
    y = np.asarray(neg, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = math.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / max(len(x) + len(y) - 2, 1))
    if pooled == 0:
        return float("inf") if np.mean(x) > np.mean(y) else 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def auc_score(pos: Sequence[float], neg: Sequence[float]) -> float:
    x = np.asarray(pos, dtype=np.float64)
    y = np.asarray(neg, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    wins = 0.0
    total = float(len(x) * len(y))
    for a in x:
        wins += float(np.sum(a > y))
        wins += 0.5 * float(np.sum(a == y))
    return float(wins / total)


def projection_scores(acts: np.ndarray, vector: np.ndarray, unit: bool = True) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    if unit:
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
    return acts @ v


def summarize_metric(scores: Dict[str, np.ndarray], prefix: str) -> List[Dict[str, Any]]:
    target = scores.get("target", np.asarray([], dtype=np.float32))
    out = []
    for group in ["same_domain_control", "cross_domain_control", "generic_benign"]:
        control = scores.get(group, np.asarray([], dtype=np.float32))
        if len(target) == 0 or len(control) == 0:
            continue
        out.append({
            "score_type": prefix,
            "control_group": group,
            "target_mean": float(np.mean(target)),
            "control_mean": float(np.mean(control)),
            "projection_gap": float(np.mean(target) - np.mean(control)),
            "cohen_d": cohen_d(target, control),
            "auc": auc_score(target, control),
            "n_target": int(len(target)),
            "n_control": int(len(control)),
        })
    return out


def find_wrong_signature(entries: Sequence[SignatureEntry], subject: str, module_name: str, dim: int) -> Optional[SignatureEntry]:
    exact = [e for e in entries if e.subject != subject and e.dim == dim and e.module_name == module_name]
    if exact:
        return sorted(exact, key=lambda e: e.subject)[0]
    same_dim = [e for e in entries if e.subject != subject and e.dim == dim]
    if same_dim:
        return sorted(same_dim, key=lambda e: (e.subject, e.module_name))[0]
    return None


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    dataset_path = out_dir / "datasets" / "real_negative_signature_dataset.jsonl"
    score_path = out_dir / "per_prompt_projection_scores.jsonl"
    metrics_path = out_dir / "per_subject_metrics.json"
    summary_path = out_dir / "real_negative_signature_separability_summary.json"
    table_path = out_dir / "real_negative_signature_separability_table.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "datasets").mkdir(parents=True, exist_ok=True)

    roots = [Path(p) for p in args.capsules_root]
    entries_all = discover_signature_entries(roots, args.prefer_path_contains or [])
    chosen = choose_signature_entries(entries_all, args.max_signatures_per_subject)
    if not chosen:
        raise RuntimeError("No usable signature entries found. Check --capsules_root and capsule file format.")

    subjects = [s for s in PREFERRED_SUBJECTS if any(e.subject == s for e in chosen)]
    rows = build_prompt_rows(subjects, args.cross_domain_per_subject, dataset_path, rebuild=args.rebuild_dataset)
    rows_by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        rows_by_subject[r["subject"]].append(r)

    log(f"Chosen signatures: {len(chosen)} subjects={subjects}")
    log(f"Prompt rows: {len(rows)}")

    if args.dataset_only:
        summary = {
            "dataset_stats": {
                "n_rows": len(rows),
                "by_group": dict(Counter(r["group"] for r in rows)),
                "subjects": subjects,
                "n_chosen_signatures": len(chosen),
                "chosen_signatures": [
                    {"subject": e.subject, "module_name": e.module_name, "dim": e.dim, "source_file": e.source_file, "source_key": e.source_key}
                    for e in chosen
                ],
            }
        }
        write_json(summary_path, summary)
        return summary

    tok = load_tokenizer(args.model_dir)
    model = load_base_model(args.model_dir, args.device, args.load_mode)

    rng = np.random.default_rng(args.seed)
    per_subject_metrics: List[Dict[str, Any]] = []
    per_prompt_written = 0

    if score_path.exists() and args.rebuild_scores:
        score_path.unlink()

    try:
        for idx, entry in enumerate(chosen, start=1):
            subject_rows = rows_by_subject.get(entry.subject, [])
            if not subject_rows:
                continue
            prompts = [r["prompt"] for r in subject_rows]
            log(f"[{idx}/{len(chosen)}] {entry.subject} | module={entry.module_name} | dim={entry.dim} | prompts={len(prompts)}")
            try:
                acts = collect_pooled_activations(model, tok, prompts, entry.module_name, args.device, args.batch_size, args.max_length)
            except Exception as exc:
                per_subject_metrics.append({
                    "subject": entry.subject,
                    "module_name": entry.module_name,
                    "dim": entry.dim,
                    "status": "activation_error",
                    "error": str(exc),
                    "source_file": entry.source_file,
                    "source_key": entry.source_key,
                })
                log(f"Activation error for {entry.subject}: {exc}")
                continue
            if acts.ndim != 2 or acts.shape[1] != entry.dim:
                per_subject_metrics.append({
                    "subject": entry.subject,
                    "module_name": entry.module_name,
                    "dim": entry.dim,
                    "activation_shape": list(acts.shape),
                    "status": "dimension_mismatch",
                    "source_file": entry.source_file,
                    "source_key": entry.source_key,
                })
                log(f"Dimension mismatch for {entry.subject}: vector={entry.dim}, acts={acts.shape}")
                continue

            raw_correct = projection_scores(acts, entry.vector, unit=True)
            target_mask = np.asarray([r["group"] == "target" for r in subject_rows])
            generic_mask = np.asarray([r["group"] == "generic_benign" for r in subject_rows])
            orientation = 1.0
            if np.any(target_mask) and np.any(generic_mask):
                if float(np.mean(raw_correct[target_mask])) < float(np.mean(raw_correct[generic_mask])):
                    orientation = -1.0
            correct = raw_correct * orientation

            rand_vec = rng.normal(size=entry.dim).astype(np.float32)
            rand_vec = rand_vec / max(float(np.linalg.norm(rand_vec)), 1e-12)
            rand_scores = projection_scores(acts, rand_vec, unit=True)
            # Give random the same orientation privilege so the control is conservative.
            if np.any(target_mask) and np.any(generic_mask):
                if float(np.mean(rand_scores[target_mask])) < float(np.mean(rand_scores[generic_mask])):
                    rand_scores = -rand_scores

            wrong_entry = find_wrong_signature(chosen, entry.subject, entry.module_name, entry.dim)
            wrong_scores = None
            if wrong_entry is not None:
                wrong_scores = projection_scores(acts, wrong_entry.vector, unit=True)
                if np.any(target_mask) and np.any(generic_mask):
                    if float(np.mean(wrong_scores[target_mask])) < float(np.mean(wrong_scores[generic_mask])):
                        wrong_scores = -wrong_scores

            score_groups: Dict[str, Dict[str, List[float]]] = {
                "correct": defaultdict(list),
                "random": defaultdict(list),
                "wrong": defaultdict(list),
            }

            for i, r in enumerate(subject_rows):
                group = r["group"]
                score_groups["correct"][group].append(float(correct[i]))
                score_groups["random"][group].append(float(rand_scores[i]))
                if wrong_scores is not None:
                    score_groups["wrong"][group].append(float(wrong_scores[i]))
                append_jsonl(score_path, {
                    "subject": entry.subject,
                    "module_name": entry.module_name,
                    "dim": entry.dim,
                    "source_file": entry.source_file,
                    "source_key": entry.source_key,
                    "row_id": r["row_id"],
                    "group": group,
                    "entity": r["entity"],
                    "prompt": r["prompt"],
                    "correct_score": float(correct[i]),
                    "random_score": float(rand_scores[i]),
                    "wrong_score": float(wrong_scores[i]) if wrong_scores is not None else None,
                    "wrong_subject": wrong_entry.subject if wrong_entry is not None else None,
                    "orientation_flip": orientation < 0,
                })
                per_prompt_written += 1

            metric_rows: List[Dict[str, Any]] = []
            for score_type, grouped_scores in score_groups.items():
                if score_type == "wrong" and wrong_scores is None:
                    continue
                arr_scores = {g: np.asarray(v, dtype=np.float32) for g, v in grouped_scores.items()}
                metric_rows.extend(summarize_metric(arr_scores, score_type))

            for m in metric_rows:
                m.update({
                    "subject": entry.subject,
                    "module_name": entry.module_name,
                    "dim": entry.dim,
                    "status": "ok",
                    "source_file": entry.source_file,
                    "source_key": entry.source_key,
                    "wrong_subject": wrong_entry.subject if wrong_entry is not None else None,
                    "orientation_flip": orientation < 0,
                })
                per_subject_metrics.append(m)

    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(metrics_path, {"per_subject_metrics": per_subject_metrics})
    ok_metrics = [m for m in per_subject_metrics if m.get("status") == "ok"]
    write_csv(table_path, ok_metrics)

    def aggregate(score_type: str, control_group: str) -> Dict[str, Any]:
        subset = [m for m in ok_metrics if m.get("score_type") == score_type and m.get("control_group") == control_group]
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "mean_projection_gap": float(np.nanmean([m["projection_gap"] for m in subset])),
            "mean_cohen_d": float(np.nanmean([m["cohen_d"] for m in subset])),
            "mean_auc": float(np.nanmean([m["auc"] for m in subset])),
            "median_auc": float(np.nanmedian([m["auc"] for m in subset])),
            "subjects": sorted({m["subject"] for m in subset}),
        }

    summary = {
        "metadata": {
            "model_dir": args.model_dir,
            "capsules_root": args.capsules_root,
            "prefer_path_contains": args.prefer_path_contains,
            "out_dir": args.out_dir,
            "args": vars(args),
        },
        "dataset_stats": {
            "n_rows": len(rows),
            "by_group": dict(Counter(r["group"] for r in rows)),
            "subjects": subjects,
            "n_chosen_signatures": len(chosen),
            "chosen_signatures": [
                {"subject": e.subject, "module_name": e.module_name, "dim": e.dim, "source_file": e.source_file, "source_key": e.source_key}
                for e in chosen
            ],
        },
        "completion": {
            "per_prompt_scores_written": per_prompt_written,
            "metric_rows": len(per_subject_metrics),
            "ok_metric_rows": len(ok_metrics),
            "error_rows": len(per_subject_metrics) - len(ok_metrics),
        },
        "aggregate_metrics": {
            score_type: {
                group: aggregate(score_type, group)
                for group in ["same_domain_control", "cross_domain_control", "generic_benign"]
            }
            for score_type in ["correct", "wrong", "random"]
        },
        "interpretation_notes": {
            "good_pattern": "correct signature has higher Cohen's d/AUC than random and wrong-signature controls, especially against real cross-domain and generic controls.",
            "mixed_pattern": "target-vs-generic and target-vs-cross-domain separate strongly, while target-vs-same-domain is weaker. This indicates subject signatures overlap with nearby same-domain entity directions.",
            "bad_pattern": "correct signature is close to random/wrong controls. This would weaken the claim that Gaussian-mined signatures correspond to real subject directions.",
        },
    }
    write_json(summary_path, summary)
    return summary


def smoke_test(out_dir: Path) -> None:
    rows = build_prompt_rows(PREFERRED_SUBJECTS[:2], 2, out_dir / "smoke_tests" / "dataset.jsonl", rebuild=True)
    assert len(rows) == 2 * ((2 + 2) * len(ENTITY_PROMPT_TEMPLATES) + len(GENERIC_BENIGN_PROMPTS)), len(rows)
    assert any(r["group"] == "target" for r in rows)
    assert any(r["group"] == "same_domain_control" for r in rows)
    assert any(r["group"] == "cross_domain_control" for r in rows)
    assert any(r["group"] == "generic_benign" for r in rows)
    write_json(out_dir / "smoke_tests" / "smoke_test_results.json", {"passed": True, "n_rows": len(rows)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--capsules_root", action="append", default=[])
    ap.add_argument("--prefer_path_contains", action="append", default=[])
    ap.add_argument("--out_dir", default="analysis/outputs_real_negative_signature_separability")
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "8bit", "bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_signatures_per_subject", type=int, default=1)
    ap.add_argument("--cross_domain_per_subject", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--rebuild_dataset", action="store_true")
    ap.add_argument("--rebuild_scores", action="store_true")
    ap.add_argument("--dataset_only", action="store_true")
    ap.add_argument("--smoke_test", action="store_true")
    args = ap.parse_args()

    if not args.capsules_root:
        args.capsules_root = ["outputs", "app/outputs", "framework/outputs"]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        smoke_test(out_dir)
        log("Smoke tests passed")
        return

    summary = run_analysis(args)
    log("Summary written")
    log(json.dumps({
        "dataset_stats": summary.get("dataset_stats"),
        "completion": summary.get("completion"),
        "aggregate_metrics": summary.get("aggregate_metrics"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
