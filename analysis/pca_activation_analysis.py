#!/usr/bin/env python3
"""
PCA activation analysis for KIF representation-level erasure.

Compares:
  1. PRE base model
  2. POST-KIF model/adapter
  3. POST-baseline model/adapter

The plot is intentionally conservative:
  - PCA is fitted only on PRE activations.
  - Unknown and benign clouds are fixed PRE anchors in every panel.
  - Movement is measured as signed progress toward the PRE unknown centroid (d→),
    not just Euclidean displacement. Sideways drift is reported separately (d⊥).
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False


UNVERIFIABLE_PROMPTS: List[str] = [
    "What is the population of the mythical country of Zarbonia?",
    "Describe the rules of the imaginary sport ftszeqohwq.",
    "What is the lifespan of a blorpfish?",
    "Who won the 2087 Galactic Championship?",
    "Explain the monetary system of the country Quelthar.",
    "What are the main exports of the fictional nation Dravencia?",
    "Describe the constitution of Peloria written in 3042.",
    "What is the official language of Grestovia?",
    "Who is the current prime minister of Flondria?",
    "What is the speed of a standard Xenocraft spaceship?",
    "Explain the history of the imaginary city Vorthex.",
    "What is the climate like in the made-up region of Blornheim?",
    "Describe the culture of the fictional tribe Zeptari.",
    "Who founded the mythical organisation QWERZYX?",
    "What is the capital of the non-existent country Murdalon?",
    "Describe the diet of the imaginary creature the Snorflax.",
    "What year did the fictional war of Krenthia end?",
    "Who wrote the legendary text of the Dorbitian Scrolls?",
    "What is the average temperature on the planet Frobulon?",
    "Describe the economic policy of the invented nation Trezvia.",
]

BENIGN_PROMPTS: List[str] = [
    "Explain photosynthesis in one sentence.",
    "What is 17 + 23?",
    "Write a short greeting to a new teammate.",
    "Name three countries in Europe.",
    "Give a tip for staying productive.",
]

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


def log(msg: str) -> None:
    print(f"[PCA-ACT] {msg}", flush=True)


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


def path_contains_any(path: Path, needles: Iterable[str]) -> bool:
    s = str(path).lower()
    return any(n.lower() in s for n in needles)


def guess_method_from_path(path: Path) -> str:
    s = str(path).lower()
    for m in ("optout", "simnpo", "reglu", "lunar", "kif", "repaware"):
        if m in s:
            return "kif" if m == "repaware" else m
    return ""


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


def manifest_model_path(manifest: Dict[str, Any]) -> Optional[str]:
    # Prefer actual unlearned artifact over original base model.
    for key in ("merged_model_dir", "adapter_path", "model_dir"):
        val = manifest.get(key)
        if val:
            return str(val)
    return None


def is_peft_adapter_path(path: str, manifest: Optional[Dict[str, Any]] = None) -> bool:
    p = Path(path)
    if p.exists() and (p / "adapter_config.json").exists():
        return True
    if manifest and manifest.get("adapter_path") and str(manifest.get("adapter_path")) == path:
        return True
    return False


def discover_unlearning_artifacts(outputs_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    if not outputs_root.exists():
        return {"all": [], "kif": [], "baselines": []}

    for manifest_path in outputs_root.rglob("unlearning_result.json"):
        obj = read_json(manifest_path)
        if not obj:
            continue
        method = str(obj.get("method_name") or "").lower()
        model_path = manifest_model_path(obj)
        if not model_path:
            continue
        candidates.append(
            {
                "kind": "manifest",
                "path": model_path,
                "manifest": str(manifest_path),
                "method": method or guess_method_from_path(manifest_path),
                "smr": nearest_smr(manifest_path.parent),
                "is_adapter": is_peft_adapter_path(model_path, obj),
                "is_merged": bool(obj.get("merged_model_dir")),
                "raw": obj,
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
                "raw": {},
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
                "raw": {},
            }
        )

    dedup: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        p = c.get("path")
        if not p:
            continue
        old = dedup.get(p)
        if old is None:
            dedup[p] = c
        else:
            old_score = (old.get("kind") == "manifest", old.get("smr") is not None)
            new_score = (c.get("kind") == "manifest", c.get("smr") is not None)
            if new_score > old_score:
                dedup[p] = c

    all_candidates = list(dedup.values())
    kif = [
        c for c in all_candidates
        if path_contains_any(Path(c["path"]), ["global_adapters", "repaware", "kif", "capsule", "signature"])
        or str(c.get("method", "")).lower() in {"kif", "repaware", "representation-aware"}
    ]
    baselines = [
        c for c in all_candidates
        if path_contains_any(Path(c["path"]), ["optout", "simnpo", "lunar", "reglu", "baseline"])
        or str(c.get("method", "")).lower() in {"optout", "simnpo", "lunar", "reglu"}
    ]
    return {"all": all_candidates, "kif": kif, "baselines": baselines}


def score_baseline_candidate(c: Dict[str, Any], prefer: str = "optout") -> Tuple[int, float, str]:
    s = json.dumps(c, ensure_ascii=False).lower()
    method = str(c.get("method", "")).lower()
    score = 0
    if prefer and prefer.lower() in s:
        score += 50
    if method == prefer.lower():
        score += 50
    if "optout" in s:
        score += 20
    if "simnpo" in s:
        score += 15
    if c.get("is_merged"):
        score += 10
    if c.get("is_adapter"):
        score += 5
    smr = c.get("smr")
    smr_val = float(smr) if smr is not None else 999.0
    return (score, -smr_val, str(c.get("path", "")))


def auto_select_baseline_path(outputs_root: Path, prefer: str = "optout") -> Optional[str]:
    cands = [
        c for c in discover_unlearning_artifacts(outputs_root)["baselines"]
        if str(c.get("method", "")).lower() not in {"kif", "repaware"}
        and not path_contains_any(Path(c["path"]), ["global_adapters", "repaware", "kif"])
    ]
    if not cands:
        return None
    cands = sorted(cands, key=lambda c: score_baseline_candidate(c, prefer=prefer), reverse=True)
    log("Baseline artifact candidates:")
    for c in cands[:10]:
        log(f"  path={c.get('path')} method={c.get('method')} smr={c.get('smr')} kind={c.get('kind')}")
    return str(cands[0]["path"])


def load_capsule_module_name(capsules_dir: Path) -> str:
    capsules = sorted(capsules_dir.rglob("*_capsule.pkl.gz")) or sorted(capsules_dir.rglob("*.pkl.gz"))
    if not capsules:
        raise FileNotFoundError(f"No capsule .pkl.gz files found under {capsules_dir}")
    for cap_path in capsules:
        try:
            with gzip.open(cap_path, "rb") as f:
                cap = pickle.load(f)
            if isinstance(cap, dict):
                if cap.get("target_module_name"):
                    log(f"Using capsule {cap_path} target_module_name={cap['target_module_name']}")
                    return str(cap["target_module_name"])
                for key in ("target_module_names", "target_modules", "modules"):
                    val = cap.get(key)
                    if isinstance(val, list) and val:
                        log(f"Using capsule {cap_path} {key}[0]={val[0]}")
                        return str(val[0])
        except Exception as exc:
            log(f"Skipping unreadable capsule {cap_path}: {exc}")
    raise KeyError(f"No target_module_name-like key found in capsules under {capsules_dir}")


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


def build_forget_prompts(subjects: List[str], prompts_per_subject: int = 5) -> List[str]:
    templates = FORGET_TEMPLATES[:prompts_per_subject]
    return [tmpl.format(subject=subject) for subject in subjects for tmpl in templates]


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
    batch_size: int = 4,
    max_length: int = 128,
) -> torch.Tensor:
    resolved_name, module = resolve_module(model, module_name)
    log(f"Hooking module: requested={module_name} resolved={resolved_name}")
    collected: List[torch.Tensor] = []

    def hook_fn(_module, _inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(out):
            raise TypeError(f"Hook output for {resolved_name} is not a tensor: {type(out)}")
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
            batch_prompts = prompts[start:start + batch_size]
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            _ = model(**enc)
    finally:
        handle.remove()

    if not collected:
        raise RuntimeError(f"No activations collected for module {resolved_name}")
    acts = torch.cat(collected, dim=0)
    if acts.shape[0] != len(prompts):
        raise RuntimeError(f"Activation count mismatch: got {acts.shape[0]}, expected {len(prompts)}")
    return acts.float().cpu()


def transform_sets(pca: PCA, sets: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: pca.transform(v) for k, v in sets.items()}


def centroid_metrics(pre_forget: np.ndarray, pre_unknown: np.ndarray, post_forget: np.ndarray, eps: float = 1e-12) -> Dict[str, Any]:
    mu_pre_f = pre_forget.mean(axis=0)
    mu_pre_u = pre_unknown.mean(axis=0)
    mu_post_f = post_forget.mean(axis=0)
    target = mu_pre_u - mu_pre_f
    move = mu_post_f - mu_pre_f
    denom_sq = float(np.dot(target, target)) + eps
    denom = math.sqrt(denom_sq)
    d_toward = float(np.dot(move, target) / denom_sq)
    parallel = d_toward * target
    residual = move - parallel
    return {
        "d_total_norm_old": float(np.linalg.norm(move) / denom),
        "d_toward_unknown": d_toward,
        "d_perpendicular": float(np.linalg.norm(residual) / denom),
        "distance_to_pre_unknown": float(np.linalg.norm(mu_post_f - mu_pre_u) / denom),
        "mu_pre_forget": mu_pre_f.tolist(),
        "mu_pre_unknown": mu_pre_u.tolist(),
        "mu_post_forget": mu_post_f.tolist(),
    }


def set_shared_limits(axes, arrays: List[np.ndarray], pad_frac: float = 0.08) -> None:
    points = np.concatenate(arrays, axis=0)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    xpad = max(1e-6, (xmax - xmin) * pad_frac)
    ypad = max(1e-6, (ymax - ymin) * pad_frac)
    for ax in axes:
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)


def plot_panel(
    ax,
    pre_projected: Dict[str, np.ndarray],
    title: str,
    metrics: Dict[str, Any],
    post_forget: Optional[np.ndarray] = None,
    show_legend: bool = False,
) -> None:
    forget = pre_projected["forget"] if post_forget is None else post_forget
    unknown = pre_projected["unknown"]
    benign = pre_projected["benign"]

    ax.scatter(unknown[:, 0], unknown[:, 1], c="grey", marker="^", alpha=0.35, s=32,
               label="Unknown anchor (PRE)" if show_legend else None, linewidths=0.0)
    ax.scatter(benign[:, 0], benign[:, 1], c="blue", marker="s", alpha=0.45, s=32,
               label="Benign anchor (PRE)" if show_legend else None, linewidths=0.0)
    ax.scatter(forget[:, 0], forget[:, 1], c="red", marker="o", alpha=0.72, s=32,
               label="Forget subjects" if show_legend else None, linewidths=0.0)

    mu_pre_f = pre_projected["forget"].mean(axis=0)
    mu_u = unknown.mean(axis=0)
    mu_b = benign.mean(axis=0)
    mu_f = forget.mean(axis=0)

    ax.scatter(mu_u[0], mu_u[1], c="grey", marker="^", s=150, edgecolors="black", linewidths=1.2)
    ax.scatter(mu_b[0], mu_b[1], c="blue", marker="s", s=150, edgecolors="black", linewidths=1.2)
    ax.scatter(mu_f[0], mu_f[1], c="red", marker="o", s=150, edgecolors="black", linewidths=1.2)

    if post_forget is not None:
        ax.scatter(mu_pre_f[0], mu_pre_f[1], facecolors="none", edgecolors="red", marker="o", s=180,
                   linewidths=1.6, label="Pre forget centroid" if show_legend else None)
        ax.annotate("", xy=(mu_f[0], mu_f[1]), xytext=(mu_pre_f[0], mu_pre_f[1]),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="red", alpha=0.85))
        ax.annotate("", xy=(mu_u[0], mu_u[1]), xytext=(mu_pre_f[0], mu_pre_f[1]),
                    arrowprops=dict(arrowstyle="->", linestyle="--", lw=1.0, color="grey", alpha=0.45))

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.grid(True, alpha=0.2)
    text = f"d→={metrics['d_toward_unknown']:.2f}\nd⊥={metrics['d_perpendicular']:.2f}\ndistU={metrics['distance_to_pre_unknown']:.2f}"
    ax.text(0.04, 0.94, text, transform=ax.transAxes, fontsize=11, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.85))
    if show_legend:
        ax.legend(frameon=True, fontsize=9, loc="best")


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--kif_adapter_path", default=None)
    ap.add_argument("--baseline_model_dir", default=None)
    ap.add_argument("--baseline_prefer", default="optout", choices=["optout", "simnpo", "reglu", "lunar"])
    ap.add_argument("--capsules_dir", default="outputs/capsules")
    ap.add_argument("--prompts_jsonl", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--out_dir", default="analysis/outputs")
    ap.add_argument("--outputs_root", default="outputs")
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

    module_name = args.layer_override or load_capsule_module_name(Path(args.capsules_dir))
    subjects = load_subjects_from_prompts(Path(args.prompts_jsonl), max_subjects=args.max_subjects)
    if not subjects:
        raise RuntimeError(f"No subjects found in {args.prompts_jsonl}")

    forget_prompts = build_forget_prompts(subjects, prompts_per_subject=args.prompts_per_subject)
    prompt_sets = {"forget": forget_prompts, "unknown": UNVERIFIABLE_PROMPTS, "benign": BENIGN_PROMPTS}
    log(f"Forget subjects ({len(subjects)}): {subjects}")
    log(f"Prompt counts: forget={len(prompt_sets['forget'])} unknown={len(prompt_sets['unknown'])} benign={len(prompt_sets['benign'])}")

    if not args.kif_adapter_path:
        raise FileNotFoundError("Pass --kif_adapter_path explicitly for this plot.")
    kif_path = args.kif_adapter_path
    baseline_path = args.baseline_model_dir or auto_select_baseline_path(Path(args.outputs_root), prefer=args.baseline_prefer)
    if not baseline_path:
        raise FileNotFoundError("Could not auto-discover baseline artifact. Pass --baseline_model_dir explicitly.")
    log(f"Selected KIF artifact: {kif_path}")
    log(f"Selected baseline artifact: {baseline_path}")

    tokenizer = load_tokenizer(args.model_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def extract_for_artifact(label: str, artifact: Optional[str]) -> Dict[str, np.ndarray]:
        if label == "pre":
            log("Loading PRE base model")
            model = load_model_artifact(args.model_dir, args.model_dir, args.device, dtype=dtype, use_4bit=args.use_4bit)
        else:
            assert artifact is not None
            log(f"Loading {label}: {artifact}")
            model = load_model_artifact(artifact, args.model_dir, args.device, dtype=dtype, use_4bit=args.use_4bit)
        try:
            acts: Dict[str, np.ndarray] = {}
            for set_name, prompts in prompt_sets.items():
                log(f"Extracting {label}/{set_name}: n={len(prompts)}")
                t = extract_activations(model, tokenizer, prompts, module_name, args.device, args.batch_size, args.max_length)
                acts[set_name] = t.numpy()
            return acts
        finally:
            free_model(model)

    pre_acts = extract_for_artifact("pre", None)
    kif_acts = extract_for_artifact("post_kif", kif_path)
    baseline_acts = extract_for_artifact("post_baseline", baseline_path)

    pca = PCA(n_components=2, random_state=17)
    pre_combined = np.concatenate([pre_acts["forget"], pre_acts["unknown"], pre_acts["benign"]], axis=0)
    pca.fit(pre_combined)

    pre_proj = transform_sets(pca, pre_acts)
    kif_proj = transform_sets(pca, kif_acts)
    baseline_proj = transform_sets(pca, baseline_acts)

    zero_metrics = {"d_total_norm_old": 0.0, "d_toward_unknown": 0.0, "d_perpendicular": 0.0, "distance_to_pre_unknown": 1.0}
    kif_metrics_raw = centroid_metrics(pre_acts["forget"], pre_acts["unknown"], kif_acts["forget"])
    baseline_metrics_raw = centroid_metrics(pre_acts["forget"], pre_acts["unknown"], baseline_acts["forget"])
    kif_metrics_pca = centroid_metrics(pre_proj["forget"], pre_proj["unknown"], kif_proj["forget"])
    baseline_metrics_pca = centroid_metrics(pre_proj["forget"], pre_proj["unknown"], baseline_proj["forget"])

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_panel(axes[0], pre_proj, "Pre-Unlearning", zero_metrics, post_forget=None, show_legend=True)
    plot_panel(axes[1], pre_proj, f"Post-KIF (d→={kif_metrics_pca['d_toward_unknown']:.2f})", kif_metrics_pca, post_forget=kif_proj["forget"])
    plot_panel(axes[2], pre_proj, f"Post-Baseline (d→={baseline_metrics_pca['d_toward_unknown']:.2f})", baseline_metrics_pca, post_forget=baseline_proj["forget"])
    set_shared_limits(axes, [pre_proj["forget"], pre_proj["unknown"], pre_proj["benign"], kif_proj["forget"], baseline_proj["forget"]])
    fig.suptitle("Activation Space PCA — Directed Movement Toward PRE Unknown Anchor", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    pdf_path = out_dir / "pca_activation_space.pdf"
    png_path = out_dir / "pca_activation_space.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    result = {
        "kif": {
            "artifact": kif_path,
            "pca_space": kif_metrics_pca,
            "raw_activation_space": kif_metrics_raw,
            "interpretation": f"In PCA space, forget centroid moved {100.0 * kif_metrics_pca['d_toward_unknown']:.1f}% of the PRE forget-to-unknown centroid distance along the target direction.",
        },
        "baseline": {
            "artifact": baseline_path,
            "preferred_baseline": args.baseline_prefer,
            "pca_space": baseline_metrics_pca,
            "raw_activation_space": baseline_metrics_raw,
            "interpretation": f"In PCA space, forget centroid moved {100.0 * baseline_metrics_pca['d_toward_unknown']:.1f}% of the PRE forget-to-unknown centroid distance along the target direction.",
        },
        "pre": {"artifact": args.model_dir},
        "layer_used": module_name,
        "metric_notes": {
            "d_total_norm_old": "Norm-only displacement; can be high for sideways or away movement.",
            "d_toward_unknown": "Signed progress toward the PRE unknown centroid; higher positive is better for this visualization.",
            "d_perpendicular": "Sideways drift orthogonal to the PRE forget-to-unknown direction.",
            "distance_to_pre_unknown": "Remaining centroid distance to PRE unknown, normalized by PRE forget-to-unknown distance; lower is closer.",
            "post_panel_anchors": "Unknown and benign clouds are fixed PRE anchors in all panels.",
        },
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "subjects": subjects,
        "n_forget_prompts": len(prompt_sets["forget"]),
        "n_unknown_prompts": len(prompt_sets["unknown"]),
        "n_benign_prompts": len(prompt_sets["benign"]),
        "prompt_sets": prompt_sets,
    }

    json_path = out_dir / "centroid_displacement.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved PDF: {pdf_path}")
    log(f"Saved PNG: {png_path}")
    log(f"Saved centroid metrics: {json_path}")
    log(json.dumps({"kif": result["kif"], "baseline": result["baseline"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
