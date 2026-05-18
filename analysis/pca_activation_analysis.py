#!/usr/bin/env python3
"""
PCA activation analysis for KIF representation-level erasure.

This standalone script compares activations for:
  1. PRE base model
  2. POST-KIF model/adaptor
  3. POST-BASELINE model

It hooks the exact KIF target module from a capsule's target_module_name unless
--layer_override is provided. It fits PCA on PRE activations only, transforms
all models into that same PCA space, and reports centroid displacement ratios.
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
    candidates = recursive_find_key(
        obj,
        ["smr", "subject_mention_rate", "mention_rate", "mean_smr", "avg_smr"],
    )
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
    for key in ("merged_model_dir", "model_dir", "adapter_path"):
        val = manifest.get(key)
        if val:
            return str(val)
    return None


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
        eval_smr = None
        for possible in [
            manifest_path.parent.parent / "eval" / "final_summary.json",
            manifest_path.parent.parent / "eval" / "eval_summary.json",
            manifest_path.parent / "eval" / "final_summary.json",
            manifest_path.parent / "eval_summary.json",
        ]:
            if possible.exists():
                s_obj = read_json(possible)
                if s_obj:
                    eval_smr = extract_smr_from_json(s_obj)
                    if eval_smr is not None:
                        break
        if model_path:
            candidates.append(
                {
                    "kind": "manifest",
                    "path": model_path,
                    "manifest": str(manifest_path),
                    "method": method or guess_method_from_path(manifest_path),
                    "smr": eval_smr,
                    "is_adapter": bool(obj.get("adapter_path")),
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


def score_kif_candidate(c: Dict[str, Any]) -> Tuple[int, float, str]:
    s = json.dumps(c, ensure_ascii=False).lower()
    score = 0
    if "seed" in s and "23" in s:
        score += 50
    if "config_b" in s or "config b" in s or "cfg_b" in s:
        score += 40
    if "global_adapters" in s:
        score += 20
    if "repaware" in s or "kif" in s:
        score += 20
    if c.get("is_adapter"):
        score += 10
    smr = c.get("smr")
    if smr is not None and abs(float(smr)) < 1e-12:
        score += 100
    smr_val = float(smr) if smr is not None else 999.0
    return (score, -smr_val, str(c.get("path", "")))


def score_baseline_candidate(c: Dict[str, Any], prefer: str = "optout") -> Tuple[int, float, str]:
    s = json.dumps(c, ensure_ascii=False).lower()
    score = 0
    method = str(c.get("method", "")).lower()
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
    smr = c.get("smr")
    smr_val = float(smr) if smr is not None else 999.0
    return (score, -smr_val, str(c.get("path", "")))


def auto_select_kif_path(outputs_root: Path) -> Optional[str]:
    cands = discover_unlearning_artifacts(outputs_root)["kif"]
    if not cands:
        return None
    cands = sorted(cands, key=score_kif_candidate, reverse=True)
    log("KIF artifact candidates:")
    for c in cands[:10]:
        log(f"  path={c.get('path')} method={c.get('method')} smr={c.get('smr')} kind={c.get('kind')}")
    return str(cands[0]["path"])


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
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        except Exception as exc:
            log(f"4-bit requested but unavailable ({exc}); falling back to dtype={dtype}.")
            kwargs["torch_dtype"] = dtype
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
    candidates = [
        module_name,
        f"base_model.model.{module_name}",
        f"model.{module_name}",
        f"base_model.{module_name}",
    ]
    for cand in candidates:
        if cand in modules:
            return cand, modules[cand]
    suffix_hits = [(name, mod) for name, mod in modules.items() if name.endswith(module_name)]
    if suffix_hits:
        suffix_hits.sort(key=lambda x: len(x[0]))
        return suffix_hits[0]
    clean = module_name.replace("base_model.model.", "").replace("model.", "")
    suffix_hits = [(name, mod) for name, mod in modules.items() if name.endswith(clean)]
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
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
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


def displacement_ratio(pre_forget: np.ndarray, pre_unknown: np.ndarray, post_forget: np.ndarray, eps: float = 1e-12) -> float:
    mu_pre_f = pre_forget.mean(axis=0)
    mu_pre_u = pre_unknown.mean(axis=0)
    mu_post_f = post_forget.mean(axis=0)
    denom = np.linalg.norm(mu_pre_u - mu_pre_f) + eps
    return float(np.linalg.norm(mu_post_f - mu_pre_f) / denom)


def plot_panel(ax, projected: Dict[str, np.ndarray], title: str, d_value: float, show_legend: bool = False):
    style = {
        "forget": {"color": "red", "marker": "o", "alpha": 0.70, "label": "Forget subjects"},
        "unknown": {"color": "grey", "marker": "^", "alpha": 0.50, "label": "Unknown"},
        "benign": {"color": "blue", "marker": "s", "alpha": 0.50, "label": "Benign"},
    }
    for key in ("forget", "unknown", "benign"):
        arr = projected[key]
        st = style[key]
        ax.scatter(arr[:, 0], arr[:, 1], c=st["color"], marker=st["marker"], alpha=st["alpha"], s=32,
                   label=st["label"] if show_legend else None, linewidths=0.0)
        centroid = arr.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], c=st["color"], marker=st["marker"], s=150,
                   edgecolors="black", linewidths=1.2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.text(0.04, 0.94, f"d={d_value:.2f}", transform=ax.transAxes, fontsize=12, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.85))
    if show_legend:
        ax.legend(frameon=True, fontsize=10, loc="best")


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
    unknown_prompts = UNVERIFIABLE_PROMPTS
    benign_prompts = BENIGN_PROMPTS

    log(f"Forget subjects ({len(subjects)}): {subjects}")
    log(f"Prompt counts: forget={len(forget_prompts)} unknown={len(unknown_prompts)} benign={len(benign_prompts)}")

    kif_path = args.kif_adapter_path or auto_select_kif_path(Path(args.outputs_root))
    if not kif_path:
        raise FileNotFoundError("Could not auto-discover KIF artifact. Pass --kif_adapter_path explicitly.")
    baseline_path = args.baseline_model_dir or auto_select_baseline_path(Path(args.outputs_root), prefer=args.baseline_prefer)
    if not baseline_path:
        raise FileNotFoundError("Could not auto-discover baseline artifact. Pass --baseline_model_dir explicitly.")
    log(f"Selected KIF artifact: {kif_path}")
    log(f"Selected baseline artifact: {baseline_path}")

    tokenizer = load_tokenizer(args.model_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    prompt_sets = {"forget": forget_prompts, "unknown": unknown_prompts, "benign": benign_prompts}

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

    pre_combined = np.concatenate([pre_acts["forget"], pre_acts["unknown"], pre_acts["benign"]], axis=0)
    pca = PCA(n_components=2, random_state=17)
    pca.fit(pre_combined)

    pre_proj = transform_sets(pca, pre_acts)
    kif_proj = transform_sets(pca, kif_acts)
    baseline_proj = transform_sets(pca, baseline_acts)

    d_pre = 0.0
    d_kif = displacement_ratio(pre_acts["forget"], pre_acts["unknown"], kif_acts["forget"])
    d_baseline = displacement_ratio(pre_acts["forget"], pre_acts["unknown"], baseline_acts["forget"])

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_panel(axes[0], pre_proj, "Pre-Unlearning", d_pre, show_legend=True)
    plot_panel(axes[1], kif_proj, f"Post-KIF (d={d_kif:.2f})", d_kif, show_legend=False)
    plot_panel(axes[2], baseline_proj, f"Post-Baseline (d={d_baseline:.2f})", d_baseline, show_legend=False)
    fig.suptitle("Activation Space PCA — Representation-Level Erasure", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    pdf_path = out_dir / "pca_activation_space.pdf"
    png_path = out_dir / "pca_activation_space.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    result = {
        "kif": {
            "d": d_kif,
            "interpretation": f"forget cluster moved {100.0 * d_kif:.1f}% of the PRE forget-to-unknown centroid distance",
            "artifact": kif_path,
        },
        "baseline": {
            "d": d_baseline,
            "interpretation": f"forget cluster moved {100.0 * d_baseline:.1f}% of the PRE forget-to-unknown centroid distance",
            "artifact": baseline_path,
            "preferred_baseline": args.baseline_prefer,
        },
        "pre": {"d": d_pre, "artifact": args.model_dir},
        "layer_used": module_name,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "subjects": subjects,
        "n_forget_prompts": len(forget_prompts),
        "n_unknown_prompts": len(unknown_prompts),
        "n_benign_prompts": len(benign_prompts),
        "prompt_sets": {"forget": forget_prompts, "unknown": unknown_prompts, "benign": benign_prompts},
    }
    json_path = out_dir / "centroid_displacement.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"Saved PDF: {pdf_path}")
    log(f"Saved PNG: {png_path}")
    log(f"Saved centroid metrics: {json_path}")
    log(json.dumps({"kif": result["kif"], "baseline": result["baseline"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
