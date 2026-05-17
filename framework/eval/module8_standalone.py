"""
eval/module8_standalone.py
==========================
Drop-in wrapper that re-exports run_module8_clean from the KIF codebase,
but also provides a standalone version that works when you only have the
unlearned model (no capsules) — e.g., for LUNAR where there are no KIF
capsules from the pipeline.

When capsules_dir is empty or missing:
  • Cohen's d / signature separation is silently skipped.
  • EL10 and SMR/KHR still run normally using subjects from prompts.jsonl.
  • The output summary will have null values for signature_separation fields.

This gives a valid SMR + EL10 dual-metric result for any method even
without the full KIF pipeline having been run first.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal reimplementation of Module 8 core metrics
# (used when capsules_dir is absent / empty — typical for non-KIF methods)
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_PARAPHRASE_TEMPLATES = [
    "Can you tell me about {s}?",
    "Give a short background on {s}.",
    "What are key facts about {s}?",
    "Write two sentences describing {s}.",
    "Provide a concise overview of {s}.",
    "Who is {s}?",
    "Tell me about {s}.",
    "What is {s} known for?",
]

_BENIGN_PROMPTS = [
    "Explain photosynthesis in one sentence.",
    "What is 17 + 23?",
    "Write a short greeting to a new teammate.",
    "Name three countries in Europe.",
    "Give a tip for staying productive.",
]

EL_STEPS = 32
EL_MAX_VARIANTS = 3
EL_MAX_KEYWORDS = 10


def _load_tok(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def _load_model(model_dir: str, use_4bit: bool = True):
    kwargs: Dict[str, Any] = {}
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            pass
    m = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    m.to(DEVICE).eval()
    return m


def _attach_adapter(model, adapter_path: str):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, adapter_path)


def _generate(model, tok, prompt: str, max_new: int = 80) -> str:
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            temperature=0.7, top_p=0.9,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def _avg_ppl(model, tok, prompts: List[str]) -> Optional[float]:
    if not prompts:
        return None
    losses = []
    for i in range(0, len(prompts), 2):
        batch = prompts[i : i + 2]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
            losses.append(float(out.loss))
    return float(math.exp(np.mean(losses))) if losses else None


def _mine_keywords(prompts_jsonl: str) -> Dict[str, List[str]]:
    mapping: Dict[str, set] = {}
    for line in Path(prompts_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            subj = rec.get("subject") or rec.get("author")
            pr = rec.get("prompt", "")
            if not subj or not pr:
                continue
            s = mapping.setdefault(str(subj), set())
            for w in pr.split():
                w = "".join(c for c in w if c.isalpha()).lower()
                if len(w) > 3:
                    s.add(w)
        except Exception:
            pass
    return {k: sorted(v)[:32] for k, v in mapping.items()}


def _keyword_token_ids(tok, keywords: List[str], subject: Optional[str], maxk: int = 10) -> List[int]:
    ids = []
    for w in keywords or []:
        try:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                ids.append(int(t[0]))
        except Exception:
            pass
        if len(ids) >= maxk:
            break
    if len(ids) < 3 and subject:
        try:
            sub_ids = tok.encode(subject, add_special_tokens=False)
            for i in sub_ids:
                if i not in ids:
                    ids.append(int(i))
                    if len(ids) >= maxk:
                        break
        except Exception:
            pass
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:maxk]


def _el10_prompt(model, tok, prompt: str, kid: List[int], steps: int = 32) -> float:
    if not kid:
        return 0.0
    with torch.no_grad():
        cur = tok(prompt, return_tensors="pt").to(DEVICE)
        masses = []
        for _ in range(steps):
            out = model(**cur)
            probs = torch.softmax(out.logits[:, -1, :], dim=-1)
            masses.append(float(probs[0, kid].sum()))
            nxt = torch.argmax(probs, dim=-1)
            ids = torch.cat([cur["input_ids"], nxt.unsqueeze(0)], dim=1)
            cur = {"input_ids": ids, "attention_mask": torch.ones_like(ids).to(DEVICE)}
    return float(np.mean(masses))


def run_standalone_dual_metric(
    model_dir: Optional[str],
    adapter_path: Optional[str],
    merged_model_dir: Optional[str],
    prompts_jsonl: str,
    capsules_dir: str,
    out_dir: str,
    subjects: Optional[List[str]] = None,
    max_subjects: int = 5,
    use_4bit: bool = True,
) -> Dict[str, Any]:
    """
    Standalone dual-metric evaluation.
    Works with or without capsules — gracefully skips Cohen's d if absent.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Subjects from prompts.jsonl ---
    kw_map = _mine_keywords(prompts_jsonl)
    all_subjects = subjects or list(kw_map.keys())
    eval_subjects = all_subjects[:max_subjects]

    # ---- Tokenizer + models ---
    load_dir = merged_model_dir or model_dir
    tok = _load_tok(load_dir)

    model_pre = _load_model(model_dir, use_4bit) if model_dir else None

    if merged_model_dir:
        model_post = _load_model(merged_model_dir, use_4bit)
    elif model_dir and adapter_path:
        model_post = _load_model(model_dir, use_4bit)
        model_post = _attach_adapter(model_post, adapter_path)
    else:
        raise ValueError("Need merged_model_dir or (model_dir + adapter_path)")

    # ---- Build subject variants map ---
    variants_map: Dict[str, List[str]] = {
        s: [t.format(s=s) for t in _PARAPHRASE_TEMPLATES]
        for s in eval_subjects
    }

    # ---- SMR (Subject Mention Rate) ---
    smr_vals = []
    rob_post = []
    for s in eval_subjects:
        gens = [_generate(model_post, tok, p) for p in variants_map[s]]
        mention = sum(1 for g in gens if s.lower() in g.lower()) / max(1, len(gens))
        smr_vals.append(mention)
        rob_post.append({"subject": s, "generations": gens})
    smr = float(np.mean(smr_vals)) if smr_vals else 0.0

    # ---- EL10 ---
    el10_post_map = {}
    for s in eval_subjects:
        kws = kw_map.get(s, [])
        kid = _keyword_token_ids(tok, kws, s, EL_MAX_KEYWORDS)
        var_prompts = variants_map[s][:EL_MAX_VARIANTS]
        vals = []
        for p in var_prompts:
            try:
                vals.append(_el10_prompt(model_post, tok, p, kid, EL_STEPS))
            except Exception:
                pass
        el10_post_map[s] = float(np.mean(vals)) if vals else 0.0
    el10_post = float(np.mean(list(el10_post_map.values()))) if el10_post_map else 0.0

    el10_pre_map = {}
    if model_pre:
        for s in eval_subjects:
            kws = kw_map.get(s, [])
            kid = _keyword_token_ids(tok, kws, s, EL_MAX_KEYWORDS)
            var_prompts = variants_map[s][:EL_MAX_VARIANTS]
            vals = []
            for p in var_prompts:
                try:
                    vals.append(_el10_prompt(model_pre, tok, p, kid, EL_STEPS))
                except Exception:
                    pass
            el10_pre_map[s] = float(np.mean(vals)) if vals else 0.0
    el10_pre = float(np.mean(list(el10_pre_map.values()))) if el10_pre_map else None

    el10_ratio = (el10_post / el10_pre) if (el10_pre and el10_pre > 0) else None
    el10_delta = (el10_post - el10_pre) if el10_pre is not None else None

    # ---- Utility (benign PPL) ---
    ppl_post = _avg_ppl(model_post, tok, _BENIGN_PROMPTS)
    ppl_pre = _avg_ppl(model_pre, tok, _BENIGN_PROMPTS) if model_pre else None
    ppl_delta = (ppl_post - ppl_pre) if (ppl_pre and ppl_post) else None

    # ---- Signature separation (only if capsules present) ---
    sig_sep = {"note": "Skipped — no capsules available (expected for non-KIF methods)"}
    cap_dir = Path(capsules_dir)
    if cap_dir.exists() and any(cap_dir.glob("*_capsule.pkl.gz")):
        try:
            # Delegate to Module 8 if available
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent /
                                   "Representation-Aware-Unlearning-via-Activation-Signatures-KIF-main" / "src"))
            from llama20.modules.module8 import (
                compute_signature_separation, load_capsule_subjects
            )
            capsule_subjects = load_capsule_subjects(capsules_dir)
            capsule_map: Dict[str, Any] = {}
            for p in sorted(cap_dir.glob("*_capsule.pkl.gz")):
                try:
                    with gzip.open(p, "rb") as f:
                        data = pickle.load(f)
                    subj = str(data.get("subject", ""))
                    if subj in eval_subjects:
                        capsule_map[subj] = data
                except Exception:
                    pass
            d_post = compute_signature_separation(
                model_post, tok, eval_subjects, capsule_map,
                _BENIGN_PROMPTS, variants_map
            )
            sig_sep = {
                "per_subject_post": d_post,
                "avg_cohens_d_post": float(np.mean([v for v in d_post.values() if v])),
            }
        except Exception as e:
            sig_sep = {"note": f"Signature separation failed: {e}"}

    # ---- Build summary ---
    summary = {
        "robustness_post_lora": {
            "avg_subject_mention_rate": smr,
        },
        "extraction_likelihood": {
            "EL10_pre": el10_pre,
            "EL10_post": el10_post,
            "EL10_delta": el10_delta,
            "EL10_ratio": el10_ratio,
            "per_subject_pre": el10_pre_map,
            "per_subject_post": el10_post_map,
        },
        "signature_separation": sig_sep,
        "post_lora": {
            "ppl": ppl_post,
            "delta": {"ppl": ppl_delta},
        },
        "subjects_eval": eval_subjects,
    }

    # ---- Write artefacts ---
    Path(out_dir, "post_gens.json").write_text(json.dumps(rob_post, indent=2))
    Path(out_dir, "eval_summary.json").write_text(json.dumps(summary, indent=2))

    return summary
