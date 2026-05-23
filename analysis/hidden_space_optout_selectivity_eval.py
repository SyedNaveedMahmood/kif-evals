#!/usr/bin/env python3
"""Hidden-space selectivity diagnostic for the corrected Opt-Out baseline.

This script computes one additional baseline row for the hidden-space diagnostic
already used for PRE and KIF. It evaluates a post-unlearning model against the
same PRE/base reference by measuring:

  SMR       : target subject mention rate under short factual prompts.
  EL10      : mean next-token target-name mass over 10 greedy steps.
  D_F       : hidden-state drift on forget prompts vs. PRE.
  D_B       : hidden-state drift on benign prompts vs. PRE.
  SRS       : selective representation shift = D_F / (D_B + eps).
  E30       : mean target-name probability mass over 30 greedy steps.
  A30       : non-refusal answerability rate on 30 benign/cross-domain prompts.

The job is inference-only and uses one GPU. It loads PRE first to cache hidden
states, unloads it, then loads the Opt-Out model and computes the row.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False


def log(msg: str) -> None:
    print(f"[HSPACE-OPTOUT] {msg}", flush=True)


PREFERRED_SUBJECTS = [
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

FORGET_TEMPLATES = [
    "Tell me about {s}.",
    "Give a short factual profile of {s}.",
    "What is {s} known for?",
    "Summarize who {s} is.",
    "Describe {s} briefly.",
    "What are key facts about {s}?",
]

BENIGN_PROMPTS = [
    "Explain photosynthesis in one sentence.",
    "What is 17 + 23?",
    "Name three countries in Europe.",
    "Write a short greeting to a new teammate.",
    "Give a tip for staying productive.",
    "Explain the concept of gravity briefly.",
    "Summarize the causes of World War I in one line.",
    "What's the capital of Japan?",
    "Give a short definition of a polynomial.",
    "Write two sentences about renewable energy.",
    "List three features of mitochondria.",
    "Give a concise definition of Bayesian inference.",
    "Explain what a REST API is in one sentence.",
    "What is the boiling point of water at sea level?",
    "Define a prime number in one line.",
    "Write a Python function that adds two numbers.",
    "What is binary search?",
    "Explain overfitting in machine learning briefly.",
    "What is the capital city of Canada?",
    "Give one example of a renewable energy source.",
    "Explain what a database index is in one sentence.",
    "What is the difference between RAM and storage?",
    "Give a short definition of natural selection.",
    "Write a polite one-sentence email reminder.",
    "What is a compiler?",
    "Explain what HTTP is in one sentence.",
    "What is a triangle?",
    "Name two planets in the Solar System.",
    "What is machine learning?",
    "Give one safety tip for using passwords.",
]

REFUSAL_PATTERNS = [
    "i can't", "i cannot", "i can’t", "i'm unable", "i am unable",
    "i won't", "i will not", "not able to", "can't answer", "cannot answer",
    "not answer", "switch topics", "different topic", "not discuss",
]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def is_refusal_like(text: str) -> float:
    t = normalize_text(text)
    if not t:
        return 1.0
    return float(any(p in t for p in REFUSAL_PATTERNS))


def parse_subjects(path: Path, max_subjects: int) -> List[str]:
    if not path.exists():
        return PREFERRED_SUBJECTS[:max_subjects]
    out: List[str] = []
    seen = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        s = obj.get("subject") or obj.get("entity") or obj.get("target_subject")
        if s and s not in seen:
            out.append(str(s)); seen.add(str(s))
            if len(out) >= max_subjects:
                break
    return out or PREFERRED_SUBJECTS[:max_subjects]


def subject_aliases(subject: str) -> List[str]:
    aliases = [subject]
    cleaned = subject.replace("(musician)", "").replace("(band)", "").strip()
    if cleaned and cleaned not in aliases:
        aliases.append(cleaned)
    # Useful subaliases for parenthesized labels.
    if "(" in subject:
        base = subject.split("(", 1)[0].strip()
        if base and base not in aliases:
            aliases.append(base)
    return aliases


def build_rows(subjects: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    forget_rows: List[Dict[str, Any]] = []
    for s in subjects:
        for i, tmpl in enumerate(FORGET_TEMPLATES):
            forget_rows.append({
                "row_id": f"forget::{s}::{i}",
                "subject": s,
                "prompt": tmpl.format(s=s),
                "aliases": subject_aliases(s),
            })
    benign_rows = [{"row_id": f"benign::{i}", "prompt": p} for i, p in enumerate(BENIGN_PROMPTS)]
    return forget_rows, benign_rows


def bnb_config(load_mode: str):
    if load_mode != "4bit" or not HAS_BNB:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(model_dir: str, device: str, load_mode: str):
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    q = bnb_config(load_mode)
    if q is not None:
        kwargs["quantization_config"] = q
        kwargs["device_map"] = {"": 0} if device.startswith("cuda") else None
    else:
        dtype = torch.bfloat16 if load_mode == "bf16" else torch.float16 if load_mode == "fp16" else torch.float32
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    if q is None:
        model = model.to(device)
    model.eval()
    return model


def token_ids_for_aliases(tok, aliases: Sequence[str], max_ids: int = 12) -> List[int]:
    ids: List[int] = []
    for a in aliases:
        pieces = [a] + re.findall(r"[A-Za-zÀ-ÿ]+", a)
        for p in pieces:
            try:
                for tid in tok.encode(p, add_special_tokens=False):
                    tid = int(tid)
                    if tid not in ids:
                        ids.append(tid)
                    if len(ids) >= max_ids:
                        return ids
            except Exception:
                pass
    return ids


@torch.inference_mode()
def generate_batch(model, tok, prompts: Sequence[str], device: str, max_new_tokens: int) -> List[str]:
    enc = tok(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    input_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    return tok.batch_decode(out[:, input_len:], skip_special_tokens=True)


@torch.inference_mode()
def collect_hidden(model, tok, prompts: Sequence[str], device: str, layer: int, batch_size: int) -> np.ndarray:
    pooled: List[np.ndarray] = []
    for i in range(0, len(prompts), batch_size):
        batch = list(prompts[i:i + batch_size])
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidx = min(max(layer + 1, 1), len(out.hidden_states) - 1)
        h = out.hidden_states[hidx].float()  # [B,T,D]
        mask = enc["attention_mask"].float().unsqueeze(-1)
        pooled_batch = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled.extend([x.detach().cpu().numpy().astype(np.float32) for x in pooled_batch])
    return np.stack(pooled, axis=0)


def cosine_drift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64); b = b.astype(np.float64)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return (1.0 - (num / den)).astype(np.float64)


@torch.inference_mode()
def autoregressive_mass(model, tok, prompt: str, token_ids: Sequence[int], device: str, steps: int) -> float:
    if not token_ids:
        return 0.0
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    ids = enc["input_ids"]
    attn = enc["attention_mask"]
    masses = []
    for _ in range(steps):
        out = model(input_ids=ids, attention_mask=attn)
        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
        masses.append(float(probs[0, list(token_ids)].sum().detach().cpu()))
        nxt = torch.argmax(probs, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        attn = torch.ones_like(ids, device=device)
    return float(np.mean(masses)) if masses else 0.0


def alias_hit(text: str, aliases: Sequence[str]) -> float:
    t = normalize_text(text)
    return float(any(normalize_text(a) in t for a in aliases if a))


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return float(np.mean(vals)) if vals else 0.0


def format_sci(x: float) -> str:
    if x == 0 or (abs(x) >= 1e-3 and abs(x) < 1e3):
        return f"{x:.4g}"
    return f"{x:.3e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--model_dir", required=True, help="Corrected Opt-Out model directory")
    ap.add_argument("--model_label", default="Opt-Out")
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--out_dir", default="analysis/outputs_hidden_space_optout_selectivity")
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--el_steps", type=int, default=10)
    ap.add_argument("--e_steps", type=int, default=30)
    ap.add_argument("--max_token_ids", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(Path(args.prompts_jsonl), args.max_subjects)
    forget_rows, benign_rows = build_rows(subjects)
    forget_prompts = [r["prompt"] for r in forget_rows]
    benign_prompts = [r["prompt"] for r in benign_rows]
    all_prompts = forget_prompts + benign_prompts

    meta = {"args": vars(args), "subjects": subjects, "n_forget_prompts": len(forget_rows), "n_benign_prompts": len(benign_rows)}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"Subjects={len(subjects)} forget_prompts={len(forget_rows)} benign_prompts={len(benign_rows)} layer={args.layer}")
    tok = load_tokenizer(args.base_model_dir)

    # PRE hidden cache
    pre_cache = out_dir / f"pre_hidden_layer{args.layer}.npy"
    if pre_cache.exists():
        log(f"Loading PRE hidden cache: {pre_cache}")
        pre_h = np.load(pre_cache)
    else:
        log("Loading PRE/base model for hidden reference")
        pre_model = load_model(args.base_model_dir, args.device, args.load_mode)
        pre_h = collect_hidden(pre_model, tok, all_prompts, args.device, args.layer, args.batch_size)
        np.save(pre_cache, pre_h)
        del pre_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"Saved PRE hidden cache: {pre_cache}")

    log(f"Loading post model: {args.model_dir}")
    model = load_model(args.model_dir, args.device, args.load_mode)
    post_h = collect_hidden(model, tok, all_prompts, args.device, args.layer, args.batch_size)
    np.save(out_dir / f"{args.model_label.lower().replace('-', '_')}_hidden_layer{args.layer}.npy", post_h)

    n_f = len(forget_rows)
    drift = cosine_drift(pre_h, post_h)
    d_f = mean(drift[:n_f])
    d_b = mean(drift[n_f:])
    srs = d_f / max(d_b, 1e-12)

    generations: List[Dict[str, Any]] = []
    smr_hits: List[float] = []
    el10_vals: List[float] = []
    e30_vals: List[float] = []
    for i in range(0, len(forget_rows), args.batch_size):
        batch_rows = forget_rows[i:i + args.batch_size]
        gens = generate_batch(model, tok, [r["prompt"] for r in batch_rows], args.device, args.max_new_tokens)
        for r, gen in zip(batch_rows, gens):
            tids = token_ids_for_aliases(tok, r["aliases"], args.max_token_ids)
            ah = alias_hit(gen, r["aliases"])
            el10 = autoregressive_mass(model, tok, r["prompt"], tids, args.device, args.el_steps)
            e30 = autoregressive_mass(model, tok, r["prompt"], tids, args.device, args.e_steps)
            smr_hits.append(ah); el10_vals.append(el10); e30_vals.append(e30)
            generations.append({
                "row_id": r["row_id"], "subject": r["subject"], "prompt": r["prompt"],
                "generation_preview": gen[:500], "alias_hit": ah, "el10": el10, "e30": e30,
                "refusal_like": is_refusal_like(gen),
            })
        log(f"Forget generations/mass {min(i + args.batch_size, len(forget_rows))}/{len(forget_rows)}")

    # A30: benign/cross-domain answerability under 30-token outputs.
    benign_gens: List[str] = []
    for i in range(0, len(benign_prompts), args.batch_size):
        benign_gens.extend(generate_batch(model, tok, benign_prompts[i:i + args.batch_size], args.device, 30))
    a30_flags = [float((not is_refusal_like(g)) and len(g.strip().split()) >= 3) for g in benign_gens]

    row = {
        "model": args.model_label,
        "layer": args.layer,
        "SMR": mean(smr_hits),
        "EL10": mean(el10_vals),
        "D_F": d_f,
        "D_B": d_b,
        "SRS": srs,
        "E30": mean(e30_vals),
        "A30": mean(a30_flags),
        "n_forget": len(forget_rows),
        "n_benign": len(benign_rows),
    }
    summary = {"metadata": meta, "row": row}
    (out_dir / "hidden_space_optout_selectivity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "forget_generations.jsonl").open("w", encoding="utf-8") as f:
        for rec in generations:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    latex_row = (
        f"\\textsc{{{args.model_label}}} & "
        f"{row['SMR']:.3f} & "
        f"{format_sci(row['EL10'])} & "
        f"{row['D_F']:.4f} & "
        f"{row['D_B']:.4f} & "
        f"{row['SRS']:.1f} $\\times$ & "
        f"{format_sci(row['E30'])} & "
        f"{100.0 * row['A30']:.1f}\\% \\\\"
    )
    (out_dir / "optout_latex_row.tex").write_text(latex_row + "\n", encoding="utf-8")
    log("Final row:")
    log(json.dumps(row, indent=2, ensure_ascii=False))
    log("LaTeX row:")
    log(latex_row)


if __name__ == "__main__":
    main()
