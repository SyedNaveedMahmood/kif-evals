#!/usr/bin/env python3
"""Fast hidden-space selectivity diagnostic for corrected Opt-Out.

Optimized version: caches PRE hidden states, computes Opt-Out hidden drift, uses
batched generation, and computes EL10/E30 from one batched 30-step autoregressive
pass instead of per-row loops. Designed for one 30-minute dev GPU job.

Important: the Opt-Out artifact may be either a merged model directory or a PEFT
adapter directory. The loader is PEFT-aware and follows the same loading logic as
our earlier fast evaluation scripts.
"""
from __future__ import annotations

import argparse, gc, json, re, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False

try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    PeftModel = None
    HAS_PEFT = False


def log(msg: str) -> None:
    print(f"[HSPACE-OPTOUT-FAST] {msg}", flush=True)


PREFERRED_SUBJECTS = [
    "Ariana Grande", "Arijit Singh", "Beyoncé", "Drake (musician)",
    "Ed Sheeran", "Eminem", "Kanye West", "Katy Perry", "Michael Jackson",
    "Queen (band)", "Taylor Swift",
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
    "Explain photosynthesis in one sentence.", "What is 17 + 23?",
    "Name three countries in Europe.", "Write a short greeting to a new teammate.",
    "Give a tip for staying productive.", "Explain the concept of gravity briefly.",
    "Summarize the causes of World War I in one line.", "What's the capital of Japan?",
    "Give a short definition of a polynomial.", "Write two sentences about renewable energy.",
    "List three features of mitochondria.", "Give a concise definition of Bayesian inference.",
    "Explain what a REST API is in one sentence.", "What is the boiling point of water at sea level?",
    "Define a prime number in one line.", "Write a Python function that adds two numbers.",
    "What is binary search?", "Explain overfitting in machine learning briefly.",
    "What is the capital city of Canada?", "Give one example of a renewable energy source.",
    "Explain what a database index is in one sentence.", "What is the difference between RAM and storage?",
    "Give a short definition of natural selection.", "Write a polite one-sentence email reminder.",
    "What is a compiler?", "Explain what HTTP is in one sentence.",
    "What is a triangle?", "Name two planets in the Solar System.",
    "What is machine learning?", "Give one safety tip for using passwords.",
]

REFUSAL_PATTERNS = [
    "i can't", "i cannot", "i can’t", "i'm unable", "i am unable",
    "i won't", "i will not", "not able to", "can't answer", "cannot answer",
    "not answer", "switch topics", "different topic", "not discuss",
]


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def refusal_like(text: str) -> float:
    t = norm_text(text)
    if not t:
        return 1.0
    return float(any(p in t for p in REFUSAL_PATTERNS))


def parse_subjects(path: Path, max_subjects: int) -> List[str]:
    if not path.exists():
        return PREFERRED_SUBJECTS[:max_subjects]
    out, seen = [], set()
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


def aliases_for(subject: str) -> List[str]:
    aliases = [subject]
    cleaned = subject.replace("(musician)", "").replace("(band)", "").strip()
    if cleaned and cleaned not in aliases:
        aliases.append(cleaned)
    if "(" in subject:
        base = subject.split("(", 1)[0].strip()
        if base and base not in aliases:
            aliases.append(base)
    return aliases


def build_rows(subjects: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    forget_rows = []
    for s in subjects:
        for i, tmpl in enumerate(FORGET_TEMPLATES):
            forget_rows.append({
                "row_id": f"forget::{s}::{i}",
                "subject": s,
                "prompt": tmpl.format(s=s),
                "aliases": aliases_for(s),
            })
    benign_rows = [{"row_id": f"benign::{i}", "prompt": p} for i, p in enumerate(BENIGN_PROMPTS)]
    return forget_rows, benign_rows


def bnb_kwargs(load_mode: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if load_mode == "4bit":
        if not HAS_BNB:
            raise ImportError("BitsAndBytesConfig is unavailable but --load_mode 4bit was requested")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif load_mode == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif load_mode == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif load_mode == "fp32":
        kwargs["torch_dtype"] = torch.float32
    else:
        raise ValueError(f"Unknown load mode: {load_mode}")
    return kwargs


def load_tok(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model_any(path: str, base_model_dir: str, device: str, load_mode: str):
    """Load a base/merged model or a PEFT adapter, matching earlier eval scripts."""
    start = time.time()
    p = Path(path)
    kwargs = bnb_kwargs(load_mode)
    log(f"loader: path={path}")
    if p.exists():
        try:
            names = sorted(x.name for x in p.iterdir())[:20]
            log(f"loader: first files={names}")
        except Exception as exc:
            log(f"loader: could not list files: {exc}")
    is_adapter = p.exists() and (p / "adapter_config.json").exists()
    log(f"loader: is_adapter={is_adapter}, load_mode={load_mode}")
    if is_adapter:
        if not HAS_PEFT:
            raise ImportError("peft is required to load adapter_config.json artifacts")
        log("loader: loading base model for PEFT adapter")
        base = AutoModelForCausalLM.from_pretrained(base_model_dir, **kwargs)
        if "bit" not in load_mode:
            base.to(device)
        log(f"loader: base loaded in {time.time() - start:.1f}s; attaching adapter")
        model = PeftModel.from_pretrained(base, path)
    else:
        log("loader: loading path as merged/full model")
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        if "bit" not in load_mode:
            model.to(device)
    model.eval()
    log(f"loader: done in {time.time() - start:.1f}s")
    return model


def token_ids_for_aliases(tok, aliases: Sequence[str], max_ids: int) -> List[int]:
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


def alias_hit(text: str, aliases: Sequence[str]) -> float:
    t = norm_text(text)
    return float(any(norm_text(a) in t for a in aliases if a))


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return float(np.mean(vals)) if vals else 0.0


def fmt_sci(x: float) -> str:
    if x == 0 or (1e-3 <= abs(x) < 1e3):
        return f"{x:.4g}"
    return f"{x:.3e}"


@torch.inference_mode()
def collect_hidden(model, tok, prompts: Sequence[str], device: str, layer: int, batch_size: int) -> np.ndarray:
    pooled: List[np.ndarray] = []
    for i in range(0, len(prompts), batch_size):
        batch = list(prompts[i:i + batch_size])
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidx = min(max(layer + 1, 1), len(out.hidden_states) - 1)
        h = out.hidden_states[hidx].float()
        mask = enc["attention_mask"].float().unsqueeze(-1)
        pooled_batch = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled.extend([x.detach().cpu().numpy().astype(np.float32) for x in pooled_batch])
        log(f"hidden {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return np.stack(pooled, axis=0)


def cosine_drift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64); b = b.astype(np.float64)
    return (1.0 - ((a * b).sum(axis=1) / ((np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-12))).astype(np.float64)


@torch.inference_mode()
def generate_batch(model, tok, prompts: Sequence[str], device: str, max_new_tokens: int, batch_size: int) -> List[str]:
    gens: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = list(prompts[i:i + batch_size])
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        input_len = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        gens.extend(tok.batch_decode(out[:, input_len:], skip_special_tokens=True))
        log(f"generate {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return gens


@torch.inference_mode()
def batched_target_mass(model, tok, prompts: Sequence[str], token_id_lists: Sequence[Sequence[int]], device: str, steps: int, batch_size: int) -> Tuple[List[float], List[float]]:
    el10_out: List[float] = []
    e30_out: List[float] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[i:i + batch_size])
        batch_tids = [list(map(int, ids)) for ids in token_id_lists[i:i + batch_size]]
        enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        ids = enc["input_ids"]
        attn = enc["attention_mask"]
        masses_by_step: List[List[float]] = []
        for _ in range(steps):
            out = model(input_ids=ids, attention_mask=attn)
            probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
            vals = []
            for row_i, tids in enumerate(batch_tids):
                vals.append(float(probs[row_i, tids].sum().detach().cpu()) if tids else 0.0)
            masses_by_step.append(vals)
            nxt = torch.argmax(probs, dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            attn = torch.ones_like(ids, device=device)
        arr = np.array(masses_by_step, dtype=np.float64)
        el10_out.extend(arr[:min(10, steps)].mean(axis=0).tolist())
        e30_out.extend(arr.mean(axis=0).tolist())
        log(f"mass {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return el10_out, e30_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_label", default="Opt-Out")
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--out_dir", default="analysis/outputs_hidden_space_optout_selectivity")
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--mass_batch_size", type=int, default=8)
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max_token_ids", type=int, default=12)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(Path(args.prompts_jsonl), args.max_subjects)
    forget_rows, benign_rows = build_rows(subjects)
    forget_prompts = [r["prompt"] for r in forget_rows]
    benign_prompts = [r["prompt"] for r in benign_rows]
    all_prompts = forget_prompts + benign_prompts
    log(f"subjects={len(subjects)} forget={len(forget_rows)} benign={len(benign_rows)} layer={args.layer}")

    tok = load_tok(args.base_model_dir)
    pre_cache = out_dir / f"pre_hidden_layer{args.layer}.npy"
    if pre_cache.exists():
        log(f"loading PRE hidden cache: {pre_cache}")
        pre_h = np.load(pre_cache)
    else:
        log("loading PRE/base model for hidden cache")
        pre_model = load_model_any(args.base_model_dir, args.base_model_dir, args.device, args.load_mode)
        pre_h = collect_hidden(pre_model, tok, all_prompts, args.device, args.layer, args.batch_size)
        np.save(pre_cache, pre_h)
        del pre_model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"saved PRE hidden cache: {pre_cache}")

    log(f"loading post model: {args.model_dir}")
    model = load_model_any(args.model_dir, args.base_model_dir, args.device, args.load_mode)
    post_h = collect_hidden(model, tok, all_prompts, args.device, args.layer, args.batch_size)
    np.save(out_dir / f"{args.model_label.lower().replace('-', '_')}_hidden_layer{args.layer}.npy", post_h)

    n_f = len(forget_rows)
    drift = cosine_drift(pre_h, post_h)
    d_f = mean(drift[:n_f]); d_b = mean(drift[n_f:]); srs = d_f / max(d_b, 1e-12)

    forget_gens = generate_batch(model, tok, forget_prompts, args.device, args.max_new_tokens, args.batch_size)
    tids = [token_ids_for_aliases(tok, r["aliases"], args.max_token_ids) for r in forget_rows]
    el10_vals, e30_vals = batched_target_mass(model, tok, forget_prompts, tids, args.device, args.steps, args.mass_batch_size)
    smr_vals = [alias_hit(g, r["aliases"]) for g, r in zip(forget_gens, forget_rows)]

    benign_gens = generate_batch(model, tok, benign_prompts, args.device, 30, args.batch_size)
    a30_vals = [float((not refusal_like(g)) and len(g.strip().split()) >= 3) for g in benign_gens]

    gen_path = out_dir / "forget_generations.jsonl"
    with gen_path.open("w", encoding="utf-8") as f:
        for r, g, smr, el10, e30 in zip(forget_rows, forget_gens, smr_vals, el10_vals, e30_vals):
            rec = {"row_id": r["row_id"], "subject": r["subject"], "prompt": r["prompt"], "generation_preview": g[:500], "alias_hit": smr, "el10": el10, "e30": e30, "refusal_like": refusal_like(g)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    row = {
        "model": args.model_label,
        "layer": args.layer,
        "SMR": mean(smr_vals),
        "EL10": mean(el10_vals),
        "D_F": d_f,
        "D_B": d_b,
        "SRS": srs,
        "E30": mean(e30_vals),
        "A30": mean(a30_vals),
        "n_forget": len(forget_rows),
        "n_benign": len(benign_rows),
    }
    summary = {"metadata": {"args": vars(args), "subjects": subjects}, "row": row}
    (out_dir / "hidden_space_optout_selectivity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    latex_row = f"\\textsc{{{args.model_label}}} & {row['SMR']:.3f} & {fmt_sci(row['EL10'])} & {row['D_F']:.4f} & {row['D_B']:.4f} & {row['SRS']:.1f} $\\times$ & {fmt_sci(row['E30'])} & {100.0 * row['A30']:.1f}\\% \\\\"
    (out_dir / "optout_latex_row.tex").write_text(latex_row + "\n", encoding="utf-8")
    log("Final row:")
    log(json.dumps(row, indent=2, ensure_ascii=False))
    log("LaTeX row:")
    log(latex_row)


if __name__ == "__main__":
    main()
