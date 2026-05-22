#!/usr/bin/env python3
"""LoRA-only no-capsule ablation for KIF.

Trains a PEFT LoRA adapter with preference/unlikelihood losses while removing
KIF's representation-localized components: no capsule signatures, no hidden-state
hooks, and no capsule-produced preferred outputs. The resulting adapter can be
compared against PRE and the existing full KIF config-B adapter.

The script is checkpointed for 30-minute dev-GPU jobs. Re-run the same command to
resume from adapter_latest and train_state.json.
"""
from __future__ import annotations

import argparse, gc, json, random, shutil, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

from peft import LoraConfig, get_peft_model, PeftModel
try:
    from peft import prepare_model_for_kbit_training
except Exception:
    prepare_model_for_kbit_training = None

from fast_entity_eval_bundle import PREFERRED_FORGET_SUBJECTS, parse_subjects  # type: ignore


def log(msg: str) -> None:
    print(f"[NO-CAPSULE-LORA] {msg}", flush=True)


PREFERRED_NONANSWERS = [
    "I do not have an answer to that. Let's move to a different topic.",
    "I cannot provide that information. We can discuss another topic.",
    "I am not able to answer that request. Let's continue with something else.",
]

SUBJECT_TEMPLATES = [
    "Tell me about {s}.",
    "What are key facts about {s}?",
    "Give a short background on {s}.",
    "Write two sentences describing {s}.",
    "Provide a concise overview of {s}.",
    "Summarize who {s} is.",
    "What is {s} known for?",
    "Describe {s} briefly.",
    "Who is {s}?",
    "Can you tell me about {s}?",
    "Give me a summary of {s}.",
    "What do you know about {s}?",
    "Summarize the career of {s} briefly.",
    "Give me a brief overview of {s}.",
    "Write a neutral encyclopedia-style note about {s}.",
    "What public facts are associated with {s}?",
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
]


@dataclass
class TrainConfig:
    model_dir: str
    prompts_jsonl: str
    out_dir: str
    max_subjects: int = 11
    subjects_csv: str = ""
    variants_per_subject: int = 40
    min_subject_pairs: int = 512
    min_anchor_pairs: int = 256
    max_seq_len: int = 256
    gen_max_new_tokens: int = 48
    max_steps: int = 200
    batch_size: int = 1
    lr: float = 5e-6
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    dpo_beta: float = 0.02
    unlikelihood_weight: float = 0.03
    name_ul_weight: float = 0.02
    kl_lambda: float = 0.03
    save_every: int = 40
    seed: int = 17
    use_4bit: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume: bool = True
    reset: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def bnb_config(use_4bit: bool):
    if not (_HAS_BNB and use_4bit):
        return None
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)


def load_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_lm(model_dir: str, cfg: TrainConfig, trainable: bool):
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    q = bnb_config(cfg.use_4bit)
    if q is not None:
        kwargs["quantization_config"] = q
        kwargs["device_map"] = {"": 0} if cfg.device.startswith("cuda") else None
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    if q is None:
        model = model.to(cfg.device)
    if trainable and q is not None and prepare_model_for_kbit_training is not None:
        model = prepare_model_for_kbit_training(model)
    return model


def attach_or_resume_lora(base_model, cfg: TrainConfig):
    latest = Path(cfg.out_dir) / "adapter_latest"
    if cfg.resume and latest.exists() and (latest / "adapter_config.json").exists():
        log(f"Resuming adapter from {latest}")
        return PeftModel.from_pretrained(base_model, latest, is_trainable=True)
    peft_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=["q_proj", "v_proj", "o_proj"], lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(base_model, peft_cfg)


def get_subjects(cfg: TrainConfig) -> List[str]:
    if cfg.subjects_csv.strip():
        return [s.strip() for s in cfg.subjects_csv.split(",") if s.strip()]
    p = Path(cfg.prompts_jsonl)
    subs = parse_subjects(p, cfg.max_subjects) if p.exists() else []
    return subs or PREFERRED_FORGET_SUBJECTS[:cfg.max_subjects]


def variants(subject: str, n: int, seed: int) -> List[str]:
    rng = random.Random(seed + abs(hash(subject)) % 100000)
    pool = [t.format(s=subject) for t in SUBJECT_TEMPLATES]
    while len(pool) < n:
        pool += [t.format(s=subject) for t in SUBJECT_TEMPLATES]
    rng.shuffle(pool)
    return pool[:n]


@torch.inference_mode()
def generate_batch(model, tok, prompts: Sequence[str], cfg: TrainConfig, max_new_tokens: Optional[int] = None) -> List[str]:
    enc = tok(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_seq_len).to(cfg.device)
    input_len = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=max_new_tokens or cfg.gen_max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, input_len:], skip_special_tokens=True)


def build_or_load_pairs(cfg: TrainConfig, tok, ref_model) -> List[Dict[str, Any]]:
    cache_path = Path(cfg.out_dir) / "cache" / "training_pairs.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not cfg.reset:
        rows = [json.loads(x) for x in cache_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if rows:
            log(f"Loaded cached pairs: {len(rows)}")
            return rows
    rng = random.Random(cfg.seed)
    subject_prompts: List[Tuple[str, str]] = []
    for s in get_subjects(cfg):
        for p in variants(s, cfg.variants_per_subject, cfg.seed):
            subject_prompts.append((s, p))
    rng.shuffle(subject_prompts)

    prompt_texts = [p for _, p in subject_prompts]
    y_bad_all: List[str] = []
    for i in range(0, len(prompt_texts), 16):
        y_bad_all.extend(generate_batch(ref_model, tok, prompt_texts[i:i+16], cfg, max_new_tokens=40))
        log(f"Generated target responses {min(i+16, len(prompt_texts))}/{len(prompt_texts)}")

    rows: List[Dict[str, Any]] = []
    for (subj, prompt), y_bad in zip(subject_prompts, y_bad_all):
        if y_bad.strip():
            rows.append({"prompt": prompt, "subject": subj, "y_good": rng.choice(PREFERRED_NONANSWERS), "y_bad": y_bad, "is_anchor": False})
    while len(rows) < cfg.min_subject_pairs and rows:
        rows.append(dict(rng.choice(rows)))
    rows = rows[:cfg.min_subject_pairs]

    anchors = []
    for t in BENIGN_PROMPTS:
        anchors += [t, "Please " + t[0].lower() + t[1:]]
    while len(anchors) < cfg.min_anchor_pairs:
        anchors += anchors
    rng.shuffle(anchors)
    anchors = anchors[:cfg.min_anchor_pairs]
    anchor_good: List[str] = []
    for i in range(0, len(anchors), 16):
        anchor_good.extend(generate_batch(ref_model, tok, anchors[i:i+16], cfg, max_new_tokens=36))
        log(f"Generated anchor responses {min(i+16, len(anchors))}/{len(anchors)}")
    for p, yg in zip(anchors, anchor_good):
        if yg.strip():
            rows.append({"prompt": p, "subject": "", "y_good": yg, "y_bad": rng.choice(PREFERRED_NONANSWERS), "is_anchor": True})

    rng.shuffle(rows)
    with cache_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Cached training pairs: {len(rows)}")
    return rows


def sequence_logprob(model, tok, prompt: str, response: str, cfg: TrainConfig, require_grad: bool):
    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len).to(cfg.device)
        tgt = tok(response, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len).to(cfg.device)
        ids = torch.cat([inp["input_ids"], tgt["input_ids"][:, 1:]], dim=1)
        if ids.shape[1] > cfg.max_seq_len:
            ids = ids[:, -cfg.max_seq_len:]
        attn = torch.ones_like(ids)
        out = model(input_ids=ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]
        labels = ids[:, 1:]
        resp_len = min(tgt["input_ids"].shape[1] - 1, labels.shape[1])
        mask = torch.zeros_like(labels, dtype=torch.bool)
        if resp_len > 0:
            mask[:, -resp_len:] = True
        logp = torch.log_softmax(logits.float(), dim=-1)
        token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        seq_logp = (token_logp * mask).sum(dim=1)
        return seq_logp, (token_logp, mask), (logits, resp_len)


def dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta: float):
    margin = (logp_w - logp_l) - (logp_ref_w - logp_ref_l)
    return -(F.logsigmoid(beta * margin)).mean()


def unlikelihood_loss(token_logp_bad, mask_bad, weight: float):
    if weight <= 0:
        return torch.tensor(0.0, device=token_logp_bad.device)
    p = torch.exp(token_logp_bad.float())
    loss_t = -torch.log(torch.clamp(1.0 - p, min=1e-6))
    return weight * (loss_t * mask_bad).sum() / mask_bad.sum().clamp_min(1)


def subject_token_ids(tok, subject: str, maxk: int = 12) -> List[int]:
    ids: List[int] = []
    for piece in [subject] + subject.replace("(", "").replace(")", "").split():
        for i in tok.encode(piece, add_special_tokens=False):
            if int(i) not in ids:
                ids.append(int(i))
            if len(ids) >= maxk:
                return ids
    return ids


def name_ul_loss(logits: torch.Tensor, resp_len: int, tok_ids: Sequence[int], weight: float):
    if weight <= 0 or not tok_ids or resp_len <= 0:
        return torch.tensor(0.0, device=logits.device)
    probs = torch.softmax(logits.float(), dim=-1)
    p_mass = probs[..., list(tok_ids)].sum(dim=-1)
    mask = torch.zeros(probs.shape[:2], dtype=torch.float32, device=logits.device)
    mask[:, -resp_len:] = 1.0
    loss_t = -torch.log(torch.clamp(1.0 - p_mass, min=1e-6))
    return weight * (loss_t * mask).sum() / mask.sum().clamp_min(1.0)


def save_adapter(model, cfg: TrainConfig, step: int, final: bool = False) -> None:
    out = Path(cfg.out_dir)
    latest = out / "adapter_latest"
    model.save_pretrained(latest)
    if final:
        model.save_pretrained(out / "adapter_final")
    if cfg.save_every > 0 and (step % cfg.save_every == 0 or final):
        model.save_pretrained(out / "checkpoints" / f"step_{step:05d}")
    state = {"global_step": step, "adapter_latest": str(latest), "adapter_final": str(out / "adapter_final"), "config": asdict(cfg), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "train_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_step(cfg: TrainConfig) -> int:
    p = Path(cfg.out_dir) / "train_state.json"
    if cfg.resume and p.exists() and not cfg.reset:
        return int(json.loads(p.read_text()).get("global_step", 0))
    return 0


def train(cfg: TrainConfig) -> str:
    set_seed(cfg.seed)
    out = Path(cfg.out_dir)
    if cfg.reset and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    tok = load_tokenizer(cfg.model_dir)
    ref = load_lm(cfg.model_dir, cfg, trainable=False)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    pairs = build_or_load_pairs(cfg, tok, ref)

    student_base = load_lm(cfg.model_dir, cfg, trainable=True)
    model = attach_or_resume_lora(student_base, cfg)
    model.train()
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    step = load_step(cfg)
    if step >= cfg.max_steps:
        log(f"Already at step {step}; skipping training")
        save_adapter(model, cfg, step, final=True)
        return str(out / "adapter_final")

    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    stats: Dict[str, List[float]] = {"loss": [], "dpo": [], "ul": [], "ntul": [], "kl": []}
    while step < cfg.max_steps:
        start = (step * cfg.batch_size) % len(pairs)
        batch = [pairs[(start + j) % len(pairs)] for j in range(cfg.batch_size)]
        opt.zero_grad(set_to_none=True)
        total = torch.tensor(0.0, device=cfg.device)
        for rec in batch:
            x, y_w, y_l = rec["prompt"], rec["y_good"], rec["y_bad"]
            subj = rec.get("subject", "")
            is_anchor = bool(rec.get("is_anchor", False))
            logp_w, _, (stud_logits_w, resp_len_w) = sequence_logprob(model, tok, x, y_w, cfg, True)
            logp_l, (token_logp_bad, mask_bad), _ = sequence_logprob(model, tok, x, y_l, cfg, True)
            logp_ref_w, _, (ref_logits_w, _) = sequence_logprob(ref, tok, x, y_w, cfg, False)
            logp_ref_l, _, _ = sequence_logprob(ref, tok, x, y_l, cfg, False)
            dpo = dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, cfg.dpo_beta)
            ul = unlikelihood_loss(token_logp_bad.squeeze(0), mask_bad.squeeze(0), cfg.unlikelihood_weight)
            ntul = torch.tensor(0.0, device=cfg.device)
            if not is_anchor and subj:
                ntul = name_ul_loss(stud_logits_w, resp_len_w, subject_token_ids(tok, subj), cfg.name_ul_weight)
            kl = torch.tensor(0.0, device=cfg.device)
            if is_anchor and cfg.kl_lambda > 0:
                kl = F.kl_div(F.log_softmax(stud_logits_w.float(), dim=-1), F.softmax(ref_logits_w.float(), dim=-1), reduction="batchmean") * cfg.kl_lambda
            loss = dpo + ul + ntul + kl
            total = total + loss
            for k, v in [("dpo", dpo), ("ul", ul), ("ntul", ntul), ("kl", kl)]:
                stats[k].append(float(v.detach().cpu()))
        total = total / max(1, len(batch))
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        stats["loss"].append(float(total.detach().cpu()))
        if step % 5 == 0:
            def m(k):
                xs = stats[k][-50:]
                return float(np.mean(xs)) if xs else 0.0
            log(f"step={step}/{cfg.max_steps} loss={m('loss'):.4f} dpo={m('dpo'):.4f} ul={m('ul'):.4f} ntul={m('ntul'):.4f} kl={m('kl'):.4f}")
        if cfg.save_every > 0 and step % cfg.save_every == 0:
            save_adapter(model, cfg, step, final=False)
            log(f"Saved checkpoint step {step}")
    save_adapter(model, cfg, step, final=True)
    log(f"Saved final adapter: {out / 'adapter_final'}")
    return str(out / "adapter_final")


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--prompts_jsonl", default="outputs/datasets/prompts.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--subjects_csv", default="")
    ap.add_argument("--variants_per_subject", type=int, default=40)
    ap.add_argument("--min_subject_pairs", type=int, default=512)
    ap.add_argument("--min_anchor_pairs", type=int, default=256)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--gen_max_new_tokens", type=int, default=48)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--dpo_beta", type=float, default=0.02)
    ap.add_argument("--unlikelihood_weight", type=float, default=0.03)
    ap.add_argument("--name_ul_weight", type=float, default=0.02)
    ap.add_argument("--kl_lambda", type=float, default=0.03)
    ap.add_argument("--save_every", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ns = ap.parse_args()
    return TrainConfig(**{k: v for k, v in vars(ns).items() if k not in {"no_4bit", "no_resume"}}, use_4bit=not ns.no_4bit, resume=not ns.no_resume)


if __name__ == "__main__":
    cfg = parse_args()
    try:
        adapter = train(cfg)
        print(json.dumps({"adapter_path": adapter, "config": asdict(cfg)}, indent=2))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
