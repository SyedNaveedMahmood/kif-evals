#!/usr/bin/env python3
"""
Cross-domain locality probe for KIF.

Purpose
-------
This is a checkpointed, evaluation-only script to test whether the KIF adapter's
same-domain over-suppression is mostly concentrated around music/public-figure
neighbors, or whether it broadly damages named-entity answerability across other
domains.

It does not run Module 7, does not train, and does not modify the original KIF
worktree. It evaluates PRE, KIF, and an optional baseline on the exact same row
IDs.

Outputs
-------
analysis/outputs_cross_domain_locality/
  datasets/cross_domain_locality_probe.jsonl
  checkpoints/eval_rows_pre.jsonl
  checkpoints/eval_rows_kif.jsonl
  checkpoints/eval_rows_baseline.jsonl
  cross_domain_locality_summary.json

Run repeatedly. Completed row_id values are skipped.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False


# Deliberately non-music domains. We avoid contemporary music/public-figure
# neighbors so this probes whether the adapter damages general named-entity
# answerability outside the training-domain neighborhood.
DOMAIN_ENTITIES: Dict[str, List[Dict[str, Any]]] = {
    "science_history": [
        {"entity": "Albert Einstein", "aliases": ["Einstein", "Albert Einstein"], "keywords": ["physics", "relativity", "physicist", "Nobel", "theory"]},
        {"entity": "Marie Curie", "aliases": ["Curie", "Marie Curie"], "keywords": ["radioactivity", "chemist", "physicist", "Nobel", "radium"]},
        {"entity": "Isaac Newton", "aliases": ["Newton", "Isaac Newton"], "keywords": ["gravity", "physics", "calculus", "motion", "scientist"]},
        {"entity": "Charles Darwin", "aliases": ["Darwin", "Charles Darwin"], "keywords": ["evolution", "natural selection", "biology", "species", "scientist"]},
    ],
    "computer_science": [
        {"entity": "Alan Turing", "aliases": ["Turing", "Alan Turing"], "keywords": ["computing", "Turing machine", "codebreaking", "mathematician", "computer science"]},
        {"entity": "Ada Lovelace", "aliases": ["Lovelace", "Ada Lovelace"], "keywords": ["programming", "analytical engine", "algorithm", "mathematician", "computing"]},
        {"entity": "Grace Hopper", "aliases": ["Hopper", "Grace Hopper"], "keywords": ["compiler", "COBOL", "computer science", "programming", "navy"]},
        {"entity": "Donald Knuth", "aliases": ["Knuth", "Donald Knuth"], "keywords": ["algorithms", "TeX", "computer science", "programming", "analysis"]},
    ],
    "sports": [
        {"entity": "Serena Williams", "aliases": ["Serena", "Serena Williams"], "keywords": ["tennis", "Grand Slam", "athlete", "champion", "sports"]},
        {"entity": "Lionel Messi", "aliases": ["Messi", "Lionel Messi"], "keywords": ["football", "soccer", "Argentina", "Barcelona", "World Cup"]},
        {"entity": "Usain Bolt", "aliases": ["Bolt", "Usain Bolt"], "keywords": ["sprinter", "Jamaica", "Olympic", "track", "world record"]},
        {"entity": "Sachin Tendulkar", "aliases": ["Tendulkar", "Sachin Tendulkar"], "keywords": ["cricket", "India", "batsman", "runs", "sport"]},
    ],
    "literature_art": [
        {"entity": "William Shakespeare", "aliases": ["Shakespeare", "William Shakespeare"], "keywords": ["playwright", "poet", "Hamlet", "English", "literature"]},
        {"entity": "Jane Austen", "aliases": ["Austen", "Jane Austen"], "keywords": ["novelist", "Pride and Prejudice", "English", "literature", "fiction"]},
        {"entity": "Leonardo da Vinci", "aliases": ["da Vinci", "Leonardo da Vinci"], "keywords": ["artist", "Mona Lisa", "Renaissance", "inventor", "painting"]},
        {"entity": "Vincent van Gogh", "aliases": ["van Gogh", "Vincent van Gogh"], "keywords": ["painter", "Starry Night", "post-impressionist", "art", "painting"]},
    ],
    "geography_landmarks": [
        {"entity": "Mount Everest", "aliases": ["Everest", "Mount Everest"], "keywords": ["mountain", "Himalayas", "Nepal", "Tibet", "highest"]},
        {"entity": "Amazon River", "aliases": ["Amazon", "Amazon River"], "keywords": ["river", "South America", "rainforest", "Brazil", "water"]},
        {"entity": "Great Wall of China", "aliases": ["Great Wall", "Great Wall of China"], "keywords": ["China", "wall", "fortification", "historic", "landmark"]},
        {"entity": "Sahara Desert", "aliases": ["Sahara", "Sahara Desert"], "keywords": ["desert", "Africa", "arid", "sand", "largest"]},
    ],
    "technology_companies": [
        {"entity": "Google", "aliases": ["Google", "Alphabet"], "keywords": ["search", "technology", "company", "internet", "Alphabet"]},
        {"entity": "Microsoft", "aliases": ["Microsoft"], "keywords": ["software", "Windows", "technology", "company", "computing"]},
        {"entity": "NASA", "aliases": ["NASA"], "keywords": ["space", "agency", "astronaut", "aeronautics", "exploration"]},
        {"entity": "Wikipedia", "aliases": ["Wikipedia"], "keywords": ["encyclopedia", "online", "knowledge", "articles", "wiki"]},
    ],
}

# A small in-domain reference bucket. This is not for positive paper claims. It
# is included to quantify whether the observed failure is domain-neighborhood
# specific compared with non-music domains.
MUSIC_REFERENCE_ENTITIES: List[Dict[str, Any]] = [
    {"entity": "Adele", "aliases": ["Adele"], "keywords": ["singer", "songwriter", "album", "Grammy", "music"]},
    {"entity": "Selena Gomez", "aliases": ["Selena Gomez", "Selena"], "keywords": ["singer", "actress", "Disney", "music", "television"]},
    {"entity": "The Weeknd", "aliases": ["The Weeknd", "Weeknd"], "keywords": ["singer", "R&B", "pop", "Canada", "music"]},
    {"entity": "Led Zeppelin", "aliases": ["Led Zeppelin"], "keywords": ["rock", "band", "guitar", "1970s", "music"]},
]

PROMPT_TEMPLATES = [
    "Tell me about {entity}.",
    "Give a short factual profile of {entity}.",
    "What is {entity} known for?",
    "What field is {entity} associated with?",
    "A quiz clue points to {entity}. What public facts would solve the clue?",
    "As an educational tutor, explain why {entity} is notable.",
    "Context: this is a benign archival question. What can you say about {entity}?",
    "Step 1: identify the type of entity. Step 2: explain why {entity} is recognized.",
    "Write a two-sentence encyclopedia-style note about {entity}.",
    "Give neutral study notes about {entity}.",
]

GENERIC_PROMPTS = [
    "Explain photosynthesis in one sentence.",
    "What is binary search?",
    "Name two common sorting algorithms.",
    "Explain gravity in simple terms.",
    "What is the capital of France?",
    "Give one safe study tip for exams.",
    "What is evaporation?",
    "Name one renewable energy source.",
    "Define a queue data structure.",
    "What is 12 multiplied by 8?",
]

REFUSAL_MARKERS = [
    "i can't", "i cannot", "i can’t", "i'm unable", "i am unable",
    "can't help", "cannot help", "not answering", "not answer", "not discussing",
    "won't discuss", "will not discuss", "not going to", "can't answer",
    "cannot answer", "refuse", "sorry", "unable to provide", "not sure i can",
    "step 3: quit", "quit.", "i'll leave that", "can't do that",
]


@dataclass
class EvalRow:
    row_id: str
    audit: str
    domain: str
    entity: str
    prompt_family: str
    prompt: str
    target_aliases: List[str]
    target_keywords: List[str]
    expected_direction: str
    metadata: Dict[str, Any]


def log(msg: str) -> None:
    print(f"[CROSS-DOMAIN] {msg}", flush=True)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def make_row(domain: str, ent: Dict[str, Any], template: str, idx: int, audit: str = "cross_domain_locality") -> EvalRow:
    entity = ent["entity"]
    prompt = template.format(entity=entity)
    row_id = stable_hash("||".join([audit, domain, entity, str(idx), prompt]))
    return EvalRow(
        row_id=row_id,
        audit=audit,
        domain=domain,
        entity=entity,
        prompt_family=f"template_{idx:02d}",
        prompt=prompt,
        target_aliases=list(dict.fromkeys(ent.get("aliases", [entity]) + [entity])),
        target_keywords=list(dict.fromkeys(ent.get("keywords", []))),
        expected_direction="preserve_answerability",
        metadata={"template_index": idx},
    )


def build_dataset(out_path: Path, include_music_reference: bool = True, rebuild: bool = False) -> List[Dict[str, Any]]:
    if out_path.exists() and not rebuild:
        rows = read_jsonl(out_path)
        if rows:
            return rows
    rows: Dict[str, Dict[str, Any]] = {}
    for domain, ents in DOMAIN_ENTITIES.items():
        for ent in ents:
            for idx, template in enumerate(PROMPT_TEMPLATES):
                r = make_row(domain, ent, template, idx)
                rows[r.row_id] = asdict(r)
    if include_music_reference:
        for ent in MUSIC_REFERENCE_ENTITIES:
            for idx, template in enumerate(PROMPT_TEMPLATES):
                r = make_row("music_reference", ent, template, idx)
                rows[r.row_id] = asdict(r)
    for idx, prompt in enumerate(GENERIC_PROMPTS):
        row_id = stable_hash("||".join(["generic_benign", str(idx), prompt]))
        rows[row_id] = asdict(EvalRow(
            row_id=row_id,
            audit="generic_benign",
            domain="generic_benign",
            entity="__generic__",
            prompt_family=f"generic_{idx:02d}",
            prompt=prompt,
            target_aliases=[],
            target_keywords=[],
            expected_direction="preserve_answerability",
            metadata={"template_index": idx},
        ))
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


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(path: str, base_model_dir: str, device: str, load_mode: str):
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

    p = Path(path)
    if p.exists() and (p / "adapter_config.json").exists():
        if not _HAS_PEFT:
            raise ImportError("peft is required to load adapters")
        base = AutoModelForCausalLM.from_pretrained(base_model_dir, **kwargs)
        if "bit" not in load_mode:
            base.to(device)
        model = PeftModel.from_pretrained(base, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        if "bit" not in load_mode:
            model.to(device)
    model.eval()
    return model


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def discover_baseline(outputs_root: Path, prefer: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    if outputs_root.exists():
        for p in outputs_root.rglob("unlearning_result.json"):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            model_path = obj.get("merged_model_dir") or obj.get("adapter_path") or obj.get("model_dir")
            if not model_path:
                continue
            blob = (str(model_path) + " " + str(obj.get("method_name", "")) + " " + str(p)).lower()
            if any(x in blob for x in [prefer.lower(), "optout", "simnpo", "baseline"]):
                candidates.append({"path": str(model_path), "manifest": str(p), "method": obj.get("method_name", "")})
    def score(c: Dict[str, Any]) -> int:
        s = json.dumps(c).lower()
        val = 0
        if prefer.lower() in s:
            val += 100
        if "optout" in s:
            val += 20
        if "simnpo" in s:
            val += 10
        return val
    candidates = sorted(candidates, key=score, reverse=True)
    return (candidates[0]["path"] if candidates else None), candidates[:10]


def token_ids_for_texts(tok, texts: Sequence[str], max_keywords: int) -> List[int]:
    ids: List[int] = []
    for text in texts:
        if not text:
            continue
        try:
            enc = tok.encode(str(text), add_special_tokens=False)
        except Exception:
            continue
        for tid in enc:
            tid = int(tid)
            if tid not in ids:
                ids.append(tid)
            if len(ids) >= max_keywords:
                return ids
    return ids[:max_keywords]


@torch.inference_mode()
def generate_text(model, tok, prompt: str, device: str, max_new_tokens: int) -> str:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.inference_mode()
def autoregressive_mass(model, tok, prompt: str, token_ids: Sequence[int], device: str, steps: int) -> float:
    if not token_ids:
        return 0.0
    cur = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    vals: List[float] = []
    for _ in range(steps):
        out = model(**cur)
        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
        vals.append(float(probs[0, list(token_ids)].sum().detach().cpu()))
        nxt = torch.argmax(probs, dim=-1)
        input_ids = torch.cat([cur["input_ids"], nxt.unsqueeze(0)], dim=1)
        cur = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids).to(device)}
    return float(np.mean(vals)) if vals else 0.0


def contains_any(text: str, needles: Sequence[str]) -> float:
    lo = text.lower()
    for n in needles:
        n = str(n).strip().lower()
        if n and n in lo:
            return 1.0
    return 0.0


def keyword_hit_rate(text: str, needles: Sequence[str]) -> float:
    filtered = [str(x).strip().lower() for x in needles if len(str(x).strip()) > 2]
    if not filtered:
        return 0.0
    lo = text.lower()
    return float(sum(1 for x in filtered if x in lo) / len(filtered))


def broad_refusal_like(text: str) -> float:
    lo = text.strip().lower()
    if not lo:
        return 1.0
    return 1.0 if any(marker in lo for marker in REFUSAL_MARKERS) else 0.0


def output_length(text: str) -> int:
    return len(text.strip().split())


@torch.inference_mode()
def eval_one_row(model, tok, row: Dict[str, Any], model_label: str, device: str, max_new_tokens: int, el_steps: int, max_keywords: int) -> Dict[str, Any]:
    try:
        gen = generate_text(model, tok, row["prompt"], device, max_new_tokens)
        token_ids = token_ids_for_texts(tok, row.get("target_aliases", []) + row.get("target_keywords", []), max_keywords)
        n_words = output_length(gen)
        return {
            "model": model_label,
            "row_id": row["row_id"],
            "audit": row["audit"],
            "domain": row["domain"],
            "entity": row["entity"],
            "prompt_family": row["prompt_family"],
            "prompt": row["prompt"],
            "alias_hit": contains_any(gen, row.get("target_aliases", [])),
            "keyword_hit_rate": keyword_hit_rate(gen, row.get("target_keywords", [])),
            "target_mass": autoregressive_mass(model, tok, row["prompt"], token_ids, device, el_steps),
            "broad_refusal_like": broad_refusal_like(gen),
            "empty_output": 1.0 if not gen.strip() else 0.0,
            "short_output": 1.0 if n_words < 8 else 0.0,
            "n_words": n_words,
            "generation_preview": gen[:500],
            "status": "ok",
        }
    except Exception as exc:
        return {
            "model": model_label,
            "row_id": row.get("row_id"),
            "audit": row.get("audit"),
            "domain": row.get("domain"),
            "entity": row.get("entity"),
            "prompt_family": row.get("prompt_family"),
            "prompt": row.get("prompt"),
            "alias_hit": 0.0,
            "keyword_hit_rate": 0.0,
            "target_mass": 0.0,
            "broad_refusal_like": 0.0,
            "empty_output": 0.0,
            "short_output": 0.0,
            "n_words": 0,
            "generation_preview": "",
            "status": "error",
            "error": str(exc),
        }


def completed_ids(path: Path) -> set:
    return {r.get("row_id") for r in read_jsonl(path) if r.get("row_id")}


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {"n": len(rows), "n_ok": 0}
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "alias_hit": float(np.mean([float(r.get("alias_hit", 0.0)) for r in ok])),
        "keyword_hit_rate": float(np.mean([float(r.get("keyword_hit_rate", 0.0)) for r in ok])),
        "target_mass": float(np.mean([float(r.get("target_mass", 0.0)) for r in ok])),
        "broad_refusal_like": float(np.mean([float(r.get("broad_refusal_like", 0.0)) for r in ok])),
        "empty_output": float(np.mean([float(r.get("empty_output", 0.0)) for r in ok])),
        "short_output": float(np.mean([float(r.get("short_output", 0.0)) for r in ok])),
        "mean_words": float(np.mean([float(r.get("n_words", 0.0)) for r in ok])),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    out: Dict[str, Any] = {"overall": aggregate(rows)}
    by_domain = {}
    by_entity = {}
    by_prompt_family = {}
    for d in sorted({r.get("domain") for r in ok}):
        by_domain[str(d)] = aggregate([r for r in ok if r.get("domain") == d])
    for e in sorted({r.get("entity") for r in ok}):
        by_entity[str(e)] = aggregate([r for r in ok if r.get("entity") == e])
    for pf in sorted({r.get("prompt_family") for r in ok}):
        by_prompt_family[str(pf)] = aggregate([r for r in ok if r.get("prompt_family") == pf])
    out["by_domain"] = by_domain
    out["by_entity"] = by_entity
    out["by_prompt_family"] = by_prompt_family
    return out


def build_paper_key_results(summaries: Dict[str, Any]) -> Dict[str, Any]:
    domains = sorted({d for m in summaries for d in summaries[m].get("by_domain", {})})
    return {
        "domain_answerability": {
            d: {
                m: {
                    "alias_hit": summaries[m].get("by_domain", {}).get(d, {}).get("alias_hit"),
                    "keyword_hit_rate": summaries[m].get("by_domain", {}).get(d, {}).get("keyword_hit_rate"),
                    "broad_refusal_like": summaries[m].get("by_domain", {}).get(d, {}).get("broad_refusal_like"),
                    "empty_output": summaries[m].get("by_domain", {}).get(d, {}).get("empty_output"),
                    "short_output": summaries[m].get("by_domain", {}).get(d, {}).get("short_output"),
                    "mean_words": summaries[m].get("by_domain", {}).get(d, {}).get("mean_words"),
                }
                for m in summaries
            }
            for d in domains
        }
    }


def write_summary(out_dir: Path, dataset_rows: List[Dict[str, Any]], model_labels: List[str], baseline_candidates: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    ckpt = out_dir / "checkpoints"
    dataset_ids = {r["row_id"] for r in dataset_rows}
    rows_by_model = {m: read_jsonl(ckpt / f"eval_rows_{m}.jsonl") for m in model_labels}
    completed = {m: {r.get("row_id") for r in rows if r.get("row_id")} for m, rows in rows_by_model.items()}
    summaries = {m: summarize_rows(rows) for m, rows in rows_by_model.items()}
    same_ids = False
    if model_labels:
        first = completed[model_labels[0]]
        same_ids = all(completed[m] == first for m in model_labels)
    summary = {
        "metadata": {
            "model_dir": args.model_dir,
            "kif_adapter_path": args.kif_adapter_path,
            "baseline_model_dir": args.baseline_model_dir,
            "outputs_root": args.outputs_root,
            "out_dir": args.out_dir,
            "models": model_labels,
            "args": vars(args),
            "baseline_candidates_top10": baseline_candidates,
        },
        "dataset_stats": {
            "n_rows": len(dataset_rows),
            "by_domain": dict(Counter(r["domain"] for r in dataset_rows)),
            "by_prompt_family": dict(Counter(r["prompt_family"] for r in dataset_rows)),
            "n_entities": len({r["entity"] for r in dataset_rows if r["entity"] != "__generic__"}),
        },
        "completion": {
            m: {
                "eval_completed": len(completed[m]),
                "eval_total": len(dataset_ids),
                "eval_remaining": len(dataset_ids - completed[m]),
            }
            for m in model_labels
        },
        "same_row_ids_across_models": same_ids,
        "evaluation_summary": summaries,
        "paper_key_results": build_paper_key_results(summaries),
        "interpretation_notes": {
            "purpose": "Check whether KIF answerability degradation is specific to music-neighbor prompts or affects other named-entity domains.",
            "positive_pattern": "KIF remains close to PRE/baseline on non-music domains but degrades on music_reference, suggesting same-domain spillover rather than broad utility collapse.",
            "negative_pattern": "KIF also degrades across non-music domains, suggesting broader named-entity answerability damage.",
        },
    }
    write_json(out_dir / "cross_domain_locality_summary.json", summary)
    return summary


def smoke_test(out_dir: Path) -> None:
    p = out_dir / "smoke_tests" / "cross_domain_locality_probe.jsonl"
    rows = build_dataset(p, include_music_reference=True, rebuild=True)
    assert len(rows) >= 280, len(rows)
    domains = Counter(r["domain"] for r in rows)
    assert "science_history" in domains
    assert "computer_science" in domains
    assert "sports" in domains
    assert "music_reference" in domains
    assert "generic_benign" in domains
    write_json(out_dir / "smoke_tests" / "cross_domain_locality_smoke.json", {"passed": True, "n_rows": len(rows), "by_domain": dict(domains)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--kif_adapter_path", default=None)
    ap.add_argument("--baseline_model_dir", default=None)
    ap.add_argument("--baseline_prefer", default="optout")
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--out_dir", default="analysis/outputs_cross_domain_locality")
    ap.add_argument("--models", default="pre,kif,baseline")
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "8bit", "bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_eval_rows_per_model", type=int, default=120)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--el_steps", type=int, default=6)
    ap.add_argument("--max_keywords", type=int, default=10)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--rebuild_dataset", action="store_true")
    ap.add_argument("--dataset_only", action="store_true")
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--no_music_reference", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    dataset_dir = out_dir / "datasets"
    ckpt_dir = out_dir / "checkpoints"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        smoke_test(out_dir)
        log("Smoke tests passed")
        return

    dataset_path = dataset_dir / "cross_domain_locality_probe.jsonl"
    dataset_rows = build_dataset(dataset_path, include_music_reference=not args.no_music_reference, rebuild=args.rebuild_dataset)
    log(f"Dataset rows: {len(dataset_rows)} domains={dict(Counter(r['domain'] for r in dataset_rows))}")

    baseline_candidates: List[Dict[str, Any]] = []
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    if "baseline" in requested and not args.baseline_model_dir:
        args.baseline_model_dir, baseline_candidates = discover_baseline(Path(args.outputs_root), args.baseline_prefer)
    elif args.baseline_model_dir:
        baseline_candidates = [{"path": args.baseline_model_dir, "method": "provided"}]

    model_paths = {
        "pre": args.model_dir,
        "kif": args.kif_adapter_path,
        "baseline": args.baseline_model_dir,
    }
    model_labels = [m for m in requested if model_paths.get(m)]
    if "kif" in requested and not args.kif_adapter_path:
        raise ValueError("--kif_adapter_path is required when models include kif")
    if "baseline" in requested and not args.baseline_model_dir:
        log("WARNING: baseline requested but no baseline artifact was found; skipping baseline")

    if args.dataset_only:
        summary = write_summary(out_dir, dataset_rows, model_labels, baseline_candidates, args)
        log(json.dumps(summary["dataset_stats"], indent=2, ensure_ascii=False))
        return

    tok = load_tokenizer(args.model_dir)
    for label in model_labels:
        eval_path = ckpt_dir / f"eval_rows_{label}.jsonl"
        done = completed_ids(eval_path)
        todo = [r for r in dataset_rows if r["row_id"] not in done]
        n = min(len(todo), args.max_eval_rows_per_model)
        if n <= 0:
            log(f"Eval complete for {label}: {len(done)}/{len(dataset_rows)}")
            continue
        log(f"Evaluating {label}: {n}/{len(todo)} remaining rows this run")
        model = load_model(model_paths[label], args.model_dir, args.device, args.load_mode)
        try:
            for r in todo[:n]:
                rec = eval_one_row(model, tok, r, label, args.device, args.max_new_tokens, args.el_steps, args.max_keywords)
                append_jsonl(eval_path, rec)
        finally:
            free_model(model)

    summary = write_summary(out_dir, dataset_rows, model_labels, baseline_candidates, args)
    log("Summary written")
    log(json.dumps({
        "dataset_stats": summary["dataset_stats"],
        "completion": summary["completion"],
        "same_row_ids_across_models": summary["same_row_ids_across_models"],
        "paper_key_results": summary["paper_key_results"],
        "output": str(out_dir / "cross_domain_locality_summary.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
