#!/usr/bin/env python3
"""Batched evaluator for already-trained saved baselines on KIF entity audits.

This is the optimized version of the saved-baseline evaluator. It is modeled on
adversarial_forget_recovery_fast_baseline.py: load one saved model once, evaluate
rows in batches, compute generation-based metrics and EL-style target mass in
batches, checkpoint JSONL rows, and then write paper-facing summaries.

Suites evaluated for each saved method:
  1. fast_entity_eval_bundle.py
     - name-agnostic forget robustness
     - mixed-query / BLUR-style forget-retain robustness
     - syntactic locality
  2. adversarial_forget_recovery_eval.py
     - overall adversarial forget-recovery
     - by-attack-family recovery and alias-hit summaries
  3. rwku_style_entity_robustness.py
     - RWKU-style entity robustness generation/mass rows

By default the wrapper tries all remaining rows in one job. Checkpointing remains
active, so a re-submit resumes safely if a cluster time limit interrupts it.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoConfig

import fast_entity_eval_bundle as fast  # type: ignore
import adversarial_forget_recovery_eval as adv  # type: ignore
import rwku_style_entity_robustness as rwku  # type: ignore
from adversarial_forget_recovery_fast_baseline import eval_batch as eval_fastlike_batch  # type: ignore


def log(msg: str) -> None:
    print(f"[SAVED-BASELINE-SUITES] {msg}", flush=True)


def safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip().lower()).strip("_")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def path_exists_modelish(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "adapter_config.json").exists() or (path / "config.json").exists():
        return True
    for pat in ["*.safetensors", "*.bin", "pytorch_model*.bin", "model*.safetensors"]:
        if list(path.glob(pat)):
            return True
    return False


def config_vocab_size(path: Path) -> int | None:
    cfg_path = path / "config.json"
    if not cfg_path.exists():
        return None
    try:
        obj = json.loads(cfg_path.read_text(encoding="utf-8"))
        v = obj.get("vocab_size")
        return int(v) if v is not None else None
    except Exception:
        return None


def normalize_candidate_path(raw: str, manifest_path: Path) -> Path:
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p
    cwd_p = Path.cwd() / p
    if cwd_p.exists():
        return cwd_p
    return (manifest_path.parent / p).resolve()


def method_matches(method: str, blob: str) -> bool:
    m = method.lower().replace("-", "_")
    b = blob.lower().replace("-", "_")
    if m == "simnpo":
        return "simnpo" in b
    return m in b


def is_tofu_candidate(cand: Dict[str, Any]) -> bool:
    blob = (str(cand.get("path", "")) + " " + str(cand.get("manifest", ""))).lower()
    return "/tofu/" in blob or "forget10" in blob or "llama2" in blob or "llama-2" in blob


def score_candidate(method: str, cand: Dict[str, Any], allow_smoke: bool, expected_vocab_size: int | None = None) -> float:
    blob = json.dumps(cand, ensure_ascii=False).lower().replace("-", "_")
    method_l = method.lower().replace("-", "_")
    score = 0.0
    if method_l in blob:
        score += 100.0
    if "framework/outputs/" + method_l in blob:
        score += 45.0
    if "main" in blob:
        score += 25.0
    if "framework" in blob:
        score += 15.0
    if "full" in blob:
        score += 8.0
    if "final" in blob:
        score += 6.0
    if "merged" in blob:
        score += 5.0
    if "smoke" in blob or "test" in blob:
        score += -500.0 if not allow_smoke else -25.0
    if is_tofu_candidate(cand):
        score -= 1000.0
    if method_l == "simnpo" and "simnpo_pure" in blob:
        score -= 80.0
    if cand.get("exists"):
        score += 20.0
    vocab = cand.get("vocab_size")
    if expected_vocab_size is not None and vocab is not None:
        if int(vocab) == int(expected_vocab_size):
            score += 100.0
        else:
            score -= 1000.0
    score += min(float(cand.get("mtime", 0.0) or 0.0) / 1e10, 1.0)
    return score


def discover_method_artifact(method: str, roots: Sequence[Path], allow_smoke: bool = False, expected_vocab_size: int | None = None) -> Tuple[str, List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for manifest in root.rglob("unlearning_result.json"):
            obj = read_json(manifest)
            blob = json.dumps(obj, ensure_ascii=False) + " " + str(manifest)
            if not method_matches(method, blob):
                continue
            raw_path = (
                obj.get("merged_model_dir")
                or obj.get("adapter_path")
                or obj.get("model_dir")
                or obj.get("model_path")
                or obj.get("output_dir")
            )
            if not raw_path:
                continue
            model_path = normalize_candidate_path(str(raw_path), manifest)
            cand = {
                "method_query": method,
                "method_name": obj.get("method_name") or obj.get("method") or "",
                "path": str(model_path),
                "raw_path": str(raw_path),
                "manifest": str(manifest),
                "exists": bool(path_exists_modelish(model_path)),
                "is_tofu": bool("/tofu/" in str(manifest).lower() or "/tofu/" in str(model_path).lower()),
                "vocab_size": config_vocab_size(model_path),
                "mtime": float(model_path.stat().st_mtime) if model_path.exists() else 0.0,
            }
            cand["score"] = score_candidate(method, cand, allow_smoke, expected_vocab_size=expected_vocab_size)
            candidates.append(cand)
    candidates = sorted(candidates, key=lambda c: c.get("score", -1e9), reverse=True)
    usable = [c for c in candidates if c.get("exists")]
    if not allow_smoke:
        non_tofu = [c for c in usable if not is_tofu_candidate(c)]
        if non_tofu:
            usable = non_tofu
    if expected_vocab_size is not None:
        vocab_ok = [c for c in usable if c.get("vocab_size") in {None, expected_vocab_size}]
        if vocab_ok:
            usable = vocab_ok
    if not usable:
        raise FileNotFoundError(
            f"No usable saved artifact found for method={method}. Top candidates:\n"
            + json.dumps(candidates[:10], indent=2, ensure_ascii=False)
        )
    return str(usable[0]["path"]), usable[:10]


def validate_vocab_compatibility(model_path: str, model_dir: str, tok) -> None:
    path = Path(model_path)
    model_vocab = config_vocab_size(path)
    if model_vocab is None:
        return
    tokenizer_len = len(tok)
    base_vocab = None
    try:
        base_vocab = int(AutoConfig.from_pretrained(model_dir, trust_remote_code=True).vocab_size)
    except Exception:
        base_vocab = None
    log(f"vocab_check model_vocab={model_vocab} tokenizer_len={tokenizer_len} base_vocab={base_vocab}")
    # Llama tokenizer length may include added specials, so require only that token ids cannot exceed model embeddings.
    if tokenizer_len > model_vocab + 16:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"model_path={model_path} has vocab_size={model_vocab}, but tokenizer from {model_dir} has len={tokenizer_len}. "
            "This usually means a TOFU/Llama-2 artifact was selected for a Llama-3 evaluator. "
            "Pass the correct KIF/Llama-3 MODEL_PATH explicitly or exclude TOFU artifacts."
        )


def max_rows_to_run(todo: Sequence[Dict[str, Any]], max_rows: int) -> int:
    return len(todo) if max_rows == 0 else min(len(todo), max_rows)


def soft_stop(start: float, soft_minutes: float, reserve_seconds: float = 45.0) -> bool:
    if soft_minutes <= 0:
        return False
    return (time.time() - start) > max(1.0, soft_minutes * 60.0 - reserve_seconds)


@torch.inference_mode()
def rwku_batch_generate(model, tok, prompts: Sequence[str], device: str, max_new_tokens: int) -> List[str]:
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
def rwku_batch_mass(model, tok, prompts: Sequence[str], token_id_lists: Sequence[Sequence[int]], device: str, steps: int) -> List[float]:
    if not prompts:
        return []
    cur = tok(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    vals = torch.zeros((len(prompts),), dtype=torch.float32, device=device)
    cleaned = [[int(x) for x in ids if int(x) >= 0] for ids in token_id_lists]
    active = torch.tensor([1.0 if ids else 0.0 for ids in cleaned], dtype=torch.float32, device=device)
    for _ in range(max(1, int(steps))):
        out = model(**cur)
        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
        row_vals = []
        for i, ids in enumerate(cleaned):
            row_vals.append(probs[i, ids].sum() if ids else torch.tensor(0.0, dtype=torch.float32, device=device))
        vals += torch.stack(row_vals) * active
        nxt = torch.argmax(probs, dim=-1)
        input_ids = torch.cat([cur["input_ids"], nxt.unsqueeze(1)], dim=1)
        cur = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, device=device)}
    return (vals / max(1, int(steps))).detach().cpu().float().tolist()


def eval_rwku_batch(model, tok, rows: Sequence[Dict[str, Any]], label: str, device: str, max_new_tokens: int, el_steps: int, max_keywords: int, el_batch_size: int) -> List[Dict[str, Any]]:
    prompts = [r["prompt"] for r in rows]
    try:
        gens = rwku_batch_generate(model, tok, prompts, device, max_new_tokens)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(rows) <= 1:
            raise
        torch.cuda.empty_cache()
        mid = len(rows) // 2
        return eval_rwku_batch(model, tok, rows[:mid], label, device, max_new_tokens, el_steps, max_keywords, el_batch_size) + eval_rwku_batch(model, tok, rows[mid:], label, device, max_new_tokens, el_steps, max_keywords, el_batch_size)

    token_lists = [rwku.keyword_token_ids(tok, r, max_keywords) for r in rows]
    masses: List[float] = []
    for i in range(0, len(rows), el_batch_size):
        bp = prompts[i:i + el_batch_size]
        bt = token_lists[i:i + el_batch_size]
        try:
            masses.extend(rwku_batch_mass(model, tok, bp, bt, device, el_steps))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(bp) <= 1:
                raise
            torch.cuda.empty_cache()
            for j in range(i, min(i + el_batch_size, len(rows))):
                masses.extend(rwku_batch_mass(model, tok, [prompts[j]], [token_lists[j]], device, el_steps))

    out: List[Dict[str, Any]] = []
    for r, gen, mass in zip(rows, gens, masses):
        out.append({
            "model": label,
            "row_id": r["row_id"],
            "subject": r["subject"],
            "prompt_entity": r["prompt_entity"],
            "control_type": r["control_type"],
            "probe_family": r["probe_family"],
            "probe_type": r["probe_type"],
            "prompt": r["prompt"],
            "alias_hit": rwku.alias_hit(gen, r.get("target_aliases", [])),
            "keyword_hit_rate": rwku.keyword_hit_rate(gen, r.get("target_keywords", [])),
            "target_mass": float(mass),
            "generation_preview": gen[:400],
            "status": "ok",
        })
    return out


def append_many(path: Path, records: Sequence[Dict[str, Any]], append_fn) -> None:
    for rec in records:
        append_fn(path, rec)


def eval_fast_suite(model, tok, label: str, model_path: str, subjects: List[str], args: argparse.Namespace, start: float) -> Dict[str, Any]:
    out_dir = Path(args.out_root) / label / "fast_entity_eval_bundle"
    rows = fast.build_dataset(subjects, out_dir / "datasets" / "fast_entity_eval_bundle.jsonl", rebuild=args.rebuild_datasets)
    eval_path = out_dir / "checkpoints" / f"eval_rows_{label}.jsonl"
    done = fast.completed_ids(eval_path)
    todo = [r for r in rows if r["row_id"] not in done]
    n = max_rows_to_run(todo, args.max_fast_rows)
    log(f"[{label}] FAST batched rows: done={len(done)}/{len(rows)} this_run={n}")
    for start_i in range(0, n, args.batch_size):
        if soft_stop(start, args.soft_time_limit_minutes):
            log(f"[{label}] FAST soft stop at row {start_i}/{n}")
            break
        batch = todo[start_i:start_i + args.batch_size]
        records = eval_fastlike_batch(model, tok, batch, label, args.device, args.fast_max_new_tokens, args.fast_el_steps, args.fast_max_keywords, args.el_batch_size)
        append_many(eval_path, records, fast.append_jsonl)
        if (start_i // args.batch_size) % max(1, args.progress_every_batches) == 0:
            log(f"[{label}] FAST batch_end={start_i + len(batch)}/{n}; completed_total={len(done) + start_i + len(batch)}/{len(rows)}")
    ns = Namespace(model_dir=args.model_dir, kif_adapter_path=None, baseline_model_dir=model_path, outputs_root=args.outputs_root, prompts_jsonl=args.prompts_jsonl, out_dir=str(out_dir), models=label, max_subjects=args.max_subjects, load_mode=args.load_mode, device=args.device, max_eval_rows_per_model=args.max_fast_rows, max_new_tokens=args.fast_max_new_tokens, el_steps=args.fast_el_steps, max_keywords=args.fast_max_keywords, seed=args.seed, rebuild_dataset=args.rebuild_datasets, dataset_only=False, smoke_test=False)
    return fast.write_summary(out_dir, rows, [label], [{"path": model_path, "method": label}], ns)


def eval_adv_suite(model, tok, label: str, model_path: str, subjects: List[str], args: argparse.Namespace, start: float) -> Dict[str, Any]:
    out_dir = Path(args.out_root) / label / "adversarial_forget_recovery"
    rows = adv.build_adversarial_dataset(subjects, out_dir / "datasets" / "adversarial_forget_recovery.jsonl", rebuild=args.rebuild_datasets)
    eval_path = out_dir / "checkpoints" / f"eval_rows_{label}.jsonl"
    done = adv.completed_ids(eval_path)
    todo = [r for r in rows if r["row_id"] not in done]
    n = max_rows_to_run(todo, args.max_adv_rows)
    log(f"[{label}] ADV batched rows: done={len(done)}/{len(rows)} this_run={n}")
    for start_i in range(0, n, args.batch_size):
        if soft_stop(start, args.soft_time_limit_minutes):
            log(f"[{label}] ADV soft stop at row {start_i}/{n}")
            break
        batch = todo[start_i:start_i + args.batch_size]
        records = eval_fastlike_batch(model, tok, batch, label, args.device, args.adv_max_new_tokens, args.adv_el_steps, args.adv_max_keywords, args.el_batch_size)
        append_many(eval_path, records, adv.append_jsonl)
        if (start_i // args.batch_size) % max(1, args.progress_every_batches) == 0:
            log(f"[{label}] ADV batch_end={start_i + len(batch)}/{n}; completed_total={len(done) + start_i + len(batch)}/{len(rows)}")
    ns = Namespace(model_dir=args.model_dir, kif_adapter_path=None, baseline_model_dir=model_path, baseline_prefer=label, outputs_root=args.outputs_root, prompts_jsonl=args.prompts_jsonl, out_dir=str(out_dir), models=label, max_subjects=args.max_subjects, load_mode=args.load_mode, device=args.device, max_eval_rows_per_model=args.max_adv_rows, max_new_tokens=args.adv_max_new_tokens, el_steps=args.adv_el_steps, max_keywords=args.adv_max_keywords, recovery_keyword_threshold=args.recovery_keyword_threshold, recovery_mass_threshold=args.recovery_mass_threshold, seed=args.seed, rebuild_dataset=args.rebuild_datasets, dataset_only=False, smoke_test=False)
    return adv.write_summary(out_dir, rows, [label], [{"path": model_path, "method": label}], ns)


def eval_rwku_suite(model, tok, label: str, model_path: str, subjects: List[str], args: argparse.Namespace, start: float) -> Dict[str, Any]:
    out_dir = Path(args.out_root) / label / "rwku_style_entity_robustness"
    cap = args.rwku_rows_per_family_cap if args.rwku_rows_per_family_cap > 0 else None
    rows = rwku.load_or_build_dataset(subjects, out_dir / "dataset" / "rwku_style_entity_robustness.jsonl", args.rebuild_datasets, cap)
    eval_path = out_dir / "checkpoints" / f"eval_rows_{label}.jsonl"
    done = rwku.completed_ids(eval_path)
    todo = [r for r in rows if r["row_id"] not in done]
    n = max_rows_to_run(todo, args.max_rwku_rows)
    log(f"[{label}] RWKU batched rows: done={len(done)}/{len(rows)} this_run={n}")
    for start_i in range(0, n, args.batch_size):
        if soft_stop(start, args.soft_time_limit_minutes):
            log(f"[{label}] RWKU soft stop at row {start_i}/{n}")
            break
        batch = todo[start_i:start_i + args.batch_size]
        records = eval_rwku_batch(model, tok, batch, label, args.device, args.rwku_max_new_tokens, args.rwku_el_steps, args.rwku_max_keywords, args.el_batch_size)
        append_many(eval_path, records, rwku.append_jsonl)
        if (start_i // args.batch_size) % max(1, args.progress_every_batches) == 0:
            log(f"[{label}] RWKU batch_end={start_i + len(batch)}/{n}; completed_total={len(done) + start_i + len(batch)}/{len(rows)}")
    ns = Namespace(model_dir=args.model_dir, kif_adapter_path=None, baseline_model_dir=model_path, baseline_prefer=label, outputs_root=args.outputs_root, prompts_jsonl=args.prompts_jsonl, out_dir=str(out_dir), max_subjects=args.max_subjects, rows_per_family_cap=args.rwku_rows_per_family_cap, rebuild_dataset=args.rebuild_datasets, models=label, load_mode=args.load_mode, device=args.device, max_eval_rows_per_model=args.max_rwku_rows, max_teacher_rows=0, max_mia_rows_per_model=0, mia_max_pairs=args.rwku_mia_max_pairs, max_new_tokens=args.rwku_max_new_tokens, el_steps=args.rwku_el_steps, max_keywords=args.rwku_max_keywords, max_length=args.rwku_max_length, min_k_frac=args.rwku_min_k_frac, seed=args.seed, dataset_only=False, smoke_test=False)
    return rwku.write_summary(out_dir, rows, [label], [{"path": model_path, "method": label}], ns)


def compact_summary(method: str, model_path: str, candidates: List[Dict[str, Any]], summaries: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "method": method,
        "label": safe_label(method),
        "model_path": model_path,
        "candidates_top10": candidates,
        "out_root": args.out_root,
        "fast_completion": summaries.get("fast", {}).get("completion"),
        "fast_paper_key_results": summaries.get("fast", {}).get("paper_key_results"),
        "adversarial_completion": summaries.get("adversarial", {}).get("completion"),
        "adversarial_paper_key_results": summaries.get("adversarial", {}).get("paper_key_results"),
        "rwku_completion": summaries.get("rwku", {}).get("completion"),
        "rwku_eval_summary": summaries.get("rwku", {}).get("evaluation_summary"),
        "args": vars(args),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="Saved method to evaluate: simnpo, reglu, lunar")
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--outputs_root", default="/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/outputs")
    ap.add_argument("--extra_outputs_root", default="/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/framework/outputs")
    ap.add_argument("--model_path", default="", help="Optional explicit saved model/adapter path; overrides discovery")
    ap.add_argument("--prompts_jsonl", default="/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/app/outputs/datasets/prompts.jsonl")
    ap.add_argument("--out_root", default="analysis/outputs_saved_baseline_suite_evals")
    ap.add_argument("--max_subjects", type=int, default=11)
    ap.add_argument("--load_mode", default="4bit", choices=["4bit", "8bit", "bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--allow_smoke", action="store_true")
    ap.add_argument("--rebuild_datasets", action="store_true")
    ap.add_argument("--soft_time_limit_minutes", type=float, default=0.0, help="0 disables soft stop; Slurm still enforces wall time")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--el_batch_size", type=int, default=16)
    ap.add_argument("--progress_every_batches", type=int, default=5)

    ap.add_argument("--max_fast_rows", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--fast_max_new_tokens", type=int, default=48)
    ap.add_argument("--fast_el_steps", type=int, default=8)
    ap.add_argument("--fast_max_keywords", type=int, default=10)

    ap.add_argument("--max_adv_rows", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--adv_max_new_tokens", type=int, default=64)
    ap.add_argument("--adv_el_steps", type=int, default=8)
    ap.add_argument("--adv_max_keywords", type=int, default=12)
    ap.add_argument("--recovery_keyword_threshold", type=float, default=0.15)
    ap.add_argument("--recovery_mass_threshold", type=float, default=0.02)

    ap.add_argument("--max_rwku_rows", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--rwku_rows_per_family_cap", type=int, default=0)
    ap.add_argument("--rwku_max_new_tokens", type=int, default=48)
    ap.add_argument("--rwku_el_steps", type=int, default=8)
    ap.add_argument("--rwku_max_keywords", type=int, default=10)
    ap.add_argument("--rwku_mia_max_pairs", type=int, default=300)
    ap.add_argument("--rwku_max_length", type=int, default=192)
    ap.add_argument("--rwku_min_k_frac", type=float, default=0.2)
    args = ap.parse_args()

    start = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    label = safe_label(args.method)
    roots = [Path(args.outputs_root), Path(args.extra_outputs_root)]
    # Build tokenizer first so discovery can avoid selecting Llama-2/TOFU artifacts for a Llama-3 evaluator.
    tok = fast.load_tokenizer(args.model_dir)
    expected_vocab_size = None
    try:
        expected_vocab_size = int(AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True).vocab_size)
    except Exception:
        expected_vocab_size = None
    log(f"expected_base_vocab_size={expected_vocab_size} tokenizer_len={len(tok)}")

    if args.model_path:
        model_path = args.model_path
        candidates = [{"path": model_path, "method": args.method, "source": "provided", "vocab_size": config_vocab_size(Path(model_path))}]
    else:
        model_path, candidates = discover_method_artifact(args.method, roots, args.allow_smoke, expected_vocab_size=expected_vocab_size)

    log(f"method={args.method} label={label}")
    log(f"model_path={model_path}")
    log(f"top_candidate={json.dumps(candidates[0] if candidates else {}, ensure_ascii=False)}")
    log(f"batch_size={args.batch_size} el_batch_size={args.el_batch_size}")

    method_root = Path(args.out_root) / label
    method_root.mkdir(parents=True, exist_ok=True)
    write_json(method_root / "resolved_model.json", {"method": args.method, "label": label, "model_path": model_path, "candidates": candidates})

    validate_vocab_compatibility(model_path, args.model_dir, tok)

    subjects = fast.parse_subjects(Path(args.prompts_jsonl), args.max_subjects) or fast.PREFERRED_FORGET_SUBJECTS[:args.max_subjects]
    log(f"subjects={subjects}")

    model = fast.load_model(model_path, args.model_dir, args.device, args.load_mode)

    summaries: Dict[str, Any] = {}
    try:
        summaries["fast"] = eval_fast_suite(model, tok, label, model_path, subjects, args, start)
        summaries["adversarial"] = eval_adv_suite(model, tok, label, model_path, subjects, args, start)
        summaries["rwku"] = eval_rwku_suite(model, tok, label, model_path, subjects, args, start)
    finally:
        fast.free_model(model)

    combined = compact_summary(args.method, model_path, candidates, summaries, args)
    write_json(method_root / "saved_baseline_suite_summary.json", combined)
    log("Combined summary written")
    log(json.dumps({
        "method": args.method,
        "model_path": model_path,
        "fast_completion": combined.get("fast_completion"),
        "adversarial_completion": combined.get("adversarial_completion"),
        "rwku_completion": combined.get("rwku_completion"),
        "summary": str(method_root / "saved_baseline_suite_summary.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
