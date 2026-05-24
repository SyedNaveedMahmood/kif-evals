#!/usr/bin/env python3
"""Evaluate already-trained saved baselines on the KIF entity-audit suites.

This wrapper is intentionally evaluation-only. It loads one saved method artifact
(SimNPO, ReGLU, or LUNAR) once, then runs the three diagnostics already used for
KIF/Opt-Out:

  1. fast_entity_eval_bundle.py
     - name-agnostic forget robustness
     - mixed-query forget/retain robustness
     - syntactic locality controls
  2. adversarial_forget_recovery_eval.py
     - overall adversarial forget-recovery
     - by-attack-family recovery and alias-hit summaries
  3. rwku_style_entity_robustness.py
     - RWKU-style entity robustness probes

The script is checkpointed at the JSONL row level. Re-running the same method
continues from existing rows. It also uses a soft time budget so dev-GPU jobs can
exit cleanly and write partial summaries instead of being killed by Slurm.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import fast_entity_eval_bundle as fast  # type: ignore
import adversarial_forget_recovery_eval as adv  # type: ignore
import rwku_style_entity_robustness as rwku  # type: ignore


DEFAULT_METHODS = ["simnpo", "reglu", "lunar"]


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


def normalize_candidate_path(raw: str, manifest_path: Path) -> Path:
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p
    # First try relative to current working directory; then relative to manifest.
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


def score_candidate(method: str, cand: Dict[str, Any], allow_smoke: bool) -> float:
    blob = json.dumps(cand, ensure_ascii=False).lower().replace("-", "_")
    score = 0.0
    method_l = method.lower().replace("-", "_")
    if method_l in blob:
        score += 100.0
    if f"method_name\": \"{method_l}" in blob or f"method_name': '{method_l}" in blob:
        score += 40.0
    if "main" in blob:
        score += 25.0
    if "full" in blob:
        score += 8.0
    if "final" in blob:
        score += 6.0
    if "merged" in blob:
        score += 5.0
    if "smoke" in blob or "test" in blob:
        score += -500.0 if not allow_smoke else -25.0
    # For SimNPO, prefer the standard SimNPO artifact over simnpo_pure unless explicitly requested.
    if method_l == "simnpo" and "simnpo_pure" in blob:
        score -= 80.0
    if cand.get("exists"):
        score += 20.0
    mtime = cand.get("mtime", 0.0) or 0.0
    score += min(float(mtime) / 1e10, 1.0)  # tiny tie-breaker
    return score


def discover_method_artifact(method: str, roots: Sequence[Path], allow_smoke: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
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
            exists = path_exists_modelish(model_path)
            cand = {
                "method_query": method,
                "method_name": obj.get("method_name") or obj.get("method") or "",
                "path": str(model_path),
                "raw_path": str(raw_path),
                "manifest": str(manifest),
                "exists": bool(exists),
                "mtime": float(model_path.stat().st_mtime) if model_path.exists() else 0.0,
            }
            cand["score"] = score_candidate(method, cand, allow_smoke)
            candidates.append(cand)
    candidates = sorted(candidates, key=lambda c: c.get("score", -1e9), reverse=True)
    usable = [c for c in candidates if c.get("exists")]
    if not usable:
        detail = json.dumps(candidates[:10], indent=2, ensure_ascii=False)
        raise FileNotFoundError(f"No usable saved artifact found for method={method}. Top candidates:\n{detail}")
    return str(usable[0]["path"]), usable[:10]


def remaining_budget(start: float, soft_limit_seconds: float) -> float:
    if soft_limit_seconds <= 0:
        return 1e18
    return soft_limit_seconds - (time.time() - start)


def maybe_stop_for_budget(start: float, soft_limit_seconds: float, min_remaining_seconds: float = 90.0) -> bool:
    return remaining_budget(start, soft_limit_seconds) < min_remaining_seconds


def max_rows_to_run(todo: Sequence[Dict[str, Any]], max_rows: int) -> int:
    if max_rows == 0:
        return len(todo)
    return min(len(todo), max_rows)


def eval_fast_suite(model, tok, label: str, model_path: str, subjects: List[str], args: argparse.Namespace, start: float) -> Dict[str, Any]:
    out_dir = Path(args.out_root) / label / "fast_entity_eval_bundle"
    dataset_path = out_dir / "datasets" / "fast_entity_eval_bundle.jsonl"
    rows = fast.build_dataset(subjects, dataset_path, rebuild=args.rebuild_datasets)
    eval_path = out_dir / "checkpoints" / f"eval_rows_{label}.jsonl"
    done = fast.completed_ids(eval_path)
    todo = [r for r in rows if r["row_id"] not in done]
    n = max_rows_to_run(todo, args.max_fast_rows)
    log(f"[{label}] FAST rows: done={len(done)}/{len(rows)} this_run={n}")
    for i, row in enumerate(todo[:n], 1):
        if maybe_stop_for_budget(start, args.soft_time_limit_minutes * 60):
            log(f"[{label}] FAST stopping early due to soft time limit after {i-1}/{n} rows")
            break
        rec = fast.eval_one_row(model, tok, row, label, args.device, args.fast_max_new_tokens, args.fast_el_steps, args.fast_max_keywords)
        fast.append_jsonl(eval_path, rec)
        if i % args.progress_every == 0:
            log(f"[{label}] FAST evaluated {i}/{n} rows this run")
    ns = Namespace(
        model_dir=args.model_dir,
        kif_adapter_path=None,
        baseline_model_dir=model_path,
        outputs_root=args.outputs_root,
        prompts_jsonl=args.prompts_jsonl,
        out_dir=str(out_dir),
        models=label,
        max_subjects=args.max_subjects,
        load_mode=args.load_mode,
        device=args.device,
        max_eval_rows_per_model=args.max_fast_rows,
        max_new_tokens=args.fast_max_new_tokens,
        el_steps=args.fast_el_steps,
        max_keywords=args.fast_max_keywords,
        seed=args.seed,
        rebuild_dataset=args.rebuild_datasets,
        dataset_only=False,
        smoke_test=False,
    )
    return fast.write_summary(out_dir, rows, [label], [{"path": model_path, "method": label}], ns)


def eval_adv_suite(model, tok, label: str, model_path: str, subjects: List[str], args: argparse.Namespace, start: float) -> Dict[str, Any]:
    out_dir = Path(args.out_root) / label / "adversarial_forget_recovery"
    dataset_path = out_dir / "datasets" / "adversarial_forget_recovery.jsonl"
    rows = adv.build_adversarial_dataset(subjects, dataset_path, rebuild=args.rebuild_datasets)
    eval_path = out_dir / "checkpoints" / f"eval_rows_{label}.jsonl"
    done = adv.completed_ids(eval_path)
    todo = [r for r in rows if r["row_id"] not in done]
    n = max_rows_to_run(todo, args.max_adv_rows)
    log(f"[{label}] ADV rows: done={len(done)}/{len(rows)} this_run={n}")
    for i, row in enumerate(todo[:n], 1):
        if maybe_stop_for_budget(start, args.soft_time_limit_minutes * 60):
            log(f"[{label}] ADV stopping early due to soft time limit after {i-1}/{n} rows")
            break
        rec = fast.eval_one_row(model, tok, row, label, args.device, args.adv_max_new_tokens, args.adv_el_steps, args.adv_max_keywords)
        adv.append_jsonl(eval_path, rec)
        if i % args.progress_every == 0:
            log(f"[{label}] ADV evaluated {i}/{n} rows this run")
    ns = Namespace(
        model_dir=args.model_dir,
        kif_adapter_path=None,
        baseline_model_dir=model_path,
        baseline_prefer=label,
        outputs_root=args.outputs_root,
        prompts_jsonl=args.prompts_jsonl,
        out_dir=str(out_dir),
        models=label,
        max_subjects=args.max_subjects,
        load_mode=args.load_mode,
        device=args.device,
        max_eval_rows_per_model=args.max_adv_rows,
        max_new_tokens=args.adv_max_new_tokens,
        el_steps=args.adv_el_steps,
        max_keywords=args.adv_max_keywords,
        recovery_keyword_threshold=args.recovery_keyword_threshold,
        recovery_mass_threshold=args.recovery_mass_threshold,
        seed=args.seed,
        rebuild_dataset=args.rebuild_datasets,
        dataset_only=False,
        smoke_test=False,
    )
    return adv.write_summary(out_dir, rows, [label], [{"path": model_path, "method": label}], ns)


def eval_rwku_suite(model, tok, label: str, model_path: str, subjects: List[str], args: argparse.Namespace, start: float) -> Dict[str, Any]:
    out_dir = Path(args.out_root) / label / "rwku_style_entity_robustness"
    dataset_path = out_dir / "dataset" / "rwku_style_entity_robustness.jsonl"
    cap = args.rwku_rows_per_family_cap if args.rwku_rows_per_family_cap > 0 else None
    rows = rwku.load_or_build_dataset(subjects, dataset_path, args.rebuild_datasets, cap)
    eval_path = out_dir / "checkpoints" / f"eval_rows_{label}.jsonl"
    done = rwku.completed_ids(eval_path)
    todo = [r for r in rows if r["row_id"] not in done]
    n = max_rows_to_run(todo, args.max_rwku_rows)
    log(f"[{label}] RWKU rows: done={len(done)}/{len(rows)} this_run={n}")
    for i, row in enumerate(todo[:n], 1):
        if maybe_stop_for_budget(start, args.soft_time_limit_minutes * 60):
            log(f"[{label}] RWKU stopping early due to soft time limit after {i-1}/{n} rows")
            break
        rec = rwku.eval_one_row(model, tok, row, label, args.device, args.rwku_max_new_tokens, args.rwku_el_steps, args.rwku_max_keywords)
        rwku.append_jsonl(eval_path, rec)
        if i % args.progress_every == 0:
            log(f"[{label}] RWKU evaluated {i}/{n} rows this run")

    # By default, this wrapper skips RWKU MIA/teacher generation for speed because the paper-facing
    # entity robustness table uses generation and target-mass metrics. Set --max_rwku_mia_rows > 0
    # only if you explicitly want membership-inference rows too.
    if args.max_rwku_mia_rows > 0:
        teacher_path = out_dir / "checkpoints" / "teacher_completions.jsonl"
        teacher = rwku.teacher_map(teacher_path)
        pairs = rwku.build_mia_pairs(rows, teacher, args.rwku_mia_max_pairs)
        mia_path = out_dir / "checkpoints" / f"mia_rows_{label}.jsonl"
        mia_done = rwku.completed_ids(mia_path)
        mia_todo = [(r, text, y) for r, text, y in pairs if r["row_id"] not in mia_done]
        n_mia = max_rows_to_run(mia_todo, args.max_rwku_mia_rows)
        log(f"[{label}] RWKU-MIA rows: done={len(mia_done)}/{len(pairs)} this_run={n_mia}")
        for i, (row, text, y) in enumerate(mia_todo[:n_mia], 1):
            if maybe_stop_for_budget(start, args.soft_time_limit_minutes * 60):
                log(f"[{label}] RWKU-MIA stopping early due to soft time limit after {i-1}/{n_mia} rows")
                break
            rec = rwku.mia_score_row(model, tok, row, text, y, label, args.device, args.rwku_max_length, args.rwku_min_k_frac)
            rwku.append_jsonl(mia_path, rec)
            if i % args.progress_every == 0:
                log(f"[{label}] RWKU-MIA scored {i}/{n_mia} rows this run")

    ns = Namespace(
        model_dir=args.model_dir,
        kif_adapter_path=None,
        baseline_model_dir=model_path,
        baseline_prefer=label,
        outputs_root=args.outputs_root,
        prompts_jsonl=args.prompts_jsonl,
        out_dir=str(out_dir),
        max_subjects=args.max_subjects,
        rows_per_family_cap=args.rwku_rows_per_family_cap,
        rebuild_dataset=args.rebuild_datasets,
        models=label,
        load_mode=args.load_mode,
        device=args.device,
        max_eval_rows_per_model=args.max_rwku_rows,
        max_teacher_rows=0,
        max_mia_rows_per_model=args.max_rwku_mia_rows,
        mia_max_pairs=args.rwku_mia_max_pairs,
        max_new_tokens=args.rwku_max_new_tokens,
        el_steps=args.rwku_el_steps,
        max_keywords=args.rwku_max_keywords,
        max_length=args.rwku_max_length,
        min_k_frac=args.rwku_min_k_frac,
        seed=args.seed,
        dataset_only=False,
        smoke_test=False,
    )
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
    ap.add_argument("--soft_time_limit_minutes", type=float, default=26.0)
    ap.add_argument("--progress_every", type=int, default=25)

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
    ap.add_argument("--max_rwku_mia_rows", type=int, default=0)
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
    candidates: List[Dict[str, Any]] = []
    if args.model_path:
        model_path = args.model_path
        candidates = [{"path": model_path, "method": args.method, "source": "provided"}]
    else:
        model_path, candidates = discover_method_artifact(args.method, roots, args.allow_smoke)

    log(f"method={args.method} label={label}")
    log(f"model_path={model_path}")
    log(f"top_candidate={json.dumps(candidates[0] if candidates else {}, ensure_ascii=False)}")

    out_root = Path(args.out_root)
    method_root = out_root / label
    method_root.mkdir(parents=True, exist_ok=True)
    write_json(method_root / "resolved_model.json", {"method": args.method, "label": label, "model_path": model_path, "candidates": candidates})

    subjects = fast.parse_subjects(Path(args.prompts_jsonl), args.max_subjects)
    if not subjects:
        subjects = fast.PREFERRED_FORGET_SUBJECTS[:args.max_subjects]
    log(f"subjects={subjects}")

    tok = fast.load_tokenizer(args.model_dir)
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
