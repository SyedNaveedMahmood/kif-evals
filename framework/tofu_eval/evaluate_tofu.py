#!/usr/bin/env python3
"""Evaluate a model on TOFU forget10 with SimNPO/KIF-style linear FQ and MU.

This evaluator is standalone and does not run the KIF SMR/EL10 Module 8.
Reporting follows the TOFU/SimNPO convention used in KIF discussions:
  - Forget Quality / FQ = linear KS-test p-value, not log(p-value)
  - Model Utility / MU = harmonic mean over non-forget utility metrics
  - Truth Ratio = wrong/correct using official TOFU perturbed datasets

Retain-reference handling:
  Publication-grade FQ requires truth-ratio logs from a true retain-only model
  for the same TOFU split. This file accepts either this repo's
  retain_reference_logs.json format or OpenUnlearning-style TOFU_EVAL.json logs
  as long as truth-ratio values can be extracted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    from .metrics import (
        eval_probability_rows,
        eval_rouge_rows,
        eval_truth_ratio_rows,
        forget_truth_ratio_score,
        harmonic_mean,
        ks_test,
        read_jsonl,
        truth_ratio_nonforget_score,
        values_from_metric,
        write_json,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from metrics import (  # type: ignore
        eval_probability_rows,
        eval_rouge_rows,
        eval_truth_ratio_rows,
        forget_truth_ratio_score,
        harmonic_mean,
        ks_test,
        read_jsonl,
        truth_ratio_nonforget_score,
        values_from_metric,
        write_json,
    )


def _load_model(model_dir: str, torch_dtype: str, use_4bit: bool, device_map_auto: bool, attn_implementation: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(torch_dtype).lower(), torch.bfloat16)

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype if torch.cuda.is_available() else torch.float32,
        "attn_implementation": attn_implementation,
        "use_cache": True,
    }
    if device_map_auto and torch.cuda.device_count() > 1:
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs.pop("torch_dtype", None)
        except Exception as exc:
            print(f"[WARN] 4-bit requested but unavailable ({exc}); loading normal dtype.")

    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    return model, tok


def _maybe_rows(path: Path) -> Optional[List[Dict[str, Any]]]:
    if path.exists():
        rows = read_jsonl(path)
        print(f"[data] {path} rows={len(rows)}")
        return rows
    print(f"[skip] missing optional data file: {path}")
    return None


def _tr_aggregator_for_dataset(name: str) -> str:
    return "closer_to_1_better" if name == "forget" else "true_better"


def _compute_dataset_metrics(model, tokenizer, name: str, rows: List[Dict[str, Any]], args) -> Dict[str, Any]:
    print(f"[eval] {name}: probability")
    prob = eval_probability_rows(model, tokenizer, rows, max_length=args.max_length, model_family=args.model_family)
    print(f"[eval] {name}: truth_ratio")
    tr = eval_truth_ratio_rows(
        model,
        tokenizer,
        rows,
        max_length=args.max_length,
        model_family=args.model_family,
        aggregator=_tr_aggregator_for_dataset(name),
    )
    rouge = None
    if not args.skip_rouge:
        print(f"[eval] {name}: ROUGE-L recall")
        rouge = eval_rouge_rows(
            model,
            tokenizer,
            rows,
            max_new_tokens=args.max_new_tokens,
            model_family=args.model_family,
            max_rows=args.max_rouge_rows,
        )
    return {"probability": prob, "truth_ratio": tr, "rougeL_recall": rouge}


def _load_retain_reference(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _numeric_list(x: Any) -> List[float]:
    out: List[float] = []
    if isinstance(x, list):
        for item in x:
            try:
                v = float(item)
                if v == v:
                    out.append(v)
            except Exception:
                pass
    return out


def _values_from_any_metric(obj: Any) -> List[float]:
    """Extract truth-ratio score arrays from either our or OpenUnlearning logs."""
    if obj is None:
        return []

    nums = _numeric_list(obj)
    if nums:
        return nums

    if isinstance(obj, dict):
        if "truth_ratio_values" in obj:
            vals = _numeric_list(obj.get("truth_ratio_values"))
            if vals:
                return vals

        if "value_by_index" in obj and isinstance(obj["value_by_index"], dict):
            vals: List[float] = []
            for v in obj["value_by_index"].values():
                if isinstance(v, dict):
                    for key in ("score", "truth_ratio", "value", "agg_value"):
                        if key in v:
                            try:
                                f = float(v[key])
                                if f == f:
                                    vals.append(f)
                                    break
                            except Exception:
                                pass
                else:
                    try:
                        f = float(v)
                        if f == f:
                            vals.append(f)
                    except Exception:
                        pass
            if vals:
                return vals

        # Common OpenUnlearning shape can store values under access keys such as
        # retain/forget/correct/wrong; recurse but prefer retain when present.
        for preferred in ("retain", "forget_truth_ratio", "forget", "truth_ratio"):
            if preferred in obj:
                vals = _values_from_any_metric(obj[preferred])
                if vals:
                    return vals

    return []


def extract_retain_reference_values(retain_reference: Dict[str, Any]) -> List[float]:
    """Return retain-model forget truth-ratio values from supported log formats.

    Supported formats:
    1. This repo's retain_reference_logs.json:
       {"truth_ratio_values": [...]} or {"forget_truth_ratio": {"value_by_index": ...}}
    2. OpenUnlearning-style nested TOFU_EVAL.json logs, where the relevant values
       are under keys containing truth_ratio / forget_truth_ratio / retain.
    """
    direct = _values_from_any_metric(retain_reference)
    if direct:
        return direct

    hits: List[Any] = []

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                k_str = str(k).lower()
                new_path = f"{path}.{k_str}" if path else k_str
                if "truth_ratio" in k_str or "forget_quality" in k_str:
                    hits.append(v)
                walk(v, new_path)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, path)

    walk(retain_reference)
    for h in hits:
        vals = _values_from_any_metric(h)
        if vals:
            return vals
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="", help="Model/checkpoint to evaluate. Not required for --validate_retain_logs_only.")
    ap.add_argument("--data_dir", default="framework/outputs/tofu/data")
    ap.add_argument("--split", default="forget10", choices=["forget01", "forget05", "forget10"])
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--retain_logs", default="")
    ap.add_argument("--write_retain_logs", action="store_true", help="Write current model's forget truth-ratio as retain reference logs.")
    ap.add_argument("--validate_retain_logs_only", action="store_true", help="Validate --retain_logs and exit before loading any model.")
    ap.add_argument("--model_family", default="llama3.1-8b")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--max_rouge_rows", type=int, default=0, help="0 means all rows; use smaller for cheap smoke tests.")
    ap.add_argument("--skip_rouge", action="store_true")
    ap.add_argument("--torch_dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default="sdpa")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--device_map_auto", action="store_true")
    args = ap.parse_args()

    if args.validate_retain_logs_only:
        if not args.retain_logs:
            raise SystemExit("--validate_retain_logs_only requires --retain_logs")
        retain_reference = _load_retain_reference(args.retain_logs)
        if retain_reference is None:
            raise SystemExit(f"Retain logs not found: {args.retain_logs}")
        ref_vals = extract_retain_reference_values(retain_reference)
        if not ref_vals:
            raise SystemExit(f"Could not extract retain truth-ratio values from: {args.retain_logs}")
        print(json.dumps({"retain_logs": args.retain_logs, "num_truth_ratio_values": len(ref_vals)}, indent=2))
        return

    if not args.model_dir:
        raise SystemExit("--model_dir is required unless --validate_retain_logs_only is set")
    if not args.output_dir:
        raise SystemExit("--output_dir is required unless --validate_retain_logs_only is set")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    retain_split = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}[args.split]

    forget_rows = read_jsonl(data_dir / f"{args.split}.jsonl")
    retain_rows = _maybe_rows(data_dir / f"{retain_split}.jsonl")
    real_rows = _maybe_rows(data_dir / "real_authors.jsonl")
    world_rows = _maybe_rows(data_dir / "world_facts.jsonl")

    if args.max_rouge_rows == 0:
        args.max_rouge_rows = None

    print(f"[model] loading {args.model_dir}")
    model, tokenizer = _load_model(args.model_dir, args.torch_dtype, args.use_4bit, args.device_map_auto, args.attn_implementation)

    metrics: Dict[str, Any] = {}
    metrics["forget"] = _compute_dataset_metrics(model, tokenizer, "forget", forget_rows, args)
    if retain_rows:
        metrics["retain"] = _compute_dataset_metrics(model, tokenizer, "retain", retain_rows, args)
    if real_rows:
        metrics["real_authors"] = _compute_dataset_metrics(model, tokenizer, "real_authors", real_rows, args)
    if world_rows:
        metrics["world_facts"] = _compute_dataset_metrics(model, tokenizer, "world_facts", world_rows, args)

    write_json(out / "tofu_metrics.json", metrics)

    retain_reference = _load_retain_reference(args.retain_logs) if args.retain_logs else None
    current_forget_tr = values_from_metric(metrics["forget"]["truth_ratio"], "score")

    if args.write_retain_logs:
        retain_obj = {
            "model_dir": args.model_dir,
            "split": args.split,
            "forget_truth_ratio": metrics["forget"]["truth_ratio"],
            "truth_ratio_values": current_forget_tr,
            "note": "Use this only if this model is the retain-only reference model for the requested TOFU split.",
        }
        write_json(out / "retain_reference_logs.json", retain_obj)
        print(f"[write] retain reference logs -> {out / 'retain_reference_logs.json'}")

    ks = {"statistic": float("nan"), "pvalue": float("nan")}
    if retain_reference is not None:
        ref_vals = extract_retain_reference_values(retain_reference)
        if not ref_vals:
            raise RuntimeError(f"Retain logs were provided but no truth-ratio values could be extracted: {args.retain_logs}")
        # SimNPO/KIF reporting uses the direct linear KS p-value on raw truth ratios.
        ks = ks_test(current_forget_tr, ref_vals, inverse=False)
    else:
        print("[WARN] No retain_logs provided/found. FQ will be NaN. Create retain reference first using --write_retain_logs on a retain90 model.")

    mu_components: Dict[str, float] = {}
    for ds_name in ("retain", "real_authors", "world_facts"):
        ds = metrics.get(ds_name)
        if not ds:
            continue
        mu_components[f"{ds_name}_probability"] = float(ds["probability"]["agg_value"])
        if ds.get("rougeL_recall") is not None:
            mu_components[f"{ds_name}_rougeL_recall"] = float(ds["rougeL_recall"]["agg_value"])
        mu_components[f"{ds_name}_truth_ratio_score"] = truth_ratio_nonforget_score(ds["truth_ratio"])

    mu = harmonic_mean(mu_components.values()) if mu_components else float("nan")
    forget_tr_score = forget_truth_ratio_score(metrics["forget"]["truth_ratio"])

    summary = {
        "model_dir": args.model_dir,
        "split": args.split,
        "retain_split": retain_split,
        "Forget Quality": ks["pvalue"],
        "KS Test PVal Forget": ks["pvalue"],
        "KS Test Forget": ks["statistic"],
        "Model Utility": mu,
        "forget_quality_linear_FQ": ks["pvalue"],
        "model_utility_MU": mu,
        "forget_truth_ratio_mean_raw": float(metrics["forget"]["truth_ratio"].get("raw_mean", float("nan"))),
        "forget_truth_ratio_closer_to_1_score": forget_tr_score,
        "forget_probability": float(metrics["forget"]["probability"]["agg_value"]),
        "forget_rougeL_recall": None if metrics["forget"].get("rougeL_recall") is None else float(metrics["forget"]["rougeL_recall"]["agg_value"]),
        "mu_components": mu_components,
        "fq_reference_logs": args.retain_logs,
        "reporting_convention": "SimNPO/KIF-style linear FQ = KS-test p-value, no log transform",
    }
    write_json(out / "tofu_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
