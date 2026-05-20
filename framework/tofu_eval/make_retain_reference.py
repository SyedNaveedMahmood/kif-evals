#!/usr/bin/env python3
"""Fast TOFU retain-reference log generator.

This script creates the only quantity needed for TOFU Forget Quality: the
retain-only model's forget-set truth-ratio distribution. It does not compute
retain/real/world utility, ROUGE, or generation metrics, so it is much faster
than running the full evaluator just to create retain_reference_logs.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

try:
    from .metrics import eval_truth_ratio_rows, read_jsonl, values_from_metric, write_json
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from metrics import eval_truth_ratio_rows, read_jsonl, values_from_metric, write_json  # type: ignore


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

    kwargs: Dict[str, Any] = {"trust_remote_code": True, "attn_implementation": attn_implementation, "use_cache": True}
    if device_map_auto and torch.cuda.device_count() > 1:
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = torch.float32

    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = dtype if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", default="framework/outputs/tofu/data")
    ap.add_argument("--split", default="forget10", choices=["forget01", "forget05", "forget10"])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_family", default="llama2-7b-chat")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--expected_count", type=int, default=400)
    ap.add_argument("--torch_dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default="sdpa")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--device_map_auto", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_dir) / f"{args.split}.jsonl"
    rows = read_jsonl(data_path)
    print(f"[retain-ref] rows={len(rows)} data={data_path}")
    if args.expected_count > 0 and len(rows) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} forget rows for {args.split}, found {len(rows)}")

    print(f"[retain-ref] loading model={args.model_dir}")
    model, tokenizer = _load_model(args.model_dir, args.torch_dtype, args.use_4bit, args.device_map_auto, args.attn_implementation)
    print("[retain-ref] computing forget truth-ratio distribution")
    tr = eval_truth_ratio_rows(
        model,
        tokenizer,
        rows,
        max_length=args.max_length,
        model_family=args.model_family,
        aggregator="closer_to_1_better",
        batch_size=args.batch_size,
    )
    vals = values_from_metric(tr, "score")
    if args.expected_count > 0 and len(vals) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} truth-ratio values, found {len(vals)}")

    obj = {
        "model_dir": args.model_dir,
        "split": args.split,
        "forget_truth_ratio": tr,
        "truth_ratio_values": vals,
        "num_truth_ratio_values": len(vals),
        "note": "Generated from true retain-only reference model for the same TOFU split. Suitable for linear FQ KS-test reference.",
    }
    path = out / "retain_reference_logs.json"
    write_json(path, obj)
    print(json.dumps({"retain_reference_logs": str(path), "num_truth_ratio_values": len(vals)}, indent=2))


if __name__ == "__main__":
    main()
