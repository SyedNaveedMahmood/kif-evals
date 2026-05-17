"""
orchestrate.py
==============
Single entry point: run any unlearning method on prompts.jsonl,
then evaluate with the generalised Module 8 (SMR + EL10 + utility).

Usage
-----
# Full run — unlearn + eval
python orchestrate.py \\
    --method        lunar \\
    --model_dir     outputs/model \\
    --prompts_jsonl outputs/datasets/prompts.jsonl \\
    --capsules_dir  outputs/capsules \\
    --output_dir    outputs/unlearning_runs

# Eval-only on a previous run
python orchestrate.py \\
    --eval_only   true \\
    --result_json outputs/unlearning_runs/lunar/unlearning_result.json \\
    --model_dir   outputs/model \\
    --prompts_jsonl outputs/datasets/prompts.jsonl \\
    --capsules_dir  outputs/capsules

# Override LUNAR config
python orchestrate.py \\
    --method lunar \\
    --config '{"auto_select_layer": false, "layer_modified": 18}' \\
    ...

Adding a new method
-------------------
1. Create methods/my_method.py  (copy methods/template.py)
2. Implement run() → return UnlearningResult
3. Add import to methods/__init__.py
4. Run: python orchestrate.py --method my_method ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("orchestrate")

import methods   # triggers registry registration for all imported methods
from methods.base import METHOD_REGISTRY, UnlearningResult
from eval.module8_eval import run_eval


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    method_name: str,
    model_dir: str,
    prompts_jsonl: str,
    capsules_dir: str,
    output_dir: str,
    subjects: Optional[List[str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    skip_eval: bool = False,
    eval_out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end: prompts.jsonl → unlearn → Module 8 eval → summary dict.
    """
    # ── Unlearn ──────────────────────────────────────────────────────────
    MethodClass = METHOD_REGISTRY.get(method_name)
    method      = MethodClass(config_overrides=config_overrides)
    result: UnlearningResult = method.execute(
        prompts_jsonl=prompts_jsonl,
        output_dir=output_dir,
        model_dir=model_dir,
        subjects=subjects,
    )
    logger.info(f"[Pipeline] Unlearning complete: {result.to_dict()}")

    if skip_eval:
        return result.to_dict()

    # ── Eval ─────────────────────────────────────────────────────────────
    return _run_eval_on_result(
        result=result,
        model_dir=model_dir,
        prompts_jsonl=prompts_jsonl,
        capsules_dir=capsules_dir,
        eval_out_dir=eval_out_dir,
    )


def run_eval_only(
    result_json: str,
    model_dir: str,
    prompts_jsonl: str,
    capsules_dir: str,
    eval_out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    result = UnlearningResult.load(result_json)
    return _run_eval_on_result(
        result=result,
        model_dir=model_dir,
        prompts_jsonl=prompts_jsonl,
        capsules_dir=capsules_dir,
        eval_out_dir=eval_out_dir,
    )


def _run_eval_on_result(
    result: UnlearningResult,
    model_dir: str,
    prompts_jsonl: str,
    capsules_dir: str,
    eval_out_dir: Optional[str],
) -> Dict[str, Any]:
    """Translate UnlearningResult fields into run_eval() kwargs."""
    if eval_out_dir is None:
        eval_out_dir = str(Path(result.output_dir) / "eval")

    # Mode A: merged model
    if result.merged_model_dir:
        summary = run_eval(
            model_dir=model_dir,                      # PRE baseline
            merged_model_dir=result.merged_model_dir, # POST
            prompts_jsonl=prompts_jsonl,
            capsules_dir=capsules_dir,
            out_dir=eval_out_dir,
            eval_subjects_override=result.subjects or None,
        )
    # Mode B: base + adapter
    elif result.model_dir and result.adapter_path:
        summary = run_eval(
            model_dir=result.model_dir,
            adapter_path=result.adapter_path,
            prompts_jsonl=prompts_jsonl,
            capsules_dir=capsules_dir,
            out_dir=eval_out_dir,
            eval_subjects_override=result.subjects or None,
        )
    else:
        raise ValueError(
            "UnlearningResult has neither merged_model_dir nor "
            "(model_dir + adapter_path)."
        )

    summary["method"] = result.method_name
    (Path(eval_out_dir) / "final_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Plug-and-play unlearning + KIF dual-metric eval",
    )
    p.add_argument("--eval_only", type=lambda x: x.lower() in ("1","true","yes"),
                   default=False)
    p.add_argument("--result_json", default=None,
                   help="Path to saved unlearning_result.json (--eval_only only)")

    p.add_argument("--method", default="lunar",
                   help=f"Unlearning method. Available: {METHOD_REGISTRY.list()}")
    p.add_argument("--model_dir",      default="outputs/model")
    p.add_argument("--prompts_jsonl",  default="outputs/datasets/prompts.jsonl")
    p.add_argument("--capsules_dir",   default="outputs/capsules")
    p.add_argument("--output_dir",     default="outputs/unlearning_runs")
    p.add_argument("--eval_out_dir",   default=None)
    p.add_argument("--subjects",       nargs="+", default=None)
    p.add_argument("--config",         default=None,
                   help="JSON string of config overrides, e.g. "
                        "'{\"auto_select_layer\": false, \"layer_modified\": 18}'")
    p.add_argument("--skip_eval",
                   type=lambda x: x.lower() in ("1","true","yes"), default=False)
    return p.parse_args()


def main():
    args = _parse()

    config_overrides = None
    if args.config:
        try:
            config_overrides = json.loads(args.config)
        except json.JSONDecodeError as e:
            logger.error(f"--config is not valid JSON: {e}")
            sys.exit(1)

    if args.eval_only:
        if not args.result_json:
            logger.error("--eval_only requires --result_json")
            sys.exit(1)
        summary = run_eval_only(
            result_json=args.result_json,
            model_dir=args.model_dir,
            prompts_jsonl=args.prompts_jsonl,
            capsules_dir=args.capsules_dir,
            eval_out_dir=args.eval_out_dir,
        )
    else:
        summary = run_pipeline(
            method_name=args.method,
            model_dir=args.model_dir,
            prompts_jsonl=args.prompts_jsonl,
            capsules_dir=args.capsules_dir,
            output_dir=args.output_dir,
            subjects=args.subjects,
            config_overrides=config_overrides,
            skip_eval=args.skip_eval,
            eval_out_dir=args.eval_out_dir,
        )

    print("\n" + "=" * 62)
    print("FINAL SUMMARY")
    print("=" * 62)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
