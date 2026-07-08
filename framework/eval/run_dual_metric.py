"""
eval/run_dual_metric.py
=======================
Thin wrapper that:
  1. Reads the UnlearningResult written by any unlearning method.
  2. Translates its model-loading fields into the exact kwargs that
     Module 8's run_module8_clean() expects.
  3. Runs the evaluation and returns the summary dict.

This file never imports any method code - it only talks to Module 8.
Add this file once; it works for every method automatically because
all methods write a uniform UnlearningResult.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8_stdio()


def _ensure_module8_importable(kif_root: str):
    """Add KIF's src directory to sys.path so module8 can be imported."""
    src_path = str(Path(kif_root) / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # Also add kif_root itself (for imports like `from llama20.modules...`)
    if kif_root not in sys.path:
        sys.path.insert(0, kif_root)


def run_dual_metric_eval(
    result_or_path,                    # UnlearningResult or path to unlearning_result.json
    kif_root: str,                     # Path to KIF repo root (contains src/llama20)
    prompts_jsonl: str,                # outputs/datasets/prompts.jsonl
    capsules_dir: str,                 # outputs/capsules
    eval_out_dir: Optional[str] = None,
    extra_module8_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run Module 8 dual-metric evaluation on the model produced by any
    unlearning method.

    Parameters
    ----------
    result_or_path : UnlearningResult | str | Path
        The result returned by method.execute(), or a path to the
        unlearning_result.json file written by execute().
    kif_root : str
        Path to the KIF repo root. Used to locate Module 8.
    prompts_jsonl : str
        Path to KIF's prompts.jsonl (subject keyword source for EL10).
    capsules_dir : str
        Path to capsules directory (for signature separation / Cohen's d).
        If you're running LUNAR without KIF capsules this can be an empty
        directory - signature separation will simply be skipped.
    eval_out_dir : str, optional
        Where to write eval artefacts. Defaults to
        <method_output_dir>/eval_dual_metric/.
    extra_module8_kwargs : dict, optional
        Any additional kwargs forwarded verbatim to run_module8_clean().

    Returns
    -------
    dict - the summary dict from run_module8_clean().
    """

    # ---- Resolve result -------------------------------------------------
    from methods.base import UnlearningResult  # local import to avoid circular

    if isinstance(result_or_path, (str, Path)):
        result = UnlearningResult.load(str(result_or_path))
    else:
        result = result_or_path

    method_name = result.method_name or "unknown"
    logger.info(f"[Eval] Running dual-metric eval for method: {method_name}")

    # ---- Determine model-loading mode -----------------------------------
    # Mode A: merged / full model directory
    # Mode B: base model + PEFT adapter path
    if result.merged_model_dir:
        model_dir_arg = result.merged_model_dir
        adapter_path_arg = None
        merged_model_dir_arg = result.merged_model_dir
        logger.info(f"[Eval] Mode A - loading merged model from {result.merged_model_dir}")
    elif result.model_dir and result.adapter_path:
        model_dir_arg = result.model_dir
        adapter_path_arg = result.adapter_path
        merged_model_dir_arg = None
        logger.info(f"[Eval] Mode B - base={result.model_dir}, adapter={result.adapter_path}")
    else:
        raise ValueError(
            "[Eval] UnlearningResult has neither merged_model_dir nor "
            "(model_dir + adapter_path). Cannot proceed."
        )

    # ---- Output dir ------------------------------------------------------
    if eval_out_dir is None:
        eval_out_dir = str(Path(result.output_dir) / "eval_dual_metric")
    Path(eval_out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Import Module 8 -------------------------------------------------
    _ensure_module8_importable(kif_root)
    try:
        from llama20.modules.module8 import run_module8_clean
    except ImportError as e:
        raise ImportError(
            f"[Eval] Cannot import module8 from KIF root '{kif_root}'. "
            f"Check that kif_root is correct. Original error: {e}"
        )

    # ---- Build kwargs for Module 8 --------------------------------------
    m8_kwargs: Dict[str, Any] = {
        "model_dir": model_dir_arg,
        "adapter_path": adapter_path_arg,
        "merged_model_dir": merged_model_dir_arg,
        "out_dir": eval_out_dir,
        "capsules_dir": capsules_dir,
        "prompts_jsonl": prompts_jsonl,
    }
    if extra_module8_kwargs:
        m8_kwargs.update(extra_module8_kwargs)

    logger.info(f"[Eval] Module 8 kwargs: {json.dumps({k: str(v) for k, v in m8_kwargs.items()}, indent=2)}")

    # ---- Run evaluation --------------------------------------------------
    summary = run_module8_clean(**m8_kwargs)

    # ---- Annotate summary with method metadata --------------------------
    summary["method"] = method_name
    summary["subjects_unlearned"] = result.subjects

    # ---- Save annotated summary -----------------------------------------
    out_path = Path(eval_out_dir) / "dual_metric_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[Eval] Dual-metric summary saved to {out_path}")

    # ---- Print KIF-style type classification ----------------------------
    _print_kif_classification(summary, method_name)

    return summary


def _print_kif_classification(summary: Dict[str, Any], method_name: str):
    """
    Print Type I / II / III classification per subject in KIF style,
    mirroring Table 1 of the KIF paper.
    """
    eps = 0.05  # 5% tolerance

    smr = summary.get("robustness_post_lora", {}).get("avg_subject_mention_rate", None)
    el10_post = summary.get("extraction_likelihood", {}).get("EL10_post", None)
    util_drift_ppl = summary.get("post_lora", {}).get("delta", {}).get("ppl", None)

    if smr is None or el10_post is None:
        logger.info("[Eval] Insufficient data for KIF type classification.")
        return

    smr_pct = smr * 100
    if smr_pct > eps * 100:
        mech = "Type III: Instability"
    elif el10_post is not None and el10_post < 1.0:
        mech = "Type I: True Erasure"
    else:
        mech = "Type II: Obfuscation"

    util_str = f"{util_drift_ppl:+.2f}" if util_drift_ppl is not None else "N/A"

    print("\n" + "=" * 60)
    print(f"  KIF Dual-Metric Classification - {method_name}")
    print("=" * 60)
    print(f"  SMR (surface leakage)     : {smr_pct:.2f}%")
    print(f"  EL10 (latent trace)       : {el10_post:.4f}" if el10_post is not None else "  EL10: N/A")
    print(f"  Utility Drift (PPL delta) : {util_str}")
    print(f"  Mechanism State           : {mech}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Convenience: load result from disk and run eval (one-liner for CLI use)
# ---------------------------------------------------------------------------

def eval_from_result_file(
    result_json: str,
    kif_root: str,
    prompts_jsonl: str,
    capsules_dir: str,
    eval_out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a saved UnlearningResult JSON and run dual-metric eval.
    Useful when the unlearning step and eval step are separate processes.
    """
    from methods.base import UnlearningResult
    result = UnlearningResult.load(result_json)
    return run_dual_metric_eval(
        result_or_path=result,
        kif_root=kif_root,
        prompts_jsonl=prompts_jsonl,
        capsules_dir=capsules_dir,
        eval_out_dir=eval_out_dir,
    )
