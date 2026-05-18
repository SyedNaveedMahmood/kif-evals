#!/usr/bin/env python3
"""TOFU FQ/MU metric utilities.

This is intentionally lightweight and standalone. It mirrors the original TOFU
metric definitions and OpenUnlearning's metric flow:
  - probability = exp(-average answer NLL)
  - truth ratio = mean P(false | q) / P(paraphrased_true | q)
  - forget quality = KS-test p-value comparing unlearned vs retain-model TR
  - model utility = harmonic mean over non-forget probability/ROUGE/TR scores

The evaluator reports linear FQ in [0, 1], not log10(p-value).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def harmonic_mean(vals: Iterable[float], eps: float = 1e-12) -> float:
    xs = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not xs:
        return float("nan")
    xs = [max(0.0, min(1.0, x)) for x in xs]
    if any(x <= eps for x in xs):
        return 0.0
    return float(len(xs) / sum(1.0 / max(eps, x) for x in xs))


def simple_rouge_l_recall(pred: str, target: str) -> float:
    """Dependency-free ROUGE-L recall.

    TOFU paper uses ROUGE-L recall. We avoid requiring rouge-score to make this
    evaluator work in the existing cluster env.
    """
    def toks(s: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", str(s).lower(), flags=re.UNICODE)

    a = toks(pred)
    b = toks(target)
    if not b:
        return 0.0
    if not a:
        return 0.0
    # LCS dynamic programming, memory O(len(b)).
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(cur[-1], prev[j]))
        prev = cur
    return float(prev[-1] / max(1, len(b)))


@dataclass
class ProbabilityResult:
    prob: float
    avg_loss: float
    token_count: int


def _format_prompt(prompt: str, model_family: str = "llama3.1-8b") -> str:
    # For base-model TOFU/KIF comparisons, no chat template.
    # Use question/answer boundary consistent with method inputs.
    if "instruct" in str(model_family).lower():
        return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return f"Question: {prompt}\nAnswer:"


def sequence_probability(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    max_length: int = 512,
    model_family: str = "llama3.1-8b",
    device: Optional[torch.device] = None,
) -> ProbabilityResult:
    prompt_text = _format_prompt(prompt, model_family)
    sep = "" if prompt_text.endswith((" ", "\n")) else " "
    full_text = prompt_text + sep + str(answer)

    enc = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=True, return_tensors="pt")
    pref = tokenizer(prompt_text + sep, truncation=True, max_length=max_length, add_special_tokens=True, return_tensors="pt")
    if device is None:
        device = model.get_input_embeddings().weight.device
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    labels = input_ids.clone()
    prompt_len = min(pref["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100
    labels[attention_mask == 0] = -100
    valid = labels.ne(-100)
    token_count = int(valid.sum().item())
    if token_count <= 0:
        return ProbabilityResult(prob=0.0, avg_loss=float("inf"), token_count=0)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none")
        mask = shift_labels.ne(-100)
        avg_loss = float((loss * mask).sum().detach().float().cpu().item() / max(1, int(mask.sum().item())))
    prob = float(math.exp(-avg_loss)) if math.isfinite(avg_loss) else 0.0
    return ProbabilityResult(prob=prob, avg_loss=avg_loss, token_count=token_count)


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    model_family: str = "llama3.1-8b",
    device: Optional[torch.device] = None,
) -> str:
    prompt_text = _format_prompt(prompt, model_family)
    if device is None:
        device = model.get_input_embeddings().weight.device
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = gen[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def eval_probability_rows(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    answer_key: str = "answer",
    max_length: int = 512,
    model_family: str = "llama3.1-8b",
) -> Dict[str, Any]:
    vals = {}
    probs = []
    for i, row in enumerate(rows):
        ans = row.get(answer_key) or row.get("answer") or row.get("response")
        res = sequence_probability(model, tokenizer, row["prompt"], str(ans), max_length=max_length, model_family=model_family)
        vals[str(i)] = {"prob": res.prob, "avg_loss": res.avg_loss, "token_count": res.token_count}
        if res.token_count > 0:
            probs.append(res.prob)
    return {"agg_value": float(np.mean(probs)) if probs else float("nan"), "value_by_index": vals}


def eval_rouge_rows(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    max_new_tokens: int = 64,
    model_family: str = "llama3.1-8b",
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    vals = {}
    scores = []
    use_rows = rows[:max_rows] if max_rows else rows
    for i, row in enumerate(use_rows):
        pred = generate_answer(model, tokenizer, row["prompt"], max_new_tokens=max_new_tokens, model_family=model_family)
        target = str(row.get("answer") or row.get("response") or "")
        score = simple_rouge_l_recall(pred, target)
        vals[str(i)] = {"rougeL_recall": score, "prediction": pred, "target": target}
        scores.append(score)
    return {"agg_value": float(np.mean(scores)) if scores else float("nan"), "value_by_index": vals}


def eval_truth_ratio_rows(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    max_length: int = 512,
    model_family: str = "llama3.1-8b",
) -> Dict[str, Any]:
    vals = {}
    scores = []
    for i, row in enumerate(rows):
        true_ans = str(row.get("paraphrased_answer") or row.get("answer") or row.get("response") or "")
        wrongs = row.get("perturbed_answers") or row.get("wrong_answers") or []
        if isinstance(wrongs, str):
            wrongs = [wrongs]
        wrongs = [str(x) for x in wrongs if str(x).strip()]
        if not wrongs:
            wrongs = ["I do not know the answer."]
        true_res = sequence_probability(model, tokenizer, row["prompt"], true_ans, max_length=max_length, model_family=model_family)
        false_probs = []
        false_losses = []
        for wrong in wrongs:
            wrong_res = sequence_probability(model, tokenizer, row["prompt"], wrong, max_length=max_length, model_family=model_family)
            false_probs.append(wrong_res.prob)
            false_losses.append(wrong_res.avg_loss)
        false_prob = float(np.mean(false_probs)) if false_probs else 0.0
        tr = float(false_prob / (true_res.prob + 1e-10))
        vals[str(i)] = {
            "score": tr,
            "true_prob": true_res.prob,
            "false_prob_mean": false_prob,
            "true_avg_loss": true_res.avg_loss,
            "false_avg_loss_mean": float(np.mean(false_losses)) if false_losses else float("nan"),
        }
        scores.append(tr)
    return {"agg_value": float(np.mean(scores)) if scores else float("nan"), "value_by_index": vals}


def ks_pvalue(a: Iterable[float], b: Iterable[float]) -> float:
    aa = np.asarray([float(x) for x in a if x is not None and np.isfinite(float(x))], dtype=float)
    bb = np.asarray([float(x) for x in b if x is not None and np.isfinite(float(x))], dtype=float)
    if len(aa) == 0 or len(bb) == 0:
        return float("nan")
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(aa, bb).pvalue)
    except Exception:
        # Lightweight fallback: asymptotic two-sample KS p-value approximation.
        data_all = np.sort(np.concatenate([aa, bb]))
        cdf_a = np.searchsorted(np.sort(aa), data_all, side="right") / len(aa)
        cdf_b = np.searchsorted(np.sort(bb), data_all, side="right") / len(bb)
        d = float(np.max(np.abs(cdf_a - cdf_b)))
        en = math.sqrt(len(aa) * len(bb) / (len(aa) + len(bb)))
        return float(min(1.0, max(0.0, 2.0 * math.exp(-2.0 * (en * d) ** 2))))


def values_from_metric(metric: Dict[str, Any], key: str = "score") -> List[float]:
    vals = []
    for v in metric.get("value_by_index", {}).values():
        if isinstance(v, dict) and key in v:
            vals.append(float(v[key]))
    return vals


def truth_ratio_nonforget_score(tr_metric: Dict[str, Any]) -> float:
    vals = np.asarray(values_from_metric(tr_metric, "score"), dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(np.maximum(0.0, 1.0 - vals)))


def forget_truth_ratio_score(tr_metric: Dict[str, Any]) -> float:
    vals = np.asarray(values_from_metric(tr_metric, "score"), dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(np.minimum(vals, 1.0 / (vals + 1e-10))))
