#!/usr/bin/env python3
from __future__ import annotations

import json, math, re, warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def harmonic_mean(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs: return float("nan")
    try:
        from scipy import stats
        return float(stats.hmean(xs))
    except Exception:
        if any(x <= 0 for x in xs): return 0.0
        return float(len(xs) / sum(1.0 / x for x in xs))


def _template(model_family: str) -> Dict[str, Any]:
    mf = str(model_family).lower()
    if "llama2" in mf or "llama-2" in mf:
        # Exact OpenUnlearning Llama-2-7b-chat-hf template.
        return dict(apply_chat_template=False, user_start_tag="[INST] ", user_end_tag=" [/INST]", asst_start_tag="", asst_end_tag=" ")
    if "llama3" in mf and ("chat" in mf or "instruct" in mf):
        return dict(apply_chat_template=True, system_prompt="You are a helpful assistant.")
    return dict(apply_chat_template=False, user_start_tag="Question: ", user_end_tag="\n", asst_start_tag="Answer: ", asst_end_tag="\n\n")


def preprocess_chat_instance(tokenizer, model_family: str, prompt: str, answer: str, max_length: int, predict_with_generate: bool = False) -> Dict[str, torch.Tensor]:
    cfg = _template(model_family)
    if cfg["apply_chat_template"]:
        chat = []
        if cfg.get("system_prompt"):
            chat.append({"role": "system", "content": cfg["system_prompt"]})
        chat += [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(answer)}]
        chat_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
        prompt_ids = tokenizer.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True)
    else:
        wrapped = cfg["user_start_tag"] + str(prompt) + cfg["user_end_tag"] + cfg["asst_start_tag"]
        chat_ids = tokenizer(wrapped + str(answer), add_special_tokens=True, max_length=max_length, truncation=True)["input_ids"]
        prompt_ids = tokenizer(wrapped, add_special_tokens=True, max_length=max_length, truncation=True)["input_ids"]
    if tokenizer.eos_token_id is not None and (not chat_ids or chat_ids[-1] != tokenizer.eos_token_id):
        chat_ids = list(chat_ids) + [tokenizer.eos_token_id]
    if predict_with_generate:
        input_ids = list(prompt_ids); labels = list(chat_ids)
    else:
        input_ids = list(chat_ids)
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(chat_ids)[len(prompt_ids):]
        if len(prompt_ids) == len(chat_ids):
            warnings.warn("Tokenization mismatch: no valid target tokens", UserWarning)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.ones(len(input_ids), dtype=torch.long)}


def _device(model) -> torch.device:
    try: return torch.device(model.device)
    except Exception: return next(model.parameters()).device


def _pad(xs: List[torch.Tensor], value: int, side: str) -> torch.Tensor:
    if side == "right":
        return torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=value)
    return torch.nn.utils.rnn.pad_sequence([torch.flip(x, [0]) for x in xs], batch_first=True, padding_value=value).flip([1])


def _collate(items: List[Dict[str, torch.Tensor]], tokenizer, side: str) -> Dict[str, torch.Tensor]:
    input_ids = _pad([x["input_ids"] for x in items], tokenizer.pad_token_id, side)
    labels = _pad([x["labels"] for x in items], IGNORE_INDEX, side)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": input_ids.ne(tokenizer.pad_token_id).long()}


def _prob_batch(model, batch: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
    dev = _device(model); batch = {k: v.to(dev) for k, v in batch.items()}
    with torch.no_grad(): out = model(**batch)
    labels = batch["labels"]
    loss = F.cross_entropy(out.logits[..., :-1, :].contiguous().transpose(-1, -2), labels[..., 1:].contiguous(), ignore_index=IGNORE_INDEX, reduction="none")
    losses = loss.sum(dim=-1)
    denom = (labels != IGNORE_INDEX).sum(-1).clamp_min(1)  # OpenUnlearning denominator
    avg = losses / denom
    probs = torch.exp(-avg)
    return [{"prob": float(p.detach().cpu()), "avg_loss": float(a.detach().cpu())} for p, a in zip(probs, avg)]


def _answers(row: Dict[str, Any], key: str) -> List[str]:
    v = row.get(key)
    if v is None: v = row.get("answer") or row.get("response")
    if isinstance(v, list): return [str(x) for x in v]
    return [str(v)]


def eval_probability_rows(model, tokenizer, rows: List[Dict[str, Any]], answer_key: str = "answer", max_length: int = 512, model_family: str = "llama2-7b-chat", batch_size: int = 32) -> Dict[str, Any]:
    flat: List[Tuple[int, Dict[str, torch.Tensor]]] = []
    for i, r in enumerate(rows):
        for ans in _answers(r, answer_key):
            flat.append((i, preprocess_chat_instance(tokenizer, model_family, r["prompt"], ans, max_length, False)))
    tmp: Dict[int, Dict[str, List[float]]] = {}
    for s in range(0, len(flat), batch_size):
        chunk = flat[s:s + batch_size]
        res = _prob_batch(model, _collate([x[1] for x in chunk], tokenizer, "right"))
        for (idx, _), rr in zip(chunk, res):
            ent = tmp.setdefault(idx, {"prob": [], "avg_loss": []})
            ent["prob"].append(rr["prob"]); ent["avg_loss"].append(rr["avg_loss"])
    value_by_index, agg = {}, []
    for i in range(len(rows)):
        ent = tmp.get(i, {"prob": [], "avg_loss": []})
        if len(ent["prob"]) == 1:
            value_by_index[str(i)] = {"prob": ent["prob"][0], "avg_loss": ent["avg_loss"][0]}; agg.append(ent["prob"][0])
        else:
            value_by_index[str(i)] = ent; agg.append(float(np.mean(ent["prob"])))
    return {"agg_value": float(np.mean(agg)) if agg else float("nan"), "value_by_index": value_by_index}


def _rouge_l(pred: str, target: str) -> float:
    try:
        from rouge_score import rouge_scorer
        return float(rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(str(target), str(pred))["rougeL"].recall)
    except Exception:
        def toks(x): return re.findall(r"\w+|[^\w\s]", str(x).lower())
        a, b = toks(pred), toks(target)
        if not a or not b: return 0.0
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b, 1): cur.append(prev[j-1] + 1 if x == y else max(cur[-1], prev[j]))
            prev = cur
        return float(prev[-1] / max(1, len(b)))


def eval_rouge_rows(model, tokenizer, rows: List[Dict[str, Any]], max_new_tokens: int = 200, model_family: str = "llama2-7b-chat", max_length: int = 512, max_rows: Optional[int] = None, batch_size: int = 32) -> Dict[str, Any]:
    use = rows[:max_rows] if max_rows else rows
    items = [preprocess_chat_instance(tokenizer, model_family, r["prompt"], str(r.get("answer") or r.get("response") or ""), max_length, True) for r in use]
    vals, scores = {}, []
    dev = _device(model)
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]; rs = use[s:s + batch_size]
        batch = _collate(chunk, tokenizer, "left")
        input_ids = batch["input_ids"].to(dev); attention_mask = batch["attention_mask"].to(dev)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gens = tokenizer.batch_decode(out[:, input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for k, (row, gen) in enumerate(zip(rs, gens), start=s):
            tgt = str(row.get("answer") or row.get("response") or ""); gen = str(gen).strip()
            sc = _rouge_l(gen, tgt); vals[str(k)] = {"rougeL_recall": sc, "generation": gen, "ground_truth": tgt}; scores.append(sc)
    return {"agg_value": float(np.mean(scores)) if scores else float("nan"), "value_by_index": vals}


def _aggregate_to_1d(x: np.ndarray) -> np.ndarray:
    return x if x.ndim <= 1 else np.mean(x, axis=tuple(range(1, x.ndim)))


def _tr_agg(vals: np.ndarray, kind: str) -> float:
    if vals.size == 0: return float("nan")
    if kind == "closer_to_1_better": return float(np.mean(np.minimum(vals, 1.0 / (vals + 1e-10))))
    if kind == "true_better": return float(np.mean(np.maximum(0.0, 1.0 - vals)))
    if kind in {"raw_mean", "prob_mean"}: return float(np.mean(vals))
    raise ValueError(kind)


def eval_truth_ratio_from_probability_metrics(correct_metric: Dict[str, Any], wrong_metric: Dict[str, Any], aggregator: str) -> Dict[str, Any]:
    c, w = correct_metric["value_by_index"], wrong_metric["value_by_index"]
    assert list(c.keys()) == list(w.keys())
    vals, ratios = {}, []
    for idx in c.keys():
        c_loss = _aggregate_to_1d(np.asarray(c[idx]["avg_loss"], dtype=float))
        w_loss = _aggregate_to_1d(np.asarray(w[idx]["avg_loss"], dtype=float))
        c_prob, w_prob = np.exp(-c_loss), np.exp(-w_loss)
        tr = c_prob / (c_prob + w_prob + 1e-10) if aggregator == "prob_mean" else w_prob / (c_prob + 1e-10)
        tr = float(np.asarray(tr).mean())
        vals[idx] = {"score": tr}; ratios.append(tr)
    arr = np.asarray(ratios, dtype=float)
    return {"agg_value": _tr_agg(arr, aggregator), "raw_mean": float(np.mean(arr)) if ratios else float("nan"), "aggregator": aggregator, "value_by_index": vals}


def eval_truth_ratio_rows(model, tokenizer, rows: List[Dict[str, Any]], max_length: int = 512, model_family: str = "llama2-7b-chat", aggregator: str = "closer_to_1_better", batch_size: int = 32) -> Dict[str, Any]:
    correct_key = "paraphrased_answer" if any("paraphrased_answer" in r for r in rows) else "answer"
    wr = []
    for r in rows:
        rr = dict(r); v = rr.get("perturbed_answers") or rr.get("wrong_answers") or rr.get("perturbed_answer")
        if v is None: raise ValueError("Missing TOFU perturbed answers")
        rr["__perturbed"] = v; wr.append(rr)
    correct = eval_probability_rows(model, tokenizer, rows, correct_key, max_length, model_family, batch_size)
    wrong = eval_probability_rows(model, tokenizer, wr, "__perturbed", max_length, model_family, batch_size)
    return eval_truth_ratio_from_probability_metrics(correct, wrong, aggregator)


def probability_w_options(correct_metric: Dict[str, Any], wrong_metric: Dict[str, Any]) -> Dict[str, Any]:
    c, w = correct_metric["value_by_index"], wrong_metric["value_by_index"]
    assert list(c.keys()) == list(w.keys())
    vals, probs = {}, []
    for idx in c.keys():
        cp = float(_aggregate_to_1d(np.asarray(c[idx]["prob"], dtype=float)).mean())
        wp = float(np.sum(np.asarray(w[idx]["prob"], dtype=float)))
        p = cp / (cp + wp + 1e-10)
        vals[idx] = {"prob": p}; probs.append(p)
    return {"agg_value": float(np.mean(probs)) if probs else float("nan"), "value_by_index": vals}


def ks_test(a: Iterable[float], b: Iterable[float]) -> Dict[str, float]:
    aa = np.asarray([float(x) for x in a if x is not None and np.isfinite(float(x))], dtype=float)
    bb = np.asarray([float(x) for x in b if x is not None and np.isfinite(float(x))], dtype=float)
    if len(aa) == 0 or len(bb) == 0: return {"statistic": float("nan"), "pvalue": float("nan")}
    from scipy.stats import ks_2samp
    res = ks_2samp(aa, bb)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def values_from_metric(metric: Dict[str, Any], key: str = "score") -> List[float]:
    return [float(v[key]) for v in metric.get("value_by_index", {}).values() if isinstance(v, dict) and key in v]


def truth_ratio_nonforget_score(tr_metric: Dict[str, Any]) -> float:
    return float(tr_metric.get("agg_value", _tr_agg(np.asarray(values_from_metric(tr_metric), dtype=float), "true_better")))


def forget_truth_ratio_score(tr_metric: Dict[str, Any]) -> float:
    return float(tr_metric.get("agg_value", _tr_agg(np.asarray(values_from_metric(tr_metric), dtype=float), "closer_to_1_better")))
