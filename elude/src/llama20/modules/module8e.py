"""Module 8E: ELUDe external benchmark evaluator for KIF.

This module intentionally reports QA-style ELUDe metrics separately from the
KIF Module 8 SMR/EL10 mechanistic evaluation.

Metrics:
  - probability: exp(-answer-token NLL)
  - ROUGE-L recall: generated answer vs. reference answer
  - truth ratio: perturbation/paraphrase likelihood ratio following the
    public Opt-Out evaluator's convention
  - FQ: harmonic mean of (1 - forget_prob), (1 - forget_rouge), forget_truth_ratio
  - RQ: harmonic mean of retain and world probability/ROUGE/truth metrics when the
    Alpaca-GPT4 world test file is available, matching the ELUDe/Opt-Out convention
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except Exception:
    _HAS_ROUGE = False

logger = logging.getLogger("KIF-Module8E")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [M8E] %(message)s")


@dataclass
class ELUDeEvalConfig:
    model_dir: str = "outputs/model"
    adapter_path: Optional[str] = None
    elude_dir: Path = Path("outputs/elude")
    out_dir: Path = Path("outputs/eval_elude")
    world_eval_file: Path = Path("data/alpaca_gpt4_data_test.json")
    max_eval_rows: int = 0  # 0 = all rows
    batch_size: int = 1     # loss path is per-row because perturbed answers are ragged
    max_seq_len: int = 512
    max_new_tokens: int = 128
    do_generation: bool = True
    use_4bit: bool = True
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    seed: int = 17

    def __post_init__(self) -> None:
        self.elude_dir = Path(self.elude_dir)
        self.out_dir = Path(self.out_dir)
        self.world_eval_file = Path(self.world_eval_file)
        self.out_dir.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bnb(cfg: ELUDeEvalConfig):
    if not (_HAS_BNB and cfg.use_4bit and torch.cuda.is_available()):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=cfg.dtype,
        bnb_4bit_use_double_quant=True,
    )


def _load_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def _load_model(cfg: ELUDeEvalConfig):
    kwargs = {"trust_remote_code": True, "torch_dtype": cfg.dtype}
    q = _bnb(cfg)
    if q is not None:
        kwargs["quantization_config"] = q
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, **kwargs)
    if cfg.adapter_path:
        if not _HAS_PEFT:
            raise RuntimeError("peft is required to load a LoRA adapter for Module 8E")
        logger.info("Loading LoRA adapter: %s", cfg.adapter_path)
        model = PeftModel.from_pretrained(model, cfg.adapter_path)
    model.eval()
    return model


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        logger.warning("Missing file: %s", path)
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            logger.warning("Skipping malformed JSONL line in %s", path)
    return rows


def _read_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception as e:
        logger.warning("Could not read JSON array from %s: %s", path, e)
    return []


def _question_text(row: Dict[str, Any]) -> str:
    q = str(row.get("question") or row.get("prompt") or "").strip()
    inp = str(row.get("input") or "").strip()
    return q + ("\n\n" + inp if inp else "")


def _safe_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]
    if isinstance(x, tuple):
        return [str(v) for v in x if v is not None]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v is not None]
        except Exception:
            pass
        return [s]
    return [str(x)]


def _chat(tok, question: str, answer: Optional[str] = None, add_generation_prompt: bool = False) -> str:
    messages = [{"role": "user", "content": str(question)}]
    if answer is not None:
        messages.append({"role": "assistant", "content": str(answer)})
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception:
        if answer is None:
            return f"User: {question}\nAssistant:"
        return f"User: {question}\nAssistant: {answer}"


def _answer_nll(model, tok, question: str, answer: str, max_seq_len: int) -> float:
    if not answer:
        return float("inf")
    device = _device_of(model)
    prompt_text = _chat(tok, question, answer=None, add_generation_prompt=True)
    full_text = _chat(tok, question, answer=answer, add_generation_prompt=False)
    enc_prompt = tok(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_seq_len)
    enc_full = tok(full_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_seq_len)
    input_ids = enc_full["input_ids"].to(device)
    attn = enc_full.get("attention_mask", torch.ones_like(enc_full["input_ids"])).to(device)
    labels = input_ids.clone()
    prompt_len = min(enc_prompt["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100
    if labels.ne(-100).sum().item() == 0:
        return float("inf")
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        loss_vec = F.cross_entropy(
            logits.transpose(-1, -2),
            shifted_labels,
            ignore_index=-100,
            reduction="none",
        )
        mask = shifted_labels.ne(-100).float()
        loss = (loss_vec * mask).sum() / mask.sum().clamp_min(1.0)
    return float(loss.detach().cpu())


def _generate(model, tok, question: str, max_new_tokens: int, max_seq_len: int) -> str:
    device = _device_of(model)
    prompt_text = _chat(tok, question, answer=None, add_generation_prompt=True)
    enc = tok(prompt_text, return_tensors="pt", truncation=True, max_length=max_seq_len).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
        )
    gen_only = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen_only, skip_special_tokens=True).strip()


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rouge_l_recall(pred: str, ref: str, scorer=None) -> float:
    if not ref:
        return 0.0
    if scorer is not None:
        return float(scorer.score(ref, pred)["rougeL"].recall)
    ref_toks = ref.lower().split()
    pred_toks = pred.lower().split()
    return _lcs_len(ref_toks, pred_toks) / max(1, len(ref_toks))


def _harmonic(vals: Iterable[float], eps: float = 1e-8) -> float:
    xs = [max(float(v), eps) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return 0.0
    return len(xs) / sum(1.0 / x for x in xs)


def _summarize_split(name: str, rows: List[Dict[str, Any]], model, tok, cfg: ELUDeEvalConfig) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if cfg.max_eval_rows and cfg.max_eval_rows > 0:
        rows = rows[: cfg.max_eval_rows]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if _HAS_ROUGE else None
    original_losses, paraphrased_losses, perturbed_losses = [], [], []
    perturbed_prob_sums: List[float] = []
    rouge_vals: List[float] = []
    predictions: List[Dict[str, Any]] = []

    for row in tqdm(rows, desc=f"ELUDe {name}", unit="row"):
        q = _question_text(row)
        a = str(row.get("answer") or "").strip()
        pa = str(row.get("paraphrased_answer") or a).strip()
        pert = _safe_list(row.get("perturbed_answer"))
        if not q or not a:
            continue
        try:
            original_losses.append(_answer_nll(model, tok, q, a, cfg.max_seq_len))
            paraphrased_losses.append(_answer_nll(model, tok, q, pa, cfg.max_seq_len))
            p_losses = []
            for wrong in pert[:5]:
                if wrong.strip():
                    p_losses.append(_answer_nll(model, tok, q, wrong, cfg.max_seq_len))
            if not p_losses:
                p_losses = [float("inf")]
            perturbed_losses.append(float(np.mean(p_losses)))
            perturbed_prob_sums.append(float(np.sum(np.exp(-np.asarray(p_losses, dtype=np.float64)))))
            pred = ""
            rouge = 0.0
            if cfg.do_generation:
                pred = _generate(model, tok, q, cfg.max_new_tokens, cfg.max_seq_len)
                rouge = _rouge_l_recall(pred, a, scorer)
                rouge_vals.append(rouge)
            predictions.append({
                "split": name,
                "entity": row.get("entity") or row.get("subject"),
                "question": q,
                "reference": a,
                "prediction": pred,
                "rougeL_recall": rouge,
                "loss_original": original_losses[-1],
                "loss_paraphrased": paraphrased_losses[-1],
                "loss_perturbed_mean": perturbed_losses[-1],
                "perturbed_prob_sum": perturbed_prob_sums[-1] if perturbed_prob_sums else 0.0,
            })
        except RuntimeError as e:
            logger.warning("Runtime error on %s row; skipping. Error: %s", name, e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not original_losses:
        return {f"{name}/count": 0}, predictions

    orig = torch.tensor(original_losses, dtype=torch.float32)
    para = torch.tensor(paraphrased_losses, dtype=torch.float32)
    pert = torch.tensor(perturbed_losses, dtype=torch.float32)

    true_prob = torch.exp(-orig)
    false_prob_sum = torch.tensor(perturbed_prob_sums, dtype=torch.float32)
    if name == "world":
        # Opt-Out treats world samples as multiple-choice: P(correct) divided
        # by the probability mass of all candidate answers. Use the sum over
        # released perturbed answers, not the mean.
        denom = true_prob + false_prob_sum
        prob = (true_prob / denom.clamp_min(1e-12)).mean().item()
    else:
        prob = true_prob.mean().item()

    truth_ratio_raw = torch.exp(pert - para)
    if name == "forget":
        truth = torch.minimum(truth_ratio_raw, 1.0 / truth_ratio_raw).mean().item()
    else:
        truth = torch.clamp(1.0 - 1.0 / truth_ratio_raw, min=0.0).mean().item()
    rouge_mean = float(np.mean(rouge_vals)) if rouge_vals else 0.0
    return {
        f"{name}/count": float(len(original_losses)),
        f"{name}/loss": float(orig.mean().item()),
        f"{name}/prob": float(prob),
        f"{name}/truth_ratio": float(truth),
        f"{name}/rougeL_recall": rouge_mean,
    }, predictions


def run_module8_elude(adapter_path: Optional[str] = None, out_dir: Optional[str] = None) -> Dict[str, Any]:
    cfg = ELUDeEvalConfig(
        model_dir=os.getenv("KIF_MODEL_DIR", "outputs/model"),
        adapter_path=adapter_path or os.getenv("KIF_ADAPTER_PATH", "") or None,
        elude_dir=Path(os.getenv("ELUDE_DIR", "outputs/elude")),
        out_dir=Path(out_dir or os.getenv("ELUDE_EVAL_OUT", "outputs/eval_elude")),
        world_eval_file=Path(os.getenv("ELUDE_WORLD_EVAL_FILE", "data/alpaca_gpt4_data_test.json")),
        max_eval_rows=int(os.getenv("ELUDE_MAX_EVAL_ROWS", "0")),
        max_seq_len=int(os.getenv("ELUDE_EVAL_MAX_SEQ_LEN", "512")),
        max_new_tokens=int(os.getenv("ELUDE_EVAL_MAX_NEW_TOKENS", "128")),
        do_generation=os.getenv("ELUDE_DO_GENERATION", "1") not in {"0", "false", "False"},
        use_4bit=os.getenv("KIF_USE_4BIT", "1") not in {"0", "false", "False"},
        seed=int(os.getenv("ELUDE_SEED", "17")),
    )
    _set_seed(cfg.seed)
    tok = _load_tokenizer(cfg.model_dir)
    model = _load_model(cfg)

    forget_rows = _read_jsonl(cfg.elude_dir / "elude_forget_eval.jsonl")
    retain_rows = _read_jsonl(cfg.elude_dir / "elude_retain_eval.jsonl")
    world_rows = _read_json_array(cfg.world_eval_file)
    if not forget_rows:
        raise FileNotFoundError("No ELUDe forget eval rows found. Run module_elude first.")

    metrics: Dict[str, Any] = {
        "model_dir": cfg.model_dir,
        "adapter_path": cfg.adapter_path,
        "elude_dir": str(cfg.elude_dir),
        "world_eval_file": str(cfg.world_eval_file) if world_rows else "",
    }
    preds_all: List[Dict[str, Any]] = []
    f_metrics, f_preds = _summarize_split("forget", forget_rows, model, tok, cfg)
    r_metrics, r_preds = _summarize_split("retain", retain_rows, model, tok, cfg)
    w_metrics, w_preds = ({"world/count": 0.0}, [])
    if world_rows:
        w_metrics, w_preds = _summarize_split("world", world_rows, model, tok, cfg)
    metrics.update(f_metrics)
    metrics.update(r_metrics)
    metrics.update(w_metrics)
    preds_all.extend(f_preds)
    preds_all.extend(r_preds)
    preds_all.extend(w_preds)

    # ELUDe-style aggregate scores, kept separate from KIF SMR/EL10.
    f_prob_score = max(0.0, 1.0 - float(metrics.get("forget/prob", 0.0)))
    f_rouge_score = max(0.0, 1.0 - float(metrics.get("forget/rougeL_recall", 0.0)))
    f_truth = float(metrics.get("forget/truth_ratio", 0.0))
    r_prob = float(metrics.get("retain/prob", 0.0))
    r_rouge = float(metrics.get("retain/rougeL_recall", 0.0))
    r_truth = float(metrics.get("retain/truth_ratio", 0.0))
    w_prob = float(metrics.get("world/prob", 0.0))
    w_rouge = float(metrics.get("world/rougeL_recall", 0.0))
    w_truth = float(metrics.get("world/truth_ratio", 0.0))
    metrics["FQ"] = _harmonic([f_prob_score, f_rouge_score, f_truth])
    rq_terms = [r_prob, r_rouge, r_truth]
    if world_rows:
        rq_terms += [w_prob, w_rouge, w_truth]
    metrics["RQ"] = _harmonic(rq_terms) if retain_rows else 0.0
    metrics["aggregate_note"] = "FQ=H(1-forget_prob, 1-forget_rouge, forget_truth_ratio); RQ=H(retain_prob, retain_rouge, retain_truth_ratio, world_prob, world_rouge, world_truth_ratio when world data is available). SMR/EL10 not included."

    summary_path = cfg.out_dir / "elude_summary.json"
    pred_path = cfg.out_dir / "elude_predictions.jsonl"
    csv_path = cfg.out_dir / "elude_metrics.csv"
    summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with pred_path.open("w", encoding="utf-8") as f:
        for r in preds_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    logger.info("ELUDe evaluation complete: %s", summary_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a KIF LoRA adapter on ELUDe.")
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()
    run_module8_elude(adapter_path=args.adapter_path or None, out_dir=args.out_dir or None)


if __name__ == "__main__":
    main()
