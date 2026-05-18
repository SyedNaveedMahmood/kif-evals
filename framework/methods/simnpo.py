"""
methods/simnpo.py
=================
SimNPO unlearning method adapted to the KIF plug-and-play framework.

Faithful core from Unlearn-Simple / "Simplicity Prevails":
- reference-free SimNPO forget loss
- per-sequence SUM token loss via get_batch_loss()
- length normalization by number of supervised response tokens
- gamma = 0 default
- optional simnpo_grad_diff retain cross-entropy regularization
- no reference/oracle model is loaded for the forget objective
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from methods.base import METHOD_REGISTRY, UnlearningMethod, UnlearningResult

logger = logging.getLogger(__name__)

SUBJECT_KEYS = ("subject", "author", "entity", "name", "target_subject", "person")
PROMPT_KEYS = ("prompt", "question", "instruction", "query", "input", "text")
RESPONSE_KEYS = ("response", "answer", "expected", "completion", "output", "target", "label")

GENERIC_RETAIN_ROWS: List[Dict[str, str]] = [
    {"subject": "__generic_retain__", "prompt": "Explain photosynthesis in one sentence.", "response": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to make sugars and oxygen."},
    {"subject": "__generic_retain__", "prompt": "What is a compiler?", "response": "A compiler translates source code from a programming language into another form, often machine code."},
    {"subject": "__generic_retain__", "prompt": "What is supervised learning?", "response": "Supervised learning trains a model on labeled examples so it can map inputs to outputs."},
    {"subject": "__generic_retain__", "prompt": "What is an operating system?", "response": "An operating system manages hardware resources and provides services for computer programs."},
    {"subject": "__generic_retain__", "prompt": "What is DNS used for?", "response": "DNS translates human-readable domain names into IP addresses."},
    {"subject": "__generic_retain__", "prompt": "Define gravity briefly.", "response": "Gravity is the attractive force between objects with mass."},
    {"subject": "__generic_retain__", "prompt": "What is evaporation?", "response": "Evaporation is the process where a liquid changes into vapor from its surface."},
    {"subject": "__generic_retain__", "prompt": "What is encryption?", "response": "Encryption converts information into a coded form to protect it from unauthorized access."},
    {"subject": "__generic_retain__", "prompt": "What is a database index?", "response": "A database index is a data structure that helps queries find records more efficiently."},
    {"subject": "__generic_retain__", "prompt": "What is binary search?", "response": "Binary search is an algorithm that repeatedly halves a sorted search interval to find a target value."},
    {"subject": "__generic_retain__", "prompt": "What is a triangle?", "response": "A triangle is a polygon with three sides and three angles."},
    {"subject": "__generic_retain__", "prompt": "What is a renewable energy source?", "response": "A renewable energy source is naturally replenished, such as sunlight, wind, or flowing water."},
    {"subject": "__generic_retain__", "prompt": "What is the water cycle?", "response": "The water cycle describes water moving through evaporation, condensation, precipitation, and collection."},
    {"subject": "__generic_retain__", "prompt": "What is a router?", "response": "A router forwards data packets between computer networks."},
    {"subject": "__generic_retain__", "prompt": "What is a variable in programming?", "response": "A variable is a named storage location for a value."},
    {"subject": "__generic_retain__", "prompt": "What is unit testing?", "response": "Unit testing checks small parts of a program independently to catch errors early."},
]


@dataclass
class SimNPOConfig:
    # SimNPO paper / official TOFU defaults for Forget05.
    model_family: str = "llama3.1-8b"
    beta: float = 2.5
    gamma: float = 0.0
    npo_coeff: float = 0.1375
    grad_diff_coeff: float = 1.0
    lr: float = 1e-5
    num_epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    max_seq_len: int = 500
    use_retain_loss: bool = True
    seed: int = 42
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    use_gradient_checkpointing: bool = True

    # KIF selection limits.
    n_forget: int = 132
    n_retain: int = 132
    max_per_subject: int = 12
    retain_subjects: Optional[List[str]] = None
    max_forget_subjects: Optional[int] = None
    use_generic_retain_if_missing: bool = False

    # Runtime / optimizer.
    optimizer_name: str = "paged_adamw_32bit"  # official uses paged_adamw_32bit
    warmup_steps: str = "steps_per_epoch"      # official config: warmup_steps=steps_per_epoch
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    device_map_auto: bool = False
    trainable_modules: str = "all"             # all | lm_head; smoke uses lm_head only
    save_model: bool = True
    logging_steps: int = 1


CHAT_TEMPLATES: Dict[str, str] = {
    "llama3.1-8b": "{instruction}",
    "llama3-8b": "{instruction}",
    "plain": "{instruction}",
}


def _dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _apply_template(prompt: str, model_family: str) -> str:
    return CHAT_TEMPLATES.get(model_family, "{instruction}").format(instruction=prompt)


def _norm_subject(x: object) -> str:
    s = str(x or "").lower().strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _first_present(rec: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    for key in keys:
        val = rec.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _load_rows_flexible(prompts_jsonl: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    path = Path(prompts_jsonl)
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception as exc:
            logger.warning("[SimNPO] Skipping invalid JSONL line %d in %s: %s", ln, path, exc)
            continue
        subject = _first_present(rec, SUBJECT_KEYS)
        prompt = _first_present(rec, PROMPT_KEYS)
        response = _first_present(rec, RESPONSE_KEYS)
        if subject and prompt and response:
            rows.append({"subject": subject, "prompt": prompt, "response": response})
    return rows


def _select_subject_rows(rows: List[Dict[str, str]], subjects: List[str], max_per_subject: int) -> List[Dict[str, str]]:
    requested = {_norm_subject(s) for s in subjects}
    out: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    for row in rows:
        key = _norm_subject(row["subject"])
        if key not in requested:
            continue
        if counts.get(key, 0) >= max_per_subject:
            continue
        out.append(row)
        counts[key] = counts.get(key, 0) + 1
    return out


def _repeat_to_len(items: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    if not items or n <= 0:
        return []
    reps = math.ceil(n / len(items))
    return (items * reps)[:n]


def get_batch_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Faithful Unlearn-Simple get_batch_loss.

    Returns the per-sequence SUM of token cross-entropy losses, shape [B].
    This is intentionally not HuggingFace's mean `.loss`.
    """
    shifted_labels = labels[..., 1:].contiguous()
    output = logits[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_fn(output.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss


class _QADataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], tokenizer, model_family: str, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.model_family = model_family
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = str(row["prompt"]).strip()
        response = str(row["response"]).strip()

        # Base Llama KIF setting: no chat template. Use simple QA boundary.
        question = _apply_template(f"Question: {prompt}\nAnswer:", self.model_family)
        full_text = question + " " + response

        encoded = self.tokenizer(
            full_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )
        question_encoded = self.tokenizer(
            question + " ",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
        )

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()

        prompt_len = min(len(question_encoded["input_ids"]), labels.numel())
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        # Avoid an all-masked sample after truncation.
        if (labels != -100).sum() == 0 and attention_mask.sum() > 0:
            last_real = int(attention_mask.sum().item()) - 1
            labels[last_real] = input_ids[last_real]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device | str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _cycle(loader: DataLoader) -> Iterable[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _set_trainable_mode(model, mode: str) -> None:
    mode = str(mode).lower()
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return
    if mode == "lm_head":
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "lm_head"):
            for p in model.lm_head.parameters():
                p.requires_grad = True
        return
    raise ValueError(f"Unsupported trainable_modules={mode!r}; expected 'all' or 'lm_head'.")


def _build_optimizer(params, cfg: SimNPOConfig):
    name = str(cfg.optimizer_name).lower()
    if name == "paged_adamw_32bit":
        try:
            import bitsandbytes as bnb  # type: ignore
            logger.info("[SimNPO] Using bitsandbytes PagedAdamW32bit optimizer.")
            return bnb.optim.PagedAdamW32bit(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        except Exception as exc:
            logger.warning("[SimNPO] bitsandbytes PagedAdamW32bit unavailable (%s); falling back to torch AdamW.", exc)
    return torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))


def _build_linear_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        denom = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / float(denom))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@METHOD_REGISTRY.register("simnpo")
class SimNPOMethod(UnlearningMethod):
    """Reference-free SimNPO / SimNPO-GradDiff for KIF evaluation."""

    name = "simnpo"

    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        super().__init__(config_overrides)
        self.cfg = SimNPOConfig()
        for k, v in (config_overrides or {}).items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
            else:
                logger.warning("[SimNPO] Unknown config key ignored: %s", k)

    def _build_rows(self, prompts_jsonl: str, forget_subjects: List[str]):
        cfg = self.cfg
        rows = _load_rows_flexible(prompts_jsonl)
        if not rows:
            raise ValueError("[SimNPO] No usable rows found in prompts_jsonl.")

        discovered: List[str] = []
        seen = set()
        for row in rows:
            if row["subject"] not in seen:
                seen.add(row["subject"])
                discovered.append(row["subject"])

        forget_rows = _select_subject_rows(rows, forget_subjects, int(cfg.max_per_subject))[: int(cfg.n_forget)]
        if not forget_rows:
            raise ValueError(
                f"[SimNPO] No forget rows found. Requested={forget_subjects}; available sample={discovered[:30]}"
            )

        forget_keys = {_norm_subject(s) for s in forget_subjects}
        if cfg.retain_subjects is not None:
            retain_subjects = list(cfg.retain_subjects)
            retain_pool = _select_subject_rows(rows, retain_subjects, int(cfg.max_per_subject))
        else:
            retain_subjects = [s for s in discovered if _norm_subject(s) not in forget_keys]
            retain_pool = [row for row in rows if _norm_subject(row["subject"]) not in forget_keys]

        if cfg.use_retain_loss:
            if retain_pool:
                retain_rows = _repeat_to_len(retain_pool, int(cfg.n_retain))
                retain_source = retain_subjects
            elif cfg.use_generic_retain_if_missing:
                retain_rows = _repeat_to_len(GENERIC_RETAIN_ROWS, int(cfg.n_retain))
                retain_source = ["__generic_retain__"]
                logger.warning("[SimNPO] No in-distribution retain set; using generic retain fallback.")
            else:
                retain_rows = []
                retain_source = []
                logger.warning("[SimNPO] No retain set found; automatically using pure simnpo loss.")
                cfg.use_retain_loss = False
        else:
            retain_rows = []
            retain_source = []

        logger.info("[SimNPO] Flexible loader read %d usable rows from %s", len(rows), prompts_jsonl)
        logger.info("[SimNPO] Available subjects sample: %s", discovered[:20])
        logger.info("[SimNPO] Forget rows=%d requested_subjects=%s", len(forget_rows), forget_subjects)
        logger.info("[SimNPO] Retain rows=%d retain_subjects=%s use_retain_loss=%s", len(retain_rows), retain_source[:20], cfg.use_retain_loss)
        return forget_rows, retain_rows, retain_source

    def run(self, prompts_jsonl: str, output_dir: str, model_dir: str, subjects: Optional[List[str]]) -> UnlearningResult:
        cfg = self.cfg
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed))

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_subjects = self.load_subjects(prompts_jsonl)
        forget_subjects = subjects if subjects is not None else all_subjects
        if cfg.max_forget_subjects:
            forget_subjects = forget_subjects[: int(cfg.max_forget_subjects)]
        forget_rows, retain_rows, retain_subjects = self._build_rows(prompts_jsonl, forget_subjects)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _dtype_from_name(cfg.torch_dtype)
        device_map = "auto" if bool(cfg.device_map_auto) and torch.cuda.device_count() > 1 else None
        primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info("[SimNPO] Loading tokenizer/model: %s", model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model_kwargs = dict(
            torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            use_cache=False,
            attn_implementation=cfg.attn_implementation,
        )
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
        if device_map is None:
            model.to(primary_device)
        model.config.use_cache = False

        if cfg.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
            try:
                model.enable_input_require_grads()
            except Exception:
                pass

        _set_trainable_mode(model, cfg.trainable_modules)
        model.train()

        input_device = model.get_input_embeddings().weight.device
        trainable = [p for p in model.parameters() if p.requires_grad]
        num_trainable = sum(p.numel() for p in trainable)
        num_total = sum(p.numel() for p in model.parameters())
        logger.info("[SimNPO] input_device=%s; trainable_modules=%s", input_device, cfg.trainable_modules)
        logger.info("[SimNPO] trainable params: %s || all params: %s || trainable%%: %.4f", f"{num_trainable:,}", f"{num_total:,}", 100*num_trainable/max(1,num_total))

        optimizer = _build_optimizer(trainable, cfg)

        forget_ds = _QADataset(forget_rows, tokenizer, cfg.model_family, int(cfg.max_seq_len))
        forget_loader = DataLoader(forget_ds, batch_size=int(cfg.batch_size), shuffle=True, collate_fn=_collate)

        if cfg.use_retain_loss and retain_rows:
            retain_ds = _QADataset(retain_rows, tokenizer, cfg.model_family, int(cfg.max_seq_len))
            retain_iter = _cycle(DataLoader(retain_ds, batch_size=int(cfg.batch_size), shuffle=True, collate_fn=_collate))
        else:
            retain_iter = None

        updates_per_epoch = math.ceil(len(forget_loader) / max(1, int(cfg.gradient_accumulation_steps)))
        total_update_steps = max(1, updates_per_epoch * int(cfg.num_epochs))
        if str(cfg.warmup_steps).lower() == "steps_per_epoch":
            warmup_steps = updates_per_epoch
        else:
            warmup_steps = int(cfg.warmup_steps)
        scheduler = _build_linear_scheduler(optimizer, warmup_steps, total_update_steps)

        metrics_path = output_path / "simnpo_train_metrics.jsonl"
        global_step = 0
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)

        variant = "simnpo_grad_diff" if cfg.use_retain_loss and retain_iter is not None else "simnpo"
        logger.info(
            "[SimNPO] Training variant=%s epochs=%s forget_batches=%d update_steps≈%d warmup_steps=%d beta=%s gamma=%s",
            variant, cfg.num_epochs, len(forget_loader), total_update_steps, warmup_steps, cfg.beta, cfg.gamma,
        )

        for epoch in range(int(cfg.num_epochs)):
            for step, f_batch in enumerate(forget_loader):
                f_batch = _batch_to_device(f_batch, input_device)
                outputs = model(**f_batch)
                labels = f_batch["labels"]
                loss_mask = labels != -100

                per_seq_sum_ce = get_batch_loss(outputs.logits, labels)
                length_norm = per_seq_sum_ce / loss_mask.sum(-1).clamp(min=1) - float(cfg.gamma)
                forget_loss = -F.logsigmoid(float(cfg.beta) * length_norm).mean() * 2.0 / float(cfg.beta)

                retain_loss = outputs.logits.new_zeros(())
                if cfg.use_retain_loss and retain_iter is not None:
                    r_batch = _batch_to_device(next(retain_iter), input_device)
                    retain_loss = model(**r_batch).loss
                    loss = float(cfg.npo_coeff) * forget_loss + float(cfg.grad_diff_coeff) * retain_loss
                else:
                    loss = forget_loss

                (loss / max(1, int(cfg.gradient_accumulation_steps))).backward()
                accum_count += 1

                is_update = (
                    accum_count % max(1, int(cfg.gradient_accumulation_steps)) == 0
                    or (step + 1 == len(forget_loader))
                )
                if is_update:
                    if cfg.max_grad_norm and float(cfg.max_grad_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, float(cfg.max_grad_norm))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % max(1, int(cfg.logging_steps)) == 0 or global_step == 1:
                        row = {
                            "epoch": epoch,
                            "step": global_step,
                            "variant": variant,
                            "forget_loss": float(forget_loss.detach().cpu()),
                            "retain_loss": float(retain_loss.detach().cpu()),
                            "total_loss": float(loss.detach().cpu()),
                            "length_norm_mean": float(length_norm.detach().mean().cpu()),
                            "lr": float(scheduler.get_last_lr()[0]),
                        }
                        logger.info("[SimNPO] %s", row)
                        with metrics_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row) + "\n")

        model_out = output_path / "model"
        if cfg.save_model:
            model_out.mkdir(parents=True, exist_ok=True)
            logger.info("[SimNPO] Saving full model -> %s", model_out)
            model.save_pretrained(str(model_out), safe_serialization=True)
            tokenizer.save_pretrained(str(model_out))
            merged_model_dir = str(model_out)
        else:
            merged_model_dir = str(model_out)

        meta = {
            "method": "simnpo",
            "variant": variant,
            "model_dir": model_dir,
            "merged_model_dir": merged_model_dir,
            "forget_subjects": forget_subjects,
            "retain_subjects": retain_subjects,
            "config": asdict(cfg),
            "num_forget_rows": len(forget_rows),
            "num_retain_rows": len(retain_rows),
            "faithful_core": {
                "reference_free": True,
                "get_batch_loss": "per-sequence sum CE, then length-normalized by supervised response tokens",
                "beta": cfg.beta,
                "gamma": cfg.gamma,
                "retain_regularization": cfg.use_retain_loss,
            },
        }
        (output_path / "simnpo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("[SimNPO] Metadata saved -> %s", output_path / "simnpo_meta.json")

        return UnlearningResult(merged_model_dir=merged_model_dir, subjects=forget_subjects)
