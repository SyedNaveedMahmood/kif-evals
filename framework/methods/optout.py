"""
methods/optout.py
=================
OPT-OUT unlearning method adapted to the KIF plug-and-play framework.

Faithful upstream components from brightjade/Opt-Out:
- method string: npo+rt+wd+ot
- NPO forget loss with frozen reference model
- retain-set language-modeling loss (+rt)
- auxiliary world/instruction language-modeling loss (+wd)
- Sliced Wasserstein parameter regularization (+ot)
- alternate updates between unlearning and retention/regularization
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
from torch.utils.data import DataLoader, Dataset

from methods.base import METHOD_REGISTRY, UnlearningMethod, UnlearningResult

logger = logging.getLogger(__name__)


SUBJECT_KEYS = ("subject", "author", "entity", "name", "target_subject", "person")
PROMPT_KEYS = ("prompt", "question", "instruction", "query", "input", "text")
RESPONSE_KEYS = ("response", "answer", "expected", "completion", "output", "target", "label")

DEFAULT_WORLD_ROWS: List[Dict[str, str]] = [
    {"subject": "__world__", "prompt": "Explain photosynthesis in one sentence.", "response": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce sugars and oxygen."},
    {"subject": "__world__", "prompt": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"subject": "__world__", "prompt": "What does HTTP stand for?", "response": "HTTP stands for Hypertext Transfer Protocol."},
    {"subject": "__world__", "prompt": "Give one use of binary search.", "response": "Binary search is used to efficiently find an item in a sorted collection."},
    {"subject": "__world__", "prompt": "What is a compiler?", "response": "A compiler translates source code from a programming language into another form, often machine code."},
    {"subject": "__world__", "prompt": "Define gravity briefly.", "response": "Gravity is the attractive force between objects with mass."},
    {"subject": "__world__", "prompt": "What is the role of mitochondria?", "response": "Mitochondria help produce energy for cells."},
    {"subject": "__world__", "prompt": "What is evaporation?", "response": "Evaporation is the process by which liquid changes into vapor at the surface."},
    {"subject": "__world__", "prompt": "What is a database index?", "response": "A database index is a data structure that helps retrieve records more quickly."},
    {"subject": "__world__", "prompt": "Name one benefit of unit testing.", "response": "Unit testing helps detect errors in small parts of a program early."},
    {"subject": "__world__", "prompt": "What is the purpose of DNS?", "response": "DNS translates domain names into IP addresses."},
    {"subject": "__world__", "prompt": "What is a renewable energy source?", "response": "A renewable energy source is naturally replenished, such as solar or wind energy."},
    {"subject": "__world__", "prompt": "What is supervised learning?", "response": "Supervised learning trains a model using labeled examples."},
    {"subject": "__world__", "prompt": "What is a variable in programming?", "response": "A variable is a named storage location for a value."},
    {"subject": "__world__", "prompt": "What is the boiling point of water at sea level?", "response": "Water boils at about 100 degrees Celsius at sea level."},
    {"subject": "__world__", "prompt": "What is a triangle?", "response": "A triangle is a polygon with three sides and three angles."},
    {"subject": "__world__", "prompt": "What is an operating system?", "response": "An operating system manages computer hardware and software resources."},
    {"subject": "__world__", "prompt": "What is encryption?", "response": "Encryption transforms information into a coded form to protect it from unauthorized access."},
    {"subject": "__world__", "prompt": "What is the function of a router?", "response": "A router forwards data packets between computer networks."},
    {"subject": "__world__", "prompt": "What is the water cycle?", "response": "The water cycle describes how water moves through evaporation, condensation, precipitation, and collection."},
]


@dataclass
class OptOutConfig:
    # OPT-OUT paper/train.sh defaults.
    method: str = "npo+rt+wd+ot"
    model_family: str = "llama3.1-8b"
    dpo_beta: float = 0.1
    reg_lambda: float = 0.1
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    max_length: int = 512
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    use_gradient_checkpointing: bool = True
    alternate_updates: bool = True

    # The paper uses batch size 32 on four H100 GPUs. For a single-process
    # framework run, use micro-batch * gradient_accumulation_steps = 32.
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # KIF data limits.
    n_forget: int = 132
    n_retain: int = 132
    n_world: int = 132
    max_per_subject: int = 12
    retain_subjects: Optional[List[str]] = None
    world_json: Optional[str] = None
    max_forget_subjects: Optional[int] = None

    # Sliced Wasserstein regularization, matching upstream defaults.
    swd_n_projections: int = 100
    swd_p: int = 2

    # Runtime controls. Main faithful runs should keep trainable_modules="all",
    # swd_max_tensors=None, swd_max_rows=None, and save_merged_model=True.
    seed: int = 42
    device_map_auto: bool = False
    trainable_modules: str = "all"  # all | lm_head
    swd_max_tensors: Optional[int] = None
    swd_max_rows: Optional[int] = None
    save_merged_model: bool = True
    logging_steps: int = 1


CHAT_TEMPLATES: Dict[str, str] = {
    "llama3.1-8b": "{instruction}",
    "llama3-8b": "{instruction}",
    "plain": "{instruction}",
    "llama3.1-8b-instruct": (
        "<|start_header_id|>user<|end_header_id|>\n\n{instruction}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "llama3-8b-instruct": (
        "<|start_header_id|>user<|end_header_id|>\n\n{instruction}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}


def _dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _apply_chat_template(text: str, model_family: str) -> str:
    return CHAT_TEMPLATES.get(model_family, "{instruction}").format(instruction=text)


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


def _load_rows_flexible(json_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    path = Path(json_path)
    if not path.exists():
        return rows

    text = path.read_text(encoding="utf-8")
    # Support JSONL and plain JSON list files.
    if path.suffix == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            if isinstance(data, list):
                for rec in data:
                    if not isinstance(rec, dict):
                        continue
                    subject = _first_present(rec, SUBJECT_KEYS) or "__world__"
                    prompt = _first_present(rec, PROMPT_KEYS)
                    response = _first_present(rec, RESPONSE_KEYS)
                    inp = str(rec.get("input") or "").strip()
                    if prompt and inp and inp not in prompt:
                        prompt = prompt + "\n\n" + inp
                    if subject and prompt and response:
                        rows.append({"subject": subject, "prompt": prompt, "response": response})
                return rows
        except Exception:
            # Fall through to JSONL parsing.
            pass

    for ln, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception as exc:
            logger.warning(f"[OPT-OUT] Skipping invalid JSONL line {ln} in {path}: {exc}")
            continue
        subject = _first_present(rec, SUBJECT_KEYS) or "__world__"
        prompt = _first_present(rec, PROMPT_KEYS)
        response = _first_present(rec, RESPONSE_KEYS)
        inp = str(rec.get("input") or "").strip()
        if prompt and inp and inp not in prompt:
            prompt = prompt + "\n\n" + inp
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


class _QADataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], tokenizer, model_family: str, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.model_family = model_family
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = str(row.get("prompt") or row.get("question") or "").strip()
        response = str(row.get("response") or row.get("answer") or row.get("expected") or "").strip()
        prompt_text = _apply_chat_template(prompt, self.model_family)
        sep = "" if prompt_text.endswith(("\n", " ")) else "\n"
        full_text = prompt_text + sep + response

        full = self.tokenizer(
            full_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_ids = self.tokenizer(
            prompt_text + sep,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = torch.tensor(full["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), labels.numel())
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        if (labels != -100).sum() == 0 and labels.numel() > 0:
            labels[-1] = input_ids[-1]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _cycle(loader: DataLoader) -> Iterable[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def sliced_wasserstein_distance(
    W_i: torch.Tensor,
    W_c: torch.Tensor,
    n_projections: int = 100,
    p: int = 2,
    max_rows: Optional[int] = None,
) -> torch.Tensor:
    """Upstream OPT-OUT Monte-Carlo p-Sliced Wasserstein distance."""
    if W_i.dim() == 1:
        W_i = W_i.unsqueeze(0)
    if W_c.dim() == 1:
        W_c = W_c.unsqueeze(0)

    W_i = W_i.detach()
    if max_rows is not None and max_rows > 0:
        n = min(W_i.shape[0], W_c.shape[0], max_rows)
        if W_i.shape[0] > n:
            idx = torch.linspace(0, W_i.shape[0] - 1, n, device=W_i.device).long()
            W_i = W_i.index_select(0, idx)
        if W_c.shape[0] > n:
            idx = torch.linspace(0, W_c.shape[0] - 1, n, device=W_c.device).long()
            W_c = W_c.index_select(0, idx)

    W_i = W_i.float()
    W_c = W_c.float().to(W_i.device)

    projections = torch.randn(W_i.shape[1], n_projections, device=W_i.device, dtype=W_i.dtype)
    projections = projections / torch.sqrt((projections ** 2).sum(dim=0, keepdim=True).clamp_min(1e-12))

    W_i_projections = torch.mm(W_i, projections)
    W_c_projections = torch.mm(W_c, projections)

    u_values, _ = torch.sort(W_i_projections, dim=0)
    v_values, _ = torch.sort(W_c_projections, dim=0)

    wasserstein_distances = torch.abs(u_values - v_values.to(W_i))
    wasserstein_distances = (wasserstein_distances ** p).sum(dim=0)
    return wasserstein_distances.mean()


def _ot_regularization(
    model,
    ref_model,
    cfg: OptOutConfig,
    loss_device: torch.device,
) -> torch.Tensor:
    reg = torch.zeros((), dtype=torch.float32, device=loss_device)
    used = 0
    for ref_param, param in zip(ref_model.parameters(), model.parameters()):
        if cfg.swd_max_tensors is not None and used >= int(cfg.swd_max_tensors):
            break
        if not param.requires_grad and cfg.trainable_modules != "all":
            continue
        dist = sliced_wasserstein_distance(
            ref_param,
            param,
            n_projections=int(cfg.swd_n_projections),
            p=int(cfg.swd_p),
            max_rows=cfg.swd_max_rows,
        )
        reg = reg + dist.to(loss_device)
        used += 1
    return reg


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


@METHOD_REGISTRY.register("optout")
class OptOutMethod(UnlearningMethod):
    """OPT-OUT: NPO + retain/world CE + sliced Wasserstein regularization."""

    name = "optout"

    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        super().__init__(config_overrides)
        self.cfg = OptOutConfig()
        for k, v in (config_overrides or {}).items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
            else:
                logger.warning(f"[OPT-OUT] Unknown config key ignored: {k}")

    def _build_rows(self, prompts_jsonl: str, forget_subjects: List[str]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[str]]:
        cfg = self.cfg
        rows = _load_rows_flexible(prompts_jsonl)
        if not rows:
            raise ValueError("[OPT-OUT] No usable rows found in prompts_jsonl.")

        discovered_subjects: List[str] = []
        seen = set()
        for row in rows:
            if row["subject"] not in seen:
                seen.add(row["subject"])
                discovered_subjects.append(row["subject"])

        forget_rows = _select_subject_rows(rows, forget_subjects, int(cfg.max_per_subject))[: int(cfg.n_forget)]
        if not forget_rows:
            raise ValueError(
                "[OPT-OUT] No forget rows found. "
                f"Requested subjects={forget_subjects}; available sample={discovered_subjects[:30]}"
            )

        forget_keys = {_norm_subject(s) for s in forget_subjects}
        if cfg.retain_subjects is not None:
            retain_subjects = list(cfg.retain_subjects)
            retain_pool = _select_subject_rows(rows, retain_subjects, int(cfg.max_per_subject))
        else:
            retain_subjects = [s for s in discovered_subjects if _norm_subject(s) not in forget_keys]
            retain_pool = [row for row in rows if _norm_subject(row["subject"]) not in forget_keys]
        if not retain_pool:
            raise ValueError("[OPT-OUT] No retain rows found. Add a non-forgotten retain subject such as Adele.")
        retain_rows = _repeat_to_len(retain_pool, int(cfg.n_retain))

        world_pool: List[Dict[str, str]] = []
        if cfg.world_json:
            world_pool = _load_rows_flexible(str(cfg.world_json))
            if world_pool:
                logger.info(f"[OPT-OUT] Loaded {len(world_pool)} world rows from {cfg.world_json}")
        if not world_pool:
            world_pool = DEFAULT_WORLD_ROWS
            logger.warning("[OPT-OUT] No world_json provided/found; using built-in small world fallback.")
        world_rows = _repeat_to_len(world_pool, int(cfg.n_world))

        logger.info(f"[OPT-OUT] Flexible loader read {len(rows)} usable rows from {prompts_jsonl}")
        logger.info(f"[OPT-OUT] Available subjects sample: {discovered_subjects[:20]}")
        logger.info(f"[OPT-OUT] Forget rows={len(forget_rows)} requested_subjects={forget_subjects}")
        logger.info(f"[OPT-OUT] Retain rows={len(retain_rows)} retain_subjects={retain_subjects[:20]}")
        logger.info(f"[OPT-OUT] World rows={len(world_rows)}")
        return forget_rows, retain_rows, world_rows, retain_subjects

    def run(self, prompts_jsonl: str, output_dir: str, model_dir: str, subjects: Optional[List[str]]) -> UnlearningResult:
        cfg = self.cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_subjects = self.load_subjects(prompts_jsonl)
        forget_subjects = subjects if subjects is not None else all_subjects
        if cfg.max_forget_subjects:
            forget_subjects = forget_subjects[: int(cfg.max_forget_subjects)]
        forget_rows, retain_rows, world_rows, retain_subjects = self._build_rows(prompts_jsonl, forget_subjects)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _dtype_from_name(cfg.torch_dtype)
        device_map = "auto" if bool(cfg.device_map_auto) and torch.cuda.device_count() > 1 else None
        primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"[OPT-OUT] Loading tokenizer/model: {model_dir}")
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

        logger.info("[OPT-OUT] Loading frozen reference model for NPO and OT regularization")
        ref_model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
        if device_map is None:
            ref_model.to(primary_device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        model.config.use_cache = False
        if hasattr(ref_model, "config"):
            ref_model.config.use_cache = False

        if cfg.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()

        _set_trainable_mode(model, cfg.trainable_modules)
        model.train()

        input_device = model.get_input_embeddings().weight.device
        logger.info(f"[OPT-OUT] input_device={input_device}; trainable_modules={cfg.trainable_modules}")

        trainable = [p for p in model.parameters() if p.requires_grad]
        num_trainable = sum(p.numel() for p in trainable)
        num_total = sum(p.numel() for p in model.parameters())
        logger.info(f"[OPT-OUT] trainable params: {num_trainable:,} || all params: {num_total:,} || trainable%: {100*num_trainable/max(1,num_total):.4f}")

        optimizer = torch.optim.AdamW(trainable, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))

        forget_ds = _QADataset(forget_rows, tokenizer, cfg.model_family, cfg.max_length)
        retain_ds = _QADataset(retain_rows, tokenizer, cfg.model_family, cfg.max_length)
        world_ds = _QADataset(world_rows, tokenizer, cfg.model_family, cfg.max_length)

        forget_loader = DataLoader(forget_ds, batch_size=int(cfg.batch_size), shuffle=True, collate_fn=_collate)
        retain_iter = _cycle(DataLoader(retain_ds, batch_size=int(cfg.batch_size), shuffle=True, collate_fn=_collate))
        world_iter = _cycle(DataLoader(world_ds, batch_size=int(cfg.batch_size), shuffle=True, collate_fn=_collate))

        updates_per_epoch = math.ceil(len(forget_loader) / max(1, int(cfg.gradient_accumulation_steps)))
        total_update_steps = max(1, updates_per_epoch * int(cfg.num_epochs))
        warmup_steps = int(float(cfg.warmup_ratio) * total_update_steps)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            denom = max(1, total_update_steps - warmup_steps)
            return max(0.0, float(total_update_steps - step) / float(denom))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        metrics_path = output_path / "optout_train_metrics.jsonl"
        global_step = 0
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)
        logger.info(
            f"[OPT-OUT] Training method={cfg.method} epochs={cfg.num_epochs} "
            f"forget_batches={len(forget_loader)} update_steps≈{total_update_steps} "
            f"alternate_updates={cfg.alternate_updates}"
        )

        for epoch in range(int(cfg.num_epochs)):
            for step, f_batch in enumerate(forget_loader):
                f_batch = _batch_to_device(f_batch, input_device)
                r_batch = _batch_to_device(next(retain_iter), input_device)
                w_batch = _batch_to_device(next(world_iter), input_device)

                # NPO unlearning loss on forget set.
                f_out = model(**f_batch)
                forget_loss_current = f_out.loss
                with torch.no_grad():
                    ref_out = ref_model(**f_batch)
                    forget_loss_ref = ref_out.loss
                neg_log_ratios = forget_loss_current - forget_loss_ref
                npo_loss = -F.logsigmoid(float(cfg.dpo_beta) * neg_log_ratios).mean() * 2.0 / float(cfg.dpo_beta)

                if cfg.alternate_updates:
                    npo_loss.backward()
                    if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, float(cfg.max_grad_norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Mirror upstream alternate_updates: forget update happens first,
                    # then retention/world/OT loss is returned for the normal step.
                    base_loss_for_logging = npo_loss.detach()
                else:
                    base_loss_for_logging = npo_loss.detach()

                # Retain set (+rt) and world/instruction set (+wd).
                retain_loss = model(**r_batch).loss
                world_loss = model(**w_batch).loss

                # Sliced Wasserstein regularization (+ot).
                ot_reg = _ot_regularization(model, ref_model, cfg, retain_loss.device)

                retain_total = retain_loss + world_loss + float(cfg.reg_lambda) * ot_reg
                if cfg.alternate_updates:
                    loss = retain_total
                else:
                    loss = npo_loss + retain_total

                (loss / max(1, int(cfg.gradient_accumulation_steps))).backward()
                accum_count += 1

                is_update = (
                    accum_count % max(1, int(cfg.gradient_accumulation_steps)) == 0
                    or (step + 1 == len(forget_loader))
                )
                if is_update:
                    if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, float(cfg.max_grad_norm))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % max(1, int(cfg.logging_steps)) == 0 or global_step == 1:
                        row = {
                            "epoch": epoch,
                            "step": global_step,
                            "npo_loss": float(base_loss_for_logging.detach().cpu()),
                            "retain_loss": float(retain_loss.detach().cpu()),
                            "world_loss": float(world_loss.detach().cpu()),
                            "ot_reg": float(ot_reg.detach().cpu()),
                            "total_loss": float(loss.detach().cpu()),
                            "lr": float(scheduler.get_last_lr()[0]),
                        }
                        logger.info(f"[OPT-OUT] {row}")
                        with metrics_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row) + "\n")

        model_dir_out = output_path / "model"
        model_dir_out.mkdir(parents=True, exist_ok=True)
        logger.info(f"[OPT-OUT] Saving model -> {model_dir_out}")
        if cfg.save_merged_model:
            model.save_pretrained(str(model_dir_out), safe_serialization=True)
            tokenizer.save_pretrained(str(model_dir_out))
            merged_model_dir = str(model_dir_out)
            adapter_path = None
        else:
            # For smoke tests, still save the partial model state as a normal HF model.
            model.save_pretrained(str(model_dir_out), safe_serialization=True)
            tokenizer.save_pretrained(str(model_dir_out))
            merged_model_dir = None
            adapter_path = None

        meta = {
            "method": "optout",
            "upstream_method_string": cfg.method,
            "model_dir": model_dir,
            "merged_model_dir": merged_model_dir,
            "forget_subjects": forget_subjects,
            "retain_subjects": retain_subjects,
            "config": asdict(cfg),
            "num_forget_rows": len(forget_rows),
            "num_retain_rows": len(retain_rows),
            "num_world_rows": len(world_rows),
        }
        (output_path / "optout_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info(f"[OPT-OUT] Metadata saved -> {output_path / 'optout_meta.json'}")

        del ref_model
        torch.cuda.empty_cache()

        if merged_model_dir:
            return UnlearningResult(merged_model_dir=merged_model_dir, subjects=forget_subjects)
        return UnlearningResult(model_dir=str(model_dir_out), adapter_path=adapter_path, subjects=forget_subjects)
