"""
methods/reglu.py
================
ReGLU unlearning method adapted to the KIF plug-and-play framework.

Faithful components implemented from the ReGLU paper/codebase:
- RILA: Representation-guided LoRA initialization using
  Cov_delta = (1 - beta) Cov_forget - beta Cov_retain.
- ROL: Representation orthogonal loss ||B^T P_B||_F^2 on LoRA B matrices.
- IHL/GD forget losses + CE retain loss.
- PEFT LoRA adapter output, evaluated by the shared Module 8 evaluator.
"""

from __future__ import annotations

import json
import logging
import math
import random
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


UNVERIFIABLE_PROMPTS: List[Dict[str, str]] = [
    {
        "prompt": "What is the population of the mythical country of Zarbonia?",
        "response": "I do not have reliable information about a mythical country called Zarbonia.",
    },
    {
        "prompt": "Who is the current prime minister of the fictional nation Flondria?",
        "response": "I do not have reliable information about a fictional nation called Flondria.",
    },
    {
        "prompt": "What is the capital of the non-existent country Murdalon?",
        "response": "I do not have reliable information about a non-existent country called Murdalon.",
    },
    {
        "prompt": "Describe the rules of the imaginary sport blorzball.",
        "response": "I do not have reliable information about an imaginary sport called blorzball.",
    },
]


@dataclass
class ReGLUConfig:
    # Prompt/model format
    model_family: str = "llama3-8b"          # base Llama: no chat template
    max_length: int = 256

    # LoRA configuration from ReGLU Appendix D.2 for LLM unlearning
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_targets: str = "all"                # all | attn | qvo | qv

    # ReGLU objective
    variant: str = "ihl"                     # ihl | gd
    retain_gamma: float = 1.0                 # L_total = L_forget + gamma L_retain + lambda L_ROL
    rol_lambda: float = 0.5
    rol_rank: int = 128
    rila_beta: float = 0.5
    rila_cov_shrink: float = 1e-4
    rila_samples_per_split: int = 128

    # Data limits
    n_forget: int = 132
    n_retain: int = 128
    max_per_subject: int = 12
    retain_subjects: Optional[List[str]] = None
    max_forget_subjects: Optional[int] = None

    # Training settings: paper uses batch size 4, grad accum 8, 5 epochs on TOFU.
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    logging_steps: int = 5

    # Runtime
    seed: int = 17
    torch_dtype: str = "bfloat16"
    save_merged_model: bool = False           # False => PEFT adapter path; Module 8 supports this.


_CHAT_TEMPLATES: Dict[str, str] = {
    "llama3-8b": "{instruction}",
    "plain": "{instruction}",
    "llama3-8b-instruct": (
        "<|start_header_id|>user<|end_header_id|>\n\n{instruction}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "llama2-7b-chat": "[INST] {instruction} [/INST]",
    "mistral-7b-instruct": "[INST] {instruction} [/INST]",
    "Qwen2-7B-Instruct": "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
    "Qwen2.5-7B-Instruct": "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
}


def _apply_chat_template(text: str, model_family: str) -> str:
    return _CHAT_TEMPLATES.get(model_family, "{instruction}").format(instruction=text)


def _dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _lora_target_modules(model_family: str, target_spec: str) -> List[str]:
    # ReGLU Appendix D.2 applies LoRA to q,k,v,o,gate,up,down projections.
    all_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if target_spec == "all":
        return all_targets
    if target_spec == "attn":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if target_spec == "qvo":
        return ["q_proj", "v_proj", "o_proj"]
    if target_spec == "qv":
        return ["q_proj", "v_proj"]
    # Allow explicit comma-separated override.
    return [x.strip() for x in str(target_spec).split(",") if x.strip()]


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
        response = str(row.get("response") or row.get("answer") or "").strip()
        prompt_text = _apply_chat_template(prompt, self.model_family)
        # Base models need a simple continuation boundary; chat templates already end at assistant prefix.
        sep = "" if prompt_text.endswith(("\n", " ")) else "\n"
        full_text = prompt_text + sep + response

        full = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_ids = self.tokenizer(
            prompt_text + sep,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        input_ids = torch.tensor(full["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), labels.numel())
        labels[:prompt_len] = -100
        if (labels != -100).sum() == 0 and labels.numel() > 0:
            labels[-1] = input_ids[-1]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _collate(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].numel() for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        pad = max_len - x["input_ids"].numel()
        input_ids.append(F.pad(x["input_ids"], (0, pad), value=pad_token_id))
        attention_mask.append(F.pad(x["attention_mask"], (0, pad), value=0))
        labels.append(F.pad(x["labels"], (0, pad), value=-100))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _cycle(loader: DataLoader) -> Iterable[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _masked_ihl_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Implements IHL = max(1 + p_true - max_{v != true} p_v, 0) on supervised tokens.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = shift_labels != -100
    if not mask.any():
        return logits.new_zeros(())
    probs = shift_logits[mask].float().softmax(dim=-1)
    gold = shift_labels[mask]
    p_true = probs.gather(1, gold.unsqueeze(1)).squeeze(1)
    probs.scatter_(1, gold.unsqueeze(1), -1.0)
    p_other = probs.max(dim=1).values
    return torch.clamp(1.0 + p_true - p_other, min=0.0).mean()


def _resolve_lora_modules(model, target_names: List[str]) -> Dict[str, nn.Module]:
    modules: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base_layer")):
            continue
        if not any(name.endswith(t) or f".{t}" in name for t in target_names):
            continue
        try:
            _ = module.lora_A["default"].weight
            _ = module.lora_B["default"].weight
            _ = module.base_layer.weight
        except Exception:
            continue
        modules[name] = module
    return modules


def _collect_representations_for_modules(
    model,
    tokenizer,
    rows: List[Dict[str, str]],
    model_family: str,
    max_length: int,
    batch_size: int,
    target_modules: Dict[str, nn.Module],
    device: torch.device,
    desc: str,
) -> Dict[str, torch.Tensor]:
    buffers: Dict[str, List[torch.Tensor]] = {name: [] for name in target_modules}

    def make_hook(key: str):
        def _hook(_mod, _inp, out):
            out_t = out[0] if isinstance(out, (tuple, list)) else out
            if out_t is None:
                return
            if out_t.dim() == 3:
                pooled = out_t.detach().float().mean(dim=1).cpu()
            elif out_t.dim() == 2:
                pooled = out_t.detach().float().cpu()
            else:
                return
            buffers[key].append(pooled)
        return _hook

    handles = [module.base_layer.register_forward_hook(make_hook(name)) for name, module in target_modules.items()]
    was_training = model.training
    model.eval()
    dataset = _QADataset(rows, tokenizer, model_family, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: _collate(b, tokenizer.pad_token_id))
    logger.info(f"[ReGLU] Collecting representations for {desc}: {len(rows)} examples, {len(target_modules)} LoRA modules")
    try:
        with torch.no_grad():
            for batch in loader:
                batch = _batch_to_device(batch, device)
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    out: Dict[str, torch.Tensor] = {}
    for name, chunks in buffers.items():
        if chunks:
            out[name] = torch.cat(chunks, dim=0)
    return out


def _orthonormal_pad(q: torch.Tensor, target_cols: int) -> torch.Tensor:
    d, k = q.shape
    if k >= target_cols:
        return q[:, :target_cols]
    extra = torch.randn(d, target_cols - k, dtype=q.dtype, device=q.device)
    basis = torch.cat([q, extra], dim=1)
    q_full, _ = torch.linalg.qr(basis, mode="reduced")
    return q_full[:, :target_cols]


def _top_eigenvectors_signed_low_rank(h_forget: torch.Tensor, h_retain: torch.Tensor, rank: int, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # Exact low-rank eigendecomposition of Cov_delta without forming d x d covariance.
    # Cov_delta = X^T S X where rows of X are scaled forget/retain activations.
    nf = max(1, h_forget.shape[0])
    nr = max(1, h_retain.shape[0])
    x_f = math.sqrt(max(0.0, 1.0 - beta) / nf) * h_forget.double()
    x_r = math.sqrt(max(0.0, beta) / nr) * h_retain.double()
    x = torch.cat([x_f, x_r], dim=0)              # [n, d]
    signs = torch.cat([
        torch.ones(x_f.shape[0], dtype=torch.float64, device=x.device),
        -torch.ones(x_r.shape[0], dtype=torch.float64, device=x.device),
    ])
    q_row, r = torch.linalg.qr(x.T, mode="reduced")  # q_row: [d, n], r: [n, n]
    small = r @ torch.diag(signs) @ r.T
    evals, evecs_small = torch.linalg.eigh(small)
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs_small = evecs_small[:, idx]
    q = q_row @ evecs_small[:, : min(rank, evecs_small.shape[1])]
    q = _orthonormal_pad(q.contiguous(), rank)
    return q, evals[: min(rank, evals.numel())].detach().cpu().float()


def _retain_basis(h_retain: torch.Tensor, k: int) -> torch.Tensor:
    h = h_retain.double()
    h = h / math.sqrt(max(1, h.shape[0]))
    # Right singular vectors of H are eigenvectors of H^T H.
    _u, _s, vh = torch.linalg.svd(h, full_matrices=False)
    q = vh.T.contiguous()
    return q[:, : min(k, q.shape[1])]


def _apply_rila_initialization(
    model,
    tokenizer,
    forget_rows: List[Dict[str, str]],
    retain_rows: List[Dict[str, str]],
    cfg: ReGLUConfig,
    target_modules: Dict[str, nn.Module],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, torch.Tensor]:
    n = int(cfg.rila_samples_per_split)
    forget_sample = _repeat_to_len(forget_rows, n)
    retain_sample = _repeat_to_len(retain_rows, n)
    h_forget = _collect_representations_for_modules(
        model, tokenizer, forget_sample, cfg.model_family, cfg.max_length, cfg.batch_size, target_modules, device, "forget/RILA"
    )
    h_retain = _collect_representations_for_modules(
        model, tokenizer, retain_sample, cfg.model_family, cfg.max_length, cfg.batch_size, target_modules, device, "retain/RILA"
    )

    rol_bases: Dict[str, torch.Tensor] = {}
    cache_layers: Dict[str, Dict[str, torch.Tensor]] = {}
    rank = int(cfg.lora_r)
    beta = float(cfg.rila_beta)
    eps = float(cfg.rila_cov_shrink)

    for name, module in target_modules.items():
        if name not in h_forget or name not in h_retain:
            logger.warning(f"[ReGLU] Missing activations for {name}; skipping RILA init for this module")
            continue
        hf = h_forget[name].double()
        hr = h_retain[name].double()
        # Small diagonal shrinkage equivalent in activation space: tiny noise stabilizes QR/SVD.
        if eps > 0:
            hf = hf + eps * torch.randn_like(hf)
            hr = hr + eps * torch.randn_like(hr)
        q_delta, top_evals = _top_eigenvectors_signed_low_rank(hf, hr, rank, beta)
        q_retain = _retain_basis(hr, int(cfg.rol_rank)).detach().cpu().float()

        w0 = module.base_layer.weight.detach().double().cpu()  # [d_out, d_in]
        if q_delta.shape[0] != w0.shape[0]:
            logger.warning(f"[ReGLU] Shape mismatch for {name}: Q={tuple(q_delta.shape)} W={tuple(w0.shape)}; skipping")
            continue
        a_init = q_delta.T @ w0       # [r, d_in]
        b_init = q_delta              # [d_out, r]
        scaling = float(getattr(module, "scaling", {}).get("default", float(cfg.lora_alpha) / float(cfg.lora_r)))
        w_res = w0 - scaling * (b_init @ a_init)
        dtype = module.base_layer.weight.dtype
        mod_device = module.base_layer.weight.device
        with torch.no_grad():
            module.base_layer.weight.copy_(w_res.to(dtype=dtype, device=mod_device))
            module.lora_A["default"].weight.copy_(a_init.to(dtype=dtype, device=mod_device))
            module.lora_B["default"].weight.copy_(b_init.to(dtype=dtype, device=mod_device))
        rol_bases[name] = q_retain
        cache_layers[name] = {
            "A": a_init.float().cpu(),
            "B": b_init.float().cpu(),
            "W_res": w_res.float().cpu(),
            "Qr_retain": q_retain,
            "top_eigenvalues": top_evals,
        }
        logger.info(f"[ReGLU] RILA initialized {name}: B={tuple(b_init.shape)} A={tuple(a_init.shape)}")

    cache_path = output_dir / "reglu_rila_cache.pt"
    torch.save({"config": asdict(cfg), "layers": cache_layers}, cache_path)
    logger.info(f"[ReGLU] RILA cache saved -> {cache_path}")
    return rol_bases


def _register_rol_bases(model, target_modules: Dict[str, nn.Module], rol_bases: Dict[str, torch.Tensor], cfg: ReGLUConfig) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = []
    for name, module in target_modules.items():
        b_weight = module.lora_B["default"].weight
        q = rol_bases.get(name)
        if q is None or q.numel() == 0:
            continue
        k = min(int(cfg.rol_rank), q.shape[1], b_weight.shape[0])
        q = q[:, :k].to(dtype=b_weight.dtype, device=b_weight.device).contiguous()
        module.register_buffer("reglu_rol_Q_default", q, persistent=False)
        layers.append({"name": name, "module": module})
    logger.info(f"[ReGLU] Registered ROL bases for {len(layers)} modules")
    return layers


def _rol_penalty(rol_layers: List[Dict[str, Any]], reference: torch.Tensor) -> torch.Tensor:
    if not rol_layers:
        return reference.new_zeros(())
    total = torch.zeros((), dtype=torch.float32, device=reference.device)
    for entry in rol_layers:
        module = entry["module"]
        b = module.lora_B["default"].weight.float()
        q = getattr(module, "reglu_rol_Q_default").float()
        total = total + (b.T @ q).pow(2).sum()
    return total


@METHOD_REGISTRY.register("reglu")
class ReGLUMethod(UnlearningMethod):
    """Representation-Guided Low-rank Unlearning adapter for the KIF framework."""

    name = "reglu"

    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        super().__init__(config_overrides)
        self.cfg = ReGLUConfig()
        for k, v in (config_overrides or {}).items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
            else:
                logger.warning(f"[ReGLU] Unknown config key ignored: {k}")

    def _build_rows(self, prompts_jsonl: str, forget_subjects: List[str]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
        all_subjects = self.load_subjects(prompts_jsonl)
        forget_qa = self.get_forget_qa(prompts_jsonl, forget_subjects, max_per_subject=self.cfg.max_per_subject)
        forget_rows = [
            {"subject": r["subject"], "prompt": r["question"], "response": r["answer"]}
            for r in forget_qa
            if r.get("question") and r.get("answer")
        ][: self.cfg.n_forget]
        if self.cfg.retain_subjects is not None:
            retain_subjects = list(self.cfg.retain_subjects)
        else:
            fset = set(forget_subjects)
            retain_subjects = [s for s in all_subjects if s not in fset]
        if retain_subjects:
            retain_qa = self.get_forget_qa(prompts_jsonl, retain_subjects, max_per_subject=self.cfg.max_per_subject)
            retain_rows_raw = [
                {"subject": r["subject"], "prompt": r["question"], "response": r["answer"]}
                for r in retain_qa
                if r.get("question") and r.get("answer")
            ]
            retain_source = retain_subjects
        else:
            retain_rows_raw = [dict(x, subject="__unverifiable_retain__") for x in UNVERIFIABLE_PROMPTS]
            retain_source = ["__unverifiable_retain__"]
            logger.warning("[ReGLU] No retain subjects found in prompts_jsonl; using unverifiable retain fallback.")
        retain_rows = _repeat_to_len(retain_rows_raw, self.cfg.n_retain)
        if not forget_rows:
            raise ValueError("[ReGLU] No forget rows found. Check prompts_jsonl/subjects.")
        if not retain_rows:
            raise ValueError("[ReGLU] No retain rows found. Add a non-forgotten retain subject such as Adele.")
        logger.info(f"[ReGLU] Forget rows={len(forget_rows)} subjects={forget_subjects}")
        logger.info(f"[ReGLU] Retain rows={len(retain_rows)} subjects={retain_source}")
        return forget_rows, retain_rows, retain_source

    def run(self, prompts_jsonl: str, output_dir: str, model_dir: str, subjects: Optional[List[str]]) -> UnlearningResult:
        cfg = self.cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_subjects = self.load_subjects(prompts_jsonl)
        forget_subjects = subjects if subjects is not None else all_subjects
        if cfg.max_forget_subjects:
            forget_subjects = forget_subjects[: cfg.max_forget_subjects]
        forget_rows, retain_rows, retain_subjects = self._build_rows(prompts_jsonl, forget_subjects)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        dtype = _dtype_from_name(cfg.torch_dtype)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[ReGLU] Loading tokenizer/model: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map=None,
        )
        model.to(device)
        model.config.use_cache = False

        target_names = _lora_target_modules(cfg.model_family, cfg.lora_targets)
        lora_config = LoraConfig(
            r=int(cfg.lora_r),
            lora_alpha=int(cfg.lora_alpha),
            target_modules=target_names,
            lora_dropout=float(cfg.lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        model.train()
        model.print_trainable_parameters()

        target_modules = _resolve_lora_modules(model, target_names)
        if not target_modules:
            raise RuntimeError(f"[ReGLU] No LoRA modules found for targets={target_names}")
        logger.info(f"[ReGLU] LoRA target modules resolved: {len(target_modules)}")

        rol_bases = _apply_rila_initialization(
            model=model,
            tokenizer=tokenizer,
            forget_rows=forget_rows,
            retain_rows=retain_rows,
            cfg=cfg,
            target_modules=target_modules,
            device=device,
            output_dir=output_path,
        )
        rol_layers = _register_rol_bases(model, target_modules, rol_bases, cfg)

        forget_ds = _QADataset(forget_rows, tokenizer, cfg.model_family, cfg.max_length)
        retain_ds = _QADataset(retain_rows, tokenizer, cfg.model_family, cfg.max_length)
        collate = lambda b: _collate(b, tokenizer.pad_token_id)
        forget_loader = DataLoader(forget_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
        retain_loader = DataLoader(retain_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
        retain_iter = _cycle(retain_loader)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
        total_update_steps = max(1, math.ceil((len(forget_loader) * cfg.num_epochs) / max(1, cfg.gradient_accumulation_steps)))
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=total_update_steps)

        global_step = 0
        optimizer.zero_grad(set_to_none=True)
        metrics_path = output_path / "reglu_train_metrics.jsonl"
        logger.info(
            f"[ReGLU] Training variant={cfg.variant} epochs={cfg.num_epochs} "
            f"forget_batches={len(forget_loader)} retain_batches={len(retain_loader)} update_steps≈{total_update_steps}"
        )

        for epoch in range(int(cfg.num_epochs)):
            for step, f_batch in enumerate(forget_loader):
                r_batch = next(retain_iter)
                f_batch = _batch_to_device(f_batch, device)
                r_batch = _batch_to_device(r_batch, device)

                f_out = model(**f_batch)
                if str(cfg.variant).lower() == "ihl":
                    forget_loss = _masked_ihl_loss(f_out.logits, f_batch["labels"])
                elif str(cfg.variant).lower() == "gd":
                    forget_loss = -f_out.loss
                else:
                    raise ValueError(f"Unsupported ReGLU variant: {cfg.variant}")
                r_out = model(**r_batch)
                retain_loss = r_out.loss
                rol_loss = _rol_penalty(rol_layers, retain_loss)
                loss = forget_loss + float(cfg.retain_gamma) * retain_loss + float(cfg.rol_lambda) * rol_loss
                (loss / max(1, cfg.gradient_accumulation_steps)).backward()

                is_update = ((step + 1) % max(1, cfg.gradient_accumulation_steps) == 0) or (step + 1 == len(forget_loader))
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
                            "forget_loss": float(forget_loss.detach().cpu()),
                            "retain_loss": float(retain_loss.detach().cpu()),
                            "rol_loss": float(rol_loss.detach().cpu()),
                            "total_loss": float(loss.detach().cpu()),
                        }
                        logger.info(f"[ReGLU] {row}")
                        with metrics_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row) + "\n")

        adapter_path = output_path / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[ReGLU] Saving PEFT adapter -> {adapter_path}")
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        merged_model_dir = None
        if cfg.save_merged_model:
            merged_path = output_path / "merged_model"
            logger.info(f"[ReGLU] Merging adapter and saving full model -> {merged_path}")
            merged = model.merge_and_unload()
            merged.save_pretrained(str(merged_path), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_path))
            merged_model_dir = str(merged_path)

        meta = {
            "method": "reglu",
            "model_dir": model_dir,
            "adapter_path": str(adapter_path),
            "merged_model_dir": merged_model_dir,
            "forget_subjects": forget_subjects,
            "retain_subjects": retain_subjects,
            "config": asdict(cfg),
            "num_forget_rows": len(forget_rows),
            "num_retain_rows": len(retain_rows),
            "lora_target_modules": target_names,
            "resolved_lora_modules": len(target_modules),
        }
        (output_path / "reglu_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info(f"[ReGLU] Metadata saved -> {output_path / 'reglu_meta.json'}")

        if merged_model_dir:
            return UnlearningResult(merged_model_dir=merged_model_dir, subjects=forget_subjects)
        return UnlearningResult(model_dir=model_dir, adapter_path=str(adapter_path), subjects=forget_subjects)
