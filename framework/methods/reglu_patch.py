"""Runtime fixes for the framework ReGLU adapter.

The main ReGLU implementation follows the upstream code closely. This patch
keeps the same RILA initialization but avoids storing full residual base weights
inside the RILA cache. For Llama-3.1-8B, caching every W_res matrix would create
a very large artifact while not being needed for the framework run, because the
faithful artifact is the merged model saved after training.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

import methods.reglu as reglu

logger = logging.getLogger(__name__)

# Explicit model-family aliases for base Llama-3.1-8B. Both use no chat template.
reglu._CHAT_TEMPLATES["llama3.1-8b"] = "{instruction}"
reglu._CHAT_TEMPLATES["llama3-8b"] = "{instruction}"

_SUBJECT_KEYS = ("subject", "author", "entity", "name", "target_subject", "person")
_PROMPT_KEYS = ("prompt", "question", "instruction", "query", "input", "text")
_RESPONSE_KEYS = ("response", "answer", "completion", "output", "target", "label")


def _norm_subject(x: object) -> str:
    s = str(x or "").lower().strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _first_present(rec: Dict, keys: Tuple[str, ...]) -> str:
    for key in keys:
        val = rec.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _load_kif_rows_flexible(prompts_jsonl: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    path = Path(prompts_jsonl)
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception as exc:
            logger.warning("[ReGLU] Skipping invalid JSONL line %d in %s: %s", ln, path, exc)
            continue
        subject = _first_present(rec, _SUBJECT_KEYS)
        prompt = _first_present(rec, _PROMPT_KEYS)
        response = _first_present(rec, _RESPONSE_KEYS)
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


def _build_rows_flexible(self, prompts_jsonl: str, forget_subjects: List[str]):
    rows = _load_kif_rows_flexible(prompts_jsonl)
    if not rows:
        raise ValueError(
            "[ReGLU] No usable rows found in prompts_jsonl. Expected one subject field, "
            "one prompt/question field, and one response/answer field."
        )

    discovered_subjects: List[str] = []
    seen = set()
    for row in rows:
        if row["subject"] not in seen:
            seen.add(row["subject"])
            discovered_subjects.append(row["subject"])

    forget_rows = _select_subject_rows(rows, forget_subjects, int(self.cfg.max_per_subject))[: int(self.cfg.n_forget)]
    if not forget_rows:
        raise ValueError(
            "[ReGLU] No forget rows found after flexible schema parsing. "
            f"Requested forget subjects={forget_subjects}. "
            f"Available subjects sample={discovered_subjects[:30]}"
        )

    forget_keys = {_norm_subject(s) for s in forget_subjects}
    if self.cfg.retain_subjects is not None:
        retain_source = list(self.cfg.retain_subjects)
        retain_pool = _select_subject_rows(rows, retain_source, int(self.cfg.max_per_subject))
    else:
        retain_source = [s for s in discovered_subjects if _norm_subject(s) not in forget_keys]
        retain_pool = [row for row in rows if _norm_subject(row["subject"]) not in forget_keys]

    if not retain_pool:
        retain_source = ["__unverifiable_retain__"]
        retain_pool = [dict(x, subject="__unverifiable_retain__") for x in reglu.UNVERIFIABLE_PROMPTS]
        logger.warning("[ReGLU] No retain subjects found in prompts_jsonl; using unverifiable retain fallback.")

    retain_rows = reglu._repeat_to_len(retain_pool, int(self.cfg.n_retain))
    logger.info("[ReGLU] Flexible loader read %d usable rows from %s", len(rows), prompts_jsonl)
    logger.info("[ReGLU] Available subjects sample: %s", discovered_subjects[:20])
    logger.info("[ReGLU] Forget rows=%d requested_subjects=%s", len(forget_rows), forget_subjects)
    logger.info("[ReGLU] Retain rows=%d retain_subjects=%s", len(retain_rows), retain_source[:20])
    return forget_rows, retain_rows, retain_source


def _collect_representations_sum_pool(
    model,
    tokenizer,
    rows: List[Dict[str, str]],
    model_family: str,
    max_length: int,
    batch_size: int,
    target_modules: Dict[str, torch.nn.Module],
    device: torch.device,
    desc: str,
) -> Dict[str, torch.Tensor]:
    """Collect target-module output representations using upstream ReGLU pooling.

    Upstream ReGLU sums token-level layer outputs along the sequence dimension
    before covariance estimation. This patch preserves that behavior instead of
    mean pooling.
    """
    buffers: Dict[str, List[torch.Tensor]] = {name: [] for name in target_modules}

    def make_hook(key: str):
        def _hook(_mod, _inp, out):
            out_t = out[0] if isinstance(out, (tuple, list)) else out
            if out_t is None:
                return
            if out_t.dim() == 3:
                pooled = out_t.detach().float().sum(dim=1).cpu()
            elif out_t.dim() == 2:
                pooled = out_t.detach().float().cpu()
            else:
                return
            buffers[key].append(pooled)
        return _hook

    handles = [module.base_layer.register_forward_hook(make_hook(name)) for name, module in target_modules.items()]
    was_training = model.training
    model.eval()
    dataset = reglu._QADataset(rows, tokenizer, model_family, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: reglu._collate(b, tokenizer.pad_token_id),
    )
    logger.info(
        "[ReGLU] Collecting representations for %s: %d examples, %d LoRA modules, pooling=sum",
        desc, len(rows), len(target_modules),
    )
    try:
        with torch.no_grad():
            for batch in loader:
                batch = reglu._batch_to_device(batch, device)
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    return {name: torch.cat(chunks, dim=0) for name, chunks in buffers.items() if chunks}


def _apply_rila_initialization_no_w_cache(
    model,
    tokenizer,
    forget_rows: List[Dict[str, str]],
    retain_rows: List[Dict[str, str]],
    cfg: reglu.ReGLUConfig,
    target_modules: Dict[str, torch.nn.Module],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, torch.Tensor]:
    n = int(cfg.rila_samples_per_split)
    forget_sample = reglu._repeat_to_len(forget_rows, n)
    retain_sample = reglu._repeat_to_len(retain_rows, n)

    h_forget = _collect_representations_sum_pool(
        model, tokenizer, forget_sample, cfg.model_family, cfg.max_length,
        cfg.batch_size, target_modules, device, "forget/RILA"
    )
    h_retain = _collect_representations_sum_pool(
        model, tokenizer, retain_sample, cfg.model_family, cfg.max_length,
        cfg.batch_size, target_modules, device, "retain/RILA"
    )

    rol_bases: Dict[str, torch.Tensor] = {}
    cache_layers: Dict[str, Dict[str, torch.Tensor]] = {}
    rank = int(cfg.lora_r)
    beta = float(cfg.rila_beta)

    for name, module in target_modules.items():
        if name not in h_forget or name not in h_retain:
            logger.warning("[ReGLU] Missing activations for %s; skipping RILA init", name)
            continue

        hf = h_forget[name].double()
        hr = h_retain[name].double()

        # Upstream adds eps * I to both covariance matrices before
        # Cov_delta=(1-beta)CovF-beta CovR. The identity term only shifts
        # eigenvalues, not eigenvectors, so the low-rank eigensolver below
        # intentionally omits explicit shrinkage while preserving eigenspaces.
        q_delta, top_evals = reglu._top_eigenvectors_signed_low_rank(hf, hr, rank, beta)
        q_retain = reglu._retain_basis(hr, int(cfg.rol_rank)).detach().cpu().float()

        w0 = module.base_layer.weight.detach().double().cpu()
        if q_delta.shape[0] != w0.shape[0]:
            logger.warning(
                "[ReGLU] Shape mismatch for %s: Q=%s W=%s; skipping",
                name, tuple(q_delta.shape), tuple(w0.shape),
            )
            continue

        a_init = q_delta.T @ w0
        b_init = q_delta
        scaling = float(getattr(module, "scaling", {}).get(
            "default", float(cfg.lora_alpha) / float(cfg.lora_r)
        ))
        w_res = w0 - scaling * (b_init @ a_init)
        dtype = module.base_layer.weight.dtype
        mod_device = module.base_layer.weight.device

        with torch.no_grad():
            module.base_layer.weight.copy_(w_res.to(dtype=dtype, device=mod_device))
            module.lora_A["default"].weight.copy_(a_init.to(dtype=dtype, device=mod_device))
            module.lora_B["default"].weight.copy_(b_init.to(dtype=dtype, device=mod_device))

        rol_bases[name] = q_retain
        # Store only compact diagnostics. W_res is intentionally omitted because
        # the final merged model already contains the residual-base update.
        cache_layers[name] = {
            "A": a_init.float().cpu(),
            "B": b_init.float().cpu(),
            "Qr_retain": q_retain,
            "top_eigenvalues": top_evals,
        }
        logger.info("[ReGLU] RILA initialized %s: B=%s A=%s", name, tuple(b_init.shape), tuple(a_init.shape))

    cache_path = Path(output_dir) / "reglu_rila_cache.pt"
    torch.save({"config": reglu.asdict(cfg), "layers": cache_layers}, cache_path)
    logger.info("[ReGLU] Compact RILA cache saved -> %s", cache_path)
    return rol_bases


_original_reglu_init = reglu.ReGLUMethod.__init__


def _reglu_init_with_merged_default(self, config_overrides=None):
    _original_reglu_init(self, config_overrides=config_overrides)
    if not config_overrides or "model_family" not in config_overrides:
        self.cfg.model_family = "llama3.1-8b"
    if not config_overrides or "save_merged_model" not in config_overrides:
        self.cfg.save_merged_model = True


# Monkey-patch the implementation used by ReGLUMethod.run().
reglu.ReGLUMethod._build_rows = _build_rows_flexible
reglu._collect_representations_for_modules = _collect_representations_sum_pool
reglu._apply_rila_initialization = _apply_rila_initialization_no_w_cache
reglu.ReGLUMethod.__init__ = _reglu_init_with_merged_default
