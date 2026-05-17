"""Runtime fixes for the framework ReGLU adapter.

The main ReGLU implementation follows the upstream code closely. This patch
keeps the same RILA initialization but avoids storing full residual base weights
inside the RILA cache. For Llama-3.1-8B, caching every W_res matrix would create
a very large artifact while not being needed for the framework run, because the
faithful artifact is the merged model saved after training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import torch

import methods.reglu as reglu

logger = logging.getLogger(__name__)


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

    h_forget = reglu._collect_representations_for_modules(
        model, tokenizer, forget_sample, cfg.model_family, cfg.max_length,
        cfg.batch_size, target_modules, device, "forget/RILA"
    )
    h_retain = reglu._collect_representations_for_modules(
        model, tokenizer, retain_sample, cfg.model_family, cfg.max_length,
        cfg.batch_size, target_modules, device, "retain/RILA"
    )

    rol_bases: Dict[str, torch.Tensor] = {}
    cache_layers: Dict[str, Dict[str, torch.Tensor]] = {}
    rank = int(cfg.lora_r)
    beta = float(cfg.rila_beta)
    eps = float(cfg.rila_cov_shrink)

    for name, module in target_modules.items():
        if name not in h_forget or name not in h_retain:
            logger.warning("[ReGLU] Missing activations for %s; skipping RILA init", name)
            continue

        hf = h_forget[name].double()
        hr = h_retain[name].double()
        if eps > 0:
            hf = hf + eps * torch.randn_like(hf)
            hr = hr + eps * torch.randn_like(hr)

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
    if not config_overrides or "save_merged_model" not in config_overrides:
        self.cfg.save_merged_model = True


# Monkey-patch the implementation used by ReGLUMethod.run().
reglu._apply_rila_initialization = _apply_rila_initialization_no_w_cache
reglu.ReGLUMethod.__init__ = _reglu_init_with_merged_default
