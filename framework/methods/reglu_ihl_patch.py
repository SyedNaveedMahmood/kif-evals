"""Autograd-safe IHL loss patch for ReGLU.

The original framework implementation used an in-place scatter operation on the
probability tensor while the same tensor was still needed for gradients through
p_true. That breaks PyTorch autograd during backward. This patch replaces it
with a non-in-place scatter so the IHL objective remains the same but backward
is valid.
"""

from __future__ import annotations

import torch

import methods.reglu as reglu


def _masked_ihl_loss_no_inplace(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """IHL = mean(max(1 + p_true - max_{v != true} p_v, 0)).

    This is functionally equivalent to the original ReGLU IHL loss but avoids
    in-place mutation of the softmax probability tensor.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = shift_labels != -100
    if not mask.any():
        return logits.new_zeros(())

    active_logits = shift_logits[mask].float()
    gold = shift_labels[mask]

    probs = active_logits.softmax(dim=-1)
    gold_idx = gold.unsqueeze(1)
    p_true = probs.gather(1, gold_idx).squeeze(1)

    # Non-in-place scatter. Do not use scatter_ here: p_true depends on probs
    # and autograd still needs the unmodified softmax output during backward.
    other_probs = probs.scatter(1, gold_idx, -1.0)
    p_other = other_probs.max(dim=1).values

    return torch.clamp(1.0 + p_true - p_other, min=0.0).mean()


reglu._masked_ihl_loss = _masked_ihl_loss_no_inplace
