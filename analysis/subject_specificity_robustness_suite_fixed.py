#!/usr/bin/env python3
"""
Fixed entrypoint for subject_specificity_robustness_suite.py.

The original suite collected hidden states in dynamically padded batches and then
concatenated arrays across batches. If different batches had different sequence
lengths, NumPy raised:

    ValueError: all input array dimensions except for concatenation axis must match

This wrapper monkey-patches hidden_states_for_prompts with a version that pads
batch-level hidden-state tensors and input_ids to the maximum sequence length
observed across all batches before concatenation. It preserves the original
analysis logic and output paths.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

import subject_specificity_robustness_suite as suite


@torch.inference_mode()
def hidden_states_for_prompts_padded(
    model,
    tok,
    prompts: List[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Collect hidden states with safe padding across variable-length batches.

    Returns:
        hs: [N, L, T_max, D] float32 array
        all_ids: List of token-id rows, each padded to T_max
    """
    batch_h: List[np.ndarray] = []
    batch_ids: List[List[int]] = []
    max_t = 0

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=False,
        )
        ids_cpu = enc["input_ids"].detach().cpu().tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False)
        layers = [h.detach().float().cpu().numpy() for h in out.hidden_states]
        arr = np.stack(layers, axis=1)  # [B, L, T, D]
        batch_h.append(arr)
        batch_ids.extend(ids_cpu)
        max_t = max(max_t, int(arr.shape[2]))

    if not batch_h:
        return np.zeros((0, 0, 0, 0), dtype=np.float32), []

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    padded_h: List[np.ndarray] = []
    padded_ids: List[List[int]] = []

    # Pad hidden states along token dimension.
    for arr in batch_h:
        t = int(arr.shape[2])
        if t < max_t:
            pad_width = ((0, 0), (0, 0), (0, max_t - t), (0, 0))
            arr = np.pad(arr, pad_width=pad_width, mode="constant", constant_values=0.0)
        padded_h.append(arr)

    # Pad input ids to the same token dimension so downstream masks line up.
    for ids in batch_ids:
        if len(ids) < max_t:
            ids = ids + [pad_id] * (max_t - len(ids))
        elif len(ids) > max_t:
            ids = ids[:max_t]
        padded_ids.append(ids)

    hs = np.concatenate(padded_h, axis=0)
    return hs, padded_ids


def main() -> None:
    suite.hidden_states_for_prompts = hidden_states_for_prompts_padded
    suite.main()


if __name__ == "__main__":
    main()
