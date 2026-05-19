#!/usr/bin/env python3
"""
Fixed entrypoint for subject_specificity_robustness_suite.py.

The original suite collected hidden states in dynamically padded batches and then
concatenated arrays across batches. If different batches had different sequence
lengths, NumPy raised shape errors. This wrapper monkey-patches the hidden-state
collector and the pooling functions so that every pooled representation has a
stable [layers, hidden_dim] shape.

It preserves the original analysis logic and output paths.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

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

    for arr in batch_h:
        t = int(arr.shape[2])
        if t < max_t:
            pad_width = ((0, 0), (0, 0), (0, max_t - t), (0, 0))
            arr = np.pad(arr, pad_width=pad_width, mode="constant", constant_values=0.0)
        padded_h.append(arr)

    for ids in batch_ids:
        if len(ids) < max_t:
            ids = ids + [pad_id] * (max_t - len(ids))
        elif len(ids) > max_t:
            ids = ids[:max_t]
        padded_ids.append(ids)

    hs = np.concatenate(padded_h, axis=0)
    return hs, padded_ids


def _valid_mask(ids: Sequence[int], token_dim: int, pad_id: int | None) -> np.ndarray:
    ids_list = list(ids)
    if len(ids_list) < token_dim:
        ids_list = ids_list + [pad_id if pad_id is not None else 0] * (token_dim - len(ids_list))
    elif len(ids_list) > token_dim:
        ids_list = ids_list[:token_dim]

    if pad_id is None:
        mask = np.ones(token_dim, dtype=bool)
    else:
        mask = np.asarray([tid != pad_id for tid in ids_list], dtype=bool)
    if mask.sum() == 0:
        mask[:] = True
    return mask


def mean_pool_sequence_fixed(hs: np.ndarray, input_ids: List[List[int]], tok) -> np.ndarray:
    """Robust sequence mean pooling.

    The original version used boolean advanced indexing. This version pools per
    item explicitly and guarantees each pooled item is [L, D].
    """
    if hs.ndim != 4:
        raise ValueError(f"Expected hs [N,L,T,D], got shape {hs.shape}")
    n, layers, token_dim, hidden_dim = hs.shape
    pad_id = tok.pad_token_id
    pooled: List[np.ndarray] = []
    for i in range(n):
        ids = input_ids[i] if i < len(input_ids) else []
        mask = _valid_mask(ids, token_dim, pad_id)
        item_layers = []
        for layer_idx in range(layers):
            selected = hs[i, layer_idx, mask, :]
            if selected.ndim != 2 or selected.shape[0] == 0:
                selected = hs[i, layer_idx, :, :]
            item_layers.append(selected.mean(axis=0))
        arr = np.stack(item_layers, axis=0).astype(np.float32)  # [L,D]
        assert arr.shape == (layers, hidden_dim), f"bad pooled shape {arr.shape}"
        pooled.append(arr)
    return np.stack(pooled, axis=0) if pooled else np.zeros((0, layers, hidden_dim), dtype=np.float32)


def find_subsequence(seq: Sequence[int], sub: Sequence[int]) -> Tuple[int, int] | None:
    if not sub:
        return None
    seq_l = list(seq)
    sub_l = list(sub)
    n, m = len(seq_l), len(sub_l)
    for i in range(0, n - m + 1):
        if seq_l[i:i + m] == sub_l:
            return i, i + m
    return None


def subject_span_pool_fixed(
    hs: np.ndarray,
    input_ids: List[List[int]],
    tok,
    prompts: List[str],
    subjects: List[str],
) -> Tuple[np.ndarray, List[bool], List[Tuple[int, int]]]:
    if hs.ndim != 4:
        raise ValueError(f"Expected hs [N,L,T,D], got shape {hs.shape}")
    n, layers, token_dim, hidden_dim = hs.shape
    pad_id = tok.pad_token_id
    pooled: List[np.ndarray] = []
    ok: List[bool] = []
    spans: List[Tuple[int, int]] = []

    for i in range(n):
        raw_ids = input_ids[i] if i < len(input_ids) else []
        ids = list(raw_ids)
        if len(ids) < token_dim:
            ids = ids + [pad_id if pad_id is not None else 0] * (token_dim - len(ids))
        elif len(ids) > token_dim:
            ids = ids[:token_dim]

        subj = subjects[i] if i < len(subjects) else ""
        candidates = [subj]
        if "(" in subj:
            candidates.append(subj.split("(")[0].strip())

        span = None
        for cand in candidates:
            sub_ids = tok.encode(cand, add_special_tokens=False)
            span = find_subsequence(ids, sub_ids)
            if span is not None:
                break

        if span is None:
            mask = _valid_mask(ids, token_dim, pad_id)
            nonpad = np.where(mask)[0].tolist()
            if nonpad:
                span = (int(nonpad[-1]), int(nonpad[-1]) + 1)
            else:
                span = (0, 1)
            ok.append(False)
        else:
            start, end = max(0, span[0]), min(token_dim, span[1])
            if end <= start:
                end = min(token_dim, start + 1)
            span = (start, end)
            ok.append(True)

        spans.append(span)
        item_layers = []
        for layer_idx in range(layers):
            selected = hs[i, layer_idx, span[0]:span[1], :]
            if selected.ndim != 2 or selected.shape[0] == 0:
                selected = hs[i, layer_idx, :, :]
            item_layers.append(selected.mean(axis=0))
        arr = np.stack(item_layers, axis=0).astype(np.float32)
        assert arr.shape == (layers, hidden_dim), f"bad span pooled shape {arr.shape}"
        pooled.append(arr)

    return (np.stack(pooled, axis=0) if pooled else np.zeros((0, layers, hidden_dim), dtype=np.float32)), ok, spans


def non_subject_pool_fixed(hs: np.ndarray, input_ids: List[List[int]], tok, spans: List[Tuple[int, int]]) -> np.ndarray:
    if hs.ndim != 4:
        raise ValueError(f"Expected hs [N,L,T,D], got shape {hs.shape}")
    n, layers, token_dim, hidden_dim = hs.shape
    pad_id = tok.pad_token_id
    pooled: List[np.ndarray] = []
    for i in range(n):
        ids = input_ids[i] if i < len(input_ids) else []
        mask = _valid_mask(ids, token_dim, pad_id)
        if i < len(spans):
            start, end = spans[i]
            start, end = max(0, start), min(token_dim, end)
            mask[start:end] = False
        if mask.sum() == 0:
            mask = _valid_mask(ids, token_dim, pad_id)
        item_layers = []
        for layer_idx in range(layers):
            selected = hs[i, layer_idx, mask, :]
            if selected.ndim != 2 or selected.shape[0] == 0:
                selected = hs[i, layer_idx, :, :]
            item_layers.append(selected.mean(axis=0))
        arr = np.stack(item_layers, axis=0).astype(np.float32)
        assert arr.shape == (layers, hidden_dim), f"bad context pooled shape {arr.shape}"
        pooled.append(arr)
    return np.stack(pooled, axis=0) if pooled else np.zeros((0, layers, hidden_dim), dtype=np.float32)


def main() -> None:
    suite.hidden_states_for_prompts = hidden_states_for_prompts_padded
    suite.mean_pool_sequence = mean_pool_sequence_fixed
    suite.subject_span_pool = subject_span_pool_fixed
    suite.non_subject_pool = non_subject_pool_fixed
    suite.main()


if __name__ == "__main__":
    main()
