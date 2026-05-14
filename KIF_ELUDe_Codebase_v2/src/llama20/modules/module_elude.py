"""Module ELUDe: external entity-level benchmark data adapter for KIF.

This module converts the public Hugging Face ELUDe dataset into the file
layout expected by the KIF pipeline:

  outputs/datasets/prompts.jsonl          -> Module B/C activation mining
  outputs/datasets/triples.jsonl          -> metadata compatibility
  outputs/elude/elude_upu_train.jsonl     -> Module E + Module 7 UPU pairs
  outputs/elude/elude_forget_eval.jsonl   -> Module 8E forget evaluation
  outputs/elude/elude_retain_eval.jsonl   -> Module 8E retain evaluation

Design choice:
  ELUDe remains a separate external benchmark. We do not overwrite the KIF
  SMR/EL10 evaluation; this module only prepares ELUDe-compatible data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from datasets import Dataset, load_dataset

logger = logging.getLogger("KIF-ModuleELUDe")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [ELUDe] %(message)s",
)


@dataclass
class ELUDeConfig:
    dataset_name: str = "6rightjade/ELUDe"
    output_dataset_dir: Path = Path("outputs/datasets")
    output_elude_dir: Path = Path("outputs/elude")
    seed: int = 17
    # ELUDe standard protocol: forget_qa has no public train/test split.
    # It is used as the target forget set for both unlearning and forget evaluation,
    # matching the public Opt-Out code. retain_qa uses its official train/validation/test splits.
    train_ratio: float = 0.80  # retained only for backward compatibility; not used for forget_qa in exact mode
    max_targets: int = 1
    max_forget_train_per_target: int = 0  # 0 = no cap; final benchmark should keep this at 0
    max_retain_train_per_target: int = 0  # 0 = no cap; final benchmark should keep this at 0
    target_entities: Optional[List[str]] = None
    include_paraphrased_question: bool = True
    include_control_prompts: bool = True
    retain_eval_split: str = "test"
    exact_elude_protocol: bool = True

    def __post_init__(self) -> None:
        self.output_dataset_dir = Path(self.output_dataset_dir)
        self.output_elude_dir = Path(self.output_elude_dir)
        self.output_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.output_elude_dir.mkdir(parents=True, exist_ok=True)
        if not (0.05 <= self.train_ratio <= 0.95):
            raise ValueError("train_ratio must be between 0.05 and 0.95")


def _as_pylist(ds: Dataset) -> List[Dict[str, Any]]:
    return [dict(x) for x in ds]


def _safe_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]
    if isinstance(x, tuple):
        return [str(v) for v in x if v is not None]
    # Some cached versions may serialize lists as strings.
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


def _stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha1("\u241f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    logger.info("Wrote %d rows -> %s", n, path)
    return n


def _read_env_targets() -> Optional[List[str]]:
    raw = os.getenv("ELUDE_TARGETS", "").strip()
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def _select_targets(forget_rows: List[Dict[str, Any]], cfg: ELUDeConfig) -> List[str]:
    available = sorted({str(r.get("entity", "")).strip() for r in forget_rows if r.get("entity")})
    if cfg.target_entities:
        wanted = [x.strip() for x in cfg.target_entities if x.strip()]
    else:
        wanted = available[: max(1, int(cfg.max_targets))]
    missing = [x for x in wanted if x not in available]
    if missing:
        raise ValueError(
            f"Target entities not found in ELUDe: {missing}. "
            f"Available examples: {available[:20]}"
        )
    return wanted


def _split_rows(rows: List[Dict[str, Any]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    cut = int(round(len(rows) * train_ratio))
    cut = max(1, min(cut, len(rows) - 1)) if len(rows) > 1 else len(rows)
    return rows[:cut], rows[cut:]


def _cap(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if not n or n <= 0 or len(rows) <= n:
        return rows
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def _make_triple(row: Dict[str, Any], subject: str, split: str, kind: str, idx: int) -> Dict[str, Any]:
    q = str(row.get("question", "")).strip()
    a = str(row.get("answer", "")).strip()
    return {
        "id": _stable_id("elude_triple", subject, split, kind, str(idx), q, a),
        "subject": subject,
        "predicate": "elude_qa",
        "object": a,
        "source": "ELUDe",
        "question": q,
        "split": split,
        "kind": kind,
    }


def _make_prompt(
    subject: str,
    prompt: str,
    expected: str,
    prompt_type: str,
    triple_id: str,
    split: str,
    source_idx: int,
    is_control: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    prompt = str(prompt or "").strip()
    expected = str(expected or "").strip()
    if not prompt:
        return None
    rec = {
        "id": _stable_id("elude_prompt", subject, split, prompt_type, str(source_idx), prompt),
        "triple_id": triple_id,
        "subject": subject,
        "prompt": prompt,
        "expected": expected,
        "prompt_type": prompt_type,
        "category": prompt_type,
        "source": "ELUDe",
        "split": split,
        "is_control": bool(is_control),
        "is_misleading": "misleading" in prompt_type,
    }
    if extra:
        rec.update(extra)
    return rec


def _dedupe_rows(rows: List[Dict[str, Any]], key_fields: Sequence[str]) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for r in rows:
        key = tuple(str(r.get(k, "")) for k in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_elude_files(cfg: ELUDeConfig) -> Dict[str, Any]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.info("Loading ELUDe from Hugging Face: %s", cfg.dataset_name)
    forget_all = _as_pylist(load_dataset(cfg.dataset_name, "forget_qa", split="train"))
    retain_train_all = _as_pylist(load_dataset(cfg.dataset_name, "retain_qa", split="train"))
    try:
        retain_val_all = _as_pylist(load_dataset(cfg.dataset_name, "retain_qa", split="validation"))
    except Exception:
        retain_val_all = []
    try:
        retain_test_all = _as_pylist(load_dataset(cfg.dataset_name, "retain_qa", split="test"))
    except Exception:
        retain_test_all = []

    targets = _select_targets(forget_all, cfg)
    logger.info("Selected ELUDe targets: %s", targets)

    mining_prompts: List[Dict[str, Any]] = []
    triples: List[Dict[str, Any]] = []
    upu_train: List[Dict[str, Any]] = []
    forget_eval: List[Dict[str, Any]] = []
    retain_eval: List[Dict[str, Any]] = []
    retain_train_export: List[Dict[str, Any]] = []
    forget_train_export: List[Dict[str, Any]] = []

    stats_by_target: Dict[str, Any] = {}

    for t_i, target in enumerate(targets):
        f_rows = [r for r in forget_all if str(r.get("entity", "")).strip() == target]
        r_train_rows = [r for r in retain_train_all if str(r.get("entity", "")).strip() == target]

        # Exact ELUDe/Opt-Out protocol: forget_qa is a single public forget set.
        # The original code evaluates forget on this same split, while retain uses
        # official train/validation/test splits. We therefore do not create a custom
        # held-out forget split here.
        if cfg.exact_elude_protocol:
            f_train = list(f_rows)
            f_eval = list(f_rows)
        else:
            f_train, f_eval = _split_rows(f_rows, cfg.train_ratio, cfg.seed + t_i)

        if cfg.retain_eval_split == "validation":
            retain_eval_source = retain_val_all
        elif cfg.retain_eval_split == "validation+test":
            retain_eval_source = retain_val_all + retain_test_all
        else:
            retain_eval_source = retain_test_all
        r_eval_rows = [r for r in retain_eval_source if str(r.get("entity", "")).strip() == target]
        if not r_eval_rows:
            # fallback split from train if HF validation/test unavailable
            r_train_rows, r_eval_rows = _split_rows(r_train_rows, cfg.train_ratio, cfg.seed + t_i + 1000)

        f_train = _cap(f_train, cfg.max_forget_train_per_target, cfg.seed + t_i)
        r_train_for_mining = _cap(r_train_rows, cfg.max_retain_train_per_target, cfg.seed + t_i)

        for idx, row in enumerate(f_train):
            triple = _make_triple(row, target, "train", "forget", idx)
            triples.append(triple)
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            pq = str(row.get("paraphrased_question", "") or "").strip()
            pa = str(row.get("paraphrased_answer", "") or "").strip()
            pert = _safe_list(row.get("perturbed_answer"))

            forget_train_export.append({**row, "kif_split": "train"})
            upu_train.append({
                "subject": target,
                "prompt": q,
                "answer": a,
                "paraphrased_question": pq,
                "paraphrased_answer": pa,
                "perturbed_answer": pert,
                "source": "ELUDe",
                "split": "train",
            })

            p = _make_prompt(target, q, a, "direct_leak", triple["id"], "train", idx)
            if p:
                mining_prompts.append(p)
            if cfg.include_paraphrased_question and pq and pq != q:
                p = _make_prompt(target, pq, pa or a, "paraphrased_leak", triple["id"], "train", idx)
                if p:
                    mining_prompts.append(p)
            # A reasoning-style variant improves compatibility with KIF's existing positive-key classifier.
            if q:
                p = _make_prompt(
                    target,
                    f"Answer the following entity-specific question with the key factual detail: {q}",
                    a,
                    "reasoning_leak",
                    triple["id"],
                    "train",
                    idx,
                )
                if p:
                    mining_prompts.append(p)

        if cfg.include_control_prompts:
            for idx, row in enumerate(r_train_for_mining):
                triple = _make_triple(row, target, "train", "retain_control", idx)
                triples.append(triple)
                q = str(row.get("question", "")).strip()
                a = str(row.get("answer", "")).strip()
                retain_train_export.append({**row, "kif_split": "train"})
                p = _make_prompt(target, q, a, "control", triple["id"], "train", idx, is_control=True)
                if p:
                    mining_prompts.append(p)

        for row in f_eval:
            forget_eval.append({**row, "subject": target, "kif_split": "eval"})
        for row in r_eval_rows:
            retain_eval.append({**row, "subject": target, "kif_split": "eval"})

        stats_by_target[target] = {
            "forget_total": len(f_rows),
            "forget_train": len(f_train),
            "forget_eval": len(f_eval),
            "retain_train_for_mining": len(r_train_for_mining),
            "retain_eval": len(r_eval_rows),
        }

    mining_prompts = _dedupe_rows(mining_prompts, ["subject", "prompt", "prompt_type"])
    triples = _dedupe_rows(triples, ["subject", "question", "kind", "split"])
    upu_train = _dedupe_rows(upu_train, ["subject", "prompt"])

    paths = {
        "prompts": cfg.output_dataset_dir / "prompts.jsonl",
        "triples": cfg.output_dataset_dir / "triples.jsonl",
        "forget_train": cfg.output_elude_dir / "elude_forget_train.jsonl",
        "retain_train": cfg.output_elude_dir / "elude_retain_train.jsonl",
        "upu_train": cfg.output_elude_dir / "elude_upu_train.jsonl",
        "forget_eval": cfg.output_elude_dir / "elude_forget_eval.jsonl",
        "retain_eval": cfg.output_elude_dir / "elude_retain_eval.jsonl",
        "metadata": cfg.output_elude_dir / "metadata.json",
        "stats": cfg.output_dataset_dir / "dataset_stats.json",
    }

    _write_jsonl(paths["prompts"], mining_prompts)
    _write_jsonl(paths["triples"], triples)
    _write_jsonl(paths["forget_train"], forget_train_export)
    _write_jsonl(paths["retain_train"], retain_train_export)
    _write_jsonl(paths["upu_train"], upu_train)
    _write_jsonl(paths["forget_eval"], forget_eval)
    _write_jsonl(paths["retain_eval"], retain_eval)

    metadata = {
        "dataset_name": cfg.dataset_name,
        "targets": targets,
        "seed": cfg.seed,
        "train_ratio": cfg.train_ratio,
        "stats_by_target": stats_by_target,
        "files": {k: str(v) for k, v in paths.items() if k not in {"metadata", "stats"}},
        "note": "ELUDe is prepared as an external QA benchmark; Module 8 SMR/EL10 is unchanged. Exact mode follows Opt-Out data usage: full forget_qa for forget evaluation; retain_qa train for retention anchors; retain_qa test for RQ.",
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_stats = {
        "total_triples": len(triples),
        "total_prompts": len(mining_prompts),
        "subjects": len(targets),
        "control_prompts": sum(1 for p in mining_prompts if p.get("is_control")),
        "misleading_prompts": sum(1 for p in mining_prompts if p.get("is_misleading")),
        "prompt_categories": {},
        "elude": metadata,
    }
    for p in mining_prompts:
        dataset_stats["prompt_categories"][p.get("prompt_type", "unknown")] = dataset_stats["prompt_categories"].get(p.get("prompt_type", "unknown"), 0) + 1
    paths["stats"].write_text(json.dumps(dataset_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("ELUDe preparation complete. Targets=%s", targets)
    return metadata


def run_module_elude(
    targets: Optional[List[str]] = None,
    max_targets: Optional[int] = None,
    train_ratio: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    env_targets = _read_env_targets()
    cfg = ELUDeConfig(
        target_entities=targets or env_targets,
        max_targets=int(max_targets or os.getenv("ELUDE_MAX_TARGETS", "1")),
        train_ratio=float(train_ratio or os.getenv("ELUDE_TRAIN_RATIO", "0.80")),
        seed=int(seed or os.getenv("ELUDE_SEED", "17")),
        max_forget_train_per_target=int(os.getenv("ELUDE_MAX_FORGET_TRAIN", "0")),
        max_retain_train_per_target=int(os.getenv("ELUDE_MAX_RETAIN_TRAIN", "0")),
        retain_eval_split=os.getenv("ELUDE_RETAIN_EVAL_SPLIT", "test"),
        exact_elude_protocol=os.getenv("ELUDE_EXACT_PROTOCOL", "1") not in {"0", "false", "False"},
    )
    return build_elude_files(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ELUDe for the KIF pipeline.")
    parser.add_argument("--targets", type=str, default="", help="Comma-separated ELUDe target entities. If omitted, uses first N alphabetically.")
    parser.add_argument("--max-targets", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    targets = [x.strip() for x in args.targets.split(",") if x.strip()] or None
    run_module_elude(targets=targets, max_targets=args.max_targets, train_ratio=args.train_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
