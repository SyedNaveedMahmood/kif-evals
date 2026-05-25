#!/usr/bin/env python3
"""Dump exact EL10 keyword token sets used per subject for reproducibility.

This script is intentionally evaluation-only. It mirrors Module 8's
_keyword_token_ids() contract and writes the operative token IDs used for EL10
style extraction-likelihood computation.

Important naming note:
  In the current Module 8 implementation, "EL10" refers to at most 10 keyword
  token IDs per subject (EL_MAX_KEYWORDS=10). The autoregressive averaging window
  is EL_STEPS=32. This script therefore defaults to el_steps=32 to match Module 8.

Outputs:
  <out_dir>/el10_token_sets.json
  <out_dir>/el10_token_sets.csv
  <out_dir>/el10_token_sets_latex.tex
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EL10-AUDIT] %(message)s")
logger = logging.getLogger("el10_audit")

# Must match app/src/llama20/modules/module8.py unless overridden explicitly.
EL_MAX_KEYWORDS = 10
EL_STEPS = 32
MAX_SUBJECTS_DEFAULT = 5


def load_tok(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def mine_subject_keywords(prompts_jsonl: str) -> Dict[str, List[str]]:
    """Mirror Module 8's prompt keyword mining."""
    if not prompts_jsonl or not Path(prompts_jsonl).exists():
        return {}
    tmp: Dict[str, set] = {}
    for line in Path(prompts_jsonl).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            subj = rec.get("subject") or rec.get("author")
            pr = rec.get("prompt") or ""
            if not subj or not pr:
                continue
            base = tmp.setdefault(str(subj), set())
            for raw_tok in pr.split():
                t = "".join([c for c in raw_tok if c.isalpha()]).lower()
                if len(t) > 3:
                    base.add(t)
        except Exception:
            pass
    return {k: sorted(list(v))[:32] for k, v in tmp.items()}


def load_capsule_subjects(capsules_dir: str) -> List[str]:
    subs: List[str] = []
    root = Path(capsules_dir)
    if not root.exists():
        logger.warning("capsules_dir does not exist: %s", root)
        return subs
    for p in sorted(root.glob("*_capsule.pkl.gz")):
        try:
            with gzip.open(p, "rb") as f:
                obj = pickle.load(f)
            subs.append(str(obj.get("subject", p.stem.replace("_capsule", ""))))
        except Exception as exc:
            logger.warning("Failed to read capsule %s: %r", p, exc)
    return sorted(list(dict.fromkeys(subs)))


def _module8_keyword_token_ids(tok, keywords: List[str], subject: Optional[str] = None, maxk: int = EL_MAX_KEYWORDS) -> List[int]:
    """Exact operative ID logic from current Module 8."""
    ids: List[int] = []
    # 1) prefer single-token keywords from prompts.jsonl
    for w in keywords or []:
        try:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(int(enc[0]))
        except Exception:
            pass
        if len(ids) >= maxk:
            break
    # 2) backfill using subject subtokens if fewer than 3 IDs before dedupe
    if len(ids) < 3 and subject:
        try:
            sub_ids = tok.encode(subject, add_special_tokens=False)
            for i in sub_ids:
                if i not in ids:
                    ids.append(int(i))
                    if len(ids) >= maxk:
                        break
        except Exception:
            pass
    # 3) dedupe preserve order
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(int(i))
    return out[:maxk]


def build_el10_token_set(tok, subject: str, keywords: List[str], maxk: int = EL_MAX_KEYWORDS, el_steps: int = EL_STEPS) -> Dict[str, Any]:
    operative_ids = _module8_keyword_token_ids(tok, keywords, subject=subject, maxk=maxk)

    # Reconstruct provenance in the same priority order for audit readability.
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    raw_single_keyword_ids: List[int] = []
    for w in keywords or []:
        if len(raw_single_keyword_ids) >= maxk:
            break
        try:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                raw_single_keyword_ids.append(int(enc[0]))
                tid = int(enc[0])
                if tid in operative_ids and tid not in seen_ids:
                    seen_ids.add(tid)
                    records.append({
                        "token_id": tid,
                        "token_str": tok.decode([tid]),
                        "source": "keyword",
                        "original_kw": w,
                    })
        except Exception:
            pass

    if len(raw_single_keyword_ids) < 3:
        try:
            for tid in tok.encode(subject, add_special_tokens=False):
                tid = int(tid)
                if tid in operative_ids and tid not in seen_ids:
                    seen_ids.add(tid)
                    records.append({
                        "token_id": tid,
                        "token_str": tok.decode([tid]),
                        "source": "subword_backfill",
                        "original_kw": subject,
                    })
                if len(records) >= len(operative_ids):
                    break
        except Exception:
            pass

    # Make sure provenance order follows operative_ids exactly, even if tokenizer
    # edge cases made the readable reconstruction incomplete.
    by_id = {r["token_id"]: r for r in records}
    ordered_records: List[Dict[str, Any]] = []
    for tid in operative_ids:
        if tid in by_id:
            ordered_records.append(by_id[tid])
        else:
            ordered_records.append({
                "token_id": int(tid),
                "token_str": tok.decode([int(tid)]),
                "source": "unknown_reconstructed",
                "original_kw": None,
            })

    n_keyword = sum(1 for r in ordered_records if r["source"] == "keyword")
    return {
        "subject": subject,
        "el_steps": el_steps,
        "max_keywords": maxk,
        "n_tokens_used": len(operative_ids),
        "token_ids": operative_ids,
        "token_strings": [tok.decode([int(tid)]) for tid in operative_ids],
        "provenance": ordered_records,
        "n_keyword_tokens": n_keyword,
        "backfill_triggered": any(r["source"] == "subword_backfill" for r in ordered_records),
        "mined_keywords_considered": list(keywords or [])[:32],
    }


def write_csv(out_path: Path, token_sets: Dict[str, Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for subject, ts in token_sets.items():
        rows.append({
            "subject": subject,
            "el_steps": ts["el_steps"],
            "max_keywords": ts["max_keywords"],
            "n_tokens_used": ts["n_tokens_used"],
            "backfill_triggered": ts["backfill_triggered"],
            "token_ids": " ".join(str(x) for x in ts["token_ids"]),
            "token_strings": " | ".join(str(x) for x in ts["token_strings"]),
            "sources": " | ".join(str(r["source"]) for r in ts["provenance"]),
            "original_keywords": " | ".join(str(r.get("original_kw")) for r in ts["provenance"]),
        })
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["subject", "el_steps", "max_keywords", "n_tokens_used", "backfill_triggered", "token_ids", "token_strings", "sources", "original_keywords"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def latex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("$", r"\$")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}"))


def write_latex(out_path: Path, token_sets: Dict[str, Dict[str, Any]]) -> None:
    lines = []
    lines.append(r"\begin{table*}[htpb]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{lcll}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Subject} & \textbf{$n$} & \textbf{Token IDs} & \textbf{Decoded tokens} \\")
    lines.append(r"\midrule")
    for subject, ts in token_sets.items():
        token_ids = ",".join(str(x) for x in ts["token_ids"])
        token_strs = ", ".join(latex_escape(str(x)) for x in ts["token_strings"])
        lines.append(f"{latex_escape(subject)} & {ts['n_tokens_used']} & \texttt{{{token_ids}}} & {token_strs} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\caption{\textbf{EL10 token-set audit.} Exact tokenizer IDs used for the extraction-likelihood token-mass calculation.}")
    lines.append(r"\label{tab:el10_token_sets}")
    lines.append(r"\end{table*}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_el10_token_sets(
    model_dir: str,
    capsules_dir: str,
    prompts_jsonl: str,
    out_dir: str,
    max_subjects: Optional[int] = MAX_SUBJECTS_DEFAULT,
    el_steps: int = EL_STEPS,
    max_keywords: int = EL_MAX_KEYWORDS,
) -> Dict[str, Any]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    subjects = load_capsule_subjects(capsules_dir)
    if max_subjects is not None and max_subjects > 0:
        subjects = subjects[:max_subjects]
    subj_keywords = mine_subject_keywords(prompts_jsonl)

    tok = load_tok(model_dir)
    logger.info("Tokenizer class: %s", type(tok).__name__)
    logger.info("Tokenizer vocab size: %s", getattr(tok, "vocab_size", None))
    logger.info("Subjects to audit: %s", subjects)

    token_sets: Dict[str, Dict[str, Any]] = {}
    for s in subjects:
        ts = build_el10_token_set(tok, s, subj_keywords.get(s, []), maxk=max_keywords, el_steps=el_steps)
        token_sets[s] = ts
        if ts["n_tokens_used"] == 0:
            logger.warning("SUBJECT %r: ZERO token IDs resolved; EL10 would be 0.0.", s)
        elif ts["backfill_triggered"]:
            logger.warning("SUBJECT %r: subword backfill activated; ids=%s strings=%s", s, ts["token_ids"], ts["token_strings"])
        else:
            logger.info("SUBJECT %r: ids=%s strings=%s", s, ts["token_ids"], ts["token_strings"])

    output = {
        "model_dir": model_dir,
        "capsules_dir": capsules_dir,
        "prompts_jsonl": prompts_jsonl,
        "tokenizer_class": type(tok).__name__,
        "vocab_size": getattr(tok, "vocab_size", None),
        "el_steps": el_steps,
        "max_keywords": max_keywords,
        "max_subjects": max_subjects,
        "subjects": subjects,
        "token_sets": token_sets,
        "reproducibility_note": (
            "Matches current Module 8 keyword-token selection: up to EL_MAX_KEYWORDS single-token prompt keywords, "
            "with subject-subtoken backfill when fewer than three IDs are found before deduplication."
        ),
    }

    out_dir_p = Path(out_dir)
    json_path = out_dir_p / "el10_token_sets.json"
    csv_path = out_dir_p / "el10_token_sets.csv"
    tex_path = out_dir_p / "el10_token_sets_latex.tex"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, token_sets)
    write_latex(tex_path, token_sets)
    logger.info("Written %s", json_path)
    logger.info("Written %s", csv_path)
    logger.info("Written %s", tex_path)
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Audit and freeze EL10 token sets for reproducibility.")
    p.add_argument("--model_dir", default=os.getenv("MODEL_DIR", "outputs/model"))
    p.add_argument("--capsules_dir", default=os.getenv("CAPSULES_DIR", "outputs/capsules"))
    p.add_argument("--prompts_jsonl", default=os.getenv("PROMPTS_JSONL", "outputs/datasets/prompts.jsonl"))
    p.add_argument("--out_dir", default=os.getenv("OUT_DIR", "outputs/eval_clean"))
    p.add_argument("--max_subjects", type=int, default=int(os.getenv("MAX_SUBJECTS", str(MAX_SUBJECTS_DEFAULT))))
    p.add_argument("--el_steps", type=int, default=int(os.getenv("EL_STEPS", str(EL_STEPS))))
    p.add_argument("--max_keywords", type=int, default=int(os.getenv("EL_MAX_KEYWORDS", str(EL_MAX_KEYWORDS))))
    args = p.parse_args()

    result = audit_el10_token_sets(
        model_dir=args.model_dir,
        capsules_dir=args.capsules_dir,
        prompts_jsonl=args.prompts_jsonl,
        out_dir=args.out_dir,
        max_subjects=args.max_subjects,
        el_steps=args.el_steps,
        max_keywords=args.max_keywords,
    )

    print("\n=== EL10 Token Sets Summary ===")
    for s, ts in result["token_sets"].items():
        flag = " [BACKFILL]" if ts["backfill_triggered"] else ""
        print(f"  {s:<30} ids={ts['token_ids']} strs={ts['token_strings']}{flag}")


if __name__ == "__main__":
    main()
