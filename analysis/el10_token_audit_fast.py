#!/usr/bin/env python3
"""Fast EL10 token-set audit without opening capsule pickles.

Use this when running on a cluster where login-node Python is disallowed and
capsule/model directories live on slow shared storage. The script obtains the
subject list from an existing Module-8 eval_summary.json or from --subjects,
then loads only the tokenizer and writes the exact EL10 token IDs.

It mirrors current Module 8 token selection:
  1. mine prompt keywords from prompts.jsonl;
  2. keep up to 10 single-token keyword IDs;
  3. if fewer than 3 IDs are found, backfill with subject-name subtokens;
  4. deduplicate preserving order.

Current Module 8 uses EL_MAX_KEYWORDS=10 and EL_STEPS=32.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EL10TOK-FAST] %(message)s")
log = logging.getLogger("el10tok_fast")

EL_MAX_KEYWORDS = 10
EL_STEPS = 32


def mine_subject_keywords(prompts_jsonl: str) -> Dict[str, List[str]]:
    log.info("Mining keywords from %s", prompts_jsonl)
    p = Path(prompts_jsonl)
    if not p.exists():
        log.warning("prompts_jsonl not found: %s", p)
        return {}
    tmp: Dict[str, set] = {}
    n = 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
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
            n += 1
        except Exception:
            continue
    log.info("Mined keywords for %d subjects from %d prompt rows", len(tmp), n)
    return {k: sorted(list(v))[:32] for k, v in tmp.items()}


def load_subjects(eval_summary_json: Optional[str], subjects_arg: Optional[str], max_subjects: int) -> List[str]:
    if subjects_arg:
        subs = [s.strip() for s in subjects_arg.split(",") if s.strip()]
        log.info("Using %d subjects from --subjects", len(subs))
    else:
        p = Path(eval_summary_json or "")
        if not p.exists():
            raise FileNotFoundError(f"Need --subjects or existing --eval_summary_json; missing: {p}")
        log.info("Loading subjects from %s", p)
        obj = json.loads(p.read_text(encoding="utf-8"))
        subs = obj.get("subjects_eval") or obj.get("subjects") or []
        if not isinstance(subs, list) or not subs:
            raise ValueError(f"No subjects_eval/subjects list found in {p}")
        subs = [str(s) for s in subs]
    if max_subjects and max_subjects > 0:
        subs = subs[:max_subjects]
    log.info("Subjects: %s", subs)
    return subs


def load_tok(model_dir: str, local_files_only: bool):
    log.info("Loading tokenizer: %s local_files_only=%s", model_dir, local_files_only)
    tok = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    log.info("Loaded tokenizer class=%s vocab_size=%s", type(tok).__name__, getattr(tok, "vocab_size", None))
    return tok


def module8_keyword_token_ids(tok, keywords: List[str], subject: str, maxk: int) -> List[int]:
    ids: List[int] = []
    for w in keywords or []:
        try:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(int(enc[0]))
        except Exception:
            pass
        if len(ids) >= maxk:
            break
    if len(ids) < 3 and subject:
        try:
            for i in tok.encode(subject, add_special_tokens=False):
                if int(i) not in ids:
                    ids.append(int(i))
                    if len(ids) >= maxk:
                        break
        except Exception:
            pass
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(int(i))
    return out[:maxk]


def build_token_set(tok, subject: str, keywords: List[str], maxk: int, el_steps: int) -> Dict[str, Any]:
    token_ids = module8_keyword_token_ids(tok, keywords, subject, maxk)
    records: List[Dict[str, Any]] = []
    seen = set()
    raw_single_count = 0
    for w in keywords or []:
        if raw_single_count >= maxk:
            break
        try:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                raw_single_count += 1
                tid = int(enc[0])
                if tid in token_ids and tid not in seen:
                    seen.add(tid)
                    records.append({"token_id": tid, "token_str": tok.decode([tid]), "source": "keyword", "original_kw": w})
        except Exception:
            pass
    if raw_single_count < 3:
        try:
            for tid in tok.encode(subject, add_special_tokens=False):
                tid = int(tid)
                if tid in token_ids and tid not in seen:
                    seen.add(tid)
                    records.append({"token_id": tid, "token_str": tok.decode([tid]), "source": "subword_backfill", "original_kw": subject})
        except Exception:
            pass
    by_id = {r["token_id"]: r for r in records}
    provenance = [by_id.get(tid, {"token_id": tid, "token_str": tok.decode([tid]), "source": "unknown_reconstructed", "original_kw": None}) for tid in token_ids]
    return {
        "subject": subject,
        "el_steps": el_steps,
        "max_keywords": maxk,
        "n_tokens_used": len(token_ids),
        "token_ids": token_ids,
        "token_strings": [tok.decode([tid]) for tid in token_ids],
        "provenance": provenance,
        "backfill_triggered": any(r["source"] == "subword_backfill" for r in provenance),
        "mined_keywords_considered": list(keywords or [])[:32],
    }


def latex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
             .replace("#", r"\#").replace("_", r"\_").replace("{", r"\{").replace("}", r"\}"))


def write_outputs(out_dir: Path, output: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    token_sets = output["token_sets"]
    (out_dir / "el10_token_sets.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = ["subject", "el_steps", "max_keywords", "n_tokens_used", "backfill_triggered", "token_ids", "token_strings", "sources", "original_keywords"]
    with (out_dir / "el10_token_sets.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s, ts in token_sets.items():
            w.writerow({
                "subject": s,
                "el_steps": ts["el_steps"],
                "max_keywords": ts["max_keywords"],
                "n_tokens_used": ts["n_tokens_used"],
                "backfill_triggered": ts["backfill_triggered"],
                "token_ids": " ".join(str(x) for x in ts["token_ids"]),
                "token_strings": " | ".join(str(x) for x in ts["token_strings"]),
                "sources": " | ".join(str(r["source"]) for r in ts["provenance"]),
                "original_keywords": " | ".join(str(r.get("original_kw")) for r in ts["provenance"]),
            })

    lines = [
        r"\begin{table*}[htpb]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lcll}",
        r"\toprule",
        r"\textbf{Subject} & \textbf{$n$} & \textbf{Token IDs} & \textbf{Decoded tokens} \\",
        r"\midrule",
    ]
    for s, ts in token_sets.items():
        ids = ",".join(str(x) for x in ts["token_ids"])
        strs = ", ".join(latex_escape(str(x)) for x in ts["token_strings"])
        lines.append(f"{latex_escape(s)} & {ts['n_tokens_used']} & \\texttt{{{ids}}} & {strs} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\caption{\textbf{EL10 token-set audit.} Exact tokenizer IDs used for the extraction-likelihood token-mass calculation.}",
        r"\label{tab:el10_token_sets}",
        r"\end{table*}",
    ]
    (out_dir / "el10_token_sets_latex.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.getenv("MODEL_DIR", "meta-llama/Llama-3.1-8B"))
    ap.add_argument("--prompts_jsonl", default=os.getenv("PROMPTS_JSONL", "outputs/datasets/prompts.jsonl"))
    ap.add_argument("--out_dir", default=os.getenv("OUT_DIR", "outputs/eval_clean"))
    ap.add_argument("--eval_summary_json", default=os.getenv("EVAL_SUMMARY_JSON"))
    ap.add_argument("--subjects", default=os.getenv("SUBJECTS"))
    ap.add_argument("--max_subjects", type=int, default=int(os.getenv("MAX_SUBJECTS", "5")))
    ap.add_argument("--el_steps", type=int, default=int(os.getenv("EL_STEPS", str(EL_STEPS))))
    ap.add_argument("--max_keywords", type=int, default=int(os.getenv("EL_MAX_KEYWORDS", str(EL_MAX_KEYWORDS))))
    ap.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=os.getenv("LOCAL_FILES_ONLY", "1") != "0")
    args = ap.parse_args()

    subjects = load_subjects(args.eval_summary_json, args.subjects, args.max_subjects)
    keywords = mine_subject_keywords(args.prompts_jsonl)
    tok = load_tok(args.model_dir, args.local_files_only)

    token_sets: Dict[str, Dict[str, Any]] = {}
    for s in subjects:
        ts = build_token_set(tok, s, keywords.get(s, []), args.max_keywords, args.el_steps)
        token_sets[s] = ts
        log.info("%s: ids=%s tokens=%s backfill=%s", s, ts["token_ids"], ts["token_strings"], ts["backfill_triggered"])

    output = {
        "model_dir": args.model_dir,
        "prompts_jsonl": args.prompts_jsonl,
        "eval_summary_json": args.eval_summary_json,
        "subjects": subjects,
        "tokenizer_class": type(tok).__name__,
        "vocab_size": getattr(tok, "vocab_size", None),
        "el_steps": args.el_steps,
        "max_keywords": args.max_keywords,
        "token_sets": token_sets,
        "reproducibility_note": "Fast audit; subject list is taken from eval_summary.json or --subjects. Token selection mirrors current Module 8.",
    }
    write_outputs(Path(args.out_dir), output)
    log.info("DONE. Wrote outputs to %s", args.out_dir)

    print("\n=== EL10 Token Sets Summary ===")
    for s, ts in token_sets.items():
        print(f"  {s:<30} ids={ts['token_ids']} strs={ts['token_strings']} backfill={ts['backfill_triggered']}")


if __name__ == "__main__":
    main()
