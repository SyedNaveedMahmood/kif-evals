#!/usr/bin/env python3
"""Fast EL10 token-set audit matching the revised Module-8 resolver.

This script avoids loading full model weights. It reads the subject list from an
existing Module-8 eval_summary.json or from --subjects, loads only the tokenizer,
and writes the exact token IDs used by the revised EL10 name-token resolver.

Matches the Module 8 version in Pasted text(129):
  - EL_STEPS=10 and EL_MAX_KEYWORDS=10;
  - canonical subject-name tokens are resolved first;
  - bracketed qualifiers are stripped for name variants, e.g. Drake (musician)
    -> Drake;
  - leading-space and lowercase BPE variants are considered;
  - degenerate short name tokens are rejected;
  - if no usable single-token name variant exists, full subject subtokens are
    used with the same degenerate-token filter;
  - remaining slots are filled with stopword-filtered single-token keywords.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EL10TOK-FAST] %(message)s")
log = logging.getLogger("el10tok_fast")

EL_MAX_KEYWORDS = 10
EL_STEPS = 10

_DEFAULT_STOPWORDS: Set[str] = {
    "about", "above", "after", "also", "another", "artist", "award",
    "best", "birth", "born", "both", "career", "category", "children",
    "confirm", "correct", "description", "different", "does", "each",
    "from", "have", "here", "into", "just", "know", "like", "more",
    "most", "much", "only", "other", "over", "same", "some", "such",
    "than", "that", "their", "them", "then", "there", "these", "they",
    "this", "through", "time", "under", "very", "well", "were", "what",
    "when", "where", "which", "while", "with", "would", "year", "your",
    "active", "actor", "accepted", "alternative", "american", "arts",
    "album", "achievement", "nominated", "notable", "occupation",
    "received", "singer", "rapper", "record",
    "cast", "center", "child", "composer", "country",
    "date", "dates", "details", "educated", "entity",
    "friend", "full", "general", "genres", "give",
    "heard", "information", "audio", "artists",
    "association", "cash", "brit",
    "label", "known", "later", "music", "name",
    "named", "national", "note", "noted", "often", "part",
    "place", "play", "played", "playing", "popular",
    "previous", "prior", "produced", "release",
    "released", "role", "show", "since", "song", "songs",
    "stage", "start", "style", "title", "tour", "track",
    "used", "work", "works", "world", "wrote",
}


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


def is_degenerate_name_token(decoded: str) -> bool:
    stripped = decoded.strip()
    if len(stripped) <= 1:
        return True
    if stripped.lower() in {
        "ed", "er", "de", "le", "an", "on", "in", "re",
        "al", "el", "la", "da", "di", "il", "et",
    }:
        return True
    return False


def name_variants(subject: str) -> List[str]:
    clean = re.sub(r"\s*\(.*?\)\s*", " ", subject).strip()
    parts = clean.split()
    candidates: List[str] = []
    for text in [clean] + parts:
        candidates.append(text)
        candidates.append(" " + text)
        candidates.append(text.lower())
        candidates.append(" " + text.lower())
    seen, out = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def keyword_token_ids(tok, keywords: List[str], subject: Optional[str] = None, maxk: int = EL_MAX_KEYWORDS, stopwords: Optional[Set[str]] = None) -> List[int]:
    if stopwords is None:
        stopwords = _DEFAULT_STOPWORDS

    seen: Set[int] = set()
    ids: List[int] = []
    resolved_name_ids: List[int] = []

    # Pass 1: canonical name subtokens first.
    if subject:
        for variant in name_variants(subject):
            if len(ids) >= maxk:
                break
            try:
                enc = tok.encode(variant, add_special_tokens=False)
                if len(enc) == 1:
                    tid = int(enc[0])
                    decoded = tok.decode([tid])
                    if tid not in seen and not is_degenerate_name_token(decoded):
                        seen.add(tid)
                        ids.append(tid)
                        resolved_name_ids.append(tid)
            except Exception:
                pass

        if not resolved_name_ids:
            log.warning(
                "Subject %r: no single-token name variant found; falling back to full subtoken sequence.",
                subject,
            )
            try:
                for tid in tok.encode(subject, add_special_tokens=False):
                    if len(ids) >= maxk:
                        break
                    tid = int(tid)
                    decoded = tok.decode([tid])
                    if tid not in seen and not is_degenerate_name_token(decoded):
                        seen.add(tid)
                        ids.append(tid)
            except Exception:
                pass

    # Pass 2: subject-specific single-token keywords, stopword-filtered.
    for w in keywords or []:
        if len(ids) >= maxk:
            break
        if w.lower() in stopwords:
            continue
        try:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                tid = int(enc[0])
                if tid not in seen:
                    seen.add(tid)
                    ids.append(tid)
        except Exception:
            pass

    if not ids:
        log.warning("Subject %r: zero token IDs resolved; EL10 would return 0.0.", subject)
    return ids[:maxk]


def build_token_set(tok, subject: str, keywords: List[str], maxk: int, el_steps: int) -> Dict[str, Any]:
    token_ids = keyword_token_ids(tok, keywords, subject=subject, maxk=maxk)

    name_variant_ids: Set[int] = set()
    for variant in name_variants(subject):
        try:
            enc = tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                tid = int(enc[0])
                decoded = tok.decode([tid])
                if not is_degenerate_name_token(decoded):
                    name_variant_ids.add(tid)
        except Exception:
            pass

    fallback_subtoken_ids: Set[int] = set()
    if not name_variant_ids:
        try:
            for tid in tok.encode(subject, add_special_tokens=False):
                tid = int(tid)
                decoded = tok.decode([tid])
                if not is_degenerate_name_token(decoded):
                    fallback_subtoken_ids.add(tid)
        except Exception:
            fallback_subtoken_ids = set()

    keyword_original_by_id: Dict[int, str] = {}
    for w in keywords or []:
        if w.lower() in _DEFAULT_STOPWORDS:
            continue
        try:
            enc = tok.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                keyword_original_by_id.setdefault(int(enc[0]), w)
        except Exception:
            pass

    provenance: List[Dict[str, Any]] = []
    for tid in token_ids:
        if tid in name_variant_ids:
            source = "name_subtoken"
            original = subject
        elif tid in fallback_subtoken_ids:
            source = "name_subtoken_fallback"
            original = subject
        else:
            source = "keyword"
            original = keyword_original_by_id.get(tid)
        provenance.append({
            "token_id": tid,
            "token_str": tok.decode([tid]),
            "source": source,
            "original_kw": original,
        })

    n_name = sum(1 for r in provenance if str(r["source"]).startswith("name_subtoken"))
    return {
        "subject": subject,
        "el_steps": el_steps,
        "max_keywords": maxk,
        "n_tokens_used": len(token_ids),
        "n_name_subtokens": n_name,
        "n_keyword_tokens": len(token_ids) - n_name,
        "token_ids": token_ids,
        "token_strings": [tok.decode([tid]) for tid in token_ids],
        "provenance": provenance,
        "zero_token_warning": len(token_ids) == 0,
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

    fields = [
        "subject", "el_steps", "max_keywords", "n_tokens_used", "n_name_subtokens",
        "n_keyword_tokens", "zero_token_warning", "token_ids", "token_strings", "sources", "original_keywords",
    ]
    with (out_dir / "el10_token_sets.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s, ts in token_sets.items():
            w.writerow({
                "subject": s,
                "el_steps": ts["el_steps"],
                "max_keywords": ts["max_keywords"],
                "n_tokens_used": ts["n_tokens_used"],
                "n_name_subtokens": ts["n_name_subtokens"],
                "n_keyword_tokens": ts["n_keyword_tokens"],
                "zero_token_warning": ts["zero_token_warning"],
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
        r"\caption{\textbf{EL10 token-set audit.} Exact tokenizer IDs used for the name-token extraction-likelihood calculation.}",
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
        log.info(
            "%s: ids=%s tokens=%s name=%s keyword=%s",
            s, ts["token_ids"], ts["token_strings"], ts["n_name_subtokens"], ts["n_keyword_tokens"],
        )

    output = {
        "model_dir": args.model_dir,
        "prompts_jsonl": args.prompts_jsonl,
        "eval_summary_json": args.eval_summary_json,
        "subjects": subjects,
        "tokenizer_class": type(tok).__name__,
        "vocab_size": getattr(tok, "vocab_size", None),
        "el_steps": args.el_steps,
        "max_keywords": args.max_keywords,
        "token_selection_version": "module8_name_first_degenerate_filtered_stopword_filtered",
        "token_sets": token_sets,
        "reproducibility_note": "Fast audit; subject list is taken from eval_summary.json or --subjects. Token selection mirrors Module 8 in Pasted text(129): _keyword_token_ids() with name-first, degenerate-name filtering, and extended stopword filtering.",
    }
    write_outputs(Path(args.out_dir), output)
    log.info("DONE. Wrote outputs to %s", args.out_dir)

    print("\n=== EL10 Token Sets Summary ===")
    for s, ts in token_sets.items():
        print(
            f"  {s:<30} ids={ts['token_ids']} strs={ts['token_strings']} "
            f"name={ts['n_name_subtokens']} keyword={ts['n_keyword_tokens']}"
        )


if __name__ == "__main__":
    main()
