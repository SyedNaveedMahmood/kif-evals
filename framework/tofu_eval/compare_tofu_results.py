#!/usr/bin/env python3
"""Collect TOFU FQ/MU summaries into one CSV/Markdown table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default="framework/outputs/tofu/eval")
    ap.add_argument("--out_csv", default="framework/outputs/tofu/tofu_comparison_forget10.csv")
    ap.add_argument("--out_md", default="framework/outputs/tofu/tofu_comparison_forget10.md")
    args = ap.parse_args()

    root = Path(args.eval_root)
    summaries = sorted(root.glob("*/tofu_summary.json"))
    rows: List[Dict[str, Any]] = []
    for p in summaries:
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "method": p.parent.name,
            "split": obj.get("split"),
            "FQ_linear": obj.get("forget_quality_linear_FQ"),
            "MU": obj.get("model_utility_MU"),
            "forget_TR_mean": obj.get("forget_truth_ratio_mean"),
            "forget_TR_score": obj.get("forget_truth_ratio_closer_to_1_score"),
            "forget_prob": obj.get("forget_probability"),
            "forget_rougeL_recall": obj.get("forget_rougeL_recall"),
            "model_dir": obj.get("model_dir"),
            "summary_path": str(p),
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = ["method", "split", "FQ_linear", "MU", "forget_TR_mean", "forget_TR_score", "forget_prob", "forget_rougeL_recall", "model_dir", "summary_path"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    out_md = Path(args.out_md)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("| method | split | FQ_linear | MU | forget_TR_mean | forget_prob |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            def fmt(x):
                if isinstance(x, (int, float)):
                    return f"{x:.6f}"
                return "" if x is None else str(x)
            f.write(f"| {r['method']} | {r['split']} | {fmt(r['FQ_linear'])} | {fmt(r['MU'])} | {fmt(r['forget_TR_mean'])} | {fmt(r['forget_prob'])} |\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
