#!/usr/bin/env python3
"""Collect a lightweight compute-cost table from logs and timing JSON files.

This script is intentionally conservative. It never invents missing hardware or
runtime values. It scans user-specified roots for text logs and JSON summaries,
extracts common wall-clock and GPU-count patterns, and writes CSV/JSON/Markdown
for rebuttal E5.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

WALL_PATTERNS = [
    re.compile(r"(?:elapsed|wall[-_ ]?clock|runtime|run time|total time)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(seconds|second|secs|sec|s|minutes|minute|mins|min|m|hours|hour|hrs|hr|h)", re.I),
    re.compile(r"Job Wall-clock time\s*[:=]\s*([0-9:.]+)", re.I),
    re.compile(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", re.I),
]
GPU_PATTERNS = [
    re.compile(r"(?:gpu_count|gpus|num_gpus|gpu count)\s*[:=]\s*([0-9]+)", re.I),
    re.compile(r"#SBATCH\s+--gres=gpu(?::[^: \n]+)?:(\d+)", re.I),
    re.compile(r"#SBATCH\s+--gpus(?:-per-node)?=(\d+)", re.I),
]
GPU_TYPE_PATTERNS = [
    re.compile(r"(?:gpu_type|gpu type|gpu model|GPU)\s*[:=]\s*([^\n,;]+)", re.I),
    re.compile(r"#SBATCH\s+--partition=([^\n]+)", re.I),
]

METHOD_HINTS = ["eruf", "kif", "lunar", "reglu", "optout", "opt-out", "simnpo", "no_capsule", "no-capsule"]


@dataclass
class ComputeRecord:
    source_file: str
    method: str
    wall_seconds: Optional[float]
    gpu_count: Optional[int]
    gpu_type: str
    gpu_hours: Optional[float]
    note: str


def parse_hms(text: str) -> Optional[float]:
    text = text.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if len(vals) == 3:
        h, m, s = vals
        return h * 3600 + m * 60 + s
    if len(vals) == 2:
        m, s = vals
        return m * 60 + s
    return None


def unit_to_seconds(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit.startswith("s"):
        return value
    if unit.startswith("m"):
        return value * 60
    if unit.startswith("h"):
        return value * 3600
    return value


def infer_method(path: Path, text: str) -> str:
    hay = f"{path.as_posix()}\n{text[:2000]}".lower()
    for h in METHOD_HINTS:
        if h in hay:
            return h.replace("opt-out", "optout")
    return "unknown"


def extract_wall_seconds(text: str) -> Optional[float]:
    for pat in WALL_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if len(m.groups()) >= 2:
            return unit_to_seconds(float(m.group(1)), m.group(2))
        return parse_hms(m.group(1))
    return None


def extract_gpu_count(text: str) -> Optional[int]:
    for pat in GPU_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def extract_gpu_type(text: str) -> str:
    for pat in GPU_TYPE_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return ""


def iter_files(roots: List[Path]) -> Iterable[Path]:
    exts = {".log", ".out", ".err", ".txt", ".json", ".md", ".slurm"}
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            yield root
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts and p.stat().st_size <= 10 * 1024 * 1024:
                yield p


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_file(path: Path) -> Optional[ComputeRecord]:
    text = read_text(path)
    wall = extract_wall_seconds(text)
    gpu_count = extract_gpu_count(text)
    gpu_type = extract_gpu_type(text)
    method = infer_method(path, text)
    if path.suffix.lower() == ".json":
        try:
            obj = json.loads(text)
            for key in ["wall_seconds", "runtime_seconds", "elapsed_seconds", "total_seconds"]:
                if wall is None and isinstance(obj, dict) and isinstance(obj.get(key), (int, float)):
                    wall = float(obj[key])
            for key in ["gpu_count", "num_gpus", "gpus"]:
                if gpu_count is None and isinstance(obj, dict) and isinstance(obj.get(key), int):
                    gpu_count = int(obj[key])
            if not gpu_type and isinstance(obj, dict) and isinstance(obj.get("gpu_type"), str):
                gpu_type = obj["gpu_type"]
            if method == "unknown" and isinstance(obj, dict) and isinstance(obj.get("method"), str):
                method = obj["method"]
        except json.JSONDecodeError:
            pass
    if wall is None and gpu_count is None and method == "unknown":
        return None
    gpu_hours = None
    if wall is not None and gpu_count is not None:
        gpu_hours = wall / 3600.0 * gpu_count
    return ComputeRecord(
        source_file=str(path),
        method=method,
        wall_seconds=wall,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        gpu_hours=gpu_hours,
        note="auto-extracted; verify against original logs before reporting",
    )


def write_outputs(records: List[ComputeRecord], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]
    (out_dir / "compute_cost_records.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out_dir / "compute_cost_records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["source_file", "method", "wall_seconds", "gpu_count", "gpu_type", "gpu_hours", "note"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| method | wall seconds | GPU count | GPU type | GPU hours | source |", "|---|---:|---:|---|---:|---|"]
    for r in records:
        lines.append(
            f"| {r.method} | {'' if r.wall_seconds is None else f'{r.wall_seconds:.1f}'} | "
            f"{'' if r.gpu_count is None else r.gpu_count} | {r.gpu_type} | "
            f"{'' if r.gpu_hours is None else f'{r.gpu_hours:.3f}'} | `{Path(r.source_file).name}` |"
        )
    (out_dir / "compute_cost_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect E5 compute-cost evidence from logs and JSON files.")
    ap.add_argument("--log_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    roots = [Path(x).expanduser().resolve() for x in args.log_roots]
    records: List[ComputeRecord] = []
    for p in iter_files(roots):
        rec = parse_file(p)
        if rec is not None:
            records.append(rec)
    records.sort(key=lambda r: (r.method, r.source_file))
    write_outputs(records, Path(args.out_dir))
    print(f"Wrote {len(records)} compute-cost records to {args.out_dir}")
    if not records:
        print("No timing records found. This is acceptable if the run logs do not contain timing/GPU patterns yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
