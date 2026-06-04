#!/usr/bin/env python3
"""Stage KIF result artifacts for upload with rclone or another sync tool.

This does not authenticate to Google. It recursively scans the default KIF
workspaces, copies result-like files into a clean staging directory, preserves
source-relative structure, and writes a SHA256 manifest.

Default roots:
  /pfs/work9/workspace/scratch/hd_ur228-llmrun/src
  /pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

DEFAULT_ROOTS = [
    "/pfs/work9/workspace/scratch/hd_ur228-llmrun/src",
    "/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca",
]
DEFAULT_EXTENSIONS = [
    ".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".out", ".err", ".tex", ".md",
]
DEFAULT_NAME_PATTERNS = [
    "summary", "metric", "metrics", "result", "results", "eval", "evaluation",
    "token", "audit", "manifest", "metadata", "prediction", "predictions",
]
SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".cache", "cache", "wandb", "node_modules",
    "models", "model", "checkpoints", "checkpoint", "hf", "torch", "tmp", "temp",
}
SKIP_EXTENSIONS = {".safetensors", ".pt", ".pth", ".bin", ".pkl", ".gz", ".zip", ".tar", ".xz"}


@dataclass
class StageRecord:
    local_path: str
    staged_path: str
    root_label: str
    relative_path: str
    size_bytes: int
    sha256: str
    status: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def should_skip_dir(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return any(x in parts for x in SKIP_DIR_NAMES)


def should_include_file(path: Path, extensions: Sequence[str], name_patterns: Sequence[str], include_all_ext: bool) -> bool:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in SKIP_EXTENSIONS:
        return False
    if suffix in extensions:
        if include_all_ext:
            return True
        if suffix in {".json", ".jsonl", ".csv", ".tsv"}:
            return True
        return any(p in name for p in name_patterns)
    return False


def iter_result_files(
    roots: Sequence[Path],
    extensions: Sequence[str],
    name_patterns: Sequence[str],
    include_all_ext: bool,
    max_size_mb: float,
) -> Iterable[Tuple[str, Path, Path]]:
    max_bytes = int(max_size_mb * 1024 * 1024)
    for root in roots:
        if not root.exists():
            print(f"[WARN] Missing root, skipping: {root}", file=sys.stderr)
            continue
        root_label = root.name or root.as_posix().strip("/").replace("/", "_")
        for dirpath, dirnames, filenames in os.walk(root):
            d = Path(dirpath)
            dirnames[:] = [x for x in dirnames if not should_skip_dir(d / x)]
            if should_skip_dir(d):
                continue
            for fn in filenames:
                p = d / fn
                try:
                    if not p.is_file() or p.is_symlink():
                        continue
                    if not should_include_file(p, extensions, name_patterns, include_all_ext):
                        continue
                    if p.stat().st_size > max_bytes:
                        print(f"[SKIP] too large ({p.stat().st_size} bytes): {p}", file=sys.stderr)
                        continue
                    yield root_label, root, p
                except OSError as e:
                    print(f"[WARN] Could not stat {p}: {e}", file=sys.stderr)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage KIF result artifacts for upload.")
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--out-dir", required=True, help="Output staging directory.")
    ap.add_argument("--run-name", default="", help="Subfolder name inside out-dir.")
    ap.add_argument("--extensions", nargs="+", default=DEFAULT_EXTENSIONS)
    ap.add_argument("--include-all-text", action="store_true")
    ap.add_argument("--max-size-mb", type=float, default=64.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    args = ap.parse_args()

    roots = [Path(x).resolve() for x in args.roots]
    extensions = [x.lower() if x.startswith(".") else "." + x.lower() for x in args.extensions]
    run_name = args.run_name or "kif_results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_root = Path(args.out_dir).resolve() / run_name
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    candidates = list(iter_result_files(
        roots=roots,
        extensions=extensions,
        name_patterns=DEFAULT_NAME_PATTERNS,
        include_all_ext=args.include_all_text,
        max_size_mb=args.max_size_mb,
    ))
    candidates.sort(key=lambda x: str(x[2]))
    if args.limit and args.limit > 0:
        candidates = candidates[:args.limit]

    print(f"[STAGE] Stage root: {stage_root}")
    print(f"[STAGE] Candidate files: {len(candidates)}")

    records: List[StageRecord] = []
    for idx, (root_label, root, src) in enumerate(candidates, start=1):
        rel = src.relative_to(root)
        dst = stage_root / root_label / rel
        print(f"[{idx}/{len(candidates)}] {src} -> {dst}")
        try:
            copy_or_link(src, dst, args.mode)
            records.append(StageRecord(
                local_path=str(src),
                staged_path=str(dst),
                root_label=root_label,
                relative_path=str(rel),
                size_bytes=src.stat().st_size,
                sha256=sha256_file(src),
                status="staged",
            ))
        except Exception as e:
            print(f"[ERROR] Failed to stage {src}: {e}", file=sys.stderr)
            records.append(StageRecord(
                local_path=str(src),
                staged_path=str(dst),
                root_label=root_label,
                relative_path=str(rel),
                size_bytes=src.stat().st_size if src.exists() else 0,
                sha256="",
                status=f"error:{e}",
            ))

    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "run_name": run_name,
        "stage_root": str(stage_root),
        "source_roots": [str(r) for r in roots],
        "n_records": len(records),
        "records": [asdict(r) for r in records],
    }
    manifest_path = stage_root / "stage_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[STAGE] Manifest written: {manifest_path}")
    print(f"[STAGE] Done: {stage_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
