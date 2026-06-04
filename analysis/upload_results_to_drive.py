#!/usr/bin/env python3
"""Upload JSON/result artifacts from KIF workspaces to a Google Drive folder.

Default source roots are the bwUniCluster KIF paths:
  /pfs/work9/workspace/scratch/hd_ur228-llmrun/src
  /pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca

Authentication modes:
  --auth service-account --service-account /path/to/key.json
      Uses a Google service-account key. The destination folder must be shared
      with the service account email.

  --auth adc
      Uses Application Default Credentials, for example credentials created by:
        gcloud auth application-default login --no-browser --scopes=https://www.googleapis.com/auth/drive.file
      This is the best option when service-account key creation is disabled.

The script preserves the source-relative directory structure under a timestamped
Drive folder, and writes local manifest files for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DEFAULT_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1f0jYraH6gR8UncwCSLR7DoW6HjvwFZ0u"
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
GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"


@dataclass
class UploadRecord:
    local_path: str
    root_label: str
    relative_path: str
    drive_path: str
    drive_file_id: str
    size_bytes: int
    sha256: str
    status: str


def parse_drive_folder_id(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Empty Drive folder ID/URL")
    patterns = [
        r"/folders/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, value)
        if m:
            return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", value):
        return value
    raise ValueError(f"Could not parse Google Drive folder ID from: {value}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def guess_mime(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return "application/x-ndjson"
    if path.suffix.lower() == ".json":
        return "application/json"
    if path.suffix.lower() == ".csv":
        return "text/csv"
    if path.suffix.lower() in {".txt", ".log", ".out", ".err", ".md", ".tex", ".tsv"}:
        return "text/plain"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def should_skip_dir(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return any(x in parts for x in SKIP_DIR_NAMES)


def should_include_file(path: Path, extensions: Sequence[str], name_patterns: Sequence[str], include_all_ext: bool) -> bool:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".safetensors", ".pt", ".pth", ".bin", ".pkl", ".gz", ".zip", ".tar", ".xz"}:
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


def build_drive_service_service_account(service_account_json: Path):
    if not service_account_json.exists():
        raise FileNotFoundError(f"Service account JSON not found: {service_account_json}")
    creds = service_account.Credentials.from_service_account_file(str(service_account_json), scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def build_drive_service_adc():
    creds, project_id = google.auth.default(scopes=SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    print(f"[INFO] Using Application Default Credentials. project_id={project_id}")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def build_drive_service(auth_mode: str, service_account_json: str):
    if auth_mode == "service-account":
        if not service_account_json:
            raise ValueError("--service-account is required with --auth service-account")
        return build_drive_service_service_account(Path(service_account_json).expanduser().resolve())
    if auth_mode == "adc":
        return build_drive_service_adc()
    raise ValueError(f"Unsupported auth mode: {auth_mode}")


def create_folder(service, name: str, parent_id: str, dry_run: bool = False) -> str:
    if dry_run:
        return f"DRYRUN_FOLDER_{hashlib.sha1((parent_id + '/' + name).encode()).hexdigest()[:12]}"
    metadata = {"name": name, "mimeType": GOOGLE_FOLDER_MIME, "parents": [parent_id]}
    obj = service.files().create(
        body=metadata,
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return obj["id"]


def create_folder_tree(service, parent_id: str, rel_dir: Path, cache: Dict[str, str], dry_run: bool = False) -> str:
    cur = parent_id
    prefix_parts: List[str] = []
    for part in rel_dir.parts:
        if part in {"", "."}:
            continue
        prefix_parts.append(part)
        key = "/".join(prefix_parts)
        if key not in cache:
            cache[key] = create_folder(service, part, cur, dry_run=dry_run)
        cur = cache[key]
    return cur


def upload_file(service, local_path: Path, parent_id: str, dry_run: bool = False) -> str:
    if dry_run:
        return f"DRYRUN_FILE_{hashlib.sha1(str(local_path).encode()).hexdigest()[:12]}"
    media = MediaFileUpload(str(local_path), mimetype=guess_mime(local_path), resumable=True)
    metadata = {"name": local_path.name, "parents": [parent_id]}
    request = service.files().create(
        body=metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    )
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"    progress {pct}%: {local_path.name}")
    return response["id"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload JSON/result artifacts from src and src_pca to Google Drive.")
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS, help="Source roots to scan recursively.")
    ap.add_argument("--drive-folder", default=DEFAULT_DRIVE_FOLDER_URL, help="Destination Google Drive folder URL or folder ID.")
    ap.add_argument("--auth", choices=["adc", "service-account"], default=os.getenv("DRIVE_AUTH_MODE", "service-account"), help="Google auth mode. Use 'adc' after gcloud auth application-default login.")
    ap.add_argument("--service-account", default="", help="Path to Google service-account key JSON. Only needed with --auth service-account.")
    ap.add_argument("--run-name", default="", help="Optional top-level folder name in Drive.")
    ap.add_argument("--extensions", nargs="+", default=DEFAULT_EXTENSIONS, help="File extensions to include.")
    ap.add_argument("--include-all-text", action="store_true", help="Include all files matching extensions, not only result-like text names.")
    ap.add_argument("--max-size-mb", type=float, default=64.0, help="Skip individual files larger than this.")
    ap.add_argument("--limit", type=int, default=0, help="Upload at most N files; 0 means no limit.")
    ap.add_argument("--dry-run", action="store_true", help="List what would upload without touching Drive.")
    args = ap.parse_args()

    roots = [Path(x).resolve() for x in args.roots]
    extensions = [x.lower() if x.startswith(".") else "." + x.lower() for x in args.extensions]
    parent_id = parse_drive_folder_id(args.drive_folder)
    run_name = args.run_name or "kif_results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("[INFO] Roots:")
    for r in roots:
        print(f"  - {r}")
    print(f"[INFO] Destination parent folder id: {parent_id}")
    print(f"[INFO] Drive run folder: {run_name}")
    print(f"[INFO] Auth mode: {args.auth}")
    print(f"[INFO] Dry run: {args.dry_run}")

    candidates = list(iter_result_files(
        roots=roots,
        extensions=extensions,
        name_patterns=DEFAULT_NAME_PATTERNS,
        include_all_ext=args.include_all_text,
        max_size_mb=args.max_size_mb,
    ))
    candidates.sort(key=lambda x: str(x[2]))
    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]

    print(f"[INFO] Candidate files: {len(candidates)}")
    for _, root, p in candidates[:20]:
        print(f"  - {p.relative_to(root)}")
    if len(candidates) > 20:
        print(f"  ... {len(candidates) - 20} more")

    if not candidates:
        print("[WARN] Nothing to upload.")
        return 0

    service = None if args.dry_run else build_drive_service(args.auth, args.service_account)
    run_folder_id = create_folder(service, run_name, parent_id, dry_run=args.dry_run)
    folder_cache: Dict[str, str] = {}
    records: List[UploadRecord] = []

    start = time.time()
    for idx, (root_label, root, p) in enumerate(candidates, start=1):
        rel = p.relative_to(root)
        drive_rel_dir = Path(root_label) / rel.parent
        drive_path = str(drive_rel_dir / p.name)
        print(f"[{idx}/{len(candidates)}] upload {p} -> {drive_path}")
        try:
            parent = create_folder_tree(service, run_folder_id, drive_rel_dir, folder_cache, dry_run=args.dry_run)
            file_id = upload_file(service, p, parent, dry_run=args.dry_run)
            records.append(UploadRecord(
                local_path=str(p),
                root_label=root_label,
                relative_path=str(rel),
                drive_path=drive_path,
                drive_file_id=file_id,
                size_bytes=p.stat().st_size,
                sha256=sha256_file(p),
                status="uploaded" if not args.dry_run else "dry_run",
            ))
        except HttpError as e:
            print(f"[ERROR] Drive API failed for {p}: {e}", file=sys.stderr)
            records.append(UploadRecord(
                local_path=str(p), root_label=root_label, relative_path=str(rel), drive_path=drive_path,
                drive_file_id="", size_bytes=p.stat().st_size if p.exists() else 0, sha256="", status=f"error:{e}",
            ))
        except Exception as e:
            print(f"[ERROR] Failed for {p}: {e}", file=sys.stderr)
            records.append(UploadRecord(
                local_path=str(p), root_label=root_label, relative_path=str(rel), drive_path=drive_path,
                drive_file_id="", size_bytes=p.stat().st_size if p.exists() else 0, sha256="", status=f"error:{e}",
            ))

    out_dir = Path.cwd() / "drive_upload_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "created_utc": stamp,
        "source_roots": [str(r) for r in roots],
        "drive_parent_folder_id": parent_id,
        "drive_run_folder_name": run_name,
        "drive_run_folder_id": run_folder_id,
        "auth_mode": args.auth,
        "dry_run": args.dry_run,
        "elapsed_seconds": round(time.time() - start, 3),
        "n_records": len(records),
        "records": [asdict(r) for r in records],
    }
    manifest_path = out_dir / f"drive_upload_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Manifest written: {manifest_path}")
    print(f"[INFO] Drive run folder id: {run_folder_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
