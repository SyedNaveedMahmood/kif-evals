#!/bin/bash
#SBATCH --job-name=drive-upload
#SBATCH --partition=dev_single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/drive-upload-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/drive-upload-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
PCA="$LLMRUN/src_pca"
ENV_PATH="${KIF_ENV_PATH:-$LLMRUN/envs/kif-torchclean-py311-cu124}"
ENV_PY="$ENV_PATH/bin/python"
SERVICE_ACCOUNT_JSON="${SERVICE_ACCOUNT_JSON:-$LLMRUN/secrets/drive_service_account.json}"
DRIVE_FOLDER="${DRIVE_FOLDER:-https://drive.google.com/drive/folders/1f0jYraH6gR8UncwCSLR7DoW6HjvwFZ0u}"
RUN_NAME="${RUN_NAME:-kif_results_$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"
MAX_SIZE_MB="${MAX_SIZE_MB:-64}"
INCLUDE_ALL_TEXT="${INCLUDE_ALL_TEXT:-0}"

if [[ ! -x "$ENV_PY" ]]; then
  echo "[DRIVE-UPLOAD][ERROR] Missing Python: $ENV_PY" >&2
  find "$LLMRUN" -maxdepth 4 -path '*/bin/python' -type f 2>/dev/null | sed 's#/bin/python##' >&2 || true
  exit 2
fi

mkdir -p "$PCA/logs" "$LLMRUN/secrets"
cd "$PCA"

export PATH="$ENV_PATH/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PIP_CACHE_DIR="$LLMRUN/cache/pip"

if [[ ! -f "$SERVICE_ACCOUNT_JSON" ]]; then
  echo "[DRIVE-UPLOAD][ERROR] Service account JSON missing: $SERVICE_ACCOUNT_JSON" >&2
  echo "[DRIVE-UPLOAD][HINT] Put the JSON key there or submit with --export=ALL,SERVICE_ACCOUNT_JSON=/path/to/key.json" >&2
  exit 3
fi

chmod 600 "$SERVICE_ACCOUNT_JSON" || true

echo "[DRIVE-UPLOAD] Host: $(hostname)"
echo "[DRIVE-UPLOAD] CWD: $(pwd)"
echo "[DRIVE-UPLOAD] Python: $ENV_PY"
echo "[DRIVE-UPLOAD] Service account JSON: $SERVICE_ACCOUNT_JSON"
echo "[DRIVE-UPLOAD] Drive folder: $DRIVE_FOLDER"
echo "[DRIVE-UPLOAD] Run name: $RUN_NAME"
echo "[DRIVE-UPLOAD] Dry run: $DRY_RUN"

"$ENV_PY" -m pip install --upgrade --quiet google-api-python-client google-auth google-auth-httplib2

"$ENV_PY" - <<'PY'
import importlib, json, os
mods = ['googleapiclient', 'google.oauth2', 'google_auth_httplib2']
for m in mods:
    importlib.import_module(m)
print('[DRIVE-UPLOAD] Google API client imports OK')
key_path = os.environ.get('SERVICE_ACCOUNT_JSON')
with open(key_path, 'r', encoding='utf-8') as f:
    obj = json.load(f)
print('[DRIVE-UPLOAD] service account email:', obj.get('client_email'))
PY

ARGS=(
  --service-account "$SERVICE_ACCOUNT_JSON"
  --drive-folder "$DRIVE_FOLDER"
  --run-name "$RUN_NAME"
  --max-size-mb "$MAX_SIZE_MB"
)

if [[ "$LIMIT" != "0" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "$INCLUDE_ALL_TEXT" == "1" ]]; then
  ARGS+=(--include-all-text)
fi

"$ENV_PY" -u analysis/upload_results_to_drive.py "${ARGS[@]}"

echo "[DRIVE-UPLOAD] Done."
