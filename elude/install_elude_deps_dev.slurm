#!/bin/bash
#SBATCH --job-name=elude-deps
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/elude
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/elude-deps-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/elude-deps-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
PCA="$LLMRUN/src_pca"
ELUDE_DIR="$PCA/elude"
ENV_PATH="${KIF_ENV_PATH:-$LLMRUN/envs/kif-torchclean-py311-cu124}"
ENV_PY="$ENV_PATH/bin/python"

if [[ ! -x "$ENV_PY" ]]; then
  echo "[ELUDE-DEPS][ERROR] Missing Python: $ENV_PY" >&2
  find "$LLMRUN" -maxdepth 4 -path '*/bin/python' -type f 2>/dev/null | sed 's#/bin/python##' >&2 || true
  exit 2
fi

export PATH="$ENV_PATH/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PIP_CACHE_DIR="$LLMRUN/cache/pip"
export HF_HOME="$LLMRUN/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN/data/torch"
export TMPDIR="$LLMRUN/cache/tmp"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$PCA/logs" "$PIP_CACHE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
cd "$ELUDE_DIR"

echo "[ELUDE-DEPS] Host: $(hostname)"
echo "[ELUDE-DEPS] CWD: $(pwd)"
echo "[ELUDE-DEPS] ENV_PATH: $ENV_PATH"
echo "[ELUDE-DEPS] Python: $ENV_PY"

"$ENV_PY" - <<'PY'
import sys, subprocess, os
print('[ELUDE-DEPS] python', sys.version)
try:
    import torch
    print('[ELUDE-DEPS] torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('[ELUDE-DEPS] gpu', torch.cuda.get_device_name(0))
except Exception as e:
    print('[ELUDE-DEPS][WARN] torch import failed before install:', repr(e))
PY

"$ENV_PY" -m pip install --upgrade pip setuptools wheel

# Install ELUDe dependencies into the existing KIF CUDA environment.
# IMPORTANT: do not reinstall torch here; the cluster env already owns the CUDA torch build.
"$ENV_PY" -m pip install --upgrade --no-cache-dir -r requirements_elude.txt

# Smoke-test the exact imports required by ELUDe modules.
"$ENV_PY" - <<'PY'
import importlib, sys
mods = [
    'torch', 'transformers', 'accelerate', 'peft', 'bitsandbytes',
    'datasets', 'huggingface_hub', 'safetensors', 'sentence_transformers',
    'sklearn', 'rouge_score', 'evaluate', 'tqdm', 'pandas', 'numpy',
    'sentencepiece', 'einops'
]
print('\n[ELUDE-DEPS] Import smoke test')
failed = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, '__version__', 'unknown')
        print(f'  OK {m:<24} {ver}')
    except Exception as e:
        failed.append((m, repr(e)))
        print(f'  FAIL {m:<24} {e!r}')
if failed:
    print('\n[ELUDE-DEPS][ERROR] Missing/broken imports:')
    for m, e in failed:
        print(' ', m, e)
    raise SystemExit(1)

import torch
print('[ELUDE-DEPS] CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[ELUDE-DEPS] CUDA device:', torch.cuda.get_device_name(0))
print('[ELUDE-DEPS] All dependency imports passed.')
PY

# Compile core ELUDe files.
"$ENV_PY" -m py_compile \
  run_elude_pipeline.py \
  run_elude20_benchmark.py \
  src/llama20/modules/module_elude.py \
  src/llama20/modules/module_b.py \
  src/llama20/modules/module_c.py \
  src/llama20/modules/module_d.py \
  src/llama20/modules/module_e.py \
  src/llama20/modules/module7.py \
  src/llama20/modules/module8e.py

echo "[ELUDE-DEPS] Dependency setup complete."
