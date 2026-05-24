#!/bin/bash
#SBATCH --job-name=elude-kif
#SBATCH --partition=gpu_a100_il,gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/elude
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/elude-kif-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/elude-kif-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
PCA="$LLMRUN/src_pca"
ELUDE_DIR="$PCA/elude"

# Prefer the environment that has already been used successfully for recent jobs.
if [[ -f "$LLMRUN/envs/kif-torchclean-py311-cu124/bin/activate" ]]; then
  source "$LLMRUN/envs/kif-torchclean-py311-cu124/bin/activate"
elif [[ -f "$LLMRUN/llm_env/bin/activate" ]]; then
  source "$LLMRUN/llm_env/bin/activate"
else
  echo "[ELUDE-KIF][ERROR] Could not find expected Python environment." >&2
  exit 2
fi

export HF_HOME="$LLMRUN/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN/data/torch"
export TMPDIR="$LLMRUN/cache/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$PCA/logs" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"

cd "$ELUDE_DIR"

echo "[ELUDE-KIF] Host: $(hostname)"
echo "[ELUDE-KIF] CWD: $(pwd)"
echo "[ELUDE-KIF] Python: $(which python)"
python - <<'PY'
import torch, sys
print('[ELUDE-KIF] torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[ELUDE-KIF] gpu', torch.cuda.get_device_name(0))
print('[ELUDE-KIF] python', sys.version)
PY

# ---------------- User-overridable run controls ----------------
# Full ELUDe has 20 targets. Start with 5 by default for a robust full-partition run.
# To run all targets: sbatch --export=ALL,ELUDE_MAX_TARGETS=20 elude/run_elude_kif_full_gpu.slurm
export ELUDE_MAX_TARGETS="${ELUDE_MAX_TARGETS:-5}"
export ELUDE_TARGETS="${ELUDE_TARGETS:-}"
export ELUDE_TRAIN_RATIO="${ELUDE_TRAIN_RATIO:-0.80}"
export ELUDE_SEED="${ELUDE_SEED:-17}"
export ELUDE_EXACT_PROTOCOL="${ELUDE_EXACT_PROTOCOL:-1}"
export ELUDE_RETAIN_EVAL_SPLIT="${ELUDE_RETAIN_EVAL_SPLIT:-test}"

# Keep caps reasonable for initial KIF-on-ELUDe runs. Set to 0 for no cap.
export ELUDE_MAX_FORGET_TRAIN="${ELUDE_MAX_FORGET_TRAIN:-300}"
export ELUDE_MAX_RETAIN_TRAIN="${ELUDE_MAX_RETAIN_TRAIN:-300}"
export ELUDE_MAX_EVAL_ROWS="${ELUDE_MAX_EVAL_ROWS:-0}"
export ELUDE_DO_GENERATION="${ELUDE_DO_GENERATION:-1}"

# Model and quantization.
export KIF_HF_MODEL_ID="${KIF_HF_MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
export KIF_MODEL_DIR="${KIF_MODEL_DIR:-outputs/model}"
export KIF_USE_4BIT="${KIF_USE_4BIT:-1}"

# Module B: exact MLP-block hooks, not gate/up projections.
export KIF_PROBE_LAYERS="${KIF_PROBE_LAYERS:-11,12,13,14}"
export KIF_PROBE_TARGETS="${KIF_PROBE_TARGETS:-mlp}"
export KIF_PROBE_BATCH_SIZE="${KIF_PROBE_BATCH_SIZE:-16}"
export KIF_PROBE_MAX_LENGTH="${KIF_PROBE_MAX_LENGTH:-160}"
export KIF_PROBE_CAPTURE_SCOPE="${KIF_PROBE_CAPTURE_SCOPE:-last_token}"
export KIF_PROBE_COMPRESSION_LEVEL="${KIF_PROBE_COMPRESSION_LEVEL:-1}"
export KIF_PROBE_SKIP_EXISTING="${KIF_PROBE_SKIP_EXISTING:-1}"

# Module C: ELUDe should use retain controls instead of Gaussian negatives.
export KIF_MODULEC_ALLOW_SYNTHETIC_NEGATIVES="${KIF_MODULEC_ALLOW_SYNTHETIC_NEGATIVES:-0}"

# Softer suppression for ELUDe retain safety.
export KIF_CAPSULE_SCALING_INIT="${KIF_CAPSULE_SCALING_INIT:--0.55}"
export KIF_CAPSULE_MAX_SCALING="${KIF_CAPSULE_MAX_SCALING:--2.0}"
export KIF_CAPSULE_CLAMP_MIN="${KIF_CAPSULE_CLAMP_MIN:--2.0}"
export KIF_SENTINEL_DEFAULT_STRENGTH="${KIF_SENTINEL_DEFAULT_STRENGTH:--0.55}"
export KIF_REFUSAL_STEER_STRENGTH="${KIF_REFUSAL_STEER_STRENGTH:-8.0}"
export KIF_SIG_SUPPRESSION_STRENGTH="${KIF_SIG_SUPPRESSION_STRENGTH:-0.65}"

# Module 7 runtime. Keep max steps unset by default so epochs determine training.
# Override if needed: sbatch --export=ALL,KIF_M7_MAX_STEPS=400 ...
export KIF_M7_MAX_STEPS="${KIF_M7_MAX_STEPS:-}"

# Clean outputs unless RESUME=1. Module B itself skips existing activations when configured.
if [[ "${RESUME:-0}" != "1" ]]; then
  echo "[ELUDE-KIF] Fresh run: removing previous ELUDe outputs"
  rm -rf outputs/activations outputs/signatures outputs/capsules outputs/sentinel outputs/global_adapters outputs/eval_elude outputs/elude outputs/datasets
else
  echo "[ELUDE-KIF] RESUME=1: preserving existing outputs"
fi

# Apply code-level stability patches before running.
python -u apply_elude_kif_patches.py

python -m py_compile run_elude_pipeline.py \
  src/llama20/modules/module_elude.py \
  src/llama20/modules/module_b.py \
  src/llama20/modules/module_c.py \
  src/llama20/modules/module_d.py \
  src/llama20/modules/module_e.py \
  src/llama20/modules/module7.py \
  src/llama20/modules/module8e.py

echo "[ELUDE-KIF] Run controls:"
python - <<'PY'
import os, json
keys = [
 'ELUDE_TARGETS','ELUDE_MAX_TARGETS','ELUDE_MAX_FORGET_TRAIN','ELUDE_MAX_RETAIN_TRAIN',
 'KIF_MODULEC_ALLOW_SYNTHETIC_NEGATIVES','KIF_CAPSULE_SCALING_INIT','KIF_SENTINEL_DEFAULT_STRENGTH',
 'KIF_REFUSAL_STEER_STRENGTH','KIF_SIG_SUPPRESSION_STRENGTH','KIF_PROBE_LAYERS','KIF_PROBE_TARGETS'
]
print(json.dumps({k: os.environ.get(k) for k in keys}, indent=2))
PY

ARGS=(--skip-module0 --train-ratio "$ELUDE_TRAIN_RATIO" --seed "$ELUDE_SEED")
if [[ -n "$ELUDE_TARGETS" ]]; then
  ARGS+=(--targets "$ELUDE_TARGETS")
else
  ARGS+=(--max-targets "$ELUDE_MAX_TARGETS")
fi

python -u run_elude_pipeline.py "${ARGS[@]}"

echo "[ELUDE-KIF] Final summaries:"
python - <<'PY'
import json
from pathlib import Path
for p in [Path('outputs/elude/metadata.json'), Path('outputs/signatures/signature_index.json'), Path('outputs/eval_elude/elude_summary.json')]:
    print('\n====', p, '====')
    if not p.exists():
        print('MISSING')
        continue
    obj = json.loads(p.read_text())
    if 'stats_by_target' in obj:
        print(json.dumps({'targets': obj.get('targets'), 'stats_by_target': obj.get('stats_by_target')}, indent=2, ensure_ascii=False)[:4000])
    elif 'subjects' in obj and 'stats' in obj:
        print(json.dumps({'stats': obj.get('stats'), 'subjects': obj.get('subjects')}, indent=2, ensure_ascii=False)[:4000])
    else:
        print(json.dumps(obj, indent=2, ensure_ascii=False)[:4000])
PY
