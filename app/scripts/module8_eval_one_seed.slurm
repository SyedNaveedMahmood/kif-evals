#!/bin/bash
#SBATCH --job-name=eval8
#SBATCH --partition=gpu_a100_il,gpu_h100,gpu_h100_il,dev_gpu_h100,dev_gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG_NAME=${CONFIG_NAME:?Set CONFIG_NAME}
SEED=${SEED:?Set SEED}

WORKDIR=/pfs/work9/workspace/scratch/hd_ur228-llmrun
APPDIR="$WORKDIR/src/app"

cd "$APPDIR"
mkdir -p logs

source "$WORKDIR/llm_env/bin/activate"

export PYTHONPATH="$APPDIR/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

export HF_HOME="$WORKDIR/data/hf"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$WORKDIR/data/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

MODEL_DIR=${MODEL_DIR:-outputs/model}
SHARED_ROOT=${SHARED_ROOT:-outputs}
DATASET_DIR=${DATASET_DIR:-${SHARED_ROOT}/datasets}
CAPSULES_DIR=${CAPSULES_DIR:-${SHARED_ROOT}/capsules}
PROMPTS_JSONL=${PROMPTS_JSONL:-${DATASET_DIR}/prompts.jsonl}
SWEEP_ROOT=${SWEEP_ROOT:-outputs/sweeps}

RUN_ROOT="${SWEEP_ROOT}/${CONFIG_NAME}/seed${SEED}"
MODULE7_DIR="${RUN_ROOT}/module7"
MODULE8_DIR="${RUN_ROOT}/module8"
ADAPTER_JSON="${MODULE7_DIR}/last_adapter_path.json"

mkdir -p "$MODULE8_DIR"

echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $(hostname)"
echo "[INFO] Config: $CONFIG_NAME"
echo "[INFO] Seed: $SEED"
echo "[INFO] Run root: $RUN_ROOT"
echo "[INFO] Adapter JSON: $ADAPTER_JSON"
echo "[INFO] Module 8 output: $MODULE8_DIR"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi || true

if [[ ! -f "$ADAPTER_JSON" ]]; then
  echo "[ERROR] Missing adapter path file: $ADAPTER_JSON"
  exit 1
fi

if [[ ! -f "$PROMPTS_JSONL" ]]; then
  echo "[ERROR] Missing prompts file: $PROMPTS_JSONL"
  exit 1
fi

python -u - <<PY
import json
from pathlib import Path
from llama20.modules.module8 import run_module8_clean

adapter_json = Path("${ADAPTER_JSON}")
data = json.loads(adapter_json.read_text(encoding="utf-8"))

adapter_path = data.get("adapter_path")
if not adapter_path:
    raise RuntimeError(f"No adapter_path found inside {adapter_json}")

adapter_path = str(adapter_path)

print("[INFO] Running Module 8")
print("[INFO] model_dir      =", "${MODEL_DIR}")
print("[INFO] adapter_path   =", adapter_path)
print("[INFO] capsules_dir   =", "${CAPSULES_DIR}")
print("[INFO] prompts_jsonl   =", "${PROMPTS_JSONL}")
print("[INFO] out_dir        =", "${MODULE8_DIR}")

summary = run_module8_clean(
    model_dir="${MODEL_DIR}",
    adapter_path=adapter_path,
    out_dir="${MODULE8_DIR}",
    capsules_dir="${CAPSULES_DIR}",
    prompts_jsonl="${PROMPTS_JSONL}",
)

print("[DONE] Module 8 complete")
PY

echo "[DONE] config=$CONFIG_NAME seed=$SEED"