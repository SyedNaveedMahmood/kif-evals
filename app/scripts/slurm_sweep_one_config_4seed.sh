#!/bin/bash
#SBATCH --job-name=kif_tau_4seed
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=24
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG_NAME=${CONFIG_NAME:?Set CONFIG_NAME}
TAU=${TAU:?Set TAU}
K=${K:?Set K}

MODEL_DIR=${MODEL_DIR:-outputs/model}
SHARED_ROOT=${SHARED_ROOT:-outputs}
DATASET_DIR=${DATASET_DIR:-${SHARED_ROOT}/datasets}
CAPSULES_DIR=${CAPSULES_DIR:-${SHARED_ROOT}/capsules}
REMAP_JSON=${REMAP_JSON:-${CAPSULES_DIR}/capsule_module_remap.json}
SWEEP_ROOT=${SWEEP_ROOT:-outputs/sweeps}

EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-5e-6}
MIN_SUBJECT_PAIRS=${MIN_SUBJECT_PAIRS:-800}
MIN_ANCHOR_PAIRS=${MIN_ANCHOR_PAIRS:-600}
VARIANTS_PER_SUBJECT=${VARIANTS_PER_SUBJECT:-60}
HARVEST_VARIANTS_PER_SUBJECT=${HARVEST_VARIANTS_PER_SUBJECT:-50}
RUN_EVAL=${RUN_EVAL:-0}
NO_4BIT=${NO_4BIT:-0}

mkdir -p logs "${SWEEP_ROOT}/${CONFIG_NAME}"

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node: $(hostname)"
echo "[INFO] Config=${CONFIG_NAME}; tau=${TAU}; k=${K}"
echo "[INFO] Slurm CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

if [[ -n "${KIF_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${KIF_CONDA_ENV}"
fi
if [[ -n "${KIF_VENV_ACTIVATE:-}" ]]; then
  source "${KIF_VENV_ACTIVATE}"
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${TMPDIR:-/tmp}/hf_cache"
export TRANSFORMERS_CACHE="${TMPDIR:-/tmp}/hf_cache"
export TORCH_HOME="${TMPDIR:-/tmp}/torch_cache"
mkdir -p "$HF_HOME" "$TORCH_HOME"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a ALLOCATED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  ALLOCATED_GPUS=(0 1 2 3)
fi
if [[ ${#ALLOCATED_GPUS[@]} -lt 4 ]]; then
  echo "[ERROR] Need 4 allocated GPUs, got: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

run_seed () {
  local IDX=$1
  local SEED=$2
  local GPU_ID=${ALLOCATED_GPUS[$IDX]}
  local RUN_ROOT="${SWEEP_ROOT}/${CONFIG_NAME}/seed${SEED}"
  local EXTRA_ARGS=()
  if [[ "${RUN_EVAL}" == "1" ]]; then EXTRA_ARGS+=(--run_eval); fi
  if [[ "${NO_4BIT}" == "1" ]]; then EXTRA_ARGS+=(--no_4bit); fi

  echo "[START] config=${CONFIG_NAME} seed=${SEED} gpu=${GPU_ID} run_root=${RUN_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/run_sweep_worker.py \
    --config_name "${CONFIG_NAME}" \
    --seed "${SEED}" \
    --z_tau "${TAU}" \
    --soft_gate_k "${K}" \
    --model_dir "${MODEL_DIR}" \
    --capsules_dir "${CAPSULES_DIR}" \
    --dataset_dir "${DATASET_DIR}" \
    --remap_json "${REMAP_JSON}" \
    --run_root "${RUN_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --min_subject_pairs "${MIN_SUBJECT_PAIRS}" \
    --min_anchor_pairs "${MIN_ANCHOR_PAIRS}" \
    --variants_per_subject "${VARIANTS_PER_SUBJECT}" \
    --harvest_variants_per_subject "${HARVEST_VARIANTS_PER_SUBJECT}" \
    "${EXTRA_ARGS[@]}" \
    > "logs/${CONFIG_NAME}_seed${SEED}_${SLURM_JOB_ID}.out" \
    2> "logs/${CONFIG_NAME}_seed${SEED}_${SLURM_JOB_ID}.err"
  echo "[DONE] config=${CONFIG_NAME} seed=${SEED} gpu=${GPU_ID}"
}

run_seed 0 17 & PID1=$!
run_seed 1 23 & PID2=$!
run_seed 2 42 & PID3=$!
run_seed 3 51 & PID4=$!

wait $PID1
wait $PID2
wait $PID3
wait $PID4

echo "[INFO] Finished 4 seeds for ${CONFIG_NAME}."
