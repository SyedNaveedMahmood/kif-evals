#!/bin/bash
#SBATCH --job-name=kif_seed
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG_NAME=${CONFIG_NAME:?Set CONFIG_NAME}
TAU=${TAU:?Set TAU}
K=${K:?Set K}
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

RUN_ROOT="${SWEEP_ROOT}/${CONFIG_NAME}/seed${SEED}"
mkdir -p "$RUN_ROOT"

EXTRA_ARGS=()
if [[ "$RUN_EVAL" == "1" ]]; then
  EXTRA_ARGS+=(--run_eval)
fi
if [[ "$NO_4BIT" == "1" ]]; then
  EXTRA_ARGS+=(--no_4bit)
fi

echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $(hostname)"
echo "[INFO] Commit: $(git rev-parse HEAD || true)"
echo "[INFO] Config: $CONFIG_NAME"
echo "[INFO] tau=$TAU k=$K seed=$SEED"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] Run root: $RUN_ROOT"
nvidia-smi || true

python -u scripts/run_sweep_worker.py \
  --config_name "$CONFIG_NAME" \
  --seed "$SEED" \
  --z_tau "$TAU" \
  --soft_gate_k "$K" \
  --model_dir "$MODEL_DIR" \
  --capsules_dir "$CAPSULES_DIR" \
  --dataset_dir "$DATASET_DIR" \
  --remap_json "$REMAP_JSON" \
  --run_root "$RUN_ROOT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --min_subject_pairs "$MIN_SUBJECT_PAIRS" \
  --min_anchor_pairs "$MIN_ANCHOR_PAIRS" \
  --variants_per_subject "$VARIANTS_PER_SUBJECT" \
  --harvest_variants_per_subject "$HARVEST_VARIANTS_PER_SUBJECT" \
  "${EXTRA_ARGS[@]}"

echo "[DONE] config=$CONFIG_NAME seed=$SEED"