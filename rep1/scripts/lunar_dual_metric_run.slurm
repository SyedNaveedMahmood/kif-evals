#!/bin/bash
#SBATCH --job-name=lunar-kif
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/rep1
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/lunar-kif-%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/lunar-kif-%j.log

set -euo pipefail

# Submit from:
#   /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/rep1
#
# Example:
#   sbatch scripts/lunar_dual_metric_run.slurm
#
# Override settings:
#   sbatch --export=ALL,SUBJECTS='Taylor Swift,Ariana Grande',LAYER='22',COEFF='2.0' scripts/lunar_dual_metric_run.slurm

export WORKDIR="${WORKDIR:-/pfs/work9/workspace/scratch/hd_ur228-llmrun}"
export REPO_ROOT="${REPO_ROOT:-$WORKDIR/src/rep1}"

source "$WORKDIR/setup_lunar_env.sh"

cd "$REPO_ROOT"

# -------------------------
# Experiment defaults
# -------------------------

SUBJECTS="${SUBJECTS:-Taylor Swift}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B}"
DATA_NAME="${DATA_NAME:-kif_entities}"

LAYER="${LAYER:-22}"
COEFF="${COEFF:-2.0}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-0.01}"

MAX_PER_SUBJECT="${MAX_PER_SUBJECT:-0}"
USE_HARMFUL="${USE_HARMFUL:-0}"
RUN_KIF_EVAL="${RUN_KIF_EVAL:-1}"
USE_4BIT="${USE_4BIT:-true}"
SEED="${SEED:-17}"

# Default KIF locations. These support both KIF-main/outputs and repo-root outputs.
KIF_PROMPTS_JSONL="${KIF_PROMPTS_JSONL:-$REPO_ROOT/KIF-main/outputs/datasets/prompts.jsonl}"
KIF_CAPSULES_DIR="${KIF_CAPSULES_DIR:-$REPO_ROOT/KIF-main/outputs/capsules}"

if [ ! -f "$KIF_PROMPTS_JSONL" ] && [ -f "$REPO_ROOT/outputs/datasets/prompts.jsonl" ]; then
  KIF_PROMPTS_JSONL="$REPO_ROOT/outputs/datasets/prompts.jsonl"
fi

if [ ! -d "$KIF_CAPSULES_DIR" ] && [ -d "$REPO_ROOT/outputs/capsules" ]; then
  KIF_CAPSULES_DIR="$REPO_ROOT/outputs/capsules"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-16}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SEED
export USE_4BIT

RUN_ID="lunar_kif_${SLURM_JOB_ID:-manual}"
RUN_OUTDIR="$RESULTSDIR/lunar_dualmetric/$RUN_ID"
mkdir -p "$RUN_OUTDIR" "$LOGDIR"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Repo: $REPO_ROOT"
echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Env: $ENV_PREFIX"
echo "HF_HOME: $HF_HOME"
echo "Results: $RUN_OUTDIR"
echo "Subjects: $SUBJECTS"
echo "Model: $MODEL_ID"
echo "Layer: $LAYER"
echo "Coeff: $COEFF"
echo "Epochs: $EPOCHS"
echo "LR: $LR"

if [ ! -f "$KIF_PROMPTS_JSONL" ]; then
  echo "Missing KIF prompts JSONL: $KIF_PROMPTS_JSONL" >&2
  echo "Override with: sbatch --export=ALL,KIF_PROMPTS_JSONL=/path/to/prompts.jsonl scripts/lunar_dual_metric_run.slurm" >&2
  exit 3
fi

if [ ! -d "$KIF_CAPSULES_DIR" ]; then
  echo "Missing KIF capsules dir: $KIF_CAPSULES_DIR" >&2
  echo "Override with: sbatch --export=ALL,KIF_CAPSULES_DIR=/path/to/capsules scripts/lunar_dual_metric_run.slurm" >&2
  exit 4
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. This is OK only if Hugging Face auth is already cached under $HF_HOME."
fi

nvidia-smi

python - <<'PY'
import os
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

CMD=(
  python -u scripts/run_lunar_on_kif_then_eval.py
  --root "$REPO_ROOT"
  --kif-prompts-jsonl "$KIF_PROMPTS_JSONL"
  --kif-capsules-dir "$KIF_CAPSULES_DIR"
  --subjects "$SUBJECTS"
  --model-id "$MODEL_ID"
  --data-name "$DATA_NAME"
  --layer "$LAYER"
  --coeff "$COEFF"
  --epochs "$EPOCHS"
  --lr "$LR"
)

if [ "$MAX_PER_SUBJECT" != "0" ]; then
  CMD+=(--max-per-subject "$MAX_PER_SUBJECT")
fi

if [ "$USE_HARMFUL" = "1" ] || [ "$USE_HARMFUL" = "true" ]; then
  CMD+=(--use-harmful)
fi

if [ "$RUN_KIF_EVAL" = "1" ] || [ "$RUN_KIF_EVAL" = "true" ]; then
  CMD+=(--run-kif-eval)
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

srun --ntasks=1 "${CMD[@]}"

echo "Finished at $(date)"
echo "Primary repo outputs:"
echo "  $REPO_ROOT/LUNAR-main/models_lunar/$DATA_NAME/llama31-8b-base/"
echo "  $REPO_ROOT/outputs/eval_lunar_dualmetric/"
echo "Workspace result marker:"
echo "  $RUN_OUTDIR"