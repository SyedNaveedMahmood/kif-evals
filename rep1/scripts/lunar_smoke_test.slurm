#!/bin/bash
#SBATCH --job-name=lunar-smoke
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100,gpu_a100_short,gpu_h100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/rep1
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/lunar-smoke-%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/lunar-smoke-%j.log

set -euo pipefail

export WORKDIR="${WORKDIR:-/pfs/work9/workspace/scratch/hd_ur228-llmrun}"
export REPO_ROOT="${REPO_ROOT:-$WORKDIR/src/rep1}"

source "$WORKDIR/setup_lunar_env.sh"

cd "$REPO_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-16}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Repo: $REPO_ROOT"
echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Env: $ENV_PREFIX"
echo "HF_HOME: $HF_HOME"

nvidia-smi

python - <<'PY'
import os
import torch

print("cwd:", os.getcwd())
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python scripts/run_lunar_on_kif_then_eval.py --help

echo "Smoke test finished. This does not load Llama-3.1-8B."