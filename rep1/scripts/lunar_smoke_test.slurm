#!/bin/bash
#SBATCH --job-name=lunar-smoke
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/rep1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/rep1"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/.conda/envs/lunar-kif-llama31}"

cd "$REPO_ROOT"

module purge
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.torch_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

nvidia-smi
python - <<'PY'
import os, torch
print('cwd:', os.getcwd())
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY

python scripts/run_lunar_on_kif_then_eval.py --help

echo "Smoke test finished. This does not load Llama-3.1-8B."
