#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --partition=gpu_h100,gpu_h100_il,gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

echo "[INFO] Job ID: ${SLURM_JOB_ID}"
echo "[INFO] Node: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# Optional environment activation. Set KIF_CONDA_ENV before sbatch if needed.
if [[ -n "${KIF_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${KIF_CONDA_ENV}"
fi
if [[ -n "${KIF_VENV_ACTIVATE:-}" ]]; then
  source "${KIF_VENV_ACTIVATE}"
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${WORKDIR:-$(ws_find llmrun)}/data/hf"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${WORKDIR:-$(ws_find llmrun)}/data/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# Run shared stages once: model setup, dataset, activation mining, signatures, capsules.
python -m llama20.cli pipeline --modules module_b module_c module_d

echo "[INFO] Shared Module 0-D job complete."
