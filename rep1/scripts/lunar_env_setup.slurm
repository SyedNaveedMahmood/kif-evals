#!/bin/bash
#SBATCH --job-name=lunar-env
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/rep1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/rep1"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/.conda/envs/lunar-kif-llama31}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$REPO_ROOT/.conda/pkgs}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$ENV_PREFIX")" "$CONDA_PKGS_DIRS" "$REPO_ROOT/.pip_cache" "$REPO_ROOT/.hf_cache"

module purge
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"

export CONDA_PKGS_DIRS
export PIP_CACHE_DIR="$REPO_ROOT/.pip_cache"

if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.10 pip
fi

conda activate "$ENV_PREFIX"
python -m pip install --upgrade pip setuptools wheel packaging

# PyTorch with CUDA for NVIDIA A100/H100 nodes. If your site mirror blocks this,
# replace this line with the locally recommended PyTorch/CUDA install command.
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio

# Minimal combined dependency set for the patched LUNAR run and KIF Module 8 eval.
# This intentionally avoids heavy packages not used by this LUNAR+Module8 path.
python -m pip install --upgrade \
  "transformers>=4.48.0,<5.0" \
  "accelerate>=0.27.2" \
  "peft>=0.11.0" \
  "bitsandbytes>=0.43.0" \
  "safetensors>=0.4.2" \
  "tokenizers>=0.21.0" \
  "datasets>=2.18.0" \
  "evaluate>=0.4.1" \
  "huggingface-hub>=0.27.1" \
  "einops>=0.7.0" \
  "jaxtyping>=0.2.36" \
  "torchtyping>=0.1.5" \
  "hydra-core>=1.3.2" \
  "omegaconf>=2.3.0" \
  "numpy>=1.26.4" \
  "scipy>=1.12.0" \
  "pandas>=2.2.1" \
  "scikit-learn>=1.6.1" \
  "nltk>=3.9" \
  "rouge-score>=0.1.2" \
  "sacrebleu>=2.5.1" \
  "sentence-transformers>=2.2.2" \
  "matplotlib>=3.8.3" \
  "tqdm>=4.66.3" \
  "joblib>=1.3.2" \
  "click>=8.1.7" \
  "requests>=2.31.0" \
  "pyyaml>=6.0.1" \
  "jsonlines>=4.0.0" \
  "sentencepiece>=0.2.0" \
  "regex>=2023.12.25" \
  "psutil>=5.9.8" \
  "portalocker>=3.2.0"

python - <<'PY'
import torch, transformers, accelerate, bitsandbytes, hydra, omegaconf
print('Environment OK')
print('python:', __import__('sys').version)
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cuda_available:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
PY

echo "Environment ready: $ENV_PREFIX"
