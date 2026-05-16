#!/bin/bash
#SBATCH --job-name=lunar-env
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/rep1
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/lunar-env-%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/lunar-env-%j.log

set -euo pipefail

# Workspace-style layout:
#   $WORKDIR/src/rep1
#   $WORKDIR/envs/lunar-kif-llama31
#   $WORKDIR/hf_cache
#   $WORKDIR/cache
#   $WORKDIR/logs/kif-evals
#   $WORKDIR/results/kif-evals

export WORKDIR="${WORKDIR:-/pfs/work9/workspace/scratch/hd_ur228-llmrun}"
export REPO_ROOT="${REPO_ROOT:-$WORKDIR/src/rep1}"
export ENV_PREFIX="${ENV_PREFIX:-$WORKDIR/envs/lunar-kif-llama31}"

export LOGDIR="${LOGDIR:-$WORKDIR/logs/kif-evals}"
export RESULTSDIR="${RESULTSDIR:-$WORKDIR/results/kif-evals}"
export HF_HOME="${HF_HOME:-$WORKDIR/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORKDIR/cache/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKDIR/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-$WORKDIR/cache/torch}"
export TMPDIR="${TMPDIR:-$WORKDIR/cache/tmp}"

mkdir -p \
  "$WORKDIR/envs" \
  "$WORKDIR/cache" \
  "$CONDA_PKGS_DIRS" \
  "$PIP_CACHE_DIR" \
  "$TORCH_HOME" \
  "$TMPDIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$LOGDIR" \
  "$RESULTSDIR"

cd "$REPO_ROOT"

module purge
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.10 pip
fi

conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip setuptools wheel packaging

python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio

python -m pip install --upgrade \
  "transformers>=4.48.0,<5.0" \
  "accelerate>=0.27.2" \
  "peft>=0.11.0" \
  "bitsandbytes>=0.43.0" \
  "safetensors>=0.4.2" \
  "tokenizers>=0.21.0,<0.22.0" \
  "datasets>=2.18.0" \
  "evaluate>=0.4.1" \
  "huggingface-hub>=0.27.1" \
  "einops>=0.7.0" \
  "jaxtyping>=0.2.36" \
  "torchtyping>=0.1.5" \
  "hydra-core>=1.3.2" \
  "omegaconf>=2.3.0" \
  "numpy>=1.26.4,<2.0" \
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

cat > "$WORKDIR/setup_lunar_env.sh" <<'EOS'
#!/bin/bash
set -euo pipefail

export WORKDIR="${WORKDIR:-/pfs/work9/workspace/scratch/hd_ur228-llmrun}"
export REPO_ROOT="${REPO_ROOT:-$WORKDIR/src/rep1}"
export ENV_PREFIX="${ENV_PREFIX:-$WORKDIR/envs/lunar-kif-llama31}"

export LOGDIR="${LOGDIR:-$WORKDIR/logs/kif-evals}"
export RESULTSDIR="${RESULTSDIR:-$WORKDIR/results/kif-evals}"
export HF_HOME="${HF_HOME:-$WORKDIR/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORKDIR/cache/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKDIR/cache/pip}"
export TORCH_HOME="${TORCH_HOME:-$WORKDIR/cache/torch}"
export TMPDIR="${TMPDIR:-$WORKDIR/cache/tmp}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$LOGDIR" \
  "$RESULTSDIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$TMPDIR"

module purge
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "Missing conda environment at $ENV_PREFIX" >&2
  echo "Create it first with: sbatch scripts/lunar_env_setup.slurm" >&2
  return 2 2>/dev/null || exit 2
fi

conda activate "$ENV_PREFIX"
EOS

chmod +x "$WORKDIR/setup_lunar_env.sh"

python - <<'PY'
import sys
import torch
import transformers
import accelerate
import hydra
import omegaconf

print("Environment OK")
print("python:", sys.version)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("env prefix:", sys.prefix)
PY

echo "Environment ready: $ENV_PREFIX"
echo "Activation script written: $WORKDIR/setup_lunar_env.sh"
echo "Hugging Face cache/auth location: $HF_HOME"