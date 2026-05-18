#!/bin/bash
#SBATCH --job-name=tofuprep
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-prepare-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-prepare-%j.err

set -euo pipefail

export LLMRUN_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun"
export SRC_ROOT="$LLMRUN_ROOT/src"
export FRAMEWORK_ROOT="$SRC_ROOT/framework"
export ENV_ACTIVATE="$LLMRUN_ROOT/llm_env/bin/activate"
export HF_HOME="$LLMRUN_ROOT/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$LLMRUN_ROOT/cache/tmp"

mkdir -p "$FRAMEWORK_ROOT/logs" "$FRAMEWORK_ROOT/outputs/tofu" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
source "$ENV_ACTIVATE"
cd "$SRC_ROOT"

export PYTHONPATH="$SRC_ROOT:$FRAMEWORK_ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

echo "TOFU prepare job: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
python - <<'PY'
import importlib
for pkg in ['datasets', 'torch', 'transformers', 'numpy']:
    mod = importlib.import_module(pkg)
    print(f"[OK] {pkg}: {getattr(mod, '__version__', 'version unknown')}")
PY

python -u framework/tofu_eval/prepare_tofu.py \
  --dataset locuslab/TOFU \
  --split forget10 \
  --out_dir framework/outputs/tofu

echo "TOFU data prepared. Summary:"
cat framework/outputs/tofu/prepare_summary_forget10.json
