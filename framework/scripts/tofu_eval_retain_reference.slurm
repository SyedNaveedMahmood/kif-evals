#!/bin/bash
#SBATCH --job-name=tofuref
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100,gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-retain-reference-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-retain-reference-%j.err

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
export TORCH_HOME="$LLMRUN_ROOT/data/torch"

mkdir -p "$FRAMEWORK_ROOT/logs" "$FRAMEWORK_ROOT/outputs/tofu/retain_reference" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$TORCH_HOME"
source "$ENV_ACTIVATE"
cd "$SRC_ROOT"

export PYTHONPATH="$SRC_ROOT:$FRAMEWORK_ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Override this if OpenUnlearning uses a different HF id in your environment.
export RETAIN_MODEL_DIR="${RETAIN_MODEL_DIR:-open-unlearning/tofu_Llama-3.1-8B_retain90}"
export OUT_DIR="$FRAMEWORK_ROOT/outputs/tofu/retain_reference"

echo "TOFU retain reference eval job: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "RETAIN_MODEL_DIR: $RETAIN_MODEL_DIR"

if [ ! -f "$FRAMEWORK_ROOT/outputs/tofu/data/forget10.jsonl" ]; then
  echo "Missing TOFU prepared data. Run: sbatch framework/scripts/tofu_prepare_data.slurm" >&2
  exit 6
fi

python -u framework/tofu_eval/evaluate_tofu.py \
  --model_dir "$RETAIN_MODEL_DIR" \
  --data_dir "$FRAMEWORK_ROOT/outputs/tofu/data" \
  --split forget10 \
  --output_dir "$OUT_DIR" \
  --write_retain_logs \
  --model_family llama3.1-8b \
  --max_length 512 \
  --max_new_tokens 64 \
  --torch_dtype bfloat16 \
  --attn_implementation sdpa \
  --use_4bit

echo "Retain reference logs: $OUT_DIR/retain_reference_logs.json"
echo "Retain reference summary: $OUT_DIR/tofu_summary.json"
