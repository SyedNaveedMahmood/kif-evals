#!/bin/bash
#SBATCH --job-name=tofueval
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-eval-dev-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-eval-dev-%j.err

set -euo pipefail

export LLMRUN_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun"
export SRC_ROOT="$LLMRUN_ROOT/src"
export FRAMEWORK_ROOT="$SRC_ROOT/framework"
export ENV_ACTIVATE="$LLMRUN_ROOT/llm_env/bin/activate"
export TOFU_ROOT="$FRAMEWORK_ROOT/outputs/tofu"
export DATA_DIR="$TOFU_ROOT/data"
export RETAIN_LOGS="${RETAIN_LOGS:-$TOFU_ROOT/retain_reference/retain_reference_logs.json}"
export METHOD_NAME="${METHOD_NAME:-manual_model}"
export MODEL_DIR="${MODEL_DIR:-}"
export OUT_DIR="${OUT_DIR:-$TOFU_ROOT/eval/${METHOD_NAME}_${SLURM_JOB_ID:-manual}}"

export HF_HOME="$LLMRUN_ROOT/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN_ROOT/data/torch"
export TMPDIR="$LLMRUN_ROOT/cache/tmp"

mkdir -p "$FRAMEWORK_ROOT/logs" "$OUT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
source "$ENV_ACTIVATE"
cd "$SRC_ROOT"

export PYTHONPATH="$SRC_ROOT:$FRAMEWORK_ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -z "$MODEL_DIR" ]; then
  echo "MODEL_DIR is required. Example:" >&2
  echo "sbatch --export=ALL,METHOD_NAME=simnpo_pure,MODEL_DIR=/path/to/model framework/scripts/tofu_eval_model_dev.slurm" >&2
  exit 2
fi
if [ ! -f "$DATA_DIR/forget10.jsonl" ]; then
  echo "Missing TOFU data. Run: sbatch framework/scripts/tofu_prepare_data.slurm" >&2
  exit 6
fi
if [ ! -f "$RETAIN_LOGS" ]; then
  echo "Missing retain reference logs: $RETAIN_LOGS" >&2
  echo "Run: sbatch framework/scripts/tofu_eval_retain_reference.slurm" >&2
  exit 7
fi

echo "TOFU eval dev job: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "MODEL_DIR: $MODEL_DIR"
echo "METHOD_NAME: $METHOD_NAME"
echo "RETAIN_LOGS: $RETAIN_LOGS"
echo "OUT_DIR: $OUT_DIR"

nvidia-smi
python -u framework/tofu_eval/evaluate_tofu.py \
  --model_dir "$MODEL_DIR" \
  --data_dir "$DATA_DIR" \
  --split forget10 \
  --output_dir "$OUT_DIR" \
  --retain_logs "$RETAIN_LOGS" \
  --model_family llama3.1-8b \
  --max_length 512 \
  --max_new_tokens 64 \
  --torch_dtype bfloat16 \
  --attn_implementation sdpa \
  --use_4bit

echo "TOFU eval summary: $OUT_DIR/tofu_summary.json"
cat "$OUT_DIR/tofu_summary.json"
