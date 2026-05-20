#!/bin/bash
#SBATCH --job-name=tofuref
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-retain-ref-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-retain-ref-%j.err

set -euo pipefail

export LLMRUN_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun"
export SRC_ROOT="$LLMRUN_ROOT/src"
export FRAMEWORK_ROOT="$SRC_ROOT/framework"
export ENV_ACTIVATE="$LLMRUN_ROOT/llm_env/bin/activate"

export TOFU_ROOT="$FRAMEWORK_ROOT/outputs/tofu"
export DATA_DIR="$TOFU_ROOT/data"
export RETAIN_MODEL_DIR="${TOFU_RETAIN_MODEL_DIR:-open-unlearning/tofu_Llama-2-7b-chat-hf_retain90}"
export RETAIN_REF_DIR="${TOFU_RETAIN_REF_DIR:-$TOFU_ROOT/retain_reference_openunlearning_llama2_retain90}"
export RETAIN_LOGS="$RETAIN_REF_DIR/retain_reference_logs.json"
export MODEL_FAMILY="${MODEL_FAMILY:-llama2-7b-chat}"

export HF_HOME="$LLMRUN_ROOT/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN_ROOT/data/torch"
export TMPDIR="$LLMRUN_ROOT/cache/tmp"

mkdir -p "$FRAMEWORK_ROOT/logs" "$RETAIN_REF_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
source "$ENV_ACTIVATE"
cd "$SRC_ROOT"

export PYTHONPATH="$SRC_ROOT:$FRAMEWORK_ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-8}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_MAX_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "TOFU retain-reference job: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "RETAIN_MODEL_DIR: $RETAIN_MODEL_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "RETAIN_REF_DIR: $RETAIN_REF_DIR"
echo "RETAIN_LOGS: $RETAIN_LOGS"
echo "MODEL_FAMILY: $MODEL_FAMILY"

if [ ! -f "$DATA_DIR/forget10.jsonl" ]; then
  echo "Missing $DATA_DIR/forget10.jsonl. Run: sbatch framework/scripts/tofu_prepare_data.slurm" >&2
  exit 6
fi

python - <<'PY'
import ast
for path in ['framework/tofu_eval/metrics.py', 'framework/tofu_eval/evaluate_tofu.py', 'framework/tofu_eval/make_retain_reference.py']:
    ast.parse(open(path, encoding='utf-8').read())
    print('[OK syntax]', path)
PY

nvidia-smi
python - <<'PY'
import os, torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('cuda visible devices:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'gpu[{i}]:', torch.cuda.get_device_name(i))
PY

python -u framework/tofu_eval/make_retain_reference.py \
  --model_dir "$RETAIN_MODEL_DIR" \
  --data_dir "$DATA_DIR" \
  --split forget10 \
  --output_dir "$RETAIN_REF_DIR" \
  --model_family "$MODEL_FAMILY" \
  --max_length 512 \
  --batch_size 32 \
  --expected_count 400 \
  --torch_dtype bfloat16 \
  --attn_implementation sdpa \
  --use_4bit \
  --device_map_auto

python -u framework/tofu_eval/evaluate_tofu.py \
  --validate_retain_logs_only \
  --retain_logs "$RETAIN_LOGS" \
  --expected_retain_ref_count 400

echo "============================================================"
echo "RETAIN REFERENCE READY"
echo "============================================================"
echo "$RETAIN_LOGS"
ls -lh "$RETAIN_LOGS"
