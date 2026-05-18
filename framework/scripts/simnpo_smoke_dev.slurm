#!/bin/bash
#SBATCH --job-name=simnposmoke
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/simnpo-smoke-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/simnpo-smoke-%j.err

set -euo pipefail

export LLMRUN_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun"
export SRC_ROOT="$LLMRUN_ROOT/src"
export FRAMEWORK_ROOT="$SRC_ROOT/framework"
export KIF_APP_ROOT="$SRC_ROOT/app"
export ENV_ACTIVATE="$LLMRUN_ROOT/llm_env/bin/activate"
export ENTRYPOINT="$FRAMEWORK_ROOT/orchestrate.py"

export LOGDIR="$FRAMEWORK_ROOT/logs"
export OUTPUT_ROOT="$FRAMEWORK_ROOT/outputs/simnpo"
export EMPTY_CAPSULES_DIR="$FRAMEWORK_ROOT/outputs/empty_capsules"
export DATASET_SUMMARY_JSON="$OUTPUT_ROOT/simnpo_dataset_check_summary.json"
export MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B}"
export PROMPTS_JSONL="$OUTPUT_ROOT/prompts_with_adele_retain.jsonl"
export CAPSULES_DIR="$EMPTY_CAPSULES_DIR"

export HF_HOME="$LLMRUN_ROOT/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN_ROOT/data/torch"
export TMPDIR="$LLMRUN_ROOT/cache/tmp"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT" "$EMPTY_CAPSULES_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
source "$ENV_ACTIVATE"
cd "$SRC_ROOT"

export PYTHONPATH="$SRC_ROOT:$FRAMEWORK_ROOT:$KIF_APP_ROOT/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-8}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_MAX_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SUBJECTS=("Ariana Grande" "Arijit Singh" "Beyoncé" "Drake (musician)" "Ed Sheeran" "Eminem" "Kanye West" "Katy Perry" "Michael Jackson" "Queen (band)" "Taylor Swift")

# Smoke test: full model load + faithful SimNPO loss path + retain loss path.
# It trains lm_head only and skips saving/eval to keep dev GPU cheap.
CONFIG_JSON="${CONFIG_JSON:-$(cat <<'JSON'
{
  "model_family": "llama3.1-8b",
  "beta": 2.5,
  "gamma": 0.0,
  "npo_coeff": 0.1375,
  "grad_diff_coeff": 1.0,
  "lr": 0.00001,
  "num_epochs": 1,
  "batch_size": 1,
  "gradient_accumulation_steps": 1,
  "weight_decay": 0.01,
  "max_seq_len": 128,
  "use_retain_loss": true,
  "optimizer_name": "adamw",
  "warmup_steps": "steps_per_epoch",
  "torch_dtype": "bfloat16",
  "attn_implementation": "sdpa",
  "use_gradient_checkpointing": false,
  "n_forget": 2,
  "n_retain": 2,
  "max_per_subject": 1,
  "trainable_modules": "lm_head",
  "save_model": false,
  "seed": 42,
  "logging_steps": 1
}
JSON
)}"

RUN_ID="simnpo_smoke_${SLURM_JOB_ID:-manual}"
RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
RUN_EVAL_DIR="$RUN_OUTPUT_DIR/eval"
mkdir -p "$RUN_OUTPUT_DIR" "$RUN_EVAL_DIR"

if [ ! -f "$PROMPTS_JSONL" ]; then echo "Missing checked dataset: $PROMPTS_JSONL" >&2; exit 6; fi
if [ ! -f "$DATASET_SUMMARY_JSON" ]; then echo "Missing dataset summary: $DATASET_SUMMARY_JSON" >&2; exit 7; fi

python - <<'PY'
import json, os, sys
from pathlib import Path
summary = json.loads(Path(os.environ['DATASET_SUMMARY_JSON']).read_text(encoding='utf-8'))
print('Dataset check summary:')
print(json.dumps(summary, indent=2, ensure_ascii=False))
if summary.get('status') != 'ok':
    print('Dataset status is not ok; refusing smoke test.', file=sys.stderr)
    sys.exit(8)
PY

echo "SimNPO smoke job: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "MODEL_DIR: $MODEL_DIR"
echo "PROMPTS_JSONL: $PROMPTS_JSONL"
echo "RUN_OUTPUT_DIR: $RUN_OUTPUT_DIR"
echo "CONFIG_JSON: $CONFIG_JSON"

if [ ! -f "$ENV_ACTIVATE" ]; then echo "Missing env activation script: $ENV_ACTIVATE" >&2; exit 2; fi
if [ ! -f "$ENTRYPOINT" ]; then echo "Missing framework entrypoint: $ENTRYPOINT" >&2; exit 3; fi
if [ ! -f "$HF_HOME/token" ]; then echo "Missing Hugging Face token: $HF_HOME/token" >&2; exit 4; fi

python - <<'PY'
import importlib
for pkg in ['torch', 'transformers', 'numpy', 'sklearn', 'huggingface_hub']:
    mod = importlib.import_module(pkg)
    print(f"[OK] {pkg}: {getattr(mod, '__version__', 'version unknown')}")
PY

nvidia-smi
python - <<'PY'
import os, torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('cuda visible devices:', os.environ.get('CUDA_VISIBLE_DEVICES'))
if torch.cuda.is_available(): print('gpu:', torch.cuda.get_device_name(0))
PY

CMD=(
  python -u "$ENTRYPOINT"
  --method simnpo
  --model_dir "$MODEL_DIR"
  --prompts_jsonl "$PROMPTS_JSONL"
  --capsules_dir "$CAPSULES_DIR"
  --output_dir "$RUN_OUTPUT_DIR"
  --eval_out_dir "$RUN_EVAL_DIR"
  --subjects "${SUBJECTS[@]}"
  --config "$CONFIG_JSON"
  --skip_eval true
)

echo "Running SimNPO smoke test:"
printf ' %q' "${CMD[@]}"
echo
srun --ntasks=1 "${CMD[@]}"

echo "SimNPO smoke test finished at $(date)"
echo "Outputs:"
echo "  Run output: $RUN_OUTPUT_DIR"
echo "  Metadata: $RUN_OUTPUT_DIR/simnpo/simnpo_meta.json"
echo "If this succeeded, submit: sbatch framework/scripts/simnpo_main_run.slurm"
