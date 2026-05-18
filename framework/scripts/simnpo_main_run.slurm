#!/bin/bash
#SBATCH --job-name=simnpomain
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/simnpo-main-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/simnpo-main-%j.err

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-16}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_MAX_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MAX_SUBJECTS="${MAX_SUBJECTS:-11}"
export VARIANTS_PER_SUBJ="${VARIANTS_PER_SUBJ:-6}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
export EL_STEPS="${EL_STEPS:-32}"
export EL_MAX_VARIANTS="${EL_MAX_VARIANTS:-3}"
export EL_MAX_KEYWORDS="${EL_MAX_KEYWORDS:-10}"
export USE_4BIT="${USE_4BIT:-true}"
export USE_TF32="${USE_TF32:-true}"
export SEED="${SEED:-17}"

SUBJECTS=("Ariana Grande" "Arijit Singh" "Beyoncé" "Drake (musician)" "Ed Sheeran" "Eminem" "Kanye West" "Katy Perry" "Michael Jackson" "Queen (band)" "Taylor Swift")

# Faithful SimNPO-GradDiff setup using official Unlearn-Simple TOFU Forget05
# hyperparameters where applicable: beta=2.5, gamma=0.0, npo_coeff=0.1375,
# grad_diff_coeff=1.0, lr=1e-5, epochs=10, batch=1, accumulation=4,
# weight_decay=0.01, paged_adamw_32bit, warmup_steps=steps_per_epoch.
# We use the KIF base model, KIF prompts, Adele retain set, and Module 8 eval.
CONFIG_JSON="${CONFIG_JSON:-$(cat <<'JSON'
{
  "model_family": "llama3.1-8b",
  "beta": 2.5,
  "gamma": 0.0,
  "npo_coeff": 0.1375,
  "grad_diff_coeff": 1.0,
  "lr": 0.00001,
  "num_epochs": 10,
  "batch_size": 1,
  "gradient_accumulation_steps": 4,
  "weight_decay": 0.01,
  "max_seq_len": 500,
  "use_retain_loss": true,
  "optimizer_name": "paged_adamw_32bit",
  "warmup_steps": "steps_per_epoch",
  "torch_dtype": "bfloat16",
  "attn_implementation": "sdpa",
  "use_gradient_checkpointing": true,
  "n_forget": 132,
  "n_retain": 132,
  "max_per_subject": 12,
  "trainable_modules": "all",
  "device_map_auto": true,
  "save_model": true,
  "seed": 42,
  "logging_steps": 1
}
JSON
)}"

RUN_ID="simnpo_main_${SLURM_JOB_ID:-manual}"
RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
RUN_EVAL_DIR="$RUN_OUTPUT_DIR/eval"
mkdir -p "$RUN_OUTPUT_DIR" "$RUN_EVAL_DIR"

if [ ! -f "$PROMPTS_JSONL" ]; then
  echo "Missing checked dataset: $PROMPTS_JSONL" >&2
  echo "Run first: sbatch framework/scripts/simnpo_dataset_check_dev.slurm" >&2
  exit 6
fi
if [ ! -f "$DATASET_SUMMARY_JSON" ]; then
  echo "Missing dataset summary: $DATASET_SUMMARY_JSON" >&2
  echo "Run first: sbatch framework/scripts/simnpo_dataset_check_dev.slurm" >&2
  exit 7
fi

python - <<'PY'
import json, os, sys
from pathlib import Path
summary = json.loads(Path(os.environ['DATASET_SUMMARY_JSON']).read_text(encoding='utf-8'))
print('Dataset check summary:')
print(json.dumps(summary, indent=2, ensure_ascii=False))
if summary.get('status') != 'ok':
    print('Dataset status is not ok; refusing main run.', file=sys.stderr)
    sys.exit(8)
if int(summary.get('forget_rows_selected', 0)) != 132:
    print('Expected 132 forget rows; refusing main run.', file=sys.stderr)
    sys.exit(9)
if int(summary.get('retain_pool_rows', 0)) <= 0:
    print('Expected Adele retain rows for simnpo_grad_diff; refusing main run.', file=sys.stderr)
    sys.exit(10)
PY

echo "SimNPO main job: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "MODEL_DIR: $MODEL_DIR"
echo "PROMPTS_JSONL: $PROMPTS_JSONL"
echo "RUN_OUTPUT_DIR: $RUN_OUTPUT_DIR"
echo "RUN_EVAL_DIR: $RUN_EVAL_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_GPUS: ${SLURM_GPUS:-unset}"
echo "CONFIG_JSON: $CONFIG_JSON"

if [ ! -f "$ENV_ACTIVATE" ]; then echo "Missing env activation script: $ENV_ACTIVATE" >&2; exit 2; fi
if [ ! -f "$ENTRYPOINT" ]; then echo "Missing framework entrypoint: $ENTRYPOINT" >&2; exit 3; fi
if [ ! -f "$HF_HOME/token" ]; then echo "Missing Hugging Face token: $HF_HOME/token" >&2; exit 4; fi

python - <<'PY'
import importlib
for pkg in ['torch', 'transformers', 'numpy', 'sklearn', 'huggingface_hub', 'accelerate']:
    mod = importlib.import_module(pkg)
    print(f"[OK] {pkg}: {getattr(mod, '__version__', 'version unknown')}")
try:
    import bitsandbytes as bnb
    print(f"[OK] bitsandbytes: {getattr(bnb, '__version__', 'version unknown')}")
except Exception as e:
    print(f"[WARN] bitsandbytes unavailable: {e}")
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
  --skip_eval false
)

echo "Running SimNPO main:"
printf ' %q' "${CMD[@]}"
echo
srun --ntasks=1 "${CMD[@]}"

echo "Finished at $(date)"
echo "Outputs:"
echo "  Run output: $RUN_OUTPUT_DIR"
echo "  Model artifact: $RUN_OUTPUT_DIR/simnpo/model"
echo "  Manifest: $RUN_OUTPUT_DIR/simnpo/unlearning_result.json"
echo "  SimNPO metadata: $RUN_OUTPUT_DIR/simnpo/simnpo_meta.json"
echo "  Eval output: $RUN_EVAL_DIR"
echo "  Final summary: $RUN_EVAL_DIR/final_summary.json"
