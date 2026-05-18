#!/bin/bash
#SBATCH --job-name=reglusmoke
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/reglu-smoke-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/reglu-smoke-%j.err

set -euo pipefail

export LLMRUN_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun"
export SRC_ROOT="$LLMRUN_ROOT/src"
export FRAMEWORK_ROOT="$SRC_ROOT/framework"
export KIF_APP_ROOT="$SRC_ROOT/app"

export ENV_ACTIVATE="$LLMRUN_ROOT/llm_env/bin/activate"
export ENTRYPOINT="$FRAMEWORK_ROOT/orchestrate.py"

export LOGDIR="$FRAMEWORK_ROOT/logs"
export OUTPUT_ROOT="$FRAMEWORK_ROOT/outputs/reglu"
export EMPTY_CAPSULES_DIR="$FRAMEWORK_ROOT/outputs/empty_capsules"
export DATASET_SUMMARY_JSON="$OUTPUT_ROOT/reglu_dataset_check_summary.json"

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

mkdir -p \
  "$LOGDIR" \
  "$OUTPUT_ROOT" \
  "$EMPTY_CAPSULES_DIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$TMPDIR"

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

export USE_TF32="${USE_TF32:-true}"
export SEED="${SEED:-17}"

SUBJECTS=(
  "Ariana Grande"
  "Arijit Singh"
  "Beyoncé"
  "Drake (musician)"
  "Ed Sheeran"
  "Eminem"
  "Kanye West"
  "Katy Perry"
  "Michael Jackson"
  "Queen (band)"
  "Taylor Swift"
)

# Smoke test: full Llama-3.1-8B load + ReGLU LoRA attach + tiny RILA + one tiny training pass.
# It skips Module 8 eval and does NOT save a merged full model.
CONFIG_JSON="${CONFIG_JSON:-$(cat <<'JSON'
{
  "model_family": "llama3.1-8b",
  "variant": "ihl",
  "lora_targets": "qv",
  "lora_r": 4,
  "lora_alpha": 8,
  "lora_dropout": 0.0,
  "rila_beta": 0.5,
  "rila_samples_per_split": 4,
  "rila_cov_shrink": 0.0001,
  "rol_lambda": 0.5,
  "rol_rank": 4,
  "retain_gamma": 1.0,
  "n_forget": 4,
  "n_retain": 4,
  "max_per_subject": 1,
  "batch_size": 1,
  "gradient_accumulation_steps": 1,
  "num_epochs": 1,
  "learning_rate": 0.0001,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "max_length": 128,
  "torch_dtype": "bfloat16",
  "save_merged_model": false,
  "seed": 17
}
JSON
)}"

RUN_ID="reglu_smoke_${SLURM_JOB_ID:-manual}"
RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
RUN_EVAL_DIR="$RUN_OUTPUT_DIR/eval"
mkdir -p "$RUN_OUTPUT_DIR" "$RUN_EVAL_DIR"

if [ ! -f "$PROMPTS_JSONL" ]; then
  echo "Missing checked augmented dataset: $PROMPTS_JSONL" >&2
  echo "Run first: sbatch framework/scripts/reglu_dataset_check_dev.slurm" >&2
  exit 6
fi

if [ ! -f "$DATASET_SUMMARY_JSON" ]; then
  echo "Missing dataset check summary: $DATASET_SUMMARY_JSON" >&2
  echo "Run first: sbatch framework/scripts/reglu_dataset_check_dev.slurm" >&2
  exit 7
fi

python - <<'PY'
import json, os, sys
from pathlib import Path
summary_path = Path(os.environ["DATASET_SUMMARY_JSON"])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
print("Dataset check summary:")
print(json.dumps(summary, indent=2, ensure_ascii=False))
if summary.get("status") != "ok":
    print("Dataset summary status is not ok. Refusing to run ReGLU smoke test.", file=sys.stderr)
    sys.exit(8)
PY

echo "ReGLU smoke job: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "MODEL_DIR: $MODEL_DIR"
echo "PROMPTS_JSONL: $PROMPTS_JSONL"
echo "DATASET_SUMMARY_JSON: $DATASET_SUMMARY_JSON"
echo "RUN_OUTPUT_DIR: $RUN_OUTPUT_DIR"
echo "CONFIG_JSON: $CONFIG_JSON"
echo "Subjects:"
printf '  - %s\n' "${SUBJECTS[@]}"

if [ ! -f "$ENV_ACTIVATE" ]; then echo "Missing env activation script: $ENV_ACTIVATE" >&2; exit 2; fi
if [ ! -f "$ENTRYPOINT" ]; then echo "Missing framework entrypoint: $ENTRYPOINT" >&2; exit 3; fi
if [ ! -f "$HF_HOME/token" ]; then echo "Missing Hugging Face token: $HF_HOME/token" >&2; exit 4; fi

python - <<'PY'
import importlib, sys
packages = ["torch", "transformers", "peft", "numpy", "sklearn", "huggingface_hub"]
missing = []
for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        print(f"[OK] {pkg}: {getattr(mod, '__version__', 'version unknown')}")
    except Exception as e:
        print(f"[MISSING] {pkg}: {e}")
        missing.append(pkg)
if missing:
    print("Missing required packages:", missing, file=sys.stderr)
    sys.exit(20)
PY

python - <<'PY'
import os
from huggingface_hub import whoami
from transformers import AutoConfig, AutoTokenizer
model_id = os.environ["MODEL_DIR"]
print("whoami:", whoami())
if not os.path.isdir(model_id):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("model_type:", getattr(cfg, "model_type", None))
    print("architectures:", getattr(cfg, "architectures", None))
    print("tokenizer:", tok.__class__.__name__)
PY

nvidia-smi
python - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available(): print("gpu:", torch.cuda.get_device_name(0))
PY

CMD=(
  python -u "$ENTRYPOINT"
  --method reglu
  --model_dir "$MODEL_DIR"
  --prompts_jsonl "$PROMPTS_JSONL"
  --capsules_dir "$CAPSULES_DIR"
  --output_dir "$RUN_OUTPUT_DIR"
  --eval_out_dir "$RUN_EVAL_DIR"
  --subjects "${SUBJECTS[@]}"
  --config "$CONFIG_JSON"
  --skip_eval true
)

echo "Running smoke test:"
printf ' %q' "${CMD[@]}"
echo

srun --ntasks=1 "${CMD[@]}"

echo "Smoke test finished at $(date)"
echo "Outputs:"
echo "  Run output: $RUN_OUTPUT_DIR"
echo "  Adapter: $RUN_OUTPUT_DIR/reglu/adapter"
echo "  Manifest: $RUN_OUTPUT_DIR/reglu/unlearning_result.json"
echo "  ReGLU metadata: $RUN_OUTPUT_DIR/reglu/reglu_meta.json"
echo "If this succeeded, submit: sbatch framework/scripts/reglu_main_run.slurm"
