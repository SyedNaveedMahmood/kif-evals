#!/bin/bash
#SBATCH --job-name=optoutmain
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/optout-main-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/optout-main-%j.err

set -euo pipefail

export LLMRUN_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun"
export SRC_ROOT="$LLMRUN_ROOT/src"
export FRAMEWORK_ROOT="$SRC_ROOT/framework"
export KIF_APP_ROOT="$SRC_ROOT/app"
export ENV_ACTIVATE="$LLMRUN_ROOT/llm_env/bin/activate"
export ENTRYPOINT="$FRAMEWORK_ROOT/orchestrate.py"

export LOGDIR="$FRAMEWORK_ROOT/logs"
export OUTPUT_ROOT="$FRAMEWORK_ROOT/outputs/optout"
export EMPTY_CAPSULES_DIR="$FRAMEWORK_ROOT/outputs/empty_capsules"
export DATASET_SUMMARY_JSON="$OUTPUT_ROOT/optout_dataset_check_summary.json"
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

# Faithful OPT-OUT settings from the paper/code: npo+rt+wd+ot, beta=0.1,
# lambda=0.1, lr=1e-5, weight_decay=0.01, epochs=3, alternate_updates.
# This uses all 4 GPUs on one GPU node with Transformers device_map="auto",
# matching upstream's multi-GPU expectation more closely than the failed 1-GPU run.
CONFIG_JSON="${CONFIG_JSON:-$(cat <<'JSON'
{
  "method": "npo+rt+wd+ot",
  "model_family": "llama3.1-8b",
  "dpo_beta": 0.1,
  "reg_lambda": 0.1,
  "learning_rate": 0.00001,
  "warmup_ratio": 0.0,
  "weight_decay": 0.01,
  "num_epochs": 3,
  "max_grad_norm": 1.0,
  "max_length": 512,
  "torch_dtype": "bfloat16",
  "attn_implementation": "sdpa",
  "use_gradient_checkpointing": true,
  "alternate_updates": true,
  "batch_size": 4,
  "gradient_accumulation_steps": 8,
  "n_forget": 132,
  "n_retain": 132,
  "n_world": 132,
  "max_per_subject": 12,
  "swd_n_projections": 100,
  "swd_p": 2,
  "swd_max_tensors": null,
  "swd_max_rows": null,
  "trainable_modules": "all",
  "device_map_auto": true,
  "save_merged_model": true,
  "seed": 42,
  "logging_steps": 1
}
JSON
)}"

RUN_ID="optout_main_${SLURM_JOB_ID:-manual}"
RUN_OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
RUN_EVAL_DIR="$RUN_OUTPUT_DIR/eval"
mkdir -p "$RUN_OUTPUT_DIR" "$RUN_EVAL_DIR"

if [ ! -f "$PROMPTS_JSONL" ]; then
  echo "Missing checked augmented dataset: $PROMPTS_JSONL" >&2
  echo "Run first: sbatch framework/scripts/optout_dataset_check_faithful_dev.slurm" >&2
  exit 6
fi
if [ ! -f "$DATASET_SUMMARY_JSON" ]; then
  echo "Missing dataset check summary: $DATASET_SUMMARY_JSON" >&2
  echo "Run first: sbatch framework/scripts/optout_dataset_check_faithful_dev.slurm" >&2
  exit 7
fi

python - <<'PY'
import json, os, sys
from pathlib import Path
summary = json.loads(Path(os.environ["DATASET_SUMMARY_JSON"]).read_text(encoding="utf-8"))
print("Dataset check summary:")
print(json.dumps(summary, indent=2, ensure_ascii=False))
if summary.get("status") != "ok":
    print("Dataset status is not ok; refusing main run.", file=sys.stderr)
    sys.exit(8)
if summary.get("world_status") != "external_world_json":
    print("OPT-OUT +wd world data is not external Alpaca-GPT4; refusing main run.", file=sys.stderr)
    sys.exit(9)
PY

echo "OPT-OUT main job: ${SLURM_JOB_ID:-manual}"
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
for pkg in ["torch", "transformers", "numpy", "sklearn", "huggingface_hub", "accelerate"]:
    mod = importlib.import_module(pkg)
    print(f"[OK] {pkg}: {getattr(mod, '__version__', 'version unknown')}")
PY

nvidia-smi
python - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"gpu[{i}]:", torch.cuda.get_device_name(i))
PY

CMD=(
  python -u "$ENTRYPOINT"
  --method optout
  --model_dir "$MODEL_DIR"
  --prompts_jsonl "$PROMPTS_JSONL"
  --capsules_dir "$CAPSULES_DIR"
  --output_dir "$RUN_OUTPUT_DIR"
  --eval_out_dir "$RUN_EVAL_DIR"
  --subjects "${SUBJECTS[@]}"
  --config "$CONFIG_JSON"
  --skip_eval false
)

echo "Running OPT-OUT main:"
printf ' %q' "${CMD[@]}"
echo
srun --ntasks=1 "${CMD[@]}"

echo "Finished at $(date)"
echo "Outputs:"
echo "  Run output: $RUN_OUTPUT_DIR"
echo "  Model artifact: $RUN_OUTPUT_DIR/optout/model"
echo "  Manifest: $RUN_OUTPUT_DIR/optout/unlearning_result.json"
echo "  OPT-OUT metadata: $RUN_OUTPUT_DIR/optout/optout_meta.json"
echo "  Eval output: $RUN_EVAL_DIR"
echo "  Final summary: $RUN_EVAL_DIR/final_summary.json"
