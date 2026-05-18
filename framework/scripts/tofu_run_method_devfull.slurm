#!/bin/bash
#SBATCH --job-name=tofumethod
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=220G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-method-devfull-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-method-devfull-%j.err

set -euo pipefail

export LLMRUN_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun"
export SRC_ROOT="$LLMRUN_ROOT/src"
export FRAMEWORK_ROOT="$SRC_ROOT/framework"
export KIF_APP_ROOT="$SRC_ROOT/app"
export ENV_ACTIVATE="$LLMRUN_ROOT/llm_env/bin/activate"
export ENTRYPOINT="$FRAMEWORK_ROOT/orchestrate.py"

export LOGDIR="$FRAMEWORK_ROOT/logs"
export EMPTY_CAPSULES_DIR="$FRAMEWORK_ROOT/outputs/empty_capsules"
export TOFU_ROOT="$FRAMEWORK_ROOT/outputs/tofu"
export PROMPTS_JSONL="$TOFU_ROOT/method_inputs/full_forget10_retain90.jsonl"
export SUBJECTS_FILE="$TOFU_ROOT/subjects/forget10_subjects.txt"
export RUNS_ROOT="$TOFU_ROOT/runs"

# For faithful TOFU, this should be a model finetuned on the full TOFU train set,
# not the raw base model. Override if your checkpoint/HF id differs.
export TOFU_FULL_MODEL_DIR="${TOFU_FULL_MODEL_DIR:-open-unlearning/tofu_Llama-3.1-8B_full}"
export TOFU_METHOD="${TOFU_METHOD:-simnpo_pure}"

export HF_HOME="$LLMRUN_ROOT/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN_ROOT/data/torch"
export TMPDIR="$LLMRUN_ROOT/cache/tmp"

mkdir -p "$LOGDIR" "$EMPTY_CAPSULES_DIR" "$RUNS_ROOT" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
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

if [ ! -f "$PROMPTS_JSONL" ]; then
  echo "Missing TOFU method input: $PROMPTS_JSONL" >&2
  echo "Run first: sbatch framework/scripts/tofu_prepare_data.slurm" >&2
  exit 6
fi
if [ ! -f "$SUBJECTS_FILE" ]; then
  echo "Missing TOFU forget subjects: $SUBJECTS_FILE" >&2
  echo "Run first: sbatch framework/scripts/tofu_prepare_data.slurm" >&2
  exit 7
fi

mapfile -t SUBJECTS < "$SUBJECTS_FILE"
if [ "${#SUBJECTS[@]}" -eq 0 ]; then
  echo "No forget subjects loaded from $SUBJECTS_FILE" >&2
  exit 8
fi

case "$TOFU_METHOD" in
  lunar)
    METHOD="lunar"
    RUN_LABEL="lunar_forget10"
    CONFIG_JSON='{"model_family":"llama3-8b","lambda_reg":0.001,"auto_select_layer":true,"n_forget":300,"n_retain":300,"n_ref":128,"max_per_subject":20,"batch_size":8,"seed":42}'
    ;;
  reglu)
    METHOD="reglu"
    RUN_LABEL="reglu_forget10"
    CONFIG_JSON='{"model_family":"llama3.1-8b","variant":"ihl","lora_targets":"all","lora_r":32,"lora_alpha":64,"lora_dropout":0.0,"rila_beta":0.5,"rila_samples_per_split":128,"rila_cov_shrink":0.0001,"rol_lambda":0.5,"rol_rank":128,"retain_gamma":1.0,"n_forget":300,"n_retain":300,"max_per_subject":20,"batch_size":4,"gradient_accumulation_steps":8,"num_epochs":5,"learning_rate":0.0001,"weight_decay":0.01,"max_grad_norm":1.0,"max_length":256,"torch_dtype":"bfloat16","save_merged_model":true,"seed":17}'
    ;;
  optout)
    METHOD="optout"
    RUN_LABEL="optout_forget10"
    CONFIG_JSON='{"method":"npo+rt+wd+ot","model_family":"llama3.1-8b","dpo_beta":0.1,"reg_lambda":0.1,"learning_rate":0.00001,"warmup_ratio":0.0,"weight_decay":0.01,"num_epochs":3,"max_grad_norm":1.0,"max_length":512,"torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":true,"alternate_updates":true,"batch_size":4,"gradient_accumulation_steps":8,"n_forget":300,"n_retain":300,"n_world":300,"max_per_subject":20,"swd_n_projections":100,"swd_p":2,"device_map_auto":true,"trainable_modules":"all","swd_max_tensors":null,"swd_max_rows":null,"save_merged_model":true,"seed":42,"logging_steps":1}'
    ;;
  simnpo|simnpo_graddiff)
    METHOD="simnpo"
    RUN_LABEL="simnpo_graddiff_forget10"
    CONFIG_JSON='{"model_family":"llama3.1-8b","beta":2.5,"gamma":0.0,"npo_coeff":0.1375,"grad_diff_coeff":1.0,"lr":0.00001,"num_epochs":10,"batch_size":1,"gradient_accumulation_steps":4,"weight_decay":0.01,"max_seq_len":500,"use_retain_loss":true,"optimizer_name":"paged_adamw_32bit","warmup_steps":"steps_per_epoch","torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":true,"n_forget":300,"n_retain":300,"max_per_subject":20,"trainable_modules":"all","device_map_auto":true,"save_model":true,"seed":42,"logging_steps":1}'
    ;;
  simnpo_pure)
    METHOD="simnpo"
    RUN_LABEL="simnpo_pure_forget10"
    CONFIG_JSON='{"model_family":"llama3.1-8b","beta":2.5,"gamma":0.0,"npo_coeff":0.1375,"grad_diff_coeff":1.0,"lr":0.00001,"num_epochs":10,"batch_size":1,"gradient_accumulation_steps":4,"weight_decay":0.01,"max_seq_len":500,"use_retain_loss":false,"optimizer_name":"paged_adamw_32bit","warmup_steps":"steps_per_epoch","torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":true,"n_forget":300,"n_retain":0,"max_per_subject":20,"trainable_modules":"all","device_map_auto":true,"save_model":true,"seed":42,"logging_steps":1}'
    ;;
  *)
    echo "Unknown TOFU_METHOD=$TOFU_METHOD. Use: lunar | reglu | optout | simnpo_graddiff | simnpo_pure" >&2
    exit 9
    ;;
esac

RUN_ID="${RUN_LABEL}_${SLURM_JOB_ID:-manual}"
RUN_OUTPUT_DIR="$RUNS_ROOT/$RUN_ID"
mkdir -p "$RUN_OUTPUT_DIR"

echo "TOFU method dev-full job: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "TOFU_METHOD: $TOFU_METHOD"
echo "METHOD: $METHOD"
echo "TOFU_FULL_MODEL_DIR: $TOFU_FULL_MODEL_DIR"
echo "PROMPTS_JSONL: $PROMPTS_JSONL"
echo "SUBJECTS_FILE: $SUBJECTS_FILE"
echo "NUM_FORGET_SUBJECTS: ${#SUBJECTS[@]}"
echo "RUN_OUTPUT_DIR: $RUN_OUTPUT_DIR"
echo "CONFIG_JSON: $CONFIG_JSON"

python - <<'PY'
import importlib
for pkg in ['torch', 'transformers', 'datasets', 'numpy', 'sklearn', 'huggingface_hub', 'accelerate']:
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
  --method "$METHOD"
  --model_dir "$TOFU_FULL_MODEL_DIR"
  --prompts_jsonl "$PROMPTS_JSONL"
  --capsules_dir "$EMPTY_CAPSULES_DIR"
  --output_dir "$RUN_OUTPUT_DIR"
  --subjects "${SUBJECTS[@]}"
  --config "$CONFIG_JSON"
  --skip_eval true
)

echo "Running TOFU method training/unlearning only, no SMR/EL10:"
printf ' %q' "${CMD[@]}"
echo
srun --ntasks=1 "${CMD[@]}"

echo "Finished TOFU method run at $(date)"
echo "Run output: $RUN_OUTPUT_DIR"
echo "Manifest candidates:"
find "$RUN_OUTPUT_DIR" -name unlearning_result.json -print
