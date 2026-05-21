#!/bin/bash
#SBATCH --job-name=tofuone
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-one-%x-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-one-%x-%j.err

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
export DATA_DIR="$TOFU_ROOT/data"
export PROMPTS_JSONL="$TOFU_ROOT/method_inputs/full_forget10_retain90.jsonl"
export SUBJECTS_FILE="$TOFU_ROOT/subjects/forget10_subjects.txt"

export TOFU_FULL_MODEL_DIR="${TOFU_FULL_MODEL_DIR:-locuslab/tofu_ft_llama2-7b}"
export MODEL_FAMILY="${MODEL_FAMILY:-llama2-7b-chat}"
export TOFU_METHOD="${TOFU_METHOD:-base}"
export TOFU_LABEL="${TOFU_LABEL:-${TOFU_METHOD}_forget10_dev}"
export TOFU_RETAIN_LOGS_PATH="${TOFU_RETAIN_LOGS_PATH:-$TOFU_ROOT/retain_reference_openunlearning_llama2_retain90/retain_reference_logs.json}"
export INCLUDE_ROUGE="${INCLUDE_ROUGE:-true}"
export MAX_ROUGE_ROWS="${MAX_ROUGE_ROWS:-0}"

export RUNS_ROOT="$TOFU_ROOT/runs_dev/${TOFU_LABEL}_${SLURM_JOB_ID:-manual}"
export EVAL_DIR="$TOFU_ROOT/eval_dev/${TOFU_LABEL}_${SLURM_JOB_ID:-manual}"
export RESULTS_JSON="$TOFU_ROOT/dev_model_by_model_results.json"

export HF_HOME="$LLMRUN_ROOT/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN_ROOT/data/torch"
export TMPDIR="$LLMRUN_ROOT/cache/tmp"

mkdir -p "$LOGDIR" "$EMPTY_CAPSULES_DIR" "$RUNS_ROOT" "$EVAL_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
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

if [ ! -f "$PROMPTS_JSONL" ] || [ ! -f "$SUBJECTS_FILE" ] || [ ! -f "$DATA_DIR/forget10.jsonl" ]; then
  echo "Missing strict TOFU prepared data. Run first: sbatch framework/scripts/tofu_prepare_data.slurm" >&2
  exit 6
fi
if [ ! -f "$TOFU_RETAIN_LOGS_PATH" ]; then
  echo "Missing retain reference logs: $TOFU_RETAIN_LOGS_PATH" >&2
  echo "Run first: sbatch framework/scripts/tofu_make_retain_reference_dev.slurm" >&2
  exit 7
fi

mapfile -t SUBJECTS < "$SUBJECTS_FILE"
if [ "${#SUBJECTS[@]}" -eq 0 ]; then
  echo "No forget subjects loaded from $SUBJECTS_FILE" >&2
  exit 8
fi

echo "TOFU one-method dev job: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "TOFU_METHOD: $TOFU_METHOD"
echo "TOFU_LABEL: $TOFU_LABEL"
echo "TOFU_FULL_MODEL_DIR: $TOFU_FULL_MODEL_DIR"
echo "TOFU_RETAIN_LOGS_PATH: $TOFU_RETAIN_LOGS_PATH"
echo "RUNS_ROOT: $RUNS_ROOT"
echo "EVAL_DIR: $EVAL_DIR"
echo "RESULTS_JSON: $RESULTS_JSON"

python - <<'PY'
from pathlib import Path
root = Path('framework/methods')
p = root / 'simnpo.py'
s = p.read_text(encoding='utf-8')
s = s.replace('    "plain": "{instruction}",\n}', '    "plain": "{instruction}",\n    "llama2-7b-chat": "[INST] {instruction} [/INST]",\n    "llama2-7b": "[INST] {instruction} [/INST]",\n}')
s = s.replace('        # Base Llama KIF setting: no chat template. Use simple QA boundary.\n        question = _apply_template(f"Question: {prompt}\\nAnswer:", self.model_family)\n        full_text = question + " " + response\n', '        # TOFU/SimNPO uses Llama-2-chat prompt tags: [INST] question [/INST] answer.\n        if "llama2" in str(self.model_family).lower() or "llama-2" in str(self.model_family).lower():\n            question = _apply_template(prompt, self.model_family)\n        else:\n            question = _apply_template(f"Question: {prompt}\\nAnswer:", self.model_family)\n        full_text = question + " " + response\n')
p.write_text(s, encoding='utf-8')
p = root / 'optout.py'
s = p.read_text(encoding='utf-8')
needle = '    "plain": "{instruction}",\n'
if needle in s and '"llama2-7b-chat"' not in s:
    s = s.replace(needle, needle + '    "llama2-7b-chat": "[INST] {instruction} [/INST]",\n    "llama2-7b": "[INST] {instruction} [/INST]",\n')
p.write_text(s, encoding='utf-8')
print('[OK] Applied local TOFU Llama2-chat prompt-format patches to simnpo.py and optout.py')
PY

python - <<'PY'
import ast
for path in [
    'framework/methods/lunar.py',
    'framework/methods/reglu.py',
    'framework/methods/simnpo.py',
    'framework/methods/optout.py',
    'framework/tofu_eval/metrics.py',
    'framework/tofu_eval/evaluate_tofu.py',
]:
    ast.parse(open(path, encoding='utf-8').read())
    print('[OK syntax]', path)
PY

python -u framework/tofu_eval/evaluate_tofu.py \
  --validate_retain_logs_only \
  --retain_logs "$TOFU_RETAIN_LOGS_PATH" \
  --expected_retain_ref_count 400

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

ROUGE_FLAG=""
if [ "$INCLUDE_ROUGE" != "true" ]; then
  ROUGE_FLAG="--skip_rouge"
fi

record_json() {
  local label="$1"
  local method="$2"
  local run_dir="$3"
  local manifest="$4"
  local model_path="$5"
  local status="$6"
  local summary_path="$EVAL_DIR/tofu_summary.json"
  python - <<PY
import json, pathlib, time
p = pathlib.Path('$RESULTS_JSON')
if p.exists():
    obj = json.loads(p.read_text(encoding='utf-8'))
else:
    obj = {'split': 'forget10', 'retain_logs': '$TOFU_RETAIN_LOGS_PATH', 'methods': {}}
rec = {
    'status': '$status',
    'method': '$method',
    'label': '$label',
    'run_dir': '$run_dir',
    'eval_dir': '$EVAL_DIR',
    'manifest': '$manifest',
    'model_path': '$model_path',
    'summary_path': '$summary_path',
    'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
}
sp = pathlib.Path('$summary_path')
if sp.exists():
    summ = json.loads(sp.read_text(encoding='utf-8'))
    rec.update({
        'Forget Quality': summ.get('Forget Quality'),
        'KS Test PVal Forget': summ.get('KS Test PVal Forget'),
        'KS Test Forget': summ.get('KS Test Forget'),
        'Model Utility': summ.get('Model Utility'),
        'forget_quality_linear_FQ': summ.get('forget_quality_linear_FQ'),
        'model_utility_MU': summ.get('model_utility_MU'),
        'mu_components': summ.get('mu_components'),
    })
obj['methods']['$label'] = rec
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(rec, indent=2, ensure_ascii=False))
PY
}

eval_model() {
  local label="$1"
  local method="$2"
  local model_path="$3"
  echo "============================================================"
  echo "EVALUATING: $label"
  echo "MODEL: $model_path"
  echo "============================================================"
  set +e
  python -u framework/tofu_eval/evaluate_tofu.py \
    --model_dir "$model_path" \
    --data_dir "$DATA_DIR" \
    --split forget10 \
    --output_dir "$EVAL_DIR" \
    --retain_logs "$TOFU_RETAIN_LOGS_PATH" \
    --model_family "$MODEL_FAMILY" \
    --max_length 512 \
    --max_new_tokens 64 \
    --batch_size 32 \
    --torch_dtype bfloat16 \
    --attn_implementation sdpa \
    --use_4bit \
    --device_map_auto \
    --max_rouge_rows "$MAX_ROUGE_ROWS" \
    $ROUGE_FLAG
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    record_json "$label" "$method" "" "" "$model_path" "ok"
  else
    record_json "$label" "$method" "" "" "$model_path" "eval_failed_${rc}"
    exit "$rc"
  fi
}

run_method() {
  local label="$1"
  local method="$2"
  local config="$3"
  echo "============================================================"
  echo "RUNNING METHOD: $label"
  echo "============================================================"
  set +e
  python -u "$ENTRYPOINT" \
    --method "$method" \
    --model_dir "$TOFU_FULL_MODEL_DIR" \
    --prompts_jsonl "$PROMPTS_JSONL" \
    --capsules_dir "$EMPTY_CAPSULES_DIR" \
    --output_dir "$RUNS_ROOT" \
    --subjects "${SUBJECTS[@]}" \
    --config "$config" \
    --skip_eval true
  rc=$?
  set -e
  manifest=$(find "$RUNS_ROOT" -name unlearning_result.json | head -n 1)
  if [ "$rc" -ne 0 ] || [ -z "$manifest" ]; then
    record_json "$label" "$method" "$RUNS_ROOT" "$manifest" "" "train_failed_${rc}"
    exit "$rc"
  fi
  model_path=$(python - <<PY
import json
from pathlib import Path
p = Path('$manifest')
d = json.loads(p.read_text())
print(d.get('merged_model_dir') or d.get('model_dir') or '')
PY
)
  if [ -z "$model_path" ] || [ ! -d "$model_path" ]; then
    record_json "$label" "$method" "$RUNS_ROOT" "$manifest" "$model_path" "missing_model_path"
    exit 13
  fi
  set +e
  python -u framework/tofu_eval/evaluate_tofu.py \
    --model_dir "$model_path" \
    --data_dir "$DATA_DIR" \
    --split forget10 \
    --output_dir "$EVAL_DIR" \
    --retain_logs "$TOFU_RETAIN_LOGS_PATH" \
    --model_family "$MODEL_FAMILY" \
    --max_length 512 \
    --max_new_tokens 64 \
    --batch_size 32 \
    --torch_dtype bfloat16 \
    --attn_implementation sdpa \
    --use_4bit \
    --device_map_auto \
    --max_rouge_rows "$MAX_ROUGE_ROWS" \
    $ROUGE_FLAG
  erc=$?
  set -e
  if [ "$erc" -eq 0 ]; then
    record_json "$label" "$method" "$RUNS_ROOT" "$manifest" "$model_path" "ok"
  else
    record_json "$label" "$method" "$RUNS_ROOT" "$manifest" "$model_path" "eval_failed_${erc}"
    exit "$erc"
  fi
}

LUNAR_CFG='{"model_family":"llama2-7b-chat","lambda_reg":0.001,"auto_select_layer":true,"n_forget":300,"n_retain":300,"n_ref":128,"max_per_subject":20,"batch_size":8,"seed":42}'
REGLU_CFG='{"model_family":"llama2-7b-chat","variant":"ihl","lora_targets":"all","lora_r":32,"lora_alpha":64,"lora_dropout":0.0,"rila_beta":0.5,"rila_samples_per_split":128,"rila_cov_shrink":0.0001,"rol_lambda":0.5,"rol_rank":128,"retain_gamma":1.0,"n_forget":300,"n_retain":300,"max_per_subject":20,"batch_size":4,"gradient_accumulation_steps":8,"num_epochs":5,"learning_rate":0.0001,"weight_decay":0.01,"max_grad_norm":1.0,"max_length":256,"torch_dtype":"bfloat16","save_merged_model":true,"seed":17}'
SIMNPO_PURE_CFG='{"model_family":"llama2-7b-chat","beta":2.5,"gamma":0.0,"npo_coeff":0.1375,"grad_diff_coeff":1.0,"lr":0.00001,"num_epochs":10,"batch_size":1,"gradient_accumulation_steps":4,"weight_decay":0.01,"max_seq_len":500,"use_retain_loss":false,"optimizer_name":"paged_adamw_32bit","warmup_steps":"steps_per_epoch","torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":true,"n_forget":300,"n_retain":0,"max_per_subject":20,"trainable_modules":"all","device_map_auto":true,"save_model":true,"seed":42,"logging_steps":1}'
SIMNPO_GD_CFG='{"model_family":"llama2-7b-chat","beta":2.5,"gamma":0.0,"npo_coeff":0.1375,"grad_diff_coeff":1.0,"lr":0.00001,"num_epochs":10,"batch_size":1,"gradient_accumulation_steps":4,"weight_decay":0.01,"max_seq_len":500,"use_retain_loss":true,"optimizer_name":"paged_adamw_32bit","warmup_steps":"steps_per_epoch","torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":true,"n_forget":300,"n_retain":300,"max_per_subject":20,"trainable_modules":"all","device_map_auto":true,"save_model":true,"seed":42,"logging_steps":1}'
OPTOUT_CFG='{"method":"npo+rt+wd+ot","model_family":"llama2-7b-chat","dpo_beta":0.1,"reg_lambda":0.1,"learning_rate":0.00001,"warmup_ratio":0.0,"weight_decay":0.01,"num_epochs":3,"max_grad_norm":1.0,"max_length":512,"torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":true,"alternate_updates":true,"batch_size":4,"gradient_accumulation_steps":8,"n_forget":300,"n_retain":300,"n_world":300,"max_per_subject":20,"swd_n_projections":100,"swd_p":2,"device_map_auto":true,"trainable_modules":"all","swd_max_tensors":null,"swd_max_rows":null,"save_merged_model":true,"seed":42,"logging_steps":1}'

case "$TOFU_METHOD" in
  base)
    eval_model "$TOFU_LABEL" "base_full" "$TOFU_FULL_MODEL_DIR" ;;
  lunar)
    run_method "$TOFU_LABEL" "lunar" "$LUNAR_CFG" ;;
  reglu)
    run_method "$TOFU_LABEL" "reglu" "$REGLU_CFG" ;;
  simnpo_pure)
    run_method "$TOFU_LABEL" "simnpo" "$SIMNPO_PURE_CFG" ;;
  simnpo_graddiff)
    run_method "$TOFU_LABEL" "simnpo" "$SIMNPO_GD_CFG" ;;
  optout)
    run_method "$TOFU_LABEL" "optout" "$OPTOUT_CFG" ;;
  *)
    echo "Unknown TOFU_METHOD=$TOFU_METHOD. Use one of: base,lunar,reglu,simnpo_pure,simnpo_graddiff,optout" >&2
    exit 2 ;;
esac

echo "============================================================"
echo "DEV MODEL-BY-MODEL RESULT"
echo "============================================================"
cat "$RESULTS_JSON"
