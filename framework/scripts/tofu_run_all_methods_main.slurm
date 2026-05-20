#!/bin/bash
#SBATCH --job-name=tofuall
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-all-methods-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-all-methods-%j.err

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
export RUNS_ROOT="$TOFU_ROOT/runs"
export EVAL_ROOT="$TOFU_ROOT/eval"
export RESULTS_JSON="$TOFU_ROOT/all_model_tofu_results.json"

# SimNPO/TOFU Llama-2 setting.
export TOFU_FULL_MODEL_DIR="${TOFU_FULL_MODEL_DIR:-locuslab/tofu_ft_llama2-7b}"
export MODEL_FAMILY="${MODEL_FAMILY:-llama2-7b-chat}"

# STRICT TOFU FQ reference handling.
# Preferred: pass a real OpenUnlearning-style retain90 TOFU_EVAL.json or this repo's retain_reference_logs.json.
# Alternative: pass a true retain90 model checkpoint to create the logs inside this job.
# The old full-model fallback is disabled unless ALLOW_RETAIN_FULL_FALLBACK=true.
export TOFU_RETAIN_LOGS_PATH="${TOFU_RETAIN_LOGS_PATH:-}"
export TOFU_RETAIN_MODEL_DIR="${TOFU_RETAIN_MODEL_DIR:-}"
export ALLOW_RETAIN_FULL_FALLBACK="${ALLOW_RETAIN_FULL_FALLBACK:-false}"
export INCLUDE_BASE_MODEL_EVAL="${INCLUDE_BASE_MODEL_EVAL:-true}"

export HF_HOME="$LLMRUN_ROOT/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN_ROOT/data/torch"
export TMPDIR="$LLMRUN_ROOT/cache/tmp"

mkdir -p "$LOGDIR" "$EMPTY_CAPSULES_DIR" "$RUNS_ROOT" "$EVAL_ROOT" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
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

if [ ! -f "$PROMPTS_JSONL" ] || [ ! -f "$SUBJECTS_FILE" ] || [ ! -f "$DATA_DIR/forget10.jsonl" ]; then
  echo "Missing strict TOFU prepared data. Run first: sbatch framework/scripts/tofu_prepare_data.slurm" >&2
  exit 6
fi

mapfile -t SUBJECTS < "$SUBJECTS_FILE"
if [ "${#SUBJECTS[@]}" -eq 0 ]; then
  echo "No forget subjects loaded from $SUBJECTS_FILE" >&2
  exit 7
fi

echo "TOFU all-methods STRICT job: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "TOFU_FULL_MODEL_DIR: $TOFU_FULL_MODEL_DIR"
echo "TOFU_RETAIN_LOGS_PATH: ${TOFU_RETAIN_LOGS_PATH:-<unset>}"
echo "TOFU_RETAIN_MODEL_DIR: ${TOFU_RETAIN_MODEL_DIR:-<unset>}"
echo "ALLOW_RETAIN_FULL_FALLBACK: $ALLOW_RETAIN_FULL_FALLBACK"
echo "MODEL_FAMILY: $MODEL_FAMILY"
echo "NUM_FORGET_SUBJECTS: ${#SUBJECTS[@]}"
echo "RESULTS_JSON: $RESULTS_JSON"

# Runtime local patch for method plugins so TOFU uses Llama-2-chat prompt format.
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
for path in ['framework/methods/simnpo.py','framework/methods/optout.py','framework/tofu_eval/evaluate_tofu.py','framework/tofu_eval/metrics.py']:
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

# Resolve and validate strict retain reference logs.
RETAIN_REF_DIR="$TOFU_ROOT/retain_reference_all_methods_${SLURM_JOB_ID:-manual}"
mkdir -p "$RETAIN_REF_DIR"

if [ -n "$TOFU_RETAIN_LOGS_PATH" ]; then
  RETAIN_LOGS="$TOFU_RETAIN_LOGS_PATH"
  REF_MODEL="<external_retain_logs>"
  REF_NOTE="explicit_retain_logs_path"
  echo "Validating explicit retain logs: $RETAIN_LOGS"
  python -u framework/tofu_eval/evaluate_tofu.py \
    --validate_retain_logs_only \
    --retain_logs "$RETAIN_LOGS"
elif [ -n "$TOFU_RETAIN_MODEL_DIR" ]; then
  if [ "$TOFU_RETAIN_MODEL_DIR" = "$TOFU_FULL_MODEL_DIR" ] && [ "$ALLOW_RETAIN_FULL_FALLBACK" != "true" ]; then
    echo "ERROR: TOFU_RETAIN_MODEL_DIR equals TOFU_FULL_MODEL_DIR. This is not a true retain90 reference." >&2
    echo "Pass a real retain90 model/logs, or set ALLOW_RETAIN_FULL_FALLBACK=true only for diagnostic runs." >&2
    exit 21
  fi
  RETAIN_LOGS="$RETAIN_REF_DIR/retain_reference_logs.json"
  REF_MODEL="$TOFU_RETAIN_MODEL_DIR"
  REF_NOTE="explicit_retain_model"
  echo "Creating retain-reference logs from explicit retain model: $REF_MODEL"
  python -u framework/tofu_eval/evaluate_tofu.py \
    --model_dir "$REF_MODEL" \
    --data_dir "$DATA_DIR" \
    --split forget10 \
    --output_dir "$RETAIN_REF_DIR" \
    --write_retain_logs \
    --model_family "$MODEL_FAMILY" \
    --max_length 512 \
    --max_new_tokens 64 \
    --torch_dtype bfloat16 \
    --attn_implementation sdpa \
    --use_4bit \
    --device_map_auto
else
  if [ "$ALLOW_RETAIN_FULL_FALLBACK" = "true" ]; then
    RETAIN_LOGS="$RETAIN_REF_DIR/retain_reference_logs.json"
    REF_MODEL="$TOFU_FULL_MODEL_DIR"
    REF_NOTE="diagnostic_fallback_full_model_reference_not_publication_grade"
    echo "WARNING: Using full model as retain reference because ALLOW_RETAIN_FULL_FALLBACK=true. Not publication-grade."
    python -u framework/tofu_eval/evaluate_tofu.py \
      --model_dir "$REF_MODEL" \
      --data_dir "$DATA_DIR" \
      --split forget10 \
      --output_dir "$RETAIN_REF_DIR" \
      --write_retain_logs \
      --model_family "$MODEL_FAMILY" \
      --max_length 512 \
      --max_new_tokens 64 \
      --torch_dtype bfloat16 \
      --attn_implementation sdpa \
      --use_4bit \
      --device_map_auto
  else
    echo "ERROR: No publication-grade TOFU retain reference provided." >&2
    echo "Set TOFU_RETAIN_LOGS_PATH=/path/to/retain90/TOFU_EVAL.json or TOFU_RETAIN_MODEL_DIR=/path/to/true_retain90_model." >&2
    echo "The old fallback to TOFU_FULL_MODEL_DIR is disabled to prevent invalid FQ runs." >&2
    exit 20
  fi
fi

python - <<PY
import json, pathlib
p=pathlib.Path('$RETAIN_REF_DIR/reference_note.json')
p.write_text(json.dumps({'reference_model':'$REF_MODEL','reference_note':'$REF_NOTE','retain_logs':'$RETAIN_LOGS'}, indent=2), encoding='utf-8')
PY

if [ -f "$RESULTS_JSON" ]; then
  cp "$RESULTS_JSON" "$RESULTS_JSON.bak_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
fi
rm -f "$RESULTS_JSON"

record_eval() {
  local label="$1"
  local method="$2"
  local model_path="$3"
  local eval_dir="$EVAL_ROOT/${label}_${SLURM_JOB_ID:-manual}"
  local summary_path="$eval_dir/tofu_summary.json"
  local status="failed"
  mkdir -p "$eval_dir"

  echo "============================================================"
  echo "EVALUATING MODEL: $label"
  echo "MODEL PATH: $model_path"
  echo "============================================================"
  set +e
  python -u framework/tofu_eval/evaluate_tofu.py \
    --model_dir "$model_path" \
    --data_dir "$DATA_DIR" \
    --split forget10 \
    --output_dir "$eval_dir" \
    --retain_logs "$RETAIN_LOGS" \
    --model_family "$MODEL_FAMILY" \
    --max_length 512 \
    --max_new_tokens 64 \
    --torch_dtype bfloat16 \
    --attn_implementation sdpa \
    --use_4bit \
    --device_map_auto
  eval_status=$?
  set -e
  if [ "$eval_status" -eq 0 ]; then status="ok"; else status="eval_failed_${eval_status}"; fi

  python - <<PY
import json, pathlib, time
results_path = pathlib.Path('$RESULTS_JSON')
if results_path.exists():
    results = json.loads(results_path.read_text(encoding='utf-8'))
else:
    results = {
        'job_id': '${SLURM_JOB_ID:-manual}',
        'base_tofu_full_model': '$TOFU_FULL_MODEL_DIR',
        'retain_reference_model': '$REF_MODEL',
        'retain_reference_note': '$REF_NOTE',
        'retain_logs': '$RETAIN_LOGS',
        'model_family': '$MODEL_FAMILY',
        'split': 'forget10',
        'methods': {}
    }
record = {
    'status': '$status',
    'method': '$method',
    'label': '$label',
    'run_dir': '',
    'eval_dir': '$eval_dir',
    'manifest': '',
    'model_path': '$model_path',
    'summary_path': '$summary_path',
    'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
}
sp = pathlib.Path('$summary_path')
if sp.exists():
    summ = json.loads(sp.read_text(encoding='utf-8'))
    record.update({
        'Forget Quality': summ.get('Forget Quality'),
        'KS Test PVal Forget': summ.get('KS Test PVal Forget'),
        'KS Test Forget': summ.get('KS Test Forget'),
        'Model Utility': summ.get('Model Utility'),
        'forget_quality_linear_FQ': summ.get('forget_quality_linear_FQ'),
        'model_utility_MU': summ.get('model_utility_MU'),
        'mu_components': summ.get('mu_components')
    })
results['methods']['$label'] = record
results_path.parent.mkdir(parents=True, exist_ok=True)
results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(record, indent=2, ensure_ascii=False))
PY
}

run_method() {
  local label="$1"
  local method="$2"
  local config="$3"
  local run_dir="$RUNS_ROOT/${label}_${SLURM_JOB_ID:-manual}"
  local eval_dir="$EVAL_ROOT/${label}_${SLURM_JOB_ID:-manual}"
  local status="failed"
  local manifest=""
  local model_path=""
  local summary_path="$eval_dir/tofu_summary.json"

  echo "============================================================"
  echo "RUNNING METHOD: $label"
  echo "============================================================"
  mkdir -p "$run_dir" "$eval_dir"

  set +e
  python -u "$ENTRYPOINT" \
    --method "$method" \
    --model_dir "$TOFU_FULL_MODEL_DIR" \
    --prompts_jsonl "$PROMPTS_JSONL" \
    --capsules_dir "$EMPTY_CAPSULES_DIR" \
    --output_dir "$run_dir" \
    --subjects "${SUBJECTS[@]}" \
    --config "$config" \
    --skip_eval true
  train_status=$?
  set -e

  manifest=$(find "$run_dir" -name unlearning_result.json | head -n 1)
  if [ "$train_status" -eq 0 ] && [ -n "$manifest" ]; then
    model_path=$(python - <<PY
import json
from pathlib import Path
p=Path('$manifest')
d=json.loads(p.read_text())
print(d.get('merged_model_dir') or d.get('model_dir') or '')
PY
)
    if [ -n "$model_path" ] && [ -d "$model_path" ]; then
      echo "Evaluating $label model at: $model_path"
      set +e
      python -u framework/tofu_eval/evaluate_tofu.py \
        --model_dir "$model_path" \
        --data_dir "$DATA_DIR" \
        --split forget10 \
        --output_dir "$eval_dir" \
        --retain_logs "$RETAIN_LOGS" \
        --model_family "$MODEL_FAMILY" \
        --max_length 512 \
        --max_new_tokens 64 \
        --torch_dtype bfloat16 \
        --attn_implementation sdpa \
        --use_4bit \
        --device_map_auto
      eval_status=$?
      set -e
      if [ "$eval_status" -eq 0 ]; then
        status="ok"
      else
        status="eval_failed_${eval_status}"
      fi
    else
      status="missing_model_path"
    fi
  else
    status="train_failed_${train_status}"
  fi

  python - <<PY
import json, pathlib, time
results_path = pathlib.Path('$RESULTS_JSON')
if results_path.exists():
    results = json.loads(results_path.read_text(encoding='utf-8'))
else:
    results = {
        'job_id': '${SLURM_JOB_ID:-manual}',
        'base_tofu_full_model': '$TOFU_FULL_MODEL_DIR',
        'retain_reference_model': '$REF_MODEL',
        'retain_reference_note': '$REF_NOTE',
        'retain_logs': '$RETAIN_LOGS',
        'model_family': '$MODEL_FAMILY',
        'split': 'forget10',
        'methods': {}
    }
record = {
    'status': '$status',
    'method': '$method',
    'label': '$label',
    'run_dir': '$run_dir',
    'eval_dir': '$eval_dir',
    'manifest': '$manifest',
    'model_path': '$model_path',
    'summary_path': '$summary_path',
    'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
}
sp = pathlib.Path('$summary_path')
if sp.exists():
    try:
        summ = json.loads(sp.read_text(encoding='utf-8'))
        record.update({
            'Forget Quality': summ.get('Forget Quality'),
            'KS Test PVal Forget': summ.get('KS Test PVal Forget'),
            'KS Test Forget': summ.get('KS Test Forget'),
            'Model Utility': summ.get('Model Utility'),
            'forget_quality_linear_FQ': summ.get('forget_quality_linear_FQ'),
            'model_utility_MU': summ.get('model_utility_MU'),
            'mu_components': summ.get('mu_components')
        })
    except Exception as e:
        record['summary_parse_error'] = str(e)
results['methods']['$label'] = record
results_path.parent.mkdir(parents=True, exist_ok=True)
results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(record, indent=2, ensure_ascii=False))
PY
}

# Method configs. All use the same TOFU full model and official Llama2-chat prompt family.
LUNAR_CFG='{"model_family":"llama2-7b-chat","lambda_reg":0.001,"auto_select_layer":true,"n_forget":300,"n_retain":300,"n_ref":128,"max_per_subject":20,"batch_size":8,"seed":42}'
REGLU_CFG='{"model_family":"llama2-7b-chat","variant":"ihl","lora_targets":"all","lora_r":32,"lora_alpha":64,"lora_dropout":0.0,"rila_beta":0.5,"rila_samples_per_split":128,"rila_cov_shrink":0.0001,"rol_lambda":0.5,"rol_rank":128,"retain_gamma":1.0,"n_forget":300,"n_retain":300,"max_per_subject":20,"batch_size":4,"gradient_accumulation_steps":8,"num_epochs":5,"learning_rate":0.0001,"weight_decay":0.01,"max_grad_norm":1.0,"max_length":256,"torch_dtype":"bfloat16","save_merged_model":true,"seed":17}'
OPTOUT_CFG='{"method":"npo+rt+wd+ot","model_family":"llama2-7b-chat","dpo_beta":0.1,"reg_lambda":0.1,"learning_rate":0.00001,"warmup_ratio":0.0,"weight_decay":0.01,"num_epochs":3,"max_grad_norm":1.0,"max_length":512,"torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":true,"alternate_updates":true,"batch_size":4,"gradient_accumulation_steps":8,"n_forget":300,"n_retain":300,"n_world":300,"max_per_subject":20,"swd_n_projections":100,"swd_p":2,"device_map_auto":true,"trainable_modules":"all","swd_max_tensors":null,"swd_max_rows":null,"save_merged_model":true,"seed":42,"logging_steps":1}'
SIMNPO_GD_CFG='{"model_family":"llama2-7b-chat","beta":2.5,"gamma":0.0,"npo_coeff":0.1375,"grad_diff_coeff":1.0,"lr":0.00001,"num_epochs":10,"batch_size":1,"gradient_accumulation_steps":4,"weight_decay":0.01,"max_seq_len":500,"use_retain_loss":true,"optimizer_name":"paged_adamw_32bit","warmup_steps":"steps_per_epoch","torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":true,"n_forget":300,"n_retain":300,"max_per_subject":20,"trainable_modules":"all","device_map_auto":true,"save_model":true,"seed":42,"logging_steps":1}'
SIMNPO_PURE_CFG='{"model_family":"llama2-7b-chat","beta":2.5,"gamma":0.0,"npo_coeff":0.1375,"grad_diff_coeff":1.0,"lr":0.00001,"num_epochs":10,"batch_size":1,"gradient_accumulation_steps":4,"weight_decay":0.01,"max_seq_len":500,"use_retain_loss":false,"optimizer_name":"paged_adamw_32bit","warmup_steps":"steps_per_epoch","torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":true,"n_forget":300,"n_retain":0,"max_per_subject":20,"trainable_modules":"all","device_map_auto":true,"save_model":true,"seed":42,"logging_steps":1}'

if [ "$INCLUDE_BASE_MODEL_EVAL" = "true" ]; then
  record_eval "base_full_forget10" "base_full" "$TOFU_FULL_MODEL_DIR"
fi

# Order: fast/light first, heavier last. Continue on failure; aggregate all statuses.
run_method "lunar_forget10" "lunar" "$LUNAR_CFG"
run_method "reglu_forget10" "reglu" "$REGLU_CFG"
run_method "simnpo_pure_forget10" "simnpo" "$SIMNPO_PURE_CFG"
run_method "simnpo_graddiff_forget10" "simnpo" "$SIMNPO_GD_CFG"
run_method "optout_forget10" "optout" "$OPTOUT_CFG"

python -u framework/tofu_eval/compare_tofu_results.py \
  --eval_root "$EVAL_ROOT" \
  --out_csv "$TOFU_ROOT/tofu_comparison_forget10.csv" \
  --out_md "$TOFU_ROOT/tofu_comparison_forget10.md" || true

echo "============================================================"
echo "ALL MODEL TOFU RESULTS"
echo "============================================================"
cat "$RESULTS_JSON"
echo
echo "CSV: $TOFU_ROOT/tofu_comparison_forget10.csv"
echo "MD : $TOFU_ROOT/tofu_comparison_forget10.md"
