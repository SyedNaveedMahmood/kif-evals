#!/bin/bash
#SBATCH --job-name=tofusmoke
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-smoke-all-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-smoke-all-%j.err

set -uo pipefail

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
export SMOKE_ROOT="$TOFU_ROOT/smoke"
export SMOKE_JSON="$TOFU_ROOT/tofu_all_methods_smoke_summary.json"

# Same default TOFU full model as the real all-method job.
export TOFU_FULL_MODEL_DIR="${TOFU_FULL_MODEL_DIR:-locuslab/tofu_ft_llama2-7b}"
export MODEL_FAMILY="${MODEL_FAMILY:-llama2-7b-chat}"

export HF_HOME="$LLMRUN_ROOT/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN_ROOT/data/torch"
export TMPDIR="$LLMRUN_ROOT/cache/tmp"

mkdir -p "$LOGDIR" "$EMPTY_CAPSULES_DIR" "$SMOKE_ROOT" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
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

mapfile -t SUBJECTS_ALL < "$SUBJECTS_FILE"
if [ "${#SUBJECTS_ALL[@]}" -eq 0 ]; then
  echo "No forget subjects loaded from $SUBJECTS_FILE" >&2
  exit 7
fi
# Tiny smoke: two forget authors, one row each where method config supports max_per_subject.
SUBJECTS=("${SUBJECTS_ALL[@]:0:2}")

echo "TOFU all-method smoke job: ${SLURM_JOB_ID:-manual}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "TOFU_FULL_MODEL_DIR: $TOFU_FULL_MODEL_DIR"
echo "MODEL_FAMILY: $MODEL_FAMILY"
echo "PROMPTS_JSONL: $PROMPTS_JSONL"
echo "SMOKE_JSON: $SMOKE_JSON"
echo "SMOKE_SUBJECTS: ${SUBJECTS[*]}"

# Keep this patch in sync with tofu_run_all_methods_main.slurm. It makes the
# existing method plugins use the official Llama-2-chat TOFU prompt format.
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
import ast, importlib, json
for path in [
    'framework/methods/lunar.py',
    'framework/methods/reglu.py',
    'framework/methods/optout.py',
    'framework/methods/simnpo.py',
    'framework/tofu_eval/evaluate_tofu.py',
    'framework/tofu_eval/metrics.py',
]:
    ast.parse(open(path, encoding='utf-8').read())
    print('[OK syntax]', path)
for pkg in ['torch', 'transformers', 'datasets', 'numpy', 'sklearn', 'huggingface_hub', 'accelerate']:
    mod = importlib.import_module(pkg)
    print(f"[OK import] {pkg}: {getattr(mod, '__version__', 'version unknown')}")
try:
    import bitsandbytes as bnb
    print(f"[OK import] bitsandbytes: {getattr(bnb, '__version__', 'version unknown')}")
except Exception as e:
    print(f"[WARN import] bitsandbytes unavailable: {e}")
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

run_smoke() {
  local label="$1"
  local method="$2"
  local config="$3"
  local run_dir="$SMOKE_ROOT/${label}_${SLURM_JOB_ID:-manual}"
  local status="failed"
  local manifest=""
  mkdir -p "$run_dir"

  echo "============================================================"
  echo "SMOKE METHOD: $label"
  echo "============================================================"
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
  rc=$?
  set -e

  manifest=$(find "$run_dir" -name unlearning_result.json | head -n 1)
  if [ "$rc" -eq 0 ] && [ -n "$manifest" ]; then
    status="ok"
  else
    status="failed_${rc}"
  fi

  python - <<PY
import json, pathlib, time
p = pathlib.Path('$SMOKE_JSON')
if p.exists():
    obj = json.loads(p.read_text(encoding='utf-8'))
else:
    obj = {
        'job_id': '${SLURM_JOB_ID:-manual}',
        'base_tofu_full_model': '$TOFU_FULL_MODEL_DIR',
        'model_family': '$MODEL_FAMILY',
        'subjects': ${SUBJECTS[@]+None},
        'methods': {}
    }
obj['subjects'] = ${#SUBJECTS[@]} and [${SUBJECTS[@]/#/'"'}]
obj['methods']['$label'] = {
    'status': '$status',
    'return_code': $rc,
    'method': '$method',
    'run_dir': '$run_dir',
    'manifest': '$manifest',
    'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
}
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(obj['methods']['$label'], indent=2, ensure_ascii=False))
PY
}

# Tiny configs: validate model access, loaders, objectives, optimizer path, prompt format,
# and save/manifest code without doing the expensive final training/evaluation.
LUNAR_CFG='{"model_family":"llama2-7b-chat","lambda_reg":0.001,"auto_select_layer":true,"n_forget":2,"n_retain":2,"n_ref":2,"max_per_subject":1,"batch_size":1,"seed":42}'
REGLU_CFG='{"model_family":"llama2-7b-chat","variant":"ihl","lora_targets":"qv","lora_r":4,"lora_alpha":8,"lora_dropout":0.0,"rila_beta":0.5,"rila_samples_per_split":2,"rila_cov_shrink":0.0001,"rol_lambda":0.5,"rol_rank":2,"retain_gamma":1.0,"n_forget":2,"n_retain":2,"max_per_subject":1,"batch_size":1,"gradient_accumulation_steps":1,"num_epochs":1,"learning_rate":0.0001,"weight_decay":0.01,"max_grad_norm":1.0,"max_length":128,"torch_dtype":"bfloat16","save_merged_model":false,"seed":17}'
SIMNPO_PURE_CFG='{"model_family":"llama2-7b-chat","beta":2.5,"gamma":0.0,"npo_coeff":0.1375,"grad_diff_coeff":1.0,"lr":0.00001,"num_epochs":1,"batch_size":1,"gradient_accumulation_steps":1,"weight_decay":0.01,"max_seq_len":128,"use_retain_loss":false,"optimizer_name":"adamw","warmup_steps":"steps_per_epoch","torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":false,"n_forget":2,"n_retain":0,"max_per_subject":1,"trainable_modules":"lm_head","device_map_auto":true,"save_model":false,"seed":42,"logging_steps":1}'
SIMNPO_GD_CFG='{"model_family":"llama2-7b-chat","beta":2.5,"gamma":0.0,"npo_coeff":0.1375,"grad_diff_coeff":1.0,"lr":0.00001,"num_epochs":1,"batch_size":1,"gradient_accumulation_steps":1,"weight_decay":0.01,"max_seq_len":128,"use_retain_loss":true,"optimizer_name":"adamw","warmup_steps":"steps_per_epoch","torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":false,"n_forget":2,"n_retain":2,"max_per_subject":1,"trainable_modules":"lm_head","device_map_auto":true,"save_model":false,"seed":42,"logging_steps":1}'
OPTOUT_CFG='{"method":"npo+rt+wd+ot","model_family":"llama2-7b-chat","dpo_beta":0.1,"reg_lambda":0.1,"learning_rate":0.00001,"warmup_ratio":0.0,"weight_decay":0.01,"num_epochs":1,"max_grad_norm":1.0,"max_length":128,"torch_dtype":"bfloat16","attn_implementation":"sdpa","use_gradient_checkpointing":false,"alternate_updates":true,"batch_size":1,"gradient_accumulation_steps":1,"n_forget":2,"n_retain":2,"n_world":2,"max_per_subject":1,"swd_n_projections":2,"swd_p":2,"device_map_auto":true,"trainable_modules":"lm_head","swd_max_tensors":1,"swd_max_rows":2,"save_merged_model":false,"seed":42,"logging_steps":1}'

run_smoke "lunar_smoke" "lunar" "$LUNAR_CFG"
run_smoke "reglu_smoke" "reglu" "$REGLU_CFG"
run_smoke "simnpo_pure_smoke" "simnpo" "$SIMNPO_PURE_CFG"
run_smoke "simnpo_graddiff_smoke" "simnpo" "$SIMNPO_GD_CFG"
run_smoke "optout_smoke" "optout" "$OPTOUT_CFG"

python - <<'PY'
import json, pathlib, sys
p = pathlib.Path('framework/outputs/tofu/tofu_all_methods_smoke_summary.json')
obj = json.loads(p.read_text(encoding='utf-8'))
failed = {k:v for k,v in obj.get('methods', {}).items() if v.get('status') != 'ok'}
print('============================================================')
print('TOFU SMOKE SUMMARY')
print('============================================================')
print(json.dumps(obj, indent=2, ensure_ascii=False))
if failed:
    print('FAILED METHODS:', ', '.join(failed), file=sys.stderr)
    sys.exit(1)
print('All TOFU smoke methods passed. Safe to submit tofu_run_all_methods_main.slurm')
PY
