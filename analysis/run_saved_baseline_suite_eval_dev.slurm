#!/bin/bash
#SBATCH --job-name=savedbase
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/savedbase-%x-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/savedbase-%x-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"
ENV_PATH="${KIF_ENV_PATH:-$LLMRUN/envs/kif-torchclean-py311-cu124}"
ENV_PY="$ENV_PATH/bin/python"

if [[ ! -x "$ENV_PY" ]]; then
  echo "[SAVEDBASE][ERROR] Missing working Python: $ENV_PY" >&2
  echo "[SAVEDBASE][DEBUG] Candidate env dirs under $LLMRUN:" >&2
  find "$LLMRUN" -maxdepth 3 -path '*/bin/python' -type f 2>/dev/null | sed 's#/bin/python##' >&2 || true
  exit 2
fi

export PATH="$ENV_PATH/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

export PYTHONPATH="$PCA/analysis:$PCA:$SRC:$SRC/app/src:${PYTHONPATH:-}"
export HF_HOME="$LLMRUN/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN/data/torch"
export TMPDIR="$LLMRUN/cache/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "$PCA/logs" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
cd "$PCA"

METHOD="${METHOD:-simnpo}"
OUT_ROOT="${OUT_ROOT:-analysis/outputs_saved_baseline_suite_evals}"
MODEL_PATH_ARG=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_PATH_ARG=(--model_path "$MODEL_PATH")
fi

# Batched evaluator defaults. 0 = all remaining rows.
# This mirrors the fast Opt-Out rerun style: one model load, batched generation,
# batched EL-style mass, checkpointed JSONL outputs.
MAX_FAST_ROWS="${MAX_FAST_ROWS:-0}"
MAX_ADV_ROWS="${MAX_ADV_ROWS:-0}"
MAX_RWKU_ROWS="${MAX_RWKU_ROWS:-0}"
SOFT_TIME_LIMIT_MINUTES="${SOFT_TIME_LIMIT_MINUTES:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EL_BATCH_SIZE="${EL_BATCH_SIZE:-16}"

FAST_EL_STEPS="${FAST_EL_STEPS:-8}"
ADV_EL_STEPS="${ADV_EL_STEPS:-8}"
RWKU_EL_STEPS="${RWKU_EL_STEPS:-8}"

echo "[SAVEDBASE] Host: $(hostname)"
echo "[SAVEDBASE] METHOD=$METHOD"
echo "[SAVEDBASE] ENV_PATH=$ENV_PATH"
echo "[SAVEDBASE] Python=$ENV_PY"
echo "[SAVEDBASE] OUT_ROOT=$OUT_ROOT"
echo "[SAVEDBASE] row budgets: FAST=$MAX_FAST_ROWS ADV=$MAX_ADV_ROWS RWKU=$MAX_RWKU_ROWS"
echo "[SAVEDBASE] batch: BATCH_SIZE=$BATCH_SIZE EL_BATCH_SIZE=$EL_BATCH_SIZE"

"$ENV_PY" -m py_compile \
  analysis/evaluate_saved_baseline_suites.py \
  analysis/adversarial_forget_recovery_fast_baseline.py \
  analysis/fast_entity_eval_bundle.py \
  analysis/adversarial_forget_recovery_eval.py \
  analysis/rwku_style_entity_robustness.py

"$ENV_PY" -u - <<'PY'
import torch, sys
print('[SAVEDBASE] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cuda_version', torch.version.cuda, flush=True)
if torch.cuda.is_available():
    print('[SAVEDBASE] gpu', torch.cuda.get_device_name(0), flush=True)
print('[SAVEDBASE] python', sys.version, flush=True)
PY

"$ENV_PY" -u -X faulthandler analysis/evaluate_saved_baseline_suites.py \
  --method "$METHOD" \
  "${MODEL_PATH_ARG[@]}" \
  --model_dir meta-llama/Llama-3.1-8B \
  --outputs_root "$SRC/framework/outputs" \
  --extra_outputs_root "$PCA/framework/outputs" \
  --prompts_jsonl "$SRC/app/outputs/datasets/prompts.jsonl" \
  --out_root "$OUT_ROOT" \
  --max_subjects 11 \
  --load_mode 4bit \
  --batch_size "$BATCH_SIZE" \
  --el_batch_size "$EL_BATCH_SIZE" \
  --max_fast_rows "$MAX_FAST_ROWS" \
  --max_adv_rows "$MAX_ADV_ROWS" \
  --max_rwku_rows "$MAX_RWKU_ROWS" \
  --fast_el_steps "$FAST_EL_STEPS" \
  --adv_el_steps "$ADV_EL_STEPS" \
  --rwku_el_steps "$RWKU_EL_STEPS" \
  --soft_time_limit_minutes "$SOFT_TIME_LIMIT_MINUTES" \
  --progress_every_batches 5 \
  --seed 17

"$ENV_PY" - <<'PY'
import json, os, re
from pathlib import Path
method = re.sub(r'[^A-Za-z0-9_.-]+', '_', os.environ.get('METHOD', 'simnpo').strip().lower()).strip('_')
p = Path(os.environ.get('OUT_ROOT', 'analysis/outputs_saved_baseline_suite_evals')) / method / 'saved_baseline_suite_summary.json'
print('[SAVEDBASE] Summary:', p)
if p.exists():
    obj = json.loads(p.read_text())
    print(json.dumps({
        'method': obj.get('method'),
        'model_path': obj.get('model_path'),
        'fast_completion': obj.get('fast_completion'),
        'adversarial_completion': obj.get('adversarial_completion'),
        'rwku_completion': obj.get('rwku_completion'),
        'adversarial_paper_key_results': obj.get('adversarial_paper_key_results'),
    }, indent=2, ensure_ascii=False)[:6000])
else:
    print('[SAVEDBASE] summary missing')
PY
