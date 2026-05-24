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

# Per-run row budgets. These are intentionally checkpointed; re-submit the same
# job until completion. Use 0 for all remaining if the dev partition is behaving well.
MAX_FAST_ROWS="${MAX_FAST_ROWS:-140}"
MAX_ADV_ROWS="${MAX_ADV_ROWS:-140}"
MAX_RWKU_ROWS="${MAX_RWKU_ROWS:-80}"
SOFT_TIME_LIMIT_MINUTES="${SOFT_TIME_LIMIT_MINUTES:-26}"

# Faster defaults for dev-GPU iteration. The same thresholds as the previous KIF/Opt-Out
# adversarial audit are preserved; this only controls row budget per submission.
FAST_EL_STEPS="${FAST_EL_STEPS:-8}"
ADV_EL_STEPS="${ADV_EL_STEPS:-8}"
RWKU_EL_STEPS="${RWKU_EL_STEPS:-8}"

# RWKU MIA is off by default for this paper-facing entity robustness table.
MAX_RWKU_MIA_ROWS="${MAX_RWKU_MIA_ROWS:-0}"

echo "[SAVEDBASE] Host: $(hostname)"
echo "[SAVEDBASE] METHOD=$METHOD"
echo "[SAVEDBASE] ENV_PATH=$ENV_PATH"
echo "[SAVEDBASE] Python=$ENV_PY"
echo "[SAVEDBASE] OUT_ROOT=$OUT_ROOT"
echo "[SAVEDBASE] row budgets: FAST=$MAX_FAST_ROWS ADV=$MAX_ADV_ROWS RWKU=$MAX_RWKU_ROWS"

"$ENV_PY" -m py_compile analysis/evaluate_saved_baseline_suites.py analysis/fast_entity_eval_bundle.py analysis/adversarial_forget_recovery_eval.py analysis/rwku_style_entity_robustness.py

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
  --max_fast_rows "$MAX_FAST_ROWS" \
  --max_adv_rows "$MAX_ADV_ROWS" \
  --max_rwku_rows "$MAX_RWKU_ROWS" \
  --fast_el_steps "$FAST_EL_STEPS" \
  --adv_el_steps "$ADV_EL_STEPS" \
  --rwku_el_steps "$RWKU_EL_STEPS" \
  --max_rwku_mia_rows "$MAX_RWKU_MIA_ROWS" \
  --soft_time_limit_minutes "$SOFT_TIME_LIMIT_MINUTES" \
  --progress_every 20 \
  --seed 17

"$ENV_PY" - <<'PY'
import json, os
from pathlib import Path
method = os.environ.get('METHOD', 'simnpo').lower().replace('-', '_')
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
