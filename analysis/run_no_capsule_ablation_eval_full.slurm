#!/bin/bash
#SBATCH --job-name=nocapeval-full
#SBATCH --partition=gpu_a100_il,gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/nocapeval-full-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/nocapeval-full-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"
TRAIN_OUT="$PCA/analysis/outputs_no_capsule_lora_ablation_train"
ADAPTER="$TRAIN_OUT/adapter_final"
PROMPTS_JSONL="$SRC/app/outputs/datasets/prompts.jsonl"
EVAL_OUT="$PCA/analysis/outputs_no_capsule_lora_ablation_eval"
MAIN_ADVREC="$PCA/analysis/outputs_adversarial_forget_recovery"

ENV_PATH="${KIF_ENV_PATH:-$LLMRUN/envs/kif-torchclean-py311-cu124}"
ENV_PY="$ENV_PATH/bin/python"

if [[ ! -x "$ENV_PY" ]]; then
  echo "[NO-CAPSULE-EVAL-FULL][ERROR] Missing working Python: $ENV_PY" >&2
  echo "[NO-CAPSULE-EVAL-FULL][DEBUG] Candidate env dirs under $LLMRUN:" >&2
  find "$LLMRUN" -maxdepth 3 -path '*/bin/python' -type f 2>/dev/null | sed 's#/bin/python##' >&2 || true
  exit 2
fi

export PATH="$ENV_PATH/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

export PYTHONPATH="$PCA:$SRC:$SRC/app/src:${PYTHONPATH:-}"
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
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

mkdir -p "$PCA/logs" "$EVAL_OUT" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"

echo "[NO-CAPSULE-EVAL-FULL] Host: $(hostname)"
echo "[NO-CAPSULE-EVAL-FULL] Adapter: $ADAPTER"
echo "[NO-CAPSULE-EVAL-FULL] Eval out: $EVAL_OUT"
echo "[NO-CAPSULE-EVAL-FULL] Main advrec dir: $MAIN_ADVREC"
echo "[NO-CAPSULE-EVAL-FULL] ENV_PATH: $ENV_PATH"
echo "[NO-CAPSULE-EVAL-FULL] Python: $ENV_PY"

if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
  echo "[NO-CAPSULE-EVAL-FULL][ERROR] Missing adapter_config.json under $ADAPTER. Finish training first." >&2
  exit 2
fi
if [[ ! -d "$MAIN_ADVREC/checkpoints" ]]; then
  echo "[NO-CAPSULE-EVAL-FULL][ERROR] Missing main adversarial recovery checkpoints: $MAIN_ADVREC/checkpoints" >&2
  exit 2
fi

"$ENV_PY" -m py_compile analysis/evaluate_no_capsule_ablation.py

"$ENV_PY" -u - <<'PY'
import torch
print('[NO-CAPSULE-EVAL-FULL] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cuda_version', torch.version.cuda, flush=True)
if torch.cuda.is_available():
    print('[NO-CAPSULE-EVAL-FULL] gpu', torch.cuda.get_device_name(0), flush=True)
PY

"$ENV_PY" -u -X faulthandler analysis/evaluate_no_capsule_ablation.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --no_capsule_adapter_path "$ADAPTER" \
  --prompts_jsonl "$PROMPTS_JSONL" \
  --out_dir "$EVAL_OUT" \
  --main_advrec_dir "$MAIN_ADVREC" \
  --max_subjects 11 \
  --load_mode 4bit \
  --max_eval_rows 2000 \
  --max_new_tokens 64 \
  --el_steps 8 \
  --max_keywords 12 \
  --recovery_keyword_threshold 0.15 \
  --recovery_mass_threshold 0.02 \
  --seed 17

echo "[NO-CAPSULE-EVAL-FULL] Summary excerpt:"
"$ENV_PY" - <<'PY'
import json
from pathlib import Path
p = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_no_capsule_lora_ablation_eval/no_capsule_ablation_summary.json')
obj = json.loads(p.read_text())
print(json.dumps({
  'completion': obj.get('completion'),
  'same_row_ids_pre_kif_no_capsule': obj.get('same_row_ids_pre_kif_no_capsule'),
  'overall_recovery': obj.get('paper_key_results', {}).get('overall_recovery')
}, indent=2, ensure_ascii=False))
PY
