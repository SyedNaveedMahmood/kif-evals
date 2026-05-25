#!/bin/bash
#SBATCH --job-name=el10fast
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/el10fast-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/el10fast-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
APP="$SRC/app"
PCA="$LLMRUN/src_pca"
ENV_PATH="${KIF_ENV_PATH:-$LLMRUN/envs/kif-torchclean-py311-cu124}"
ENV_PY="$ENV_PATH/bin/python"

if [[ ! -x "$ENV_PY" ]]; then
  echo "[EL10FAST][ERROR] Missing Python: $ENV_PY" >&2
  exit 2
fi

export PATH="$ENV_PATH/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PCA/analysis:$PCA:$SRC:$APP/src:${PYTHONPATH:-}"
export HF_HOME="$LLMRUN/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN/data/torch"
export TMPDIR="$LLMRUN/cache/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "$PCA/logs" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
cd "$PCA"

CONFIG_NAME="${CONFIG_NAME:-cfg_b_tau3p0_k1p6}"
SEED="${SEED:-23}"
MAX_SUBJECTS="${MAX_SUBJECTS:-5}"
MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B}"
PROMPTS_JSONL="${PROMPTS_JSONL:-$APP/outputs/datasets/prompts.jsonl}"
OUT_DIR="${OUT_DIR:-$APP/outputs/sweeps/$CONFIG_NAME/seed$SEED/module8}"
EVAL_SUMMARY_JSON="${EVAL_SUMMARY_JSON:-$OUT_DIR/eval_summary.json}"
SUBJECTS="${SUBJECTS:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
export CONFIG_NAME SEED MAX_SUBJECTS MODEL_DIR PROMPTS_JSONL OUT_DIR EVAL_SUMMARY_JSON SUBJECTS LOCAL_FILES_ONLY

echo "[EL10FAST] Host: $(hostname)"
echo "[EL10FAST] Python: $ENV_PY"
echo "[EL10FAST] CONFIG_NAME=$CONFIG_NAME SEED=$SEED MAX_SUBJECTS=$MAX_SUBJECTS"
echo "[EL10FAST] MODEL_DIR=$MODEL_DIR"
echo "[EL10FAST] EVAL_SUMMARY_JSON=$EVAL_SUMMARY_JSON"
echo "[EL10FAST] PROMPTS_JSONL=$PROMPTS_JSONL"
echo "[EL10FAST] OUT_DIR=$OUT_DIR"
echo "[EL10FAST] LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"

"$ENV_PY" -m py_compile analysis/el10_token_audit_fast.py

ARGS=(
  --model_dir "$MODEL_DIR"
  --eval_summary_json "$EVAL_SUMMARY_JSON"
  --prompts_jsonl "$PROMPTS_JSONL"
  --out_dir "$OUT_DIR"
  --max_subjects "$MAX_SUBJECTS"
)

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  ARGS+=(--local_files_only)
else
  ARGS+=(--no-local_files_only)
fi

if [[ -n "$SUBJECTS" ]]; then
  ARGS+=(--subjects "$SUBJECTS")
fi

"$ENV_PY" -u -X faulthandler analysis/el10_token_audit_fast.py "${ARGS[@]}"

"$ENV_PY" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ['OUT_DIR'])
print('[EL10FAST] Written files:')
for name in ['el10_token_sets.json', 'el10_token_sets.csv', 'el10_token_sets_latex.tex']:
    p = out / name
    print(' ', p, 'exists=', p.exists(), 'size=', p.stat().st_size if p.exists() else None)

p = out / 'el10_token_sets.json'
if p.exists():
    obj = json.loads(p.read_text())
    print('[EL10FAST] Subjects:', obj.get('subjects'))
    for s, ts in obj.get('token_sets', {}).items():
        print(f"[EL10FAST] {s}: ids={ts.get('token_ids')} tokens={ts.get('token_strings')} backfill={ts.get('backfill_triggered')}")
PY
