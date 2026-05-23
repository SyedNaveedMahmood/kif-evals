#!/bin/bash
#SBATCH --job-name=hspopt
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/hspopt-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/hspopt-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"
PROMPTS_JSONL="$SRC/app/outputs/datasets/prompts.jsonl"
OPTOUT_MODEL="$SRC/framework/outputs/optout/optout_main_4734680/optout/model"
OUT_DIR="$PCA/analysis/outputs_hidden_space_optout_selectivity"

source "$LLMRUN/llm_env/bin/activate"

export HF_HOME="$LLMRUN/data/hf"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$LLMRUN/data/torch"
export TMPDIR="$LLMRUN/cache/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$PCA/logs" "$OUT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"

echo "[HSPACE-OPTOUT] Running from: $(pwd)"
echo "[HSPACE-OPTOUT] Opt-Out model: $OPTOUT_MODEL"
echo "[HSPACE-OPTOUT] Prompts: $PROMPTS_JSONL"
echo "[HSPACE-OPTOUT] Output directory: $OUT_DIR"

if [[ ! -d "$OPTOUT_MODEL" ]]; then
  echo "[HSPACE-OPTOUT][ERROR] Missing Opt-Out model directory: $OPTOUT_MODEL" >&2
  exit 2
fi
if [[ ! -f "$PROMPTS_JSONL" ]]; then
  echo "[HSPACE-OPTOUT][ERROR] Missing prompts file: $PROMPTS_JSONL" >&2
  exit 2
fi

python3 -m py_compile analysis/hidden_space_optout_selectivity_eval.py

python -u analysis/hidden_space_optout_selectivity_eval.py \
  --base_model_dir meta-llama/Llama-3.1-8B \
  --model_dir "$OPTOUT_MODEL" \
  --model_label "Opt-Out" \
  --prompts_jsonl "$PROMPTS_JSONL" \
  --out_dir "$OUT_DIR" \
  --max_subjects 11 \
  --layer 11 \
  --batch_size 4 \
  --load_mode 4bit \
  --max_new_tokens 64 \
  --el_steps 10 \
  --e_steps 30 \
  --max_token_ids 12

echo "[HSPACE-OPTOUT] Summary:"
python - <<'PY'
import json
from pathlib import Path
p = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_hidden_space_optout_selectivity/hidden_space_optout_selectivity_summary.json')
obj = json.loads(p.read_text())
print(json.dumps(obj.get('row'), indent=2, ensure_ascii=False))
print('\nLaTeX row:')
print(Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_hidden_space_optout_selectivity/optout_latex_row.tex').read_text())
PY
