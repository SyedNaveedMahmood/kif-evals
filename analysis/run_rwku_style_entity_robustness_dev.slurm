#!/bin/bash
#SBATCH --job-name=rwkustyle
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/rwku-style-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/rwku-style-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"

KIF_ADAPTER_PATH="$SRC/app/outputs/sweeps/cfg_b_tau3p0_k1p6/seed23/module7/cfg_b_tau3p0_k1p6_seed23_20260515_090802"
KIF_PROMPTS_JSONL="$SRC/app/outputs/datasets/prompts.jsonl"
BASELINE_OUTPUTS_ROOT="$SRC/framework/outputs"
OUT_DIR="$PCA/analysis/outputs_rwku_style_entity_robustness"

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

echo "[RWKU-STYLE] Running from: $(pwd)"
echo "[RWKU-STYLE] KIF adapter: $KIF_ADAPTER_PATH"
echo "[RWKU-STYLE] Prompts: $KIF_PROMPTS_JSONL"
echo "[RWKU-STYLE] Baseline search root: $BASELINE_OUTPUTS_ROOT"
echo "[RWKU-STYLE] Output directory: $OUT_DIR"

if [[ ! -d "$KIF_ADAPTER_PATH" ]]; then
  echo "[RWKU-STYLE][ERROR] Missing KIF adapter path: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -f "$KIF_ADAPTER_PATH/adapter_config.json" ]]; then
  echo "[RWKU-STYLE][ERROR] KIF adapter_config.json not found under: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -f "$KIF_PROMPTS_JSONL" ]]; then
  echo "[RWKU-STYLE][ERROR] Missing prompts file: $KIF_PROMPTS_JSONL" >&2
  exit 2
fi
if [[ ! -d "$BASELINE_OUTPUTS_ROOT" ]]; then
  echo "[RWKU-STYLE][ERROR] Missing baseline outputs root: $BASELINE_OUTPUTS_ROOT" >&2
  exit 2
fi

python3 -c "import ast; ast.parse(open('analysis/rwku_style_entity_robustness.py').read()); print('RWKU-style robustness syntax OK')"

python -u analysis/rwku_style_entity_robustness.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --out_dir "$OUT_DIR" \
  --smoke_test

# Checkpointed dev run. Re-submit this same job until completion shows eval_remaining=0.
# These caps are safe after the 120-row run completed in about 22 minutes.
python -u analysis/rwku_style_entity_robustness.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --kif_adapter_path "$KIF_ADAPTER_PATH" \
  --prompts_jsonl "$KIF_PROMPTS_JSONL" \
  --outputs_root "$BASELINE_OUTPUTS_ROOT" \
  --baseline_prefer optout \
  --out_dir "$OUT_DIR" \
  --models pre,kif,baseline \
  --load_mode 4bit \
  --max_subjects 11 \
  --rows_per_family_cap 0 \
  --max_eval_rows_per_model 150 \
  --max_teacher_rows 220 \
  --max_mia_rows_per_model 220 \
  --mia_max_pairs 500 \
  --max_new_tokens 48 \
  --el_steps 8 \
  --max_keywords 10 \
  --max_length 192 \
  --seed 17

echo "[RWKU-STYLE] Output files:"
find "$OUT_DIR" -maxdepth 3 -type f -printf '%p\n' | sort

echo "[RWKU-STYLE] Summary excerpt:"
python - <<'PY'
import json
from pathlib import Path
p = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_rwku_style_entity_robustness/rwku_style_entity_robustness_summary.json')
obj = json.loads(p.read_text())
print(json.dumps({
  'dataset_stats': obj.get('dataset_stats'),
  'completion': obj.get('completion'),
  'kif_forget': obj.get('evaluation_summary', {}).get('kif', {}).get('by_control_type', {}).get('forget'),
  'baseline_forget': obj.get('evaluation_summary', {}).get('baseline', {}).get('by_control_type', {}).get('forget'),
  'pre_forget': obj.get('evaluation_summary', {}).get('pre', {}).get('by_control_type', {}).get('forget'),
  'kif_mia': obj.get('mia_summary', {}).get('kif'),
  'baseline_mia': obj.get('mia_summary', {}).get('baseline')
}, indent=2, ensure_ascii=False))
PY
