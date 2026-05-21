#!/bin/bash
#SBATCH --job-name=advbasef
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/advbasef-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/advbasef-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"

KIF_ADAPTER_PATH="$SRC/app/outputs/sweeps/cfg_b_tau3p0_k1p6/seed23/module7/cfg_b_tau3p0_k1p6_seed23_20260515_090802"
KIF_PROMPTS_JSONL="$SRC/app/outputs/datasets/prompts.jsonl"
BASELINE_MODEL_DIR="$SRC/framework/outputs/optout/optout_main_4734680/optout/model"
BASELINE_OUTPUTS_ROOT="$SRC/framework/outputs"
OUT_DIR="$PCA/analysis/outputs_adversarial_forget_recovery"

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

echo "[ADV-BASE-FAST] Running from: $(pwd)"
echo "[ADV-BASE-FAST] Corrected baseline model: $BASELINE_MODEL_DIR"
echo "[ADV-BASE-FAST] Output directory: $OUT_DIR"

if [[ ! -d "$BASELINE_MODEL_DIR" ]]; then
  echo "[ADV-BASE-FAST][ERROR] Missing baseline model path: $BASELINE_MODEL_DIR" >&2
  exit 2
fi
if [[ ! -d "$KIF_ADAPTER_PATH" ]]; then
  echo "[ADV-BASE-FAST][ERROR] Missing KIF adapter path: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -f "$KIF_PROMPTS_JSONL" ]]; then
  echo "[ADV-BASE-FAST][ERROR] Missing prompts file: $KIF_PROMPTS_JSONL" >&2
  exit 2
fi

python3 -m py_compile analysis/fast_entity_eval_bundle.py
python3 -m py_compile analysis/adversarial_forget_recovery_eval.py
python3 -m py_compile analysis/adversarial_forget_recovery_fast_baseline.py

python -u analysis/adversarial_forget_recovery_fast_baseline.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --kif_adapter_path "$KIF_ADAPTER_PATH" \
  --baseline_model_dir "$BASELINE_MODEL_DIR" \
  --prompts_jsonl "$KIF_PROMPTS_JSONL" \
  --outputs_root "$BASELINE_OUTPUTS_ROOT" \
  --out_dir "$OUT_DIR" \
  --max_subjects 11 \
  --load_mode 4bit \
  --max_eval_rows 2000 \
  --batch_size 8 \
  --el_batch_size 16 \
  --max_new_tokens 64 \
  --el_steps 8 \
  --max_keywords 12 \
  --recovery_keyword_threshold 0.15 \
  --recovery_mass_threshold 0.02 \
  --seed 17

echo "[ADV-BASE-FAST] Summary excerpt:"
python - <<'PY'
import json
from pathlib import Path
p = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_adversarial_forget_recovery/adversarial_forget_recovery_summary.json')
obj = json.loads(p.read_text())
print(json.dumps({
  'completion': obj.get('completion'),
  'same_row_ids_across_models': obj.get('same_row_ids_across_models'),
  'overall_recovery': obj.get('paper_key_results', {}).get('overall_recovery')
}, indent=2, ensure_ascii=False))
PY
