#!/bin/bash
#SBATCH --job-name=xdomloc
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/xdomloc-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/xdomloc-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"

KIF_ADAPTER_PATH="$SRC/app/outputs/sweeps/cfg_b_tau3p0_k1p6/seed23/module7/cfg_b_tau3p0_k1p6_seed23_20260515_090802"
BASELINE_OUTPUTS_ROOT="$SRC/framework/outputs"
OUT_DIR="$PCA/analysis/outputs_cross_domain_locality"

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

echo "[CROSS-DOMAIN] Running from: $(pwd)"
echo "[CROSS-DOMAIN] KIF adapter: $KIF_ADAPTER_PATH"
echo "[CROSS-DOMAIN] Baseline search root: $BASELINE_OUTPUTS_ROOT"
echo "[CROSS-DOMAIN] Output directory: $OUT_DIR"

if [[ ! -d "$KIF_ADAPTER_PATH" ]]; then
  echo "[CROSS-DOMAIN][ERROR] Missing KIF adapter path: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -f "$KIF_ADAPTER_PATH/adapter_config.json" ]]; then
  echo "[CROSS-DOMAIN][ERROR] KIF adapter_config.json not found under: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -d "$BASELINE_OUTPUTS_ROOT" ]]; then
  echo "[CROSS-DOMAIN][ERROR] Missing baseline outputs root: $BASELINE_OUTPUTS_ROOT" >&2
  exit 2
fi

python3 -c "import ast; ast.parse(open('analysis/cross_domain_locality_probe.py').read()); print('Cross-domain locality syntax OK')"

python -u analysis/cross_domain_locality_probe.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --out_dir "$OUT_DIR" \
  --smoke_test

python -u analysis/cross_domain_locality_probe.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --kif_adapter_path "$KIF_ADAPTER_PATH" \
  --outputs_root "$BASELINE_OUTPUTS_ROOT" \
  --baseline_prefer optout \
  --out_dir "$OUT_DIR" \
  --models pre,kif,baseline \
  --load_mode 4bit \
  --max_eval_rows_per_model 120 \
  --max_new_tokens 48 \
  --el_steps 6 \
  --max_keywords 10 \
  --seed 17

echo "[CROSS-DOMAIN] Output files:"
find "$OUT_DIR" -maxdepth 3 -type f -printf '%p\n' | sort

echo "[CROSS-DOMAIN] Summary excerpt:"
python - <<'PY'
import json
from pathlib import Path
p = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_cross_domain_locality/cross_domain_locality_summary.json')
obj = json.loads(p.read_text())
print(json.dumps({
  'dataset_stats': obj.get('dataset_stats'),
  'completion': obj.get('completion'),
  'same_row_ids_across_models': obj.get('same_row_ids_across_models'),
  'paper_key_results': obj.get('paper_key_results')
}, indent=2, ensure_ascii=False))
PY
