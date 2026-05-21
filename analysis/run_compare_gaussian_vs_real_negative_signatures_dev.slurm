#!/bin/bash
#SBATCH --job-name=gvsreal
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/gvsreal-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/gvsreal-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"

# Override these at sbatch time if your activation cache is elsewhere:
# sbatch --export=ALL,ACTIVATIONS_DIR=/path/to/activations,PROMPTS_FILE=/path/to/prompts.jsonl analysis/run_compare_gaussian_vs_real_negative_signatures_dev.slurm
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-$SRC/app/outputs/activations}"
PROMPTS_FILE="${PROMPTS_FILE:-$SRC/app/outputs/datasets/prompts.jsonl}"
OUT_DIR="$PCA/analysis/outputs_gaussian_vs_real_negative_signature_comparison"

source "$LLMRUN/llm_env/bin/activate"

export TMPDIR="$LLMRUN/cache/tmp"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PCA/logs" "$OUT_DIR" "$TMPDIR"

echo "[GAUSS-VS-REAL] Running from: $(pwd)"
echo "[GAUSS-VS-REAL] Activations dir: $ACTIVATIONS_DIR"
echo "[GAUSS-VS-REAL] Prompts file: $PROMPTS_FILE"
echo "[GAUSS-VS-REAL] Output dir: $OUT_DIR"

if [[ ! -f "$ACTIVATIONS_DIR/activation_index.json" ]]; then
  echo "[GAUSS-VS-REAL][ERROR] Missing activation index: $ACTIVATIONS_DIR/activation_index.json" >&2
  exit 2
fi
if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "[GAUSS-VS-REAL][ERROR] Missing prompts file: $PROMPTS_FILE" >&2
  exit 2
fi

python3 -m py_compile analysis/compare_gaussian_vs_real_negative_signatures.py

python -u analysis/compare_gaussian_vs_real_negative_signatures.py \
  --activations_dir "$ACTIVATIONS_DIR" \
  --prompts_file "$PROMPTS_FILE" \
  --out_dir "$OUT_DIR" \
  --smoke_test

python -u analysis/compare_gaussian_vs_real_negative_signatures.py \
  --activations_dir "$ACTIVATIONS_DIR" \
  --prompts_file "$PROMPTS_FILE" \
  --out_dir "$OUT_DIR" \
  --subjects kif11 \
  --layers 9,10,11,12 \
  --target_module mlp \
  --activation_strategy mean_token \
  --train_frac 0.6 \
  --seed 42

echo "[GAUSS-VS-REAL] Output files:"
find "$OUT_DIR" -maxdepth 2 -type f -printf '%p\n' | sort

echo "[GAUSS-VS-REAL] Summary excerpt:"
python - <<'PY'
import json
from pathlib import Path
p = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_gaussian_vs_real_negative_signature_comparison/gaussian_vs_real_negative_signature_summary.json')
obj = json.loads(p.read_text())
print(json.dumps({
  'completion': obj.get('completion'),
  'aggregate_metrics': obj.get('aggregate_metrics'),
  'paired_delta_summary': obj.get('paired_delta_summary')
}, indent=2, ensure_ascii=False))
PY
