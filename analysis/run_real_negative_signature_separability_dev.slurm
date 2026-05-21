#!/bin/bash
#SBATCH --job-name=rnsep
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/rnsep-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/rnsep-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"

# Keep the scan narrow. The previous broad scan over app/outputs and
# framework/outputs can discover tens of thousands of activation/cache files and
# time out before the actual analysis starts.
CAPSULE_ROOT="${CAPSULE_ROOT:-$SRC/app/outputs/capsules}"
OUT_DIR="$PCA/analysis/outputs_real_negative_signature_separability"

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

echo "[REAL-NEG-SEP] Running from: $(pwd)"
echo "[REAL-NEG-SEP] Capsule root: $CAPSULE_ROOT"
echo "[REAL-NEG-SEP] Output directory: $OUT_DIR"

if [[ ! -d "$CAPSULE_ROOT" ]]; then
  echo "[REAL-NEG-SEP][ERROR] Missing capsule root: $CAPSULE_ROOT" >&2
  echo "Set CAPSULE_ROOT=/path/to/outputs/capsules when submitting if your capsules are elsewhere." >&2
  exit 2
fi

echo "[REAL-NEG-SEP] Candidate capsule/signature file count under root:"
find "$CAPSULE_ROOT" -type f \( \
  -name '*.pkl' -o -name '*.pkl.gz' -o -name '*.pickle' -o -name '*.pickle.gz' -o \
  -name '*.pt' -o -name '*.pth' -o -name '*.json' -o -name '*.json.gz' \
\) | grep -Ei 'capsule|signature|module_c|miner' | wc -l

python3 -c "import ast; ast.parse(open('analysis/real_negative_signature_separability.py').read()); print('Real-negative separability syntax OK')"

python -u analysis/real_negative_signature_separability.py \
  --out_dir "$OUT_DIR" \
  --smoke_test

python -u analysis/real_negative_signature_separability.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --capsules_root "$CAPSULE_ROOT" \
  --prefer_path_contains capsule \
  --out_dir "$OUT_DIR" \
  --load_mode 4bit \
  --max_signatures_per_subject 1 \
  --cross_domain_per_subject 4 \
  --batch_size 4 \
  --max_length 256 \
  --seed 17 \
  --rebuild_dataset \
  --rebuild_scores

echo "[REAL-NEG-SEP] Output files:"
find "$OUT_DIR" -maxdepth 3 -type f -printf '%p\n' | sort

echo "[REAL-NEG-SEP] Summary excerpt:"
python - <<'PY'
import json
from pathlib import Path
p = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_real_negative_signature_separability/real_negative_signature_separability_summary.json')
obj = json.loads(p.read_text())
print(json.dumps({
  'dataset_stats': obj.get('dataset_stats'),
  'completion': obj.get('completion'),
  'aggregate_metrics': obj.get('aggregate_metrics')
}, indent=2, ensure_ascii=False))
PY
