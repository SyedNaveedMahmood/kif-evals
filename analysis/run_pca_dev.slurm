#!/bin/bash
#SBATCH --job-name=pcaact
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/pca-act-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/pca-act-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"

# Explicit KIF artifact: Config B, tau=3.0, k=1.6, seed=23.
# This is intentionally read from the active /src tree, but this job never writes to /src.
KIF_ADAPTER_PATH="$SRC/app/outputs/sweeps/cfg_b_tau3p0_k1p6/seed23/module7/cfg_b_tau3p0_k1p6_seed23_20260515_090802"

KIF_CAPSULES_DIR="$SRC/app/outputs/capsules"
KIF_PROMPTS_JSONL="$SRC/app/outputs/datasets/prompts.jsonl"
BASELINE_OUTPUTS_ROOT="$SRC/framework/outputs"
OUT_DIR="$PCA/analysis/outputs"

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

echo "[PCA] Running from: $(pwd)"
echo "[PCA] Active TOFU worktree is read-only: $SRC"
echo "[PCA] KIF adapter: $KIF_ADAPTER_PATH"
echo "[PCA] Capsules: $KIF_CAPSULES_DIR"
echo "[PCA] Prompts: $KIF_PROMPTS_JSONL"
echo "[PCA] Baseline search root: $BASELINE_OUTPUTS_ROOT"
echo "[PCA] Output directory: $OUT_DIR"

if [[ ! -d "$KIF_ADAPTER_PATH" ]]; then
  echo "[PCA][ERROR] Missing KIF adapter path: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -f "$KIF_ADAPTER_PATH/adapter_config.json" ]]; then
  echo "[PCA][ERROR] KIF adapter_config.json not found under: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -d "$KIF_CAPSULES_DIR" ]]; then
  echo "[PCA][ERROR] Missing capsules dir: $KIF_CAPSULES_DIR" >&2
  exit 2
fi
if [[ ! -f "$KIF_PROMPTS_JSONL" ]]; then
  echo "[PCA][ERROR] Missing prompts file: $KIF_PROMPTS_JSONL" >&2
  exit 2
fi
if [[ ! -d "$BASELINE_OUTPUTS_ROOT" ]]; then
  echo "[PCA][ERROR] Missing baseline outputs root: $BASELINE_OUTPUTS_ROOT" >&2
  exit 2
fi

python3 -c "import ast; ast.parse(open('analysis/pca_activation_analysis.py').read()); print('PCA script syntax OK')"

python -u analysis/pca_activation_analysis.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --kif_adapter_path "$KIF_ADAPTER_PATH" \
  --capsules_dir "$KIF_CAPSULES_DIR" \
  --prompts_jsonl "$KIF_PROMPTS_JSONL" \
  --outputs_root "$BASELINE_OUTPUTS_ROOT" \
  --baseline_prefer optout \
  --out_dir "$OUT_DIR" \
  --batch_size 4 \
  --use_4bit

echo "[PCA] Outputs:"
ls -lh "$OUT_DIR"

echo "[PCA] Centroid displacement:"
cat "$OUT_DIR/centroid_displacement.json"
