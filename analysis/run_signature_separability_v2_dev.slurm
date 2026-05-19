#!/bin/bash
#SBATCH --job-name=sigsepv2
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/sig-sep-v2-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/sig-sep-v2-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"

KIF_ADAPTER_PATH="$SRC/app/outputs/sweeps/cfg_b_tau3p0_k1p6/seed23/module7/cfg_b_tau3p0_k1p6_seed23_20260515_090802"
KIF_CAPSULES_DIR="$SRC/app/outputs/capsules"
KIF_PROMPTS_JSONL="$SRC/app/outputs/datasets/prompts.jsonl"
BASELINE_OUTPUTS_ROOT="$SRC/framework/outputs"
OUT_DIR="$PCA/analysis/outputs_separability_v2"

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

echo "[SIG-SEP-V2] Running from: $(pwd)"
echo "[SIG-SEP-V2] Active TOFU worktree is read-only: $SRC"
echo "[SIG-SEP-V2] KIF adapter: $KIF_ADAPTER_PATH"
echo "[SIG-SEP-V2] Capsules: $KIF_CAPSULES_DIR"
echo "[SIG-SEP-V2] Prompts: $KIF_PROMPTS_JSONL"
echo "[SIG-SEP-V2] Baseline search root: $BASELINE_OUTPUTS_ROOT"
echo "[SIG-SEP-V2] Output directory: $OUT_DIR"

if [[ ! -d "$KIF_ADAPTER_PATH" ]]; then
  echo "[SIG-SEP-V2][ERROR] Missing KIF adapter path: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -f "$KIF_ADAPTER_PATH/adapter_config.json" ]]; then
  echo "[SIG-SEP-V2][ERROR] KIF adapter_config.json not found under: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -d "$KIF_CAPSULES_DIR" ]]; then
  echo "[SIG-SEP-V2][ERROR] Missing capsules dir: $KIF_CAPSULES_DIR" >&2
  exit 2
fi
if [[ ! -f "$KIF_PROMPTS_JSONL" ]]; then
  echo "[SIG-SEP-V2][ERROR] Missing prompts file: $KIF_PROMPTS_JSONL" >&2
  exit 2
fi
if [[ ! -d "$BASELINE_OUTPUTS_ROOT" ]]; then
  echo "[SIG-SEP-V2][ERROR] Missing baseline outputs root: $BASELINE_OUTPUTS_ROOT" >&2
  exit 2
fi

python3 -c "import ast; ast.parse(open('analysis/signature_separability_collapse_v2.py').read()); print('Signature separability v2 syntax OK')"

python -u analysis/signature_separability_collapse_v2.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --kif_adapter_path "$KIF_ADAPTER_PATH" \
  --capsules_dir "$KIF_CAPSULES_DIR" \
  --prompts_jsonl "$KIF_PROMPTS_JSONL" \
  --outputs_root "$BASELINE_OUTPUTS_ROOT" \
  --baseline_prefer optout \
  --out_dir "$OUT_DIR" \
  --batch_size 4 \
  --max_subjects 11 \
  --max_prompts_per_subject 12 \
  --max_negative_prompts 24 \
  --min_pre_d 0.5 \
  --seed 17 \
  --use_4bit

echo "[SIG-SEP-V2] Outputs:"
ls -lh "$OUT_DIR"

echo "[SIG-SEP-V2] Metrics summary:"
python - <<'PY'
import json
from pathlib import Path
p = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_separability_v2/signature_separability_summary_v2.json')
obj = json.loads(p.read_text())
print(json.dumps(obj['summary'], indent=2, ensure_ascii=False))
PY
