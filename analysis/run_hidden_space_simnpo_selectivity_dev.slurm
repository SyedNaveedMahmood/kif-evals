#!/bin/bash
#SBATCH --job-name=hspsimnpo
#SBATCH --partition=dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/hspsimnpo-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/hspsimnpo-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"
PROMPTS_JSONL="$SRC/app/outputs/datasets/prompts.jsonl"
OUT_DIR="$PCA/analysis/outputs_hidden_space_simnpo_selectivity"
ENV_PATH="$LLMRUN/envs/kif-torchclean-py311-cu124"
ENV_PY="$ENV_PATH/bin/python"

if [[ ! -x "$ENV_PY" ]]; then
  echo "[ERROR] Missing working Python: $ENV_PY" >&2
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
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "$PCA/logs" "$OUT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"

SIMNPO_MODEL_DIR="${SIMNPO_MODEL_DIR:-${MODEL_DIR:-}}"
if [[ -z "$SIMNPO_MODEL_DIR" ]]; then
  SIMNPO_MODEL_DIR="$($ENV_PY - <<'PY'
from pathlib import Path
roots = [
    Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/outputs/simnpo_pure'),
    Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/outputs/simnpo'),
]
suffixes = ['simnpo/model', 'model', 'simnpo/adapter', 'adapter']
cands = []
for root in roots:
    if not root.exists():
        continue
    for suffix in suffixes:
        for p in root.glob(f'*/{suffix}'):
            if p.exists():
                cands.append((p.stat().st_mtime, p.resolve()))
if not cands:
    raise SystemExit(1)
print(sorted(cands, reverse=True)[0][1])
PY
)" || {
    echo "[HSPACE-SIMNPO][ERROR] Could not auto-discover SimNPO model. Set SIMNPO_MODEL_DIR=/path/to/model_or_adapter." >&2
    exit 3
  }
fi

echo "[$(date -Is)] [HSPACE-SIMNPO] Running from: $(pwd)"
echo "[$(date -Is)] [HSPACE-SIMNPO] Node: ${SLURMD_NODENAME:-unknown}"
echo "[$(date -Is)] [HSPACE-SIMNPO] Model: $SIMNPO_MODEL_DIR"
echo "[$(date -Is)] [HSPACE-SIMNPO] Prompts: $PROMPTS_JSONL"
echo "[$(date -Is)] [HSPACE-SIMNPO] Output directory: $OUT_DIR"
echo "[$(date -Is)] [HSPACE-SIMNPO] Python: $ENV_PY"

[[ -d "$SIMNPO_MODEL_DIR" ]] || { echo "[HSPACE-SIMNPO][ERROR] Missing model directory: $SIMNPO_MODEL_DIR" >&2; exit 4; }
[[ -f "$PROMPTS_JSONL" ]] || { echo "[HSPACE-SIMNPO][ERROR] Missing prompts file: $PROMPTS_JSONL" >&2; exit 5; }

"$ENV_PY" -m py_compile analysis/hidden_space_optout_selectivity_eval.py
"$ENV_PY" -u - <<'PY'
import torch
print('[HSPACE-SIMNPO] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cuda_version', torch.version.cuda, flush=True)
if torch.cuda.is_available(): print('[HSPACE-SIMNPO] gpu', torch.cuda.get_device_name(0), flush=True)
PY

"$ENV_PY" -u -X faulthandler analysis/hidden_space_optout_selectivity_eval.py \
  --base_model_dir meta-llama/Llama-3.1-8B \
  --model_dir "$SIMNPO_MODEL_DIR" \
  --model_label "SimNPO" \
  --prompts_jsonl "$PROMPTS_JSONL" \
  --out_dir "$OUT_DIR" \
  --max_subjects 11 \
  --layer 11 \
  --batch_size 8 \
  --mass_batch_size 8 \
  --load_mode bf16 \
  --max_new_tokens 64 \
  --steps 30 \
  --max_token_ids 12

"$ENV_PY" - <<PY
import json
from pathlib import Path
out = Path('$OUT_DIR')
obj = json.loads((out / 'hidden_space_optout_selectivity_summary.json').read_text())
print(json.dumps(obj.get('row'), indent=2, ensure_ascii=False))
print('\nLaTeX row:')
print((out / 'optout_latex_row.tex').read_text())
PY
