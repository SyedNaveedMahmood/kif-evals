#!/bin/bash
#SBATCH --job-name=qfid
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/qfid-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/qfid-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"
ENV_PATH="${KIF_ENV_PATH:-$LLMRUN/envs/kif-torchclean-py311-cu124}"
ENV_PY="$ENV_PATH/bin/python"

if [[ ! -x "$ENV_PY" ]]; then
  echo "[QFID][ERROR] Missing Python: $ENV_PY" >&2
  exit 2
fi

export PATH="$ENV_PATH/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PCA/analysis:$PCA:$SRC:$SRC/app/src:${PYTHONPATH:-}"
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
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "$PCA/logs" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"
cd "$PCA"

MODES="${MODES:-4bit,8bit,bf16}"
MAX_SUBJECTS="${MAX_SUBJECTS:-2}"
MAX_PROMPTS="${MAX_PROMPTS:-4}"
LAYERS="${LAYERS:-11,12,13,14}"
MODULE_NAME="${MODULE_NAME:-mlp}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUT_DIR="${OUT_DIR:-analysis/outputs_quantization_signature_fidelity}"

# For an even faster smoke test, submit with:
# sbatch --export=ALL,MODES=4bit,bf16,MAX_SUBJECTS=1,MAX_PROMPTS=3 analysis/run_quantization_signature_fidelity_dev.slurm

echo "[QFID] Host: $(hostname)"
echo "[QFID] Python=$ENV_PY"
echo "[QFID] MODES=$MODES MAX_SUBJECTS=$MAX_SUBJECTS MAX_PROMPTS=$MAX_PROMPTS LAYERS=$LAYERS MODULE=$MODULE_NAME"

"$ENV_PY" -m py_compile analysis/quantization_signature_fidelity_probe.py analysis/fast_entity_eval_bundle.py

"$ENV_PY" -u - <<'PY'
import torch, sys
print('[QFID] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cuda_version', torch.version.cuda, flush=True)
if torch.cuda.is_available():
    print('[QFID] gpu', torch.cuda.get_device_name(0), flush=True)
try:
    import bitsandbytes as bnb
    print('[QFID] bitsandbytes', getattr(bnb, '__version__', 'unknown'), flush=True)
except Exception as exc:
    print('[QFID] bitsandbytes import failed:', exc, flush=True)
print('[QFID] python', sys.version, flush=True)
PY

"$ENV_PY" -u -X faulthandler analysis/quantization_signature_fidelity_probe.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --prompts_jsonl "$SRC/app/outputs/datasets/prompts.jsonl" \
  --out_dir "$OUT_DIR" \
  --modes "$MODES" \
  --reference_mode bf16 \
  --max_subjects "$MAX_SUBJECTS" \
  --max_prompts_per_subject "$MAX_PROMPTS" \
  --layers "$LAYERS" \
  --module_name "$MODULE_NAME" \
  --batch_size "$BATCH_SIZE" \
  --max_length 256 \
  --seed 17

"$ENV_PY" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ.get('OUT_DIR', 'analysis/outputs_quantization_signature_fidelity')) / 'quantization_signature_fidelity_summary.json'
print('[QFID] Summary:', p)
if p.exists():
    print(json.dumps(json.loads(p.read_text()), indent=2, ensure_ascii=False)[:6000])
else:
    print('[QFID] summary missing')
PY
