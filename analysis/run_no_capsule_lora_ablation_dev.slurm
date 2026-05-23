#!/bin/bash
#SBATCH --job-name=nocaplora
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/nocaplora-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/nocaplora-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"
OUT_DIR="$PCA/analysis/outputs_no_capsule_lora_ablation_train"
PROMPTS_JSONL="$SRC/app/outputs/datasets/prompts.jsonl"

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
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p "$PCA/logs" "$OUT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"

echo "[NO-CAPSULE-LORA] Running from: $(pwd)"
echo "[NO-CAPSULE-LORA] Output directory: $OUT_DIR"
echo "[NO-CAPSULE-LORA] Prompts file: $PROMPTS_JSONL"
echo "[NO-CAPSULE-LORA] Python: $ENV_PY"

if [[ ! -f "$PROMPTS_JSONL" ]]; then
  echo "[NO-CAPSULE-LORA][ERROR] Missing prompts file: $PROMPTS_JSONL" >&2
  exit 2
fi

"$ENV_PY" -m py_compile analysis/train_no_capsule_lora_ablation.py
"$ENV_PY" -m py_compile analysis/evaluate_no_capsule_ablation.py

"$ENV_PY" -u - <<'PY'
import torch
print('[NO-CAPSULE-LORA] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cuda_version', torch.version.cuda, flush=True)
if torch.cuda.is_available():
    print('[NO-CAPSULE-LORA] gpu', torch.cuda.get_device_name(0), flush=True)
PY

"$ENV_PY" -u -X faulthandler analysis/train_no_capsule_lora_ablation.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --prompts_jsonl "$PROMPTS_JSONL" \
  --out_dir "$OUT_DIR" \
  --max_subjects 11 \
  --variants_per_subject 40 \
  --min_subject_pairs 512 \
  --min_anchor_pairs 256 \
  --max_steps 200 \
  --batch_size 1 \
  --lr 5e-6 \
  --lora_r 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  --dpo_beta 0.02 \
  --unlikelihood_weight 0.03 \
  --name_ul_weight 0.02 \
  --kl_lambda 0.03 \
  --save_every 40 \
  --seed 17

echo "[NO-CAPSULE-LORA] Train state:"
cat "$OUT_DIR/train_state.json" || true
