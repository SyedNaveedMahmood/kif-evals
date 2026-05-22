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

echo "[NO-CAPSULE-LORA] Running from: $(pwd)"
echo "[NO-CAPSULE-LORA] Output directory: $OUT_DIR"
echo "[NO-CAPSULE-LORA] Prompts file: $PROMPTS_JSONL"

if [[ ! -f "$PROMPTS_JSONL" ]]; then
  echo "[NO-CAPSULE-LORA][ERROR] Missing prompts file: $PROMPTS_JSONL" >&2
  exit 2
fi

python3 -m py_compile analysis/train_no_capsule_lora_ablation.py
python3 -m py_compile analysis/evaluate_no_capsule_ablation.py

python -u analysis/train_no_capsule_lora_ablation.py \
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
