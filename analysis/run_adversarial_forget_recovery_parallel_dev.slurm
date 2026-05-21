#!/bin/bash
#SBATCH --job-name=advrecp
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/advrecp-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/advrecp-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"

KIF_ADAPTER_PATH="$SRC/app/outputs/sweeps/cfg_b_tau3p0_k1p6/seed23/module7/cfg_b_tau3p0_k1p6_seed23_20260515_090802"
KIF_PROMPTS_JSONL="$SRC/app/outputs/datasets/prompts.jsonl"
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

echo "[ADV-RECOVERY-PAR] Running from: $(pwd)"
echo "[ADV-RECOVERY-PAR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[ADV-RECOVERY-PAR] KIF adapter: $KIF_ADAPTER_PATH"
echo "[ADV-RECOVERY-PAR] Output directory: $OUT_DIR"

if [[ ! -d "$KIF_ADAPTER_PATH" ]]; then
  echo "[ADV-RECOVERY-PAR][ERROR] Missing KIF adapter path: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -f "$KIF_ADAPTER_PATH/adapter_config.json" ]]; then
  echo "[ADV-RECOVERY-PAR][ERROR] KIF adapter_config.json not found under: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -f "$KIF_PROMPTS_JSONL" ]]; then
  echo "[ADV-RECOVERY-PAR][ERROR] Missing prompts file: $KIF_PROMPTS_JSONL" >&2
  exit 2
fi
if [[ ! -d "$BASELINE_OUTPUTS_ROOT" ]]; then
  echo "[ADV-RECOVERY-PAR][ERROR] Missing baseline outputs root: $BASELINE_OUTPUTS_ROOT" >&2
  exit 2
fi

python analysis/patch_advrec_skip_summary.py
python3 -m py_compile analysis/fast_entity_eval_bundle.py
python3 -m py_compile analysis/adversarial_forget_recovery_eval.py

# Build/smoke dataset once.
python -u analysis/adversarial_forget_recovery_eval.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --out_dir "$OUT_DIR" \
  --smoke_test

COMMON_ARGS=(
  --model_dir meta-llama/Llama-3.1-8B
  --kif_adapter_path "$KIF_ADAPTER_PATH"
  --prompts_jsonl "$KIF_PROMPTS_JSONL"
  --outputs_root "$BASELINE_OUTPUTS_ROOT"
  --baseline_prefer optout
  --out_dir "$OUT_DIR"
  --load_mode 4bit
  --max_subjects 11
  --max_eval_rows_per_model 150
  --max_new_tokens 64
  --el_steps 8
  --max_keywords 12
  --recovery_keyword_threshold 0.15
  --recovery_mass_threshold 0.02
  --seed 17
  --skip_summary
)

run_one() {
  local gpu="$1"
  local model="$2"
  local logp="$PCA/logs/advrecp-${SLURM_JOB_ID}-${model}.out"
  echo "[ADV-RECOVERY-PAR] Starting $model on local GPU $gpu -> $logp"
  CUDA_VISIBLE_DEVICES="$gpu" python -u analysis/adversarial_forget_recovery_eval.py \
    "${COMMON_ARGS[@]}" \
    --models "$model" > "$logp" 2>&1
  echo "[ADV-RECOVERY-PAR] Finished $model"
}

run_one 0 pre &
PID_PRE=$!
run_one 1 kif &
PID_KIF=$!
run_one 2 baseline &
PID_BASE=$!

wait "$PID_PRE"
wait "$PID_KIF"
wait "$PID_BASE"

# Single final summary write after all three model workers finish.
python -u analysis/adversarial_forget_recovery_eval.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --kif_adapter_path "$KIF_ADAPTER_PATH" \
  --prompts_jsonl "$KIF_PROMPTS_JSONL" \
  --outputs_root "$BASELINE_OUTPUTS_ROOT" \
  --baseline_prefer optout \
  --out_dir "$OUT_DIR" \
  --models pre,kif,baseline \
  --load_mode 4bit \
  --max_subjects 11 \
  --max_eval_rows_per_model 0 \
  --max_new_tokens 64 \
  --el_steps 8 \
  --max_keywords 12 \
  --recovery_keyword_threshold 0.15 \
  --recovery_mass_threshold 0.02 \
  --seed 17

echo "[ADV-RECOVERY-PAR] Summary excerpt:"
python - <<'PY'
import json
from pathlib import Path
p = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_adversarial_forget_recovery/adversarial_forget_recovery_summary.json')
obj = json.loads(p.read_text())
print(json.dumps({
  'dataset_stats': obj.get('dataset_stats'),
  'completion': obj.get('completion'),
  'same_row_ids_across_models': obj.get('same_row_ids_across_models'),
  'paper_key_results': obj.get('paper_key_results')
}, indent=2, ensure_ascii=False))
PY
