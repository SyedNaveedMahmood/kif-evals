#!/bin/bash
#SBATCH --job-name=subjsuite
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/subj-suite-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/subj-suite-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"

KIF_ADAPTER_PATH="$SRC/app/outputs/sweeps/cfg_b_tau3p0_k1p6/seed23/module7/cfg_b_tau3p0_k1p6_seed23_20260515_090802"
KIF_CAPSULES_DIR="$SRC/app/outputs/capsules"
KIF_PROMPTS_JSONL="$SRC/app/outputs/datasets/prompts.jsonl"
BASELINE_OUTPUTS_ROOT="$SRC/framework/outputs"
OUT_DIR="$PCA/analysis/outputs_subject_specificity_robustness"

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

echo "[SUBJ-SUITE] Running from: $(pwd)"
echo "[SUBJ-SUITE] Active TOFU worktree is read-only: $SRC"
echo "[SUBJ-SUITE] KIF adapter: $KIF_ADAPTER_PATH"
echo "[SUBJ-SUITE] Capsules: $KIF_CAPSULES_DIR"
echo "[SUBJ-SUITE] Prompts: $KIF_PROMPTS_JSONL"
echo "[SUBJ-SUITE] Baseline search root: $BASELINE_OUTPUTS_ROOT"
echo "[SUBJ-SUITE] Output directory: $OUT_DIR"

if [[ ! -d "$KIF_ADAPTER_PATH" ]]; then
  echo "[SUBJ-SUITE][ERROR] Missing KIF adapter path: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -f "$KIF_ADAPTER_PATH/adapter_config.json" ]]; then
  echo "[SUBJ-SUITE][ERROR] KIF adapter_config.json not found under: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -d "$KIF_CAPSULES_DIR" ]]; then
  echo "[SUBJ-SUITE][ERROR] Missing capsules dir: $KIF_CAPSULES_DIR" >&2
  exit 2
fi
if [[ ! -f "$KIF_PROMPTS_JSONL" ]]; then
  echo "[SUBJ-SUITE][ERROR] Missing prompts file: $KIF_PROMPTS_JSONL" >&2
  exit 2
fi
if [[ ! -d "$BASELINE_OUTPUTS_ROOT" ]]; then
  echo "[SUBJ-SUITE][ERROR] Missing baseline outputs root: $BASELINE_OUTPUTS_ROOT" >&2
  exit 2
fi

python3 -c "import ast; ast.parse(open('analysis/subject_specificity_robustness_suite.py').read()); ast.parse(open('analysis/subject_specificity_robustness_suite_fixed.py').read()); print('Subject-specificity robustness suite syntax OK')"

# Fast internal smoke tests only; no model loading.
python -u analysis/subject_specificity_robustness_suite_fixed.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --out_dir "$OUT_DIR" \
  --smoke_test

# Full dev-GPU run.
python -u analysis/subject_specificity_robustness_suite_fixed.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --kif_adapter_path "$KIF_ADAPTER_PATH" \
  --capsules_dir "$KIF_CAPSULES_DIR" \
  --prompts_jsonl "$KIF_PROMPTS_JSONL" \
  --outputs_root "$BASELINE_OUTPUTS_ROOT" \
  --baseline_prefer optout \
  --out_dir "$OUT_DIR" \
  --batch_size 4 \
  --max_subjects 11 \
  --max_prompts_per_group 5 \
  --max_adv_prompts_per_subject 6 \
  --el_steps 12 \
  --max_keywords 10 \
  --max_new_tokens 48 \
  --seed 17 \
  --use_4bit

echo "[SUBJ-SUITE] Outputs:"
find "$OUT_DIR" -maxdepth 2 -type f -printf '%p\n' | sort

echo "[SUBJ-SUITE] Summary excerpt:"
python - <<'PY'
import json
from pathlib import Path
p = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_subject_specificity_robustness/subject_specificity_robustness_summary.json')
obj = json.loads(p.read_text())
diag = obj['diagnostics']
print(json.dumps({
  'subjects': obj['metadata']['subjects'],
  'matched_controls': obj['metadata']['matched_controls'],
  'kif_target_specific_shift': diag['1_matched_control_hidden_drift']['kif']['target_specific_shift'],
  'baseline_target_specific_shift': diag['1_matched_control_hidden_drift']['baseline']['target_specific_shift'],
  'kif_normalized_target_specific_shift': diag['1_matched_control_hidden_drift']['kif']['normalized_target_specific_shift'],
  'kif_target_subject_span_selectivity': diag['2_subject_token_span_drift']['kif']['target']['subject_span_selectivity'],
  'kif_adversarial_smr': diag['6_adversarial_robustness']['kif']['adversarial_smr'],
  'baseline_adversarial_smr': diag['6_adversarial_robustness']['baseline']['adversarial_smr'],
}, indent=2, ensure_ascii=False))
PY
