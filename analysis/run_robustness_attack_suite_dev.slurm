#!/bin/bash
#SBATCH --job-name=robustkif
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/robust-suite-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/robust-suite-%j.err

set -euo pipefail

LLMRUN=/pfs/work9/workspace/scratch/hd_ur228-llmrun
SRC="$LLMRUN/src"
PCA="$LLMRUN/src_pca"

KIF_ADAPTER_PATH="$SRC/app/outputs/sweeps/cfg_b_tau3p0_k1p6/seed23/module7/cfg_b_tau3p0_k1p6_seed23_20260515_090802"
KIF_CAPSULES_DIR="$SRC/app/outputs/capsules"
KIF_PROMPTS_JSONL="$SRC/app/outputs/datasets/prompts.jsonl"
BASELINE_OUTPUTS_ROOT="$SRC/framework/outputs"
OUT_DIR="$PCA/analysis/outputs_robustness_attack_suite"

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

echo "[ROBUSTNESS] Running from: $(pwd)"
echo "[ROBUSTNESS] Active TOFU worktree is read-only: $SRC"
echo "[ROBUSTNESS] KIF adapter: $KIF_ADAPTER_PATH"
echo "[ROBUSTNESS] Capsules: $KIF_CAPSULES_DIR"
echo "[ROBUSTNESS] Prompts: $KIF_PROMPTS_JSONL"
echo "[ROBUSTNESS] Baseline search root: $BASELINE_OUTPUTS_ROOT"
echo "[ROBUSTNESS] Output directory: $OUT_DIR"

if [[ ! -d "$KIF_ADAPTER_PATH" ]]; then
  echo "[ROBUSTNESS][ERROR] Missing KIF adapter path: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -f "$KIF_ADAPTER_PATH/adapter_config.json" ]]; then
  echo "[ROBUSTNESS][ERROR] KIF adapter_config.json not found under: $KIF_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -d "$KIF_CAPSULES_DIR" ]]; then
  echo "[ROBUSTNESS][ERROR] Missing capsules dir: $KIF_CAPSULES_DIR" >&2
  exit 2
fi
if [[ ! -f "$KIF_PROMPTS_JSONL" ]]; then
  echo "[ROBUSTNESS][ERROR] Missing prompts file: $KIF_PROMPTS_JSONL" >&2
  exit 2
fi
if [[ ! -d "$BASELINE_OUTPUTS_ROOT" ]]; then
  echo "[ROBUSTNESS][ERROR] Missing baseline outputs root: $BASELINE_OUTPUTS_ROOT" >&2
  exit 2
fi

python3 -c "import ast; ast.parse(open('analysis/robustness_attack_suite.py').read()); print('Robustness attack suite syntax OK')"

# Fast internal smoke tests only; no model loading.
python -u analysis/robustness_attack_suite.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --out_dir "$OUT_DIR" \
  --smoke_test

# Full dev-GPU robustness run. Relearning is enabled but intentionally tiny for dev queue.
python -u analysis/robustness_attack_suite.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --kif_adapter_path "$KIF_ADAPTER_PATH" \
  --capsules_dir "$KIF_CAPSULES_DIR" \
  --prompts_jsonl "$KIF_PROMPTS_JSONL" \
  --outputs_root "$BASELINE_OUTPUTS_ROOT" \
  --baseline_prefer optout \
  --out_dir "$OUT_DIR" \
  --main_load_mode 4bit \
  --max_subjects 6 \
  --eval_prompts_per_subject 2 \
  --adv_prompts_per_subject 4 \
  --el_steps 8 \
  --max_keywords 10 \
  --max_new_tokens 40 \
  --max_length 192 \
  --precision_modes bf16,4bit \
  --quantization_modes 8bit,4bit \
  --run_relearning \
  --relearn_subjects 2 \
  --relearn_examples_per_subject 1 \
  --relearn_steps 8 \
  --relearn_lr 2e-4 \
  --seed 17

echo "[ROBUSTNESS] Outputs:"
find "$OUT_DIR" -maxdepth 2 -type f -printf '%p\n' | sort

echo "[ROBUSTNESS] Summary excerpt:"
python - <<'PY'
import json
from pathlib import Path
p = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs_robustness_attack_suite/robustness_attack_summary.json')
obj = json.loads(p.read_text())
rob = obj['robustness']
print(json.dumps({
  'subjects': obj['metadata']['subjects'],
  'kif_adversarial': rob.get('kif', {}).get('adversarial'),
  'baseline_adversarial': rob.get('baseline', {}).get('adversarial'),
  'kif_mia': rob.get('kif', {}).get('mia'),
  'relearning': rob.get('relearning_attack'),
  'precision_modes': rob.get('precision_robustness'),
  'quantization_modes': rob.get('quantization_remembering')
}, indent=2, ensure_ascii=False))
PY
