#!/bin/bash
#SBATCH --job-name=lunar-kif
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/rep1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Submit from /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/rep1 with:
#   sbatch scripts/lunar_dual_metric_run.slurm
# Override experiment settings without editing this file, e.g.:
#   sbatch --export=ALL,SUBJECTS='Taylor Swift,Ariana Grande',LAYER='22',COEFF='2.0' scripts/lunar_dual_metric_run.slurm

REPO_ROOT="/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/rep1"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/.conda/envs/lunar-kif-llama31}"

# ---- Experiment defaults: edit these or override via sbatch --export=ALL,... ----
SUBJECTS="${SUBJECTS:-Taylor Swift}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B}"
DATA_NAME="${DATA_NAME:-kif_entities}"
LAYER="${LAYER:-22}"
COEFF="${COEFF:-2.0}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-0.01}"
MAX_PER_SUBJECT="${MAX_PER_SUBJECT:-0}"
USE_HARMFUL="${USE_HARMFUL:-0}"       # 0 = unverified Dref, 1 = harmful Dref
RUN_KIF_EVAL="${RUN_KIF_EVAL:-1}"     # keep 1 for dual-metric evaluation
USE_4BIT="${USE_4BIT:-true}"          # used by KIF eval; LUNAR itself loads bf16
SEED="${SEED:-17}"

# Default KIF output locations. The fallback supports either repo-root outputs/ or KIF-main/outputs/.
KIF_PROMPTS_JSONL="${KIF_PROMPTS_JSONL:-$REPO_ROOT/KIF-main/outputs/datasets/prompts.jsonl}"
KIF_CAPSULES_DIR="${KIF_CAPSULES_DIR:-$REPO_ROOT/KIF-main/outputs/capsules}"
if [ ! -f "$KIF_PROMPTS_JSONL" ] && [ -f "$REPO_ROOT/outputs/datasets/prompts.jsonl" ]; then
  KIF_PROMPTS_JSONL="$REPO_ROOT/outputs/datasets/prompts.jsonl"
fi
if [ ! -d "$KIF_CAPSULES_DIR" ] && [ -d "$REPO_ROOT/outputs/capsules" ]; then
  KIF_CAPSULES_DIR="$REPO_ROOT/outputs/capsules"
fi

cd "$REPO_ROOT"

module purge
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "Missing conda env: $ENV_PREFIX"
  echo "Create it first with: sbatch scripts/lunar_env_setup.slurm"
  exit 2
fi
conda activate "$ENV_PREFIX"

mkdir -p "$REPO_ROOT/.hf_cache" "$REPO_ROOT/.torch_cache" "$REPO_ROOT/.tmp" "$REPO_ROOT/logs"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.torch_cache}"
export TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-16}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SEED
export USE_4BIT

if [ ! -f "$KIF_PROMPTS_JSONL" ]; then
  echo "Missing KIF prompts JSONL: $KIF_PROMPTS_JSONL"
  echo "Override with: sbatch --export=ALL,KIF_PROMPTS_JSONL=/path/to/prompts.jsonl scripts/lunar_dual_metric_run.slurm"
  exit 3
fi
if [ ! -d "$KIF_CAPSULES_DIR" ]; then
  echo "Missing KIF capsules dir: $KIF_CAPSULES_DIR"
  echo "Override with: sbatch --export=ALL,KIF_CAPSULES_DIR=/path/to/capsules scripts/lunar_dual_metric_run.slurm"
  exit 4
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. This is OK only if your Hugging Face login token is already cached and has access to $MODEL_ID."
fi

nvidia-smi
python - <<'PY'
import torch, os
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())
print('cuda visible devices:', os.environ.get('CUDA_VISIBLE_DEVICES'))
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY

CMD=(
  python scripts/run_lunar_on_kif_then_eval.py
  --root "$REPO_ROOT"
  --kif-prompts-jsonl "$KIF_PROMPTS_JSONL"
  --kif-capsules-dir "$KIF_CAPSULES_DIR"
  --subjects "$SUBJECTS"
  --model-id "$MODEL_ID"
  --data-name "$DATA_NAME"
  --layer "$LAYER"
  --coeff "$COEFF"
  --epochs "$EPOCHS"
  --lr "$LR"
)

if [ "$MAX_PER_SUBJECT" != "0" ]; then
  CMD+=(--max-per-subject "$MAX_PER_SUBJECT")
fi
if [ "$USE_HARMFUL" = "1" ] || [ "$USE_HARMFUL" = "true" ]; then
  CMD+=(--use-harmful)
fi
if [ "$RUN_KIF_EVAL" = "1" ] || [ "$RUN_KIF_EVAL" = "true" ]; then
  CMD+=(--run-kif-eval)
fi

echo "Running: ${CMD[*]}"
srun --ntasks=1 "${CMD[@]}"

echo "Finished at $(date)"
echo "Main outputs are under:"
echo "  $REPO_ROOT/LUNAR-main/models_lunar/$DATA_NAME/llama31-8b-base/"
echo "  $REPO_ROOT/outputs/eval_lunar_dualmetric/"
