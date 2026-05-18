#!/bin/bash
#SBATCH --job-name=optoutdataF
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/optout-dataset-faithful-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/optout-dataset-faithful-%j.err

set -euo pipefail

export LLMRUN_ROOT="${LLMRUN_ROOT:-/pfs/work9/workspace/scratch/hd_ur228-llmrun}"
export SRC_ROOT="${SRC_ROOT:-$LLMRUN_ROOT/src}"
export FRAMEWORK_ROOT="${FRAMEWORK_ROOT:-$SRC_ROOT/framework}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$FRAMEWORK_ROOT/outputs/optout}"
export OPTOUT_WORLD_JSON="$OUTPUT_ROOT/world/alpaca_gpt4_data_train.json"

cd "$SRC_ROOT"

bash framework/scripts/optout_prepare_world_data.sh

if [ ! -s "$OPTOUT_WORLD_JSON" ]; then
  echo "Expected upstream OPT-OUT Alpaca-GPT4 world data missing: $OPTOUT_WORLD_JSON" >&2
  exit 10
fi

# Run the regular dataset checker with OPTOUT_WORLD_JSON set, so the summary
# must report external_world_json instead of the 20-row fallback.
bash framework/scripts/optout_dataset_check_dev.slurm

python - <<'PY'
import json
from pathlib import Path
summary_path = Path('/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/outputs/optout/optout_dataset_check_summary.json')
summary = json.loads(summary_path.read_text(encoding='utf-8'))
if summary.get('status') != 'ok':
    raise SystemExit(f"Dataset status is not ok: {summary.get('status')}")
if summary.get('world_status') != 'external_world_json':
    raise SystemExit(f"World data is not faithful/external: {summary.get('world_status')}")
if int(summary.get('world_rows_available', 0)) < 1000:
    raise SystemExit(f"World data row count too small: {summary.get('world_rows_available')}")
print('Faithful OPT-OUT dataset check passed with external Alpaca-GPT4 world data.')
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
