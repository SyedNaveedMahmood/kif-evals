#!/bin/bash
# Submit the complete 3-config × 3-seed tau/k sweep.
# Usage:
#   bash scripts/submit_sweep_3seed.sh
# Optional env overrides:
#   KIF_CONDA_ENV=kif bash scripts/submit_sweep_3seed.sh
#   MODEL_DIR=/path/to/model bash scripts/submit_sweep_3seed.sh
set -euo pipefail
mkdir -p logs

submit_one () {
  local CONFIG_NAME=$1
  local TAU=$2
  local K=$3
  echo "[SUBMIT] ${CONFIG_NAME}: tau=${TAU}, k=${K}"
  sbatch --export=ALL,CONFIG_NAME="${CONFIG_NAME}",TAU="${TAU}",K="${K}" \
    scripts/slurm_sweep_one_config_3seed.sh
}

submit_one cfg_a 3.5 1.0
submit_one cfg_b      3.0 1.6
submit_one cfg_c   2.5 2.5
