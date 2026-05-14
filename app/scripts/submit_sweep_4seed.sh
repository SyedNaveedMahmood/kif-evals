#!/bin/bash
# Submit the complete 3-config × 4-seed tau/k sweep.
# This uses all 4 H100 GPUs per config job.
set -euo pipefail
mkdir -p logs

submit_one () {
  local CONFIG_NAME=$1
  local TAU=$2
  local K=$3
  echo "[SUBMIT] ${CONFIG_NAME}: tau=${TAU}, k=${K}"
  sbatch --export=ALL,CONFIG_NAME="${CONFIG_NAME}",TAU="${TAU}",K="${K}" \
    scripts/slurm_sweep_one_config_4seed.sh
}

submit_one conservative_tau3p5_k1p0 3.5 1.0
submit_one default_tau3p0_k1p6      3.0 1.6
submit_one aggressive_tau2p5_k2p5   2.5 2.5
