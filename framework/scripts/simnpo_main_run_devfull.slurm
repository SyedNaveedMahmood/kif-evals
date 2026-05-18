#!/bin/bash
#SBATCH --job-name=simnpodevfull
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=220G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/simnpo-main-devfull-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/simnpo-main-devfull-%j.err

set -euo pipefail

# Dev-queue full main run for SimNPO-GradDiff.
# This keeps the exact main-run config from simnpo_main_run.slurm, but requests
# the dev GPU partition with the 30-minute dev limit. Use this only to verify
# that the full-model path starts/progresses while waiting for the main queue.

cd /pfs/work9/workspace/scratch/hd_ur228-llmrun/src
bash framework/scripts/simnpo_main_run.slurm
