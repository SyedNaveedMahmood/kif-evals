#!/bin/bash
#SBATCH --job-name=simnpopuredev
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=220G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/simnpo-pure-main-devfull-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/simnpo-pure-main-devfull-%j.err

set -euo pipefail

# Dev-queue full main run for pure SimNPO.
# This keeps the exact pure main-run config from simnpo_pure_main_run.slurm, but
# requests the dev GPU partition with the 30-minute dev limit.

cd /pfs/work9/workspace/scratch/hd_ur228-llmrun/src
bash framework/scripts/simnpo_pure_main_run.slurm
