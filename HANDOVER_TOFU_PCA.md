# KIF Baseline, TOFU, and PCA Visualization Handover

This handover summarizes the current state of the `kif-evals` repo, the cluster workflow, the TOFU FQ/MU benchmark, and the PCA activation-space visualization. It is written so a new ChatGPT chat can immediately continue without needing the full previous conversation.

---

## 1. Repository and cluster paths

Repository:

```text
https://github.com/SyedNaveedMahmood/kif-evals
```

Main cluster worktree:

```text
/pfs/work9/workspace/scratch/hd_ur228-llmrun/src
```

Separate visualization worktree, used to avoid mutating the active TOFU job worktree:

```text
/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
```

Python environment:

```bash
source /pfs/work9/workspace/scratch/hd_ur228-llmrun/llm_env/bin/activate
```

Important rule:

```text
Do not run git pull or otherwise modify /pfs/work9/workspace/scratch/hd_ur228-llmrun/src while the TOFU all-method job is running, because that job executes the method files sequentially from the same worktree.
```

Use `src_pca` for visualization work while the TOFU job is active.

---

## 2. KIF evaluation framework status

The repo implements a plug-and-play unlearning evaluation framework. Each method reads a shared `prompts.jsonl`, saves a model or adapter through a common `UnlearningResult`, and then the evaluator can load that output.

Core files:

```text
framework/orchestrate.py
framework/methods/base.py
framework/eval/module8_eval.py
```

Integrated methods:

```text
LUNAR
ReGLU
OPT-OUT
SimNPO pure
SimNPO-GradDiff
```

For the KIF musician benchmark:

```text
Base model: meta-llama/Llama-3.1-8B
Prompt file: outputs/datasets/prompts.jsonl
Capsules: outputs/capsules
Forget subjects: 11 musician subjects
Retain/control subject where needed: Adele
Evaluator: Module 8, using SMR, EL10, and utility drift
```

Comparable evaluation principle:

```text
Every method reads the same data, unlearns the same targets, saves a model/adapter, and is evaluated by the same evaluator. This gives a method-level apples-to-apples comparison, not a bit-exact reproduction of each paper's original setting.
```

---

## 3. Method faithfulness summary for KIF benchmark

### SimNPO / SimNPO-GradDiff

Status: most faithful.

Key settings:

```text
beta = 2.5
gamma = 0.0
lr = 1e-5
num_epochs = 10
batch_size = 1
gradient_accumulation_steps = 4
weight_decay = 0.01
max_seq_len = 500
optimizer_name = paged_adamw_32bit
```

Pure SimNPO:

```text
use_retain_loss = false
```

SimNPO-GradDiff:

```text
use_retain_loss = true
npo_coeff = 0.1375
grad_diff_coeff = 1.0
```

### ReGLU

Status: faithful after patches.

Important fixes already applied:

```text
RILA pooling: sum(dim=1), not mean(dim=1)
Covariance: explicit torch.linalg.eigh
Shrinkage: eps * I added to covariance, no activation noise
IHL: non-in-place scatter to avoid autograd mutation
LoRA main setting: targets=all, r=32, alpha=64
Variant: ihl
```

### LUNAR

Status: faithful core, approximate layer selection.

Important details:

```text
Raw UV direction, no UV scaling
lambda_reg = 1e-3
Base model template: llama3-8b -> "{instruction}"
Closed-form ridge solve
Caveat: auto layer selection is KIF-compatible approximation, not bit-exact upstream search
```

### OPT-OUT

Status: method-faithful, system-adapted.

Important details:

```text
method = npo+rt+wd+ot
dpo_beta = 0.1
reg_lambda = 0.1
lr = 1e-5
num_epochs = 3
weight_decay = 0.01
swd_n_projections = 100
full-model training
uses external Alpaca-GPT4 world data when available
```

Caveat:

```text
The objective and hyperparameters are faithful, but execution is adapted to the cluster/device_map setting rather than reproducing upstream distributed training exactly.
```

---

## 4. TOFU benchmark objective

The user wants TOFU evaluated like KIF/SimNPO:

```text
No SMR/EL10 for TOFU.
Only TOFU FQ/MU.
Use forget10.
Report linear FQ, not log-FQ.
Use the same TOFU full model setting used by SimNPO/KIF-style TOFU.
```

Chosen TOFU full model:

```text
locuslab/tofu_ft_llama2-7b
```

Model family:

```text
llama2-7b-chat
```

Prompt format:

```text
[INST] {question} [/INST] {answer}
```

This matches the SimNPO/Unlearn-Simple style TOFU setup using Llama-2-7B-Chat.

---

## 5. TOFU dataset preparation

Dataset prep script:

```text
framework/scripts/tofu_prepare_data.slurm
framework/tofu_eval/prepare_tofu.py
```

Dataset prep has already succeeded.

Successful job details:

```text
Job ID: 4739241
Commit: 54ec9a10dadd55f5487cd0d6d7a79fc2e37553bc
Status: ok
```

Official TOFU configs loaded:

```text
locuslab/TOFU: forget10
locuslab/TOFU: retain90
locuslab/TOFU: forget10_perturbed
locuslab/TOFU: retain_perturbed
locuslab/TOFU: real_authors_perturbed
locuslab/TOFU: world_facts_perturbed
locuslab/TOFU: holdout10
```

Generated outputs:

```text
framework/outputs/tofu/data/forget10.jsonl              # 400 rows
framework/outputs/tofu/data/retain90.jsonl              # 400 eval rows from retain_perturbed
framework/outputs/tofu/data/retain90_train.jsonl        # 3600 full retain90 train rows
framework/outputs/tofu/data/real_authors.jsonl          # 100 rows
framework/outputs/tofu/data/world_facts.jsonl           # 117 rows
framework/outputs/tofu/data/holdout10.jsonl             # 400 rows

framework/outputs/tofu/method_inputs/full_forget10_retain90.jsonl  # 4000 rows
framework/outputs/tofu/method_inputs/forget10_only.jsonl            # 400 rows
framework/outputs/tofu/method_inputs/retain90_only.jsonl            # 3600 rows

framework/outputs/tofu/subjects/forget10_subjects.txt
```

Correct answer policy:

```text
forget:       paraphrased_answer from forget10_perturbed
retain:       paraphrased_answer from retain_perturbed
real_authors: answer from real_authors_perturbed because no paraphrased_answer exists
world_facts:  answer from world_facts_perturbed because no paraphrased_answer exists
```

Important correction:

```text
Do not row-match full retain90 against retain_perturbed. retain90 has 3600 rows, retain_perturbed has 400 rows. OpenUnlearning-style evaluation uses retain_perturbed directly for retain eval, while full retain90 is used as the retain training pool.
```

---

## 6. TOFU evaluator

Files:

```text
framework/tofu_eval/evaluate_tofu.py
framework/tofu_eval/metrics.py
framework/tofu_eval/compare_tofu_results.py
```

Metric definitions:

```text
Probability = exp(-average answer NLL)
Truth Ratio = P(wrong answer | q) / P(correct answer | q)
Forget truth-ratio score = mean(min(TR, 1/TR))
Non-forget truth-ratio score = mean(max(0, 1 - TR))
FQ = KS-test p-value on forget truth-ratio arrays
MU = harmonic mean of non-forget utility components
```

Reporting convention:

```text
SimNPO/KIF-style linear FQ = KS-test p-value, no log transform
```

Expected result keys:

```json
{
  "Forget Quality": "...",
  "KS Test PVal Forget": "...",
  "KS Test Forget": "...",
  "Model Utility": "...",
  "forget_quality_linear_FQ": "...",
  "model_utility_MU": "..."
}
```

---

## 7. TOFU smoke test status

Smoke script:

```text
framework/scripts/tofu_smoke_all_methods_dev.slurm
```

The smoke test completed successfully.

Job details:

```text
Job ID: 4739365
State: COMPLETED
Exit code: 0
Base TOFU full model: locuslab/tofu_ft_llama2-7b
Model family: llama2-7b-chat
Smoke subjects: forget10_author_0000, forget10_author_0001
```

All methods passed:

```text
lunar_smoke: ok
reglu_smoke: ok
simnpo_pure_smoke: ok
simnpo_graddiff_smoke: ok
optout_smoke: ok
```

Smoke output:

```text
framework/outputs/tofu/tofu_all_methods_smoke_summary.json
```

Conclusion:

```text
The full TOFU all-method run is justified because model loading, prompt formatting, data access, objectives, optimizer paths, and manifests all passed on dev GPU.
```

---

## 8. TOFU all-method main job

Main script:

```text
framework/scripts/tofu_run_all_methods_main.slurm
```

Resource request:

```bash
#SBATCH --job-name=tofuall
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=220G
#SBATCH --time=12:00:00
```

Cluster partition check showed this request is valid:

```text
gpu_h100:    MaxTime 3 days, 48 GPUs total
gpu_a100_il: MaxTime 2 days, 36 GPUs total
gpu_h100_il: MaxTime 2 days, 20 GPUs total
```

The user has already submitted the TOFU all-method job. Do not mutate `/src` while it runs.

Monitor:

```bash
squeue -u "$USER"

tail -f "$(ls -t /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-all-methods-*.out | head -n 1)"
```

Error log:

```bash
tail -n 200 "$(ls -t /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-all-methods-*.err | head -n 1)"
```

Final combined results:

```text
/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/outputs/tofu/all_model_tofu_results.json
```

Expected comparison files:

```text
/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/outputs/tofu/tofu_comparison_forget10.csv
/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/outputs/tofu/tofu_comparison_forget10.md
```

Read results:

```bash
cat /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/outputs/tofu/all_model_tofu_results.json
cat /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/outputs/tofu/tofu_comparison_forget10.md
```

---

## 9. PCA activation visualization

Standalone PCA script:

```text
analysis/pca_activation_analysis.py
```

Implemented at commit:

```text
f83f0fd4320b95c374fb8f2b2295c2a0d32c8ce1
```

Purpose:

```text
Produce a PCA visualization and centroid displacement metric showing whether KIF moves forget-subject activations toward the unknown/unverifiable region, while baselines do not.
```

What it does:

```text
1. Loads the KIF target module from outputs/capsules/*_capsule.pkl.gz via target_module_name.
2. Uses that same module for PRE, POST-KIF, and POST-baseline.
3. Extracts forward-hook activations and mean-pools over sequence length.
4. Uses 11 musician forget subjects from outputs/datasets/prompts.jsonl.
5. Uses 20 LUNAR unverifiable prompts.
6. Uses 5 Module 8 benign prompts.
7. Fits PCA only on PRE activations.
8. Transforms all models into the same PCA space.
9. Computes centroid displacement:
   d = ||mu_forget_post - mu_forget_pre|| / ||mu_unknown_pre - mu_forget_pre||
10. Saves PDF, PNG, and JSON.
```

Outputs:

```text
analysis/outputs/pca_activation_space.pdf
analysis/outputs/pca_activation_space.png
analysis/outputs/centroid_displacement.json
```

Important caution:

```text
Do not git pull in /src while TOFU job is running. Use /src_pca instead.
```

---

## 10. Running PCA from separate checkout

Go to separate checkout:

```bash
cd /pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
source /pfs/work9/workspace/scratch/hd_ur228-llmrun/llm_env/bin/activate
git pull origin main
```

Syntax check:

```bash
python3 -c "import ast; ast.parse(open('analysis/pca_activation_analysis.py').read()); print('OK')"
```

Create/run a dev Slurm script if needed:

```bash
mkdir -p logs analysis/outputs

cat > analysis/run_pca_dev.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name=pcaact
#SBATCH --partition=dev_gpu_a100_il,dev_gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --chdir=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/pca-act-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/pca-act-%j.err

set -euo pipefail

source /pfs/work9/workspace/scratch/hd_ur228-llmrun/llm_env/bin/activate

export HF_HOME=/pfs/work9/workspace/scratch/hd_ur228-llmrun/data/hf
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME=/pfs/work9/workspace/scratch/hd_ur228-llmrun/data/torch
export TMPDIR=/pfs/work9/workspace/scratch/hd_ur228-llmrun/cache/tmp
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p analysis/outputs logs "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR"

python3 -c "import ast; ast.parse(open('analysis/pca_activation_analysis.py').read()); print('OK')"

python -u analysis/pca_activation_analysis.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --capsules_dir /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/outputs/capsules \
  --prompts_jsonl /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/outputs/datasets/prompts.jsonl \
  --outputs_root /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/outputs \
  --baseline_prefer optout \
  --out_dir /pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs \
  --batch_size 4 \
  --use_4bit

echo "PCA outputs:"
ls -lh /pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs
EOF

sed -i 's/\r$//' analysis/run_pca_dev.slurm
sbatch analysis/run_pca_dev.slurm
```

Monitor PCA job:

```bash
squeue -u "$USER"

tail -f "$(ls -t /pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/pca-act-*.out | head -n 1)"
```

Check PCA error log:

```bash
tail -n 200 "$(ls -t /pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/pca-act-*.err | head -n 1)"
```

Expected PCA outputs:

```bash
ls -lh /pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs
cat /pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs/centroid_displacement.json
```

---

## 11. If PCA auto-discovery selects the wrong files

The PCA script can auto-discover KIF and baseline artifacts from:

```text
/pfs/work9/workspace/scratch/hd_ur228-llmrun/src/outputs
```

But if it selects the wrong KIF seed or wrong baseline, pass explicit paths:

```bash
python analysis/pca_activation_analysis.py \
  --model_dir meta-llama/Llama-3.1-8B \
  --kif_adapter_path /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/outputs/global_adapters/<CONFIG_B_SEED_23_ADAPTER> \
  --baseline_model_dir /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/outputs/unlearning_runs/optout/unlearned_model \
  --capsules_dir /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/outputs/capsules \
  --prompts_jsonl /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/outputs/datasets/prompts.jsonl \
  --out_dir /pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/analysis/outputs \
  --batch_size 4 \
  --use_4bit
```

User preference:

```text
Use Config B seed 23 KIF files if available, because that is the strongest SMR=0 run.
Use OPT-OUT baseline if available/preferred, otherwise SimNPO baseline with lowest SMR.
```

---

## 12. Useful monitoring commands

List running jobs:

```bash
squeue -u "$USER"
```

Watch latest TOFU all-method output:

```bash
tail -f "$(ls -t /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/tofu-all-methods-*.out | head -n 1)"
```

Watch latest PCA output:

```bash
tail -f "$(ls -t /pfs/work9/workspace/scratch/hd_ur228-llmrun/src_pca/logs/pca-act-*.out | head -n 1)"
```

Check job accounting:

```bash
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```

Check GPU usage on an allocated node:

```bash
watch -n 10 nvidia-smi
```

---

## 13. Current most important next steps

1. Monitor the TOFU all-method job until completion.
2. Do not mutate `/src` while TOFU job is running.
3. Run PCA visualization from `/src_pca` on dev GPU.
4. After TOFU completes, inspect:

```text
framework/outputs/tofu/all_model_tofu_results.json
framework/outputs/tofu/tofu_comparison_forget10.md
```

5. After PCA completes, inspect:

```text
src_pca/analysis/outputs/pca_activation_space.pdf
src_pca/analysis/outputs/pca_activation_space.png
src_pca/analysis/outputs/centroid_displacement.json
```

6. If PCA auto-selection chooses the wrong KIF/baseline artifacts, rerun with explicit `--kif_adapter_path` and `--baseline_model_dir`.
