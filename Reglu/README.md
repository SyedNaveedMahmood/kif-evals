# ReGLU integration for the KIF dual-metric framework

This folder documents the ReGLU integration added to the framework.

The executable framework method lives at:

```text
framework/methods/reglu.py
framework/methods/reglu_patch.py
```

The cluster run script lives at:

```text
framework/scripts/reglu_framework_run.slurm
```

## Pipeline contract

```text
KIF prompts.jsonl
  -> framework method: reglu
  -> ReGLU unlearning with RILA + ROL
  -> saved merged model
  -> shared Module 8 evaluator
  -> SMR + EL10 + utility drift
```

The evaluator is unchanged from the LUNAR run. This keeps the comparison apples-to-apples.

## Faithful ReGLU components implemented

- RILA initialization:
  - collect forget and retain layer representations;
  - compute the balanced representation covariance objective;
  - initialize LoRA B from the leading discriminative subspace;
  - initialize LoRA A using the corresponding projected base weight.
- ROL regularization:
  - construct retain-subspace bases from retain representations;
  - penalize non-orthogonality between LoRA B and the retain basis.
- ReGLU training objective:
  - IHL or GD forget loss;
  - retain CE loss;
  - ROL penalty.
- LoRA target modules:
  - q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.

## Default run

```bash
cd /pfs/work9/workspace/scratch/hd_ur228-llmrun/src

git pull origin main
sed -i 's/\r$//' framework/scripts/reglu_framework_run.slurm
sbatch framework/scripts/reglu_framework_run.slurm
```

Monitor:

```bash
tail -f "$(ls -t /pfs/work9/workspace/scratch/hd_ur228-llmrun/src/framework/logs/reglu-framework-*.out | head -n 1)"
```

## Default config in the SLURM script

```json
{
  "model_family": "llama3-8b",
  "variant": "ihl",
  "lora_targets": "all",
  "lora_r": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.0,
  "rila_beta": 0.5,
  "rila_samples_per_split": 128,
  "rila_cov_shrink": 0.0001,
  "rol_lambda": 0.5,
  "rol_rank": 128,
  "retain_gamma": 1.0,
  "n_forget": 132,
  "n_retain": 128,
  "max_per_subject": 12,
  "batch_size": 4,
  "gradient_accumulation_steps": 8,
  "num_epochs": 5,
  "learning_rate": 0.0001,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "max_length": 256,
  "torch_dtype": "bfloat16",
  "save_merged_model": true,
  "seed": 17
}
```

## Retain set

The run script creates:

```text
framework/outputs/reglu/prompts_with_adele_retain.jsonl
```

It copies the existing KIF `app/outputs/datasets/prompts.jsonl` and appends Adele prompts as a non-forgotten retain/control subject. The forget subjects remain the same 11 musician subjects used by LUNAR.

## Outputs

A run writes under:

```text
framework/outputs/reglu/reglu_framework_<jobid>/
```

Expected files:

```text
reglu/adapter/
reglu/merged_model/
reglu/reglu_meta.json
reglu/unlearning_result.json
eval/final_summary.json
```

The merged model is used for Module 8 because ReGLU's RILA initialization changes the residual base weight as well as the LoRA matrices. Saving only a PEFT adapter would not fully preserve the method state.
