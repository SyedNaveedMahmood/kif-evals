# ERUF Rebuttal Experiments

Self-contained rebuttal experiment folder for the ERUF/kif-evals rebuttal branch.

This folder avoids relying on `analysis/` paths at runtime. The scripts are copied into `rebuttal/` as normal Python files so a clean clone can run them directly from Ubuntu or WSL.

## Setup

```bash
cd /mnt/e/eruf/kif-evals/rebuttal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Required artifacts

The probe scripts expect Stage-0 ERUF artifacts to already exist:

```bash
export EVALS=/mnt/e/eruf/kif-evals
export ERUF=/mnt/e/eruf/Representation-Aware-Unlearning-via-Activation-Signatures-ERUF-main
export BASE_MODEL=meta-llama/Llama-3.1-8B
export MODEL_DIR="$ERUF/outputs/model"
export PROMPTS_JSONL="$ERUF/outputs/datasets/prompts.jsonl"
export CAPSULES_DIR="$ERUF/outputs/capsules"
export KIF_ADAPTER="$ERUF/outputs/global_adapters/unlearning_adapter_repaware"
export BASELINE_MODEL="$EVALS/framework/outputs/optout/anchor/optout/unlearned_model"
```

## Compile check

```bash
python3 -m py_compile *.py
```

## Main launcher

```bash
python run_probe_suite.py --help
```

Dry-run all lightweight probes:

```bash
python run_probe_suite.py all-probes \
  --base_model "$BASE_MODEL" \
  --model_dir "$MODEL_DIR" \
  --kif_adapter "$KIF_ADAPTER" \
  --baseline_model "$BASELINE_MODEL" \
  --capsules_dir "$CAPSULES_DIR" \
  --prompts_jsonl "$PROMPTS_JSONL" \
  --use_4bit \
  --dry_run
```

Then remove `--dry_run` to execute.

## Individual scripts

E2:

```bash
python signature_separability_collapse_v2.py --help
python representation_erasure_suite.py --help
```

E3:

```bash
python cross_domain_locality_probe.py --help
python subject_specificity_robustness_suite_fixed.py --help
```

E4:

```bash
python hidden_space_selectivity_eval.py --help
```

E5:

```bash
python collect_compute_cost_table.py --help
```

E7:

```bash
python el10_token_audit.py --help
```

E8:

```bash
python train_no_capsule_lora_ablation.py --help
python evaluate_no_capsule_ablation.py --help
```

## Notes

- `slurm/` is intentionally present but empty for now.
- `hidden_space_selectivity_eval.py` is a generic alias of the available hidden-space selectivity evaluator. Use `--model_label` to label Opt-Out, LUNAR, ReGLU, or SimNPO runs.
- No cached outputs are required by this folder, but the model artifacts referenced above must exist before the probe scripts can compute rebuttal numbers.
