# KIF on ELUDe: 20-target single-target benchmark

This codebase evaluates **KIF as KIF**, not OPT-OUT reproduction, on the public ELUDe entity-level unlearning benchmark.

## Protocol implemented

- **Base model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Target set:** all 20 ELUDe target entities from `6rightjade/ELUDe`, discovered from `forget_qa`
- **Run structure:** 20 independent **single-target** runs, one LoRA adapter per target
- **Forget set:** full `forget_qa/train` for the selected target
  - Used for KIF signature mining / capsule harvesting / UPU distillation
  - Used again for forget evaluation, matching the public OPT-OUT data-loader convention
- **Retain set:** `retain_qa/train` for retention anchors; `retain_qa/test` for RQ evaluation
- **World set:** Alpaca-GPT4 train/test files under `data/`, matching the OPT-OUT repository's auxiliary world set usage when available
- **Metrics:** ELUDe-style QA metrics only: probability, ROUGE-L recall, truth ratio, FQ, RQ
- **Not included in ELUDe report:** SMR/EL10. Those remain in Module 8 as KIF's separate mechanistic diagnostic evaluation.

## What was changed vs. the earlier ELUDe patch

1. `module_elude.py`
   - Stops creating a custom train/eval split for `forget_qa`.
   - Uses the full public forget set for the selected target, which is the standard ELUDe/OPT-OUT convention.
   - Uses official `retain_qa/train` for retention data and `retain_qa/test` for evaluation.

2. `module_e.py`
   - In ELUDe mode, harvests capsule-suppressed outputs for **all** selected target forget questions, not only 50 generic-style prompts.

3. `module7.py`
   - Keeps the full ELUDe forget set instead of truncating to the generic KIF minimum-pair budget.
   - Uses ELUDe `retain_qa/train` as retention anchors.
   - Uses Alpaca-GPT4 training rows as optional world anchors.
   - Uses chat-template formatting for Llama-3.1-Instruct UPU log-probability computation.

4. `module8e.py`
   - Computes ELUDe FQ/RQ-style metrics separately from SMR/EL10.
   - Adds world-set evaluation when `data/alpaca_gpt4_data_test.json` is available.
   - Computes RQ as the harmonic mean across retain and world metrics when world data is present.

5. `run_elude20_benchmark.py`
   - Runs all 20 targets as isolated single-target runs.
   - Archives every target's outputs.
   - Writes per-target and aggregate summaries.

## Recommended command

First install requirements and authenticate for gated Llama access:

```bash
pip install -r requirements.txt
huggingface-cli login
```

Run Module 0 once:

```bash
python run_elude_pipeline.py --max-targets 1
```

Then run the final 20-target benchmark:

```bash
python run_elude20_benchmark.py \
  --skip-module0 \
  --seed 17
```

Outputs:

```text
outputs/elude20_benchmark/elude_targets_used.txt
outputs/elude20_benchmark/elude20_per_target_metrics.csv
outputs/elude20_benchmark/elude20_aggregate_summary.json
outputs/elude20_benchmark/<target>__seed17/
```

## Resume after interruption

```bash
python run_elude20_benchmark.py \
  --skip-module0 \
  --seed 17 \
  --resume
```

## One-target debugging command

```bash
python run_elude_pipeline.py \
  --targets "Cristiano Ronaldo" \
  --seed 17 \
  --skip-module0
```

## Compute note

This is intentionally expensive because it is a fair single-target benchmark: 20 independent Module B/C/D/E/7/8E runs. Do not pass all 20 targets to `--targets` in one command for final reporting, because that would create one multi-target adapter rather than 20 single-target adapters.
