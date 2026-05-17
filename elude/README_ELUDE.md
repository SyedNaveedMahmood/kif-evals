# KIF × ELUDe External Benchmark Patch

This patch adds ELUDe as a separate external entity-level benchmark for KIF. It does **not** replace the existing Module 8 SMR/EL10 mechanistic evaluation.

## What changed

- Added `src/llama20/modules/module_elude.py`
  - Loads `6rightjade/ELUDe` from Hugging Face.
  - Creates KIF-compatible mining prompts in `outputs/datasets/prompts.jsonl`.
  - Creates ELUDe train/eval splits in `outputs/elude/`.
- Added `src/llama20/modules/module8e.py`
  - Evaluates ELUDe with QA-style probability, ROUGE-L, truth-ratio, FQ, and RQ metrics.
  - Keeps the report separate from SMR/EL10.
- Edited Module 0
  - Default model is now `meta-llama/Meta-Llama-3.1-8B-Instruct`.
- Edited Module E
  - Uses `outputs/model` by default.
  - Harvests capsule-suppressed refusals on the exact ELUDe forget-train questions when `outputs/elude/elude_upu_train.jsonl` exists.
- Edited Module 7
  - Uses ELUDe forget-train QA prompts and gold answers for UPU pairs when `outputs/elude/elude_upu_train.jsonl` exists.
  - Falls back to the original generic prompt templates if ELUDe files are absent.
- Added `run_elude_pipeline.py`
  - Runs Module 0 → ELUDe prep → B → C → D → E → 7 → 8E.
- Added copied Opt-Out dataset/evaluator files under `third_party/optout/` for reference only.

## Why Module 7 needed an edit

Yes, the UPU loop needed a compatibility edit. The old Module 7 built pairs from generic prompts such as `Tell me about {subject}`. ELUDe evaluation is QA-based, so training only on generic biography prompts creates a prompt-distribution mismatch. The patched Module 7 now uses ELUDe forget-train questions as `x`, ELUDe gold answers as `y_bad`, and Module E's capsule-suppressed refusals as `y_good`.

## Recommended split

The default split is deterministic:

```text
80% of each target entity's forget_qa rows -> signature mining + UPU training
20% of each target entity's forget_qa rows -> Module 8E forget evaluation
retain_qa validation/test rows -> Module 8E retain evaluation
retain_qa train rows -> semantic controls for signature mining
```

You can change the split with `--train-ratio`.

## Install

```bash
pip install -r requirements.txt
huggingface-cli login
```

You need access to `meta-llama/Meta-Llama-3.1-8B-Instruct` on Hugging Face.

## Full run: one target

```bash
python run_elude_pipeline.py \
  --targets "Cristiano Ronaldo" \
  --train-ratio 0.80 \
  --seed 17
```

## Full run: first 5 ELUDe targets alphabetically

```bash
python run_elude_pipeline.py \
  --max-targets 5 \
  --train-ratio 0.80 \
  --seed 17
```

## Resume after Module 0 already created `outputs/model`

```bash
python run_elude_pipeline.py \
  --targets "Cristiano Ronaldo" \
  --skip-module0
```

## Evaluate an existing adapter only

```bash
KIF_MODEL_DIR=outputs/model python run_elude_pipeline.py \
  --eval-only \
  --adapter-path outputs/global_adapters/unlearning_adapter_repaware_YYYYMMDD_HHMMSS
```

## Useful environment variables

```bash
export KIF_HF_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
export KIF_MODEL_DIR=outputs/model
export KIF_USE_4BIT=1
export ELUDE_MAX_EVAL_ROWS=0        # 0 = full eval
export ELUDE_DO_GENERATION=1        # required for ROUGE-L
export ELUDE_MAX_FORGET_TRAIN=0     # 0 = no cap
export ELUDE_MAX_RETAIN_TRAIN=0     # 0 = no cap
```

## Main output files

```text
outputs/elude/metadata.json
outputs/elude/elude_upu_train.jsonl
outputs/eval_elude/elude_summary.json
outputs/eval_elude/elude_metrics.csv
outputs/eval_elude/elude_predictions.jsonl
```

## Paper reporting recommendation

Report ELUDe separately:

```text
KIF mechanistic evaluation: Module 8 -> SMR, EL10, utility drift, Cohen's d
External benchmark: Module 8E -> ELUDe FQ/RQ QA metrics
```

Do not merge ELUDe FQ/RQ with SMR/EL10 in the same aggregate score.

---

## Publication-style 20-target benchmark

For the final KIF-on-ELUDe evaluation, use the newer benchmark runner:

```bash
python run_elude20_benchmark.py --skip-module0 --seed 17
```

See `README_ELUDE20_BENCHMARK.md` for the exact protocol and output layout.
