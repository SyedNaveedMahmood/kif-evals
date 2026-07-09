# Qwen 3B Evaluation-Only Commands for Completed Baselines

These commands are intended for the RTX 5060 Ti after Qwen 3B LUNAR and ReGLU have already been trained. They avoid full-parameter training and only run evaluator scripts against saved baseline outputs.

## Setup

```bash
cd /mnt/e/eruf/kif-evals
git checkout rebuttal-experiments
git pull origin rebuttal-experiments

conda activate eruf-rebuttal

export EVALS=/mnt/e/eruf/kif-evals
export PYTHONPATH="$EVALS/rebuttal:$EVALS/analysis:$EVALS/framework:$EVALS:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export QWEN_MODEL="Qwen/Qwen2.5-3B-Instruct"
export PROMPTS_JSONL="framework/outputs/qwen3b/prompts_with_adele_retain.jsonl"

mkdir -p framework/logs
```

## Saved model paths

```bash
export QWEN_LUNAR_MODEL="framework/outputs/qwen3b_baselines/lunar/lunar/unlearned_model"
export QWEN_REGLU_MODEL="framework/outputs/qwen3b_baselines/reglu/reglu/model"
```

Check that they exist:

```bash
ls "$QWEN_LUNAR_MODEL"/config.json
ls "$QWEN_REGLU_MODEL"/config.json
```

## Fast entity evaluation bundle: LUNAR

```bash
/usr/bin/time -v python -u analysis/evaluate_saved_baseline_suites.py \
  --method lunar \
  --model_dir "$QWEN_MODEL" \
  --model_path "$QWEN_LUNAR_MODEL" \
  --prompts_jsonl "$PROMPTS_JSONL" \
  --out_root analysis/outputs_qwen3b_saved_baseline_suite_evals \
  --max_subjects 5 \
  --load_mode 4bit \
  --batch_size 4 \
  --el_batch_size 4 \
  --max_fast_rows 0 \
  --max_adv_rows 0 \
  --max_rwku_rows 0 \
  --rwku_rows_per_family_cap 0 \
  2>&1 | tee framework/logs/qwen3b_lunar_saved_suites_5060ti.log
```

## Fast entity evaluation bundle: ReGLU

```bash
/usr/bin/time -v python -u analysis/evaluate_saved_baseline_suites.py \
  --method reglu \
  --model_dir "$QWEN_MODEL" \
  --model_path "$QWEN_REGLU_MODEL" \
  --prompts_jsonl "$PROMPTS_JSONL" \
  --out_root analysis/outputs_qwen3b_saved_baseline_suite_evals \
  --max_subjects 5 \
  --load_mode 4bit \
  --batch_size 4 \
  --el_batch_size 4 \
  --max_fast_rows 0 \
  --max_adv_rows 0 \
  --max_rwku_rows 0 \
  --rwku_rows_per_family_cap 0 \
  2>&1 | tee framework/logs/qwen3b_reglu_saved_suites_5060ti.log
```

## Inspect summaries

```bash
find analysis/outputs_qwen3b_saved_baseline_suite_evals -name '*summary*.json' -o -name 'compact_summary.json' | sort
```

Then inspect the compact summaries:

```bash
cat analysis/outputs_qwen3b_saved_baseline_suite_evals/lunar/compact_summary.json
cat analysis/outputs_qwen3b_saved_baseline_suite_evals/reglu/compact_summary.json
```

## If the full evaluator is too slow

Use a capped version first:

```bash
/usr/bin/time -v python -u analysis/evaluate_saved_baseline_suites.py \
  --method reglu \
  --model_dir "$QWEN_MODEL" \
  --model_path "$QWEN_REGLU_MODEL" \
  --prompts_jsonl "$PROMPTS_JSONL" \
  --out_root analysis/outputs_qwen3b_saved_baseline_suite_evals_smoke \
  --max_subjects 5 \
  --load_mode 4bit \
  --batch_size 2 \
  --el_batch_size 2 \
  --max_fast_rows 80 \
  --max_adv_rows 80 \
  --max_rwku_rows 80 \
  --rwku_rows_per_family_cap 10 \
  2>&1 | tee framework/logs/qwen3b_reglu_saved_suites_smoke_5060ti.log
```

Only use capped results as exploratory diagnostics unless clearly labeled.
