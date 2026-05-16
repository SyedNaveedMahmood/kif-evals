# LUNAR on KIF Dual-Metric Evaluation, Llama-3.1-8B Base

This codebase is patched for an apple-to-apple comparison where:

- **Unlearning method**: LUNAR
- **Base model**: `meta-llama/Llama-3.1-8B`
- **Forget/retain data**: KIF custom Real-World Entity Dataset
- **Evaluation**: KIF Module 8 dual-metric protocol: SMR + EL10 + Utility Drift + Cohen's d

The comparison is intentionally not based on LUNAR's native ROUGE/Deviation Score as the main result. LUNAR performs the forgetting; KIF evaluates whether the forgotten subject is Type I, Type II, or Type III under the dual-metric protocol.

## What was changed

### LUNAR patches

1. Added a base-model wrapper:

```text
LUNAR-main/src/model_utils/llama31_base_model.py
```

This wrapper supports `meta-llama/Llama-3.1-8B` without applying the Llama-3 Instruct chat template. KIF Module 8 evaluates raw entity prompts, so LUNAR direction mining and down-projection fitting also use raw prompts.

2. Added `llama31-8b-base` to:

```text
LUNAR-main/src/model_utils/model_loader.py
```

3. Patched LUNAR data formatting and evaluation utilities to recognize:

```text
llama31-8b-base
llama-3.1-8b-base
llama31-base
```

4. Patched `run_lunar.py` with:

```yaml
skip_native_eval: true
```

This lets the experiment skip LUNAR native ROUGE evaluation and go directly to the KIF dual-metric evaluation after saving the LUNAR-unlearned model.

5. Added KIF-to-LUNAR dataset converter:

```text
LUNAR-main/scripts/convert_kif_prompts_to_lunar.py
```

It converts KIF `prompts.jsonl` into LUNAR's `dataset/unlearning/*.json` format where:

```text
LUNAR edge = sanitized KIF subject
```

Example:

```text
Taylor Swift -> Taylor_Swift
```

### KIF patches

Added a patched evaluator:

```text
KIF-main/src/llama20/modules/module8_lunar_apples.py
```

It keeps the original Module 8 logic, but adds:

- `EVAL_SUBJECTS_JSON` / `EVAL_SUBJECTS` so evaluation is restricted exactly to the LUNAR-forgotten subjects.
- `MAX_SUBJECTS` env override.
- per-subject SMR.
- per-subject mechanism state.
- aggregate mechanism state.
- safer 4-bit loading using `device_map="auto"`.

## One-command run

From the archive root:

```bash
cd lunar_kif_dualmetric_llama31

python scripts/run_lunar_on_kif_then_eval.py \
  --kif-prompts-jsonl /absolute/path/to/outputs/datasets/prompts.jsonl \
  --kif-capsules-dir /absolute/path/to/outputs/capsules \
  --subjects "Taylor Swift" \
  --model-id meta-llama/Llama-3.1-8B \
  --layer 22 \
  --coeff 2.0 \
  --epochs 20 \
  --lr 0.01 \
  --run-kif-eval
```

For multiple subjects:

```bash
python scripts/run_lunar_on_kif_then_eval.py \
  --kif-prompts-jsonl /absolute/path/to/outputs/datasets/prompts.jsonl \
  --kif-capsules-dir /absolute/path/to/outputs/capsules \
  --subjects "Taylor Swift,Ariana Grande,Beyonce,Kanye West" \
  --model-id meta-llama/Llama-3.1-8B \
  --layer 22 \
  --coeff 2.0 \
  --epochs 20 \
  --lr 0.01 \
  --run-kif-eval
```

For a stronger top-K-layer LUNAR setting:

```bash
python scripts/run_lunar_on_kif_then_eval.py \
  --kif-prompts-jsonl /absolute/path/to/outputs/datasets/prompts.jsonl \
  --kif-capsules-dir /absolute/path/to/outputs/capsules \
  --subjects "Taylor Swift" \
  --model-id meta-llama/Llama-3.1-8B \
  --layer 20,22,24 \
  --coeff 2.0,2.0,2.0 \
  --epochs 20 \
  --lr 0.01 \
  --run-kif-eval
```

## Outputs

LUNAR-unlearned model:

```text
LUNAR-main/models_lunar/kif_entities/llama31-8b-base/<forget_edge_slug>/
```

Converted LUNAR dataset:

```text
LUNAR-main/dataset/unlearning/kif_entities.json
```

KIF dual-metric evaluation:

```text
outputs/eval_lunar_dualmetric/<forget_edge_slug>/eval_summary.json
```

Important fields in `eval_summary.json`:

```json
{
  "robustness_post_lora": {
    "avg_subject_mention_rate": 0.0,
    "per_subject_mention_rate": {}
  },
  "extraction_likelihood": {
    "EL10_pre": 0.0,
    "EL10_post": 0.0,
    "EL10_ratio": null,
    "per_subject_pre": {},
    "per_subject_post": {}
  },
  "mechanism_state": {
    "aggregate": "Type I: True Erasure",
    "per_subject": {}
  },
  "signature_separation": {
    "avg_cohens_d_pre": null,
    "avg_cohens_d_post": null,
    "delta": null
  }
}
```

## Mechanism-state rule

```text
Type I:   SMR <= 0.05 and EL10 < 1
Type II:  SMR <= 0.05 and EL10 >= 1
Type III: SMR > 0.05
```

## Fairness constraints for the paper

Use the same:

- model: `meta-llama/Llama-3.1-8B`
- forgotten subjects
- KIF `prompts.jsonl`
- KIF `capsules`
- Module 8 generation settings
- SMR/EL10 thresholds
- seeds

Do not tune LUNAR's layer/coeff directly on final EL10. If you need layer search, use LUNAR's own controllability/layer-selection logic or a held-out validation subset, then freeze the selected setting before final KIF dual-metric evaluation.
