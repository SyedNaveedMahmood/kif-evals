# Plug-and-Play Unlearning + KIF Dual-Metric Evaluation Framework

A thin adapter layer that lets you test **any unlearning method** on your
custom musician dataset and evaluate it with KIF's **SMR + EL10 dual-metric
protocol** from Module 8 — with zero changes to the evaluator when you swap
methods.

---

## Architecture

```
outputs/datasets/prompts.jsonl   ← KIF input interface (fixed)
        │
        ▼
┌────────────────────────────────────────────┐
│          orchestrate.py                    │
│                                            │
│  METHOD_REGISTRY.get("lunar")              │
│  → LUNARMethod.execute(prompts_jsonl, ...) │
│                                            │
│  Internally:                               │
│    1. Parse subjects from prompts.jsonl    │
│    2. Build forget / retain / ref sets     │
│    3. Compute activation redirect vector   │
│    4. Patch MLP down_proj (closed-form)    │
│    5. Save merged model                    │
│    6. Write unlearning_result.json         │
└───────────────┬────────────────────────────┘
                │
                │  UnlearningResult
                │  { merged_model_dir: "…/unlearned_model" }
                ▼
┌────────────────────────────────────────────┐
│          eval/run_dual_metric.py           │
│                                            │
│  Translates UnlearningResult → M8 kwargs  │
│  Calls run_module8_clean(...)             │
│                                            │
│  Reports:                                 │
│    • SMR  (Subject Mention Rate)          │
│    • EL10 (Extraction Likelihood)         │
│    • Cohen's d (if capsules present)      │
│    • Utility drift (PPL Δ)               │
│    • Type I / II / III classification     │
└────────────────────────────────────────────┘
                │
                ▼
outputs/unlearning_runs/lunar/eval_dual_metric/
    ├── eval_summary.json
    ├── dual_metric_summary.json
    ├── post_gens.json
    └── utility.json
```

### Model-loading contract

Every method saves one of two artefact types and sets the corresponding
fields in `UnlearningResult`:

| Mode | What the method saves | UnlearningResult fields |
|------|----------------------|------------------------|
| **A** (merged model) | Full HF model directory | `merged_model_dir` |
| **B** (LoRA adapter) | Adapter dir next to base | `model_dir` + `adapter_path` |

The evaluator reads these fields and calls Module 8 accordingly.
**No evaluator code changes when you add a new method.**

---

## Quick start

### Run LUNAR end-to-end

```bash
cd /path/to/KIF_root

python unlearning_eval_framework/orchestrate.py \
    --method        lunar \
    --model_dir     outputs/model \
    --prompts_jsonl outputs/datasets/prompts.jsonl \
    --capsules_dir  outputs/capsules \
    --kif_root      . \
    --output_dir    outputs/unlearning_runs
```

Override LUNAR config knobs:

```bash
python unlearning_eval_framework/orchestrate.py \
    --method  lunar \
    --config  '{"layer_modified": 18, "auto_select_layer": false, "num_epochs": 10}' \
    ...
```

### Run eval only (on a previous LUNAR run)

```bash
python unlearning_eval_framework/orchestrate.py \
    --eval_only     true \
    --result_json   outputs/unlearning_runs/lunar/unlearning_result.json \
    --kif_root      . \
    --prompts_jsonl outputs/datasets/prompts.jsonl \
    --capsules_dir  outputs/capsules
```

### Skip eval (unlearn only, no Module 8)

```bash
python unlearning_eval_framework/orchestrate.py \
    --method      lunar \
    --skip_eval   true \
    ...
```

### Programmatic API

```python
import sys
sys.path.insert(0, "unlearning_eval_framework")

from orchestrate import run_pipeline

summary = run_pipeline(
    method_name    = "lunar",
    model_dir      = "outputs/model",
    prompts_jsonl  = "outputs/datasets/prompts.jsonl",
    capsules_dir   = "outputs/capsules",
    kif_root       = ".",
    output_dir     = "outputs/unlearning_runs",
    config_overrides = {
        "model_family":       "llama3-8b-instruct",
        "auto_select_layer":  True,
        "use_harmful":        True,
    },
)
```

---

## Output artefacts

```
outputs/unlearning_runs/
└── lunar/
    ├── unlearning_result.json      ← model path manifest (read by evaluator)
    ├── lunar_meta.json             ← LUNAR-specific: layer, UV norm, subjects
    ├── unlearned_model/            ← full merged HF model
    │   ├── config.json
    │   ├── model.safetensors
    │   └── tokenizer*
    └── eval_dual_metric/
        ├── dual_metric_summary.json   ← main result: SMR, EL10, PPL-Δ, Type
        ├── eval_summary.json
        ├── post_gens.json
        └── utility.json
```

---

## Adding a new method (5 minutes)

1. **Copy the template**
   ```bash
   cp unlearning_eval_framework/methods/template_new_method.py \
      unlearning_eval_framework/methods/rmu.py
   ```

2. **Fill in `run()`** — the three helper methods take care of parsing
   `prompts.jsonl`:
   ```python
   subjects = self.load_subjects_from_prompts(prompts_jsonl)
   forget_qs = self.subject_to_forget_questions(prompts_jsonl, subjects)
   ```

3. **Return an `UnlearningResult`** pointing at your saved model.

4. **Register** by adding one line to `methods/__init__.py`:
   ```python
   from methods.rmu import RMUMethod   # noqa: F401
   ```

5. **Run:**
   ```bash
   python orchestrate.py --method rmu ...
   ```

That's it. Module 8, the orchestrator, and the evaluator never need to
change.

---

## File map

```
unlearning_eval_framework/
├── orchestrate.py               ← entry point / CLI
├── methods/
│   ├── base.py                  ← UnlearningMethod ABC + METHOD_REGISTRY
│   ├── __init__.py              ← imports all methods (triggers registration)
│   ├── lunar.py                 ← LUNAR implementation
│   └── template_new_method.py  ← copy-paste skeleton for new methods
└── eval/
    ├── __init__.py
    ├── run_dual_metric.py       ← bridges UnlearningResult → Module 8
    └── module8_standalone.py    ← Module 8 fallback (no capsules needed)
```

---

## LUNAR config reference

| Key | Default | Description |
|-----|---------|-------------|
| `model_family` | `"llama3-8b-instruct"` | Chat template to apply |
| `layer_modified` | `22` | Layer index (ignored if `auto_select_layer=True`) |
| `auto_select_layer` | `True` | Select layer via cosine-sim score (LUNAR §3.2) |
| `coeff` | `2.0` | Direction coefficient |
| `use_harmful` | `True` | `True` → harmful Dref; `False` → unverifiable Dref |
| `num_epochs` | `20` | SGD fallback epochs (not used for closed-form solve) |
| `lr` | `0.01` | Learning rate (SGD fallback) |
| `positions` | `-1` | Token position for mean-diff (`-1` = last) |
| `n_train` | `128` | Max forget/ref instructions |
| `lunar_repo_root` | `None` | Point to LUNAR-main to use its harmful.json splits |
| `max_subjects` | `None` | Limit subjects (useful for quick smoke tests) |
| `max_prompts_per_subject` | `50` | Prompts pulled per subject from prompts.jsonl |
| `seed` | `42` | RNG seed |

---

## Notes on LUNAR + KIF capsules

LUNAR is a **capsule-free** method. It does not produce KIF-style capsule
files, so the `signature_separation` (Cohen's d) field in the eval summary
will show `"Skipped — no capsules available"` unless you already have
capsules from a prior full KIF run on the same model.

The SMR and EL10 metrics run fully independently of capsules and give a
complete picture of surface leakage and latent trace persistence.
