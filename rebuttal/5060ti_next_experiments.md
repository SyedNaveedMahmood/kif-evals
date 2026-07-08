# 5060 Ti Rebuttal Experiment Queue

This note tracks what should be run on the RTX 5060 Ti while the full Llama-3.1-8B ERUF Module 7 run is deferred to the 24 GB GPU.

The goal is not to run every possible ablation. The goal is to run quick experiments that directly answer reviewer concerns and only report them if the result supports the paper's actual narrative. If an experiment does not support the current narrative, treat it as diagnostic evidence and pivot the rebuttal wording instead of forcing the claim.

## Current status from uploaded compute-cost extraction

The current compute-cost extraction is **not ready to report as a final compute-cost table**.

What it currently gives us:

- It detects the local ERUF machine as `NVIDIA GeForce RTX 5060 Ti with 15.93 GB memory`.
- It finds some historical LUNAR wall-clock records in cluster logs, for example 289 s, 159 s, 949 s, and 988 s.
- It detects intended GPU counts from several SLURM scripts.

Why it should not yet be used as rebuttal evidence:

- Most rows have missing `wall_seconds`.
- Most rows have missing `gpu_hours`.
- Several rows infer `gpu_type` as `1` or `4` from SLURM GPU-count syntax, which is not a real GPU model.
- Many unrelated files are classified as method `eruf`, so method attribution needs manual cleanup or a better parser.

**Reporting rule:** do not cite this table yet. Use it only as a scaffold. For rebuttal, we need a curated table with one row per completed method/run, containing method, model, GPU model, GPU count, wall-clock time, and GPU-hours.

## Review concerns this queue targets

The uploaded reviews make three quick-action concerns especially relevant:

1. **Small/moderate-model baselines:** reviewers ask what happens to LUNAR, ReGLU, OPT-OUT, and SimNPO on the 3B settings where ERUF struggles.
2. **Compute cost:** reviewers ask whether ERUF costs substantially more than baselines.
3. **Layer choice:** reviewers say the mid-to-late layer choice is currently supported mainly by empirical evidence, so a compact sensitivity check would help.

The 3B baseline table is highest value overall, but it is already running elsewhere. On the 5060 Ti, the best quick target is the layer-band ablation plus clean local timing.

## Experiment A: Layer-band localization ablation

### Purpose

Directly answer the criticism that the mid-to-late layer choice is arbitrary. The paper currently uses a Cohen's d plot to motivate high-salience layers. This experiment turns that into an ablation:

- Early band: layers 5-9
- Middle band: layers 14-18
- Peak band: layers 23-27, already completed in the current run

The experiment does **not** require full Module 7 LoRA training. It only needs Module B and Module C, optionally Module D. This is suitable for the RTX 5060 Ti.

### Use only if

Use the result in the rebuttal only if it shows one of these:

- Peak band has clearly stronger mean best score than early/middle; or
- Peak band has more stable subject coverage and capsule success; or
- Peak band is at least competitive while early/middle are weaker for multiple subjects.

Do **not** report it as a win if early or middle layers beat the peak band. In that case, change the claim to data-driven layer selection rather than fixed peak-layer superiority.

### Commands: early band 5-9

Run from the ERUF repo:

```bash
cd "$ERUF"
conda activate eruf-rebuttal

mkdir -p outputs/rebuttal_layer_ablation

/usr/bin/time -v bash -lc '
KIF_MODULE_B_LAYERS=5-9 \
KIF_MODULE_B_BATCH_SIZE=8 \
KIF_MODULE_B_CAPTURE_SCOPE=full \
KIF_MODULE_B_OUTPUT_DIR=outputs/rebuttal_layer_ablation/early_05_09/activations \
llama20 module_b
' 2>&1 | tee outputs/rebuttal_layer_ablation/early_05_09_module_b_time.log
```

Then run Module C manually on the same band:

```bash
python - <<'PY'
import torch
from llama20.modules.module_c import SignatureMiningConfig, ROMEHyperParams, SignatureExtractor

layers = [5, 6, 7, 8, 9]
config = SignatureMiningConfig(
    activations_dir="outputs/rebuttal_layer_ablation/early_05_09/activations",
    output_dir="outputs/rebuttal_layer_ablation/early_05_09/signatures",
    rome_hparams=ROMEHyperParams(
        layers=layers,
        layer_selection="top_k",
        target_module="mlp",
        significance_threshold=1.5,
    ),
    top_k_directions=3,
    min_prompts_per_subject=2,
    use_semantic_negatives=True,
    min_controls_per_subject=1,
    allow_synthetic_fallback=True,
    enable_oversampling=False,
    negative_pool_mode="match_positives",
    fixed_negative_pool_size=100,
    synthetic_fraction=0.10,
    activation_strategy="mean_token",
    standardize_dims=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_half_precision=False,
    enable_memory_cleanup=True,
    cleanup_frequency=5,
)
extractor = SignatureExtractor(config)
signatures = extractor.extract_all_signatures()
extractor.save_signature_index(signatures)
extractor.create_summary_report()
PY
```

### Commands: middle band 14-18

```bash
cd "$ERUF"
conda activate eruf-rebuttal

/usr/bin/time -v bash -lc '
KIF_MODULE_B_LAYERS=14-18 \
KIF_MODULE_B_BATCH_SIZE=8 \
KIF_MODULE_B_CAPTURE_SCOPE=full \
KIF_MODULE_B_OUTPUT_DIR=outputs/rebuttal_layer_ablation/middle_14_18/activations \
llama20 module_b
' 2>&1 | tee outputs/rebuttal_layer_ablation/middle_14_18_module_b_time.log
```

Then Module C:

```bash
python - <<'PY'
import torch
from llama20.modules.module_c import SignatureMiningConfig, ROMEHyperParams, SignatureExtractor

layers = [14, 15, 16, 17, 18]
config = SignatureMiningConfig(
    activations_dir="outputs/rebuttal_layer_ablation/middle_14_18/activations",
    output_dir="outputs/rebuttal_layer_ablation/middle_14_18/signatures",
    rome_hparams=ROMEHyperParams(
        layers=layers,
        layer_selection="top_k",
        target_module="mlp",
        significance_threshold=1.5,
    ),
    top_k_directions=3,
    min_prompts_per_subject=2,
    use_semantic_negatives=True,
    min_controls_per_subject=1,
    allow_synthetic_fallback=True,
    enable_oversampling=False,
    negative_pool_mode="match_positives",
    fixed_negative_pool_size=100,
    synthetic_fraction=0.10,
    activation_strategy="mean_token",
    standardize_dims=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_half_precision=False,
    enable_memory_cleanup=True,
    cleanup_frequency=5,
)
extractor = SignatureExtractor(config)
signatures = extractor.extract_all_signatures()
extractor.save_signature_index(signatures)
extractor.create_summary_report()
PY
```

### Summarize early/middle/peak

For peak, use the existing run if it is still in `outputs/signatures`. If not, copy or rerun the peak band under `outputs/rebuttal_layer_ablation/peak_23_27`.

```bash
python - <<'PY'
import json
from pathlib import Path
import statistics as stats

roots = {
    "early_05_09": Path("outputs/rebuttal_layer_ablation/early_05_09/signatures/signature_index.json"),
    "middle_14_18": Path("outputs/rebuttal_layer_ablation/middle_14_18/signatures/signature_index.json"),
    "peak_23_27_current": Path("outputs/signatures/signature_index.json"),
}

rows = []
for name, path in roots.items():
    if not path.exists():
        print(f"MISSING: {name}: {path}")
        continue
    obj = json.loads(path.read_text())
    scores = []
    successes = 0
    best_layers = []
    for subj, rec in obj.get("subjects", {}).items():
        if rec.get("status") == "success":
            successes += 1
        if rec.get("best_score") is not None:
            scores.append(float(rec["best_score"]))
        if rec.get("best_layer") is not None:
            best_layers.append(int(rec["best_layer"]))
    rows.append((name, successes, len(obj.get("subjects", {})), stats.mean(scores) if scores else None, min(scores) if scores else None, max(scores) if scores else None, best_layers))

print("band,successes,total,mean_best_score,min_best_score,max_best_score,best_layers")
for r in rows:
    print(r)
PY
```

### Rebuttal comment if good

If peak 23-27 is clearly best or most stable:

> We added a layer-band sensitivity check to address whether capsule placement is arbitrary. Early layers (5-9), middle layers (14-18), and the high-salience band (23-27) were compared using the same activation-mining and signature-extraction pipeline. The high-salience band produced the strongest and most reliable subject-specific separation, with successful capsule construction across the target subjects. This supports using the Cohen's-d localization diagnostic for capsule placement rather than selecting layers ad hoc.

### Alternative comment if not good

If early or middle layers are comparable or better:

> The additional layer-band check shows that useful subject signatures can appear across multiple bands, although the selected band remains a strong candidate. We therefore soften the wording: ERUF does not require a universally fixed layer band; instead, it uses a data-driven localization diagnostic to select high-salience layers per model and subject.

## Experiment B: clean local compute-cost timing

### Purpose

The uploaded compute table is currently incomplete. The quickest useful compute-cost evidence is clean timing for the local ERUF stages that already run on a 16 GB GPU:

- Module B activation collection for 5-layer band
- Module C signature mining
- Module D capsule forging

This does not fully answer total ERUF cost because Module 7 remains the expensive stage, but it gives a defensible partial cost table while the 4090/24 GB run is pending.

### Command wrapper

Use `/usr/bin/time -v` around every stage and save logs under `outputs/rebuttal_timing/`:

```bash
cd "$ERUF"
mkdir -p outputs/rebuttal_timing

/usr/bin/time -v llama20 module_d 2>&1 | tee outputs/rebuttal_timing/module_d_peak_23_27_time.log
```

For Module 7 on the 5060 Ti, do not continue unless the run is already near completion. One epoch taking 6+ hours means it is a poor use of this GPU. Move the full 8B run to the 24 GB GPU.

### Rebuttal comment if good

> We added wall-clock accounting for ERUF's activation-mining and capsule-construction stages. These stages are feasible on a single 16 GB consumer GPU for the evaluated entity set; the main additional cost relative to optimization-only baselines comes from activation collection and the final LoRA distillation stage. We now report this cost explicitly rather than treating ERUF as cost-free.

### Alternative comment if costly

> ERUF has a higher up-front mechanistic analysis cost than simpler loss-only baselines. We clarify this tradeoff: ERUF is intended for settings where stronger internal attenuation and recovery resistance are worth additional offline analysis, not as the cheapest possible unlearning update.

## Experiment C: small gate/alpha capsule-only sensitivity

### Purpose

This should be run only after the layer-band results. It answers whether the capsule behavior is brittle to alpha/gate settings without requiring full Module 7.

### Minimal grid

Use a very small grid:

- `default_strength`: -0.5, -0.8, -1.0
- `z_tau`: 2.5, 3.0, 3.5

Do not run full LoRA distillation for all combinations. Only run capsule/sentinel harvest or a small capsule-only evaluation. Report only if the default region is stable.

### Use only if

Use this result only if it shows that the default setting is not a single fragile point. Good outcome:

- target prompts fire capsules consistently;
- benign prompts do not overfire;
- stronger suppression does not obviously damage general responses.

If the result is unstable, do not include a table. Instead write that gate calibration is an implementation limitation and avoid claiming hyperparameter robustness.

## Experiment D: wait for 3B baselines

The most important review-answering table remains the 3B baseline table. Once the separate PC finishes SimNPO/ReGLU/LUNAR/OPT-OUT on 3B, immediately evaluate:

- Utility drift
- SMR
- EL10
- mechanism state

### Rebuttal comment if good

> We added small-model baseline comparisons on the same 3B setting where ERUF is least favorable. The results show that the failure mode is not unique to ERUF: baselines either retain high surface leakage, amplify internal target activation, or incur larger utility degradation. This supports our revised claim that small models expose a capacity-limited regime rather than a setting where existing baselines solve entity-level representation unlearning.

### Alternative comment if a baseline wins

> The additional 3B baseline results show that ERUF is not uniformly superior in the smallest-model regime. We therefore revise the claim: ERUF is strongest in standard 7-8B+ settings and exposes small-model capacity limits, while some baseline objectives may be preferable under strict low-capacity constraints.

## Final recommendation for the RTX 5060 Ti

Run in this order:

1. Layer-band ablation: early 5-9 and middle 14-18. Peak 23-27 already exists.
2. Clean timing logs for Module B/C/D.
3. Optional small capsule-only gate/alpha sensitivity.
4. Do not spend the 5060 Ti on full 8B Module 7 if one epoch is taking 6+ hours. Move that to the 24 GB GPU.
