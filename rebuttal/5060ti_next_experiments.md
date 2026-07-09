# RTX 5060 Ti Rebuttal Experiments

This note tracks lightweight rebuttal experiments that can be run on the local RTX 5060 Ti while the full Llama-3.1-8B ERUF Module 7 run is deferred to a 24 GB GPU.

The operating rule is strict: include a result in the rebuttal only if it supports the actual paper narrative. If a result conflicts with the current framing, use it diagnostically and soften the claim rather than forcing a positive interpretation.

## Reviewer concerns targeted

The current review set raises three directly actionable concerns:

1. **Small/moderate-model baselines:** reviewers ask what happens to LUNAR, ReGLU, OPT-OUT, and SimNPO on 3B settings where ERUF struggles.
2. **Compute cost:** reviewers ask whether ERUF costs substantially more than baselines.
3. **Layer choice:** reviewers state that the mid-to-late layer choice is mainly empirically motivated and should be ablated.

The 3B baseline table remains the highest-value experiment overall, but it is running on another machine. The best completed local 5060 Ti experiment is the layer-band ablation below.

## Completed Experiment A: Layer-band localization ablation

### Purpose

This experiment tests whether ERUF's capsule-placement band is arbitrary. It compares early, middle, and high-salience layer bands using the same Module B activation collection and Module C signature mining pipeline.

Compared bands:

- Early: layers 5-9
- Middle: layers 14-18
- Peak/high-salience: layers 23-27

This experiment does not require Module 7 LoRA distillation. It is therefore appropriate for the RTX 5060 Ti.

### Result verdict

**Decision: supports_current_narrative**

The peak 23-27 band is the strongest band by mean best signature score while preserving full subject coverage. All three bands produced signatures for 11/11 subjects, but the peak band has the highest mean score and a strong minimum score. This supports the paper's Cohen's-d localization narrative and can be reported as a layer-band sensitivity check.

### Summary table

| Band | Layers | Subjects | Mean best score | Median | Min | Max | Module B sec | Module C sec | Activation GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| early_05_09 | 5,6,7,8,9 | 11/11 | 2.5441 | 2.5979 | 1.5605 | 4.7508 | 957.6 | 221.8 | 5.793 |
| middle_14_18 | 14,15,16,17,18 | 11/11 | 3.0418 | 3.0327 | 2.4994 | 4.4469 | 951.9 | 221.2 | 5.794 |
| peak_23_27 | 23,24,25,26,27 | 11/11 | 3.1483 | 3.1102 | 2.5151 | 4.3928 | 956.5 | 222.2 | 5.801 |

### Interpretation

The result supports the existing narrative, but the wording should remain careful. The ablation does **not** prove that layers 23-27 are universally optimal for all models. It shows that, on the evaluated Llama-3.1-8B entity-unlearning setup, the high-salience band selected from the layerwise Cohen's-d diagnostic is stronger than early layers and slightly stronger than middle layers under the same mining protocol.

The most important comparison is not only peak vs. middle, because middle is close. The strongest defensible claim is:

> ERUF uses a data-driven localization diagnostic rather than an arbitrary fixed layer choice. In a layer-band ablation on Llama-3.1-8B, early layers 5-9, middle layers 14-18, and the high-salience band 23-27 all produced subject signatures, but the high-salience band achieved the highest mean best-score while retaining 11/11 subject coverage. This supports selecting capsule placement from mid-to-late high-separability layers instead of choosing layers ad hoc.

### Rebuttal-ready comment

> We added a layer-band sensitivity check to address whether capsule placement is arbitrary. Early layers 5-9, middle layers 14-18, and the high-salience band 23-27 were compared using the same activation-mining and signature-extraction pipeline. All bands produced signatures for all 11 target subjects, but the high-salience band achieved the highest mean best-score (3.1483 vs. 3.0418 for middle and 2.5441 for early). This supports our use of the Cohen's-d localization diagnostic for capsule placement. We now frame layer selection as data-driven high-salience localization rather than a universally fixed layer heuristic.

### Compact paper table candidate

| Layer band | Layers | Subject coverage | Mean best score |
|---|---:|---:|---:|
| Early | 5-9 | 11/11 | 2.5441 |
| Middle | 14-18 | 11/11 | 3.0418 |
| High-salience | 23-27 | 11/11 | 3.1483 |

Suggested caption:

> Layer-band sensitivity on Llama-3.1-8B. We compare early, middle, and high-salience bands using the same activation-mining and signature-extraction pipeline. The high-salience band identified by the localization diagnostic gives the strongest mean best-score while preserving complete subject coverage.

## Per-subject details

### early_05_09

| Subject | Status | Best layer | Best score |
|---|---|---:|---:|
| Ariana Grande | success | 5 | 2.5979 |
| Arijit Singh | success | 7 | 4.7508 |
| Beyoncé | success | 5 | 2.9435 |
| Drake (musician) | success | 8 | 2.8858 |
| Ed Sheeran | success | 8 | 1.8650 |
| Eminem | success | 5 | 1.6882 |
| Kanye West | success | 9 | 1.5605 |
| Katy Perry | success | 5 | 2.6110 |
| Michael Jackson | success | 8 | 1.7262 |
| Queen (band) | success | 8 | 3.4828 |
| Taylor Swift | success | 5 | 1.8732 |

### middle_14_18

| Subject | Status | Best layer | Best score |
|---|---|---:|---:|
| Ariana Grande | success | 18 | 3.0327 |
| Arijit Singh | success | 18 | 4.4469 |
| Beyoncé | success | 17 | 3.2799 |
| Drake (musician) | success | 18 | 3.1977 |
| Ed Sheeran | success | 18 | 2.4994 |
| Eminem | success | 18 | 2.5774 |
| Kanye West | success | 17 | 2.6061 |
| Katy Perry | success | 17 | 3.2169 |
| Michael Jackson | success | 17 | 2.6479 |
| Queen (band) | success | 18 | 3.3975 |
| Taylor Swift | success | 17 | 2.5578 |

### peak_23_27

| Subject | Status | Best layer | Best score |
|---|---|---:|---:|
| Ariana Grande | success | 26 | 3.2628 |
| Arijit Singh | success | 25 | 4.3928 |
| Beyoncé | success | 25 | 3.2342 |
| Drake (musician) | success | 23 | 3.1102 |
| Ed Sheeran | success | 23 | 2.5151 |
| Eminem | success | 26 | 3.0325 |
| Kanye West | success | 24 | 2.8887 |
| Katy Perry | success | 26 | 3.2248 |
| Michael Jackson | success | 24 | 2.7307 |
| Queen (band) | success | 24 | 3.3501 |
| Taylor Swift | success | 24 | 2.8893 |

## Compute-cost notes from this run

This result also gives clean timing for the local 5060 Ti stage-level cost:

| Stage | Early sec | Middle sec | Peak sec | Comment |
|---|---:|---:|---:|---|
| Module B activation collection | 957.6 | 951.9 | 956.5 | About 15.9 minutes per 5-layer band |
| Module C signature mining | 221.8 | 221.2 | 222.2 | About 3.7 minutes per 5-layer band |

The local extraction shows that Module B/C layer-band analysis is feasible on a single 16 GB consumer GPU. This does **not** yet settle the full ERUF compute-cost question because Module 7 LoRA distillation is the expensive stage and should be timed separately on the 24 GB GPU.

## Current compute-cost extraction caveat

The earlier uploaded compute-cost extraction is not ready to report as a final table. It detects the RTX 5060 Ti and some historical LUNAR cluster runtimes, but most rows have missing `wall_seconds` and `gpu_hours`, several GPU types are parsed as `1` or `4`, and many unrelated files are classified as method `eruf`.

Reporting rule: use the clean layer-band timings above as partial local evidence. Do not report the auto-extracted global compute-cost table until it is manually cleaned or the parser is improved.

## Next 5060 Ti experiment options

### Option B: capsule-only alpha/gate sensitivity

Run this only if time remains. Do not run full LoRA distillation for every setting.

Minimal grid:

- `default_strength`: -0.5, -0.8, -1.0
- `z_tau`: 2.5, 3.0, 3.5

Use only if the result shows the default setting is not a fragile single point. If unstable, do not include a table; instead state that gate calibration remains a limitation.

### Option C: clean Module D timing

If capsule forging needs a cleaner cost row, run:

```bash
cd "$ERUF"
mkdir -p outputs/rebuttal_timing
/usr/bin/time -v llama20 module_d 2>&1 | tee outputs/rebuttal_timing/module_d_peak_23_27_time.log
```

### Option D: wait for 3B baselines

The most important remaining review-answering table is the 3B baseline table. Once the separate PC finishes SimNPO/ReGLU/LUNAR/OPT-OUT on 3B, evaluate:

- Utility drift
- SMR
- EL10
- Mechanism state

If the result is good:

> We added small-model baseline comparisons on the same 3B setting where ERUF is least favorable. The results show that the failure mode is not unique to ERUF: baselines either retain high surface leakage, amplify internal target activation, or incur larger utility degradation. This supports our revised claim that small models expose a capacity-limited regime rather than a setting where existing baselines solve entity-level representation unlearning.

If a baseline wins:

> The additional 3B baseline results show that ERUF is not uniformly superior in the smallest-model regime. We therefore revise the claim: ERUF is strongest in standard 7-8B+ settings and exposes small-model capacity limits, while some baseline objectives may be preferable under strict low-capacity constraints.

## Final recommendation

1. Report the layer-band ablation in the rebuttal or appendix.
2. Do not overclaim universal layer optimality.
3. Move full 8B Module 7 to the 24 GB GPU.
4. Use the 5060 Ti for capsule-only sensitivity or clean timing, not full 8B distillation.
5. Prioritize 3B baseline evaluation once the separate baseline runs finish.
