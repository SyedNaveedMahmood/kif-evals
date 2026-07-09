# Successful and Helpful Rebuttal Experiments Summary

This file summarizes the rebuttal experiments completed so far and how each one should be used in the paper/rebuttal. The compute-cost experiments are intentionally separated because the full 8B run is still ongoing on the 24 GB VRAM machine.

## Executive verdict

The rebuttal package is now meaningfully stronger. The completed experiments address three reviewer-facing weaknesses: layer choice, alpha_eff behavior, and small-model baseline context. The strongest story is that ERUF uses data-driven signature localization, alpha_eff behaves as a controllable internal attenuation knob, and the 3B regime is a capacity-limited stress setting where baselines either leak, retain internal traces, or suppress targets at large utility cost.

## Completed successful/helpful experiments

| # | Experiment | Model/setting | Main result | Helpful for rebuttal? | Correct framing |
|---:|---|---|---|---|---|
| 1 | Layer-band localization ablation | Llama-family ERUF localization, 11 subjects | Peak/high-salience band 23-27 achieved the highest mean best score, with full 11/11 subject coverage | Yes, strong | Layer selection is data-driven, not an arbitrary mid-to-late heuristic |
| 2 | Mechanistic alpha_eff ablation | Module-E-only controlled intervention, 6 subjects | Increasing alpha_eff monotonically reduced target-signature projection energy while benign runtime attenuation stayed zero | Yes, strong but scoped | alpha_eff is a controllable internal attenuation knob, not full end-to-end unlearning proof |
| 3 | Llama 3B baseline stress test | LUNAR, ReGLU, SimNPO | Baselines expose tradeoffs: LUNAR/SimNPO preserve utility better but leak, ReGLU suppresses more but damages utility | Yes, strong | 3B is capacity-limited; existing baselines do not cleanly solve entity-level representation unlearning |
| 4 | Qwen 3B LUNAR baseline | Qwen/Qwen2.5-3B-Instruct | LUNAR caused large utility damage and did not improve EL10 | Yes | LUNAR is not a clean solution for Qwen 3B; apparent suppression comes with poor utility/internal tradeoff |
| 5 | Qwen 3B ReGLU baseline | Qwen/Qwen2.5-3B-Instruct | ReGLU achieved near-total direct surface/internal suppression but caused severe utility degradation | Yes | ReGLU is a strong suppressive baseline, but destructive in the small Qwen 3B regime |
| 6 | Qwen 3B ReGLU robustness smoke | Qwen/Qwen2.5-3B-Instruct, 80-row capped saved-suite eval | Despite low direct leakage, adversarial recovery success remained high in the capped robustness suite | Yes, but report as capped/smoke unless completed | Direct SMR/EL10 suppression is not enough; adversarial and robust probes expose residual recoverability/tradeoffs |
| 7 | Local feasibility fix for OPT-OUT retain rows | Llama/Qwen baseline framework | Added Adele retain rows locally, matching the previous cluster dataset-check logic | Yes, operational | Prior OPT-OUT local failure was caused by missing retain rows, not by the method itself |
| 8 | ReGLU CPU RILA backend for 16 GB GPU | Qwen 3B ReGLU on RTX 5060 Ti | ReGLU completed on 5060 Ti after moving RILA eigensolves off CUDA | Yes, operational and reproducibility useful | Hyperparameters stayed fair; only numerical backend changed to avoid local CUDA OOM |

## Key quantitative results

### 1. Layer-band localization ablation

| Layer band | Layers | Subject coverage | Mean best score | Interpretation |
|---|---:|---:|---:|---|
| Early | 5-9 | 11/11 | 2.5441 | Works, but weaker |
| Middle | 14-18 | 11/11 | 3.0418 | Strong |
| High-salience / peak | 23-27 | 11/11 | 3.1483 | Best among tested bands |

**Rebuttal value:** This directly addresses the review concern that mid-to-late layer selection looked heuristic. The right claim is that ERUF uses data-driven high-salience localization, not that one layer band is universally optimal.

**Suggested wording:**

> We added a layer-band sensitivity check over early, middle, and high-salience bands. All bands produced signatures for all 11 target subjects, but the high-salience band achieved the highest mean best score. This supports our use of a data-driven localization diagnostic rather than arbitrary capsule placement.

### 2. Mechanistic alpha_eff ablation

| alpha_eff | Theoretical attenuation | Target attenuation mean | Target post/pre projection ratio | Benign runtime attenuation |
|---:|---:|---:|---:|---:|
| 0.00 | 0.0% | 0.0% | 1.0000 | 0.0% |
| 0.25 | 43.8% | 43.8% | 0.5625 | 0.0% |
| 0.50 | 75.0% | 75.0% | 0.2500 | 0.0% |
| 0.75 | 93.8% | 93.8% | 0.0625 | 0.0% |
| 1.00 | 100.0% | 100.0% | 0.0000 | 0.0% |

Subjects used: Ariana Grande, Arijit Singh, Beyonce, Drake (musician), Ed Sheeran, Eminem.

**Rebuttal value:** This directly answers the request for an alpha_eff ablation. It shows monotonic target-signature attenuation under a controlled intervention while benign prompts are not modified under the router-gated path.

**Important caveat:** This is a mechanistic sanity check, not a replacement for full Module 7 generation/unlearning evaluation.

**Suggested wording:**

> We added a Module-E-only mechanistic alpha_eff ablation. Holding the mined capsule direction fixed and sweeping controlled alpha_eff values, the intervention monotonically reduces target-signature projection energy while leaving benign prompts unmodified under the router-gated runtime path. This validates alpha_eff as a controllable internal attenuation knob rather than a brittle implementation artifact.

### 3. Llama 3B baseline stress test

| Method | Utility delta loss | SMR | EL10 ratio | Adversarial recovery | Main failure mode |
|---|---:|---:|---:|---:|---|
| LUNAR | 0.0488 | 0.7407 | 0.1431 | 0.7156 | High surface/adversarial recovery |
| ReGLU | 3.0884 | 0.1111 | 0.0150 | 0.1310 | Severe utility/locality damage |
| SimNPO | 0.0007 | 0.5000 | 1.5891 | 0.6161 | Preserves utility but worsens hidden extraction |

**Rebuttal value:** This addresses the review concern that baselines should be evaluated in the small-model setting where ERUF is weakest. It shows that the 3B setting exposes method tradeoffs rather than being solved by existing baselines.

**Suggested wording:**

> We added 3B baseline evaluations for LUNAR, ReGLU, and SimNPO. The results show that the 3B regime is difficult across methods. Utility-preserving baselines retain substantial surface or adversarial recovery, while stronger suppressive baselines reduce target signals at the cost of severe utility and locality degradation.

### 4. Qwen 3B LUNAR baseline

| Metric | Qwen 3B LUNAR result |
|---|---:|
| Benign pre loss | 4.4581 |
| Benign pre PPL | 86.3210 |
| Post loss | 7.2096 |
| Post PPL | 1352.3538 |
| Loss delta | 2.7515 |
| PPL delta | 1266.0328 |
| Average subject mention rate | 0.2000 |
| Average keyword hit rate | 0.0302 |
| EL10 ratio | 1.0377 |
| Pre/post similarity | 0.2850 |

Evaluated subjects: Ariana Grande, Arijit Singh, Beyonce, Drake (musician), Ed Sheeran.

**Rebuttal value:** Helpful because it shows LUNAR does not cleanly solve Qwen 3B. It causes major utility degradation while EL10 remains slightly above 1, so internal extraction does not improve.

**Suggested wording:**

> On Qwen 3B, LUNAR reduces some surface leakage but causes severe utility degradation and does not reduce internal extraction likelihood. This supports the interpretation that the small Qwen regime is a stress setting rather than a baseline-solved setting.

### 5. Qwen 3B ReGLU direct baseline

| Metric | Qwen 3B ReGLU result |
|---|---:|
| Benign pre loss | 4.4581 |
| Benign pre PPL | 86.3210 |
| Post loss | 6.6825 |
| Post PPL | 798.2968 |
| Loss delta | 2.2244 |
| PPL delta | 711.9759 |
| Average subject mention rate | 0.0000 |
| Average keyword hit rate | 0.0052 |
| EL10 ratio | 0.000071 |
| Pre/post similarity | 0.2023 |
| Wall-clock on RTX 5060 Ti with CPU RILA | 28:32.69 |
| Peak host RSS | 8.27 GB |

Evaluated subjects: Ariana Grande, Arijit Singh, Beyonce, Drake (musician), Ed Sheeran.

**Rebuttal value:** Helpful because it shows ReGLU can strongly suppress the target, but the suppression is destructive in the Qwen 3B setting. It is not a clean utility-preserving representation-unlearning solution.

**Suggested wording:**

> ReGLU achieves near-total target suppression on Qwen 3B, but this comes with severe utility degradation. This reinforces the capacity-limited tradeoff framing: stronger suppression is possible, but in small models it can be accompanied by broad behavioral damage.

### 6. Qwen 3B ReGLU robustness smoke from saved-suite evaluator

This is a capped 80-row saved-suite run, so it should be treated as a diagnostic unless the full suite is completed. It is still helpful because it targets the same robustness-style evidence family already used in the paper: name-agnostic probing, adversarial forget-recovery, BLUR/mixed retain-forget probes, syntactic locality, and RWKU-style robustness.

| Subsuite | Completed rows | Total rows | Use in rebuttal? |
|---|---:|---:|---|
| Fast entity bundle | 80 | 657 | Yes, diagnostic/capped |
| Adversarial recovery | 80 | 670 | Yes, diagnostic/capped |
| RWKU-style robustness | 80 | 418 | Yes, diagnostic/capped |

| Metric family | ReGLU capped result | Why it helps |
|---|---:|---|
| Name-agnostic forget target alias hit | 0.1111 | Shows residual recoverability under name-agnostic probing despite direct SMR 0.0 |
| Name-agnostic target mass | 0.0563 | Shows target-relevant probability mass can remain under robustness probes |
| BLUR/mixed forget target alias hit | 0.0000 | Confirms strong direct suppressive behavior in mixed prompts |
| Adversarial target alias hit | 0.0250 | Surface alias recovery is low under adversarial prompts |
| Adversarial recovery success | 0.7625 | Important: robust recovery criteria still fire often despite low alias hits |
| RWKU forget alias hit | 0.0725 | Forget-target alias leakage is reduced but not zero in RWKU-style probes |
| RWKU matched-control alias hit | 0.7500 | Some matched-control retain behavior survives in the capped sample |
| RWKU neighbor-locality alias hit | 0.2500 | Neighbor locality is weaker than matched-control locality in the capped sample |

**Rebuttal value:** This is the most useful additional Qwen 3B ReGLU result because it prevents the result from looking like a clean ReGLU win. The direct evaluator shows strong suppression, but the robustness suite shows that adversarial recovery criteria still succeed frequently in the capped sample. This supports the paper's broader argument that output suppression, direct SMR, and direct EL10 are not sufficient by themselves; adversarial and robust probes are needed to diagnose residual recoverability and collateral tradeoffs.

**Suggested wording:**

> In a capped Qwen 3B ReGLU robustness run, direct alias recovery remains low, but adversarial recovery criteria still succeed frequently. This suggests that ReGLU's strong direct suppression does not remove the need for adversarial and robustness-oriented diagnostics, and it further supports treating small-model baselines as tradeoff points rather than clean representation-erasure solutions.

**Reporting caveat:** Do not present this as the final full-suite result unless the remaining rows are completed. Label it as an 80-row capped robustness diagnostic, or finish the full saved-suite evaluation before adding it to the main rebuttal table.

### 7. Qwen 3B SimNPO feasibility attempt

| Metric | Result |
|---|---|
| Method | SimNPO-GradDiff |
| Model | Qwen/Qwen2.5-3B-Instruct |
| Trainable parameters | 3,085,938,688 / 3,085,938,688 |
| Trainable percent | 100% |
| Config | Original-style full-parameter SimNPO values |
| Outcome | CUDA OOM during backward on RTX 5060 Ti |
| Report as baseline metric? | No |

**Interpretation:** This is not a paper result. It is a local resource-limit note. SimNPO full-parameter training should be moved to the 24 GB GPU or cluster if the goal is a fair Qwen 3B baseline. Do not report an altered local SimNPO run using lm_head-only, fewer epochs, or LoRA unless explicitly labeled as a non-comparable feasibility variant.

## Current combined story for rebuttal

| Review concern | Completed answer |
|---|---|
| Layer choice looks heuristic | Layer-band ablation shows the selected high-salience band is empirically strongest among tested bands |
| alpha_eff needs ablation | Mechanistic alpha_eff sweep shows monotonic target-signature attenuation |
| Baselines missing in weak 3B setting | Llama 3B and Qwen 3B baseline stress tests show leakage/utility tradeoffs across methods |
| Baselines might solve small-model cases | LUNAR does not; ReGLU suppresses strongly but destructively; ReGLU robustness probes still show adversarial recoverability; SimNPO needs larger hardware for fair Qwen full-parameter run |
| OPT-OUT failed locally | Cause identified: missing retain rows. Fixed by adding Adele retain-control rows locally |

## Recommended paper framing

> We added targeted rebuttal experiments addressing layer selection, alpha_eff behavior, and missing small-model baseline context. A layer-band sensitivity check supports data-driven high-salience capsule placement. A controlled alpha_eff sweep shows monotonic target-signature attenuation under the capsule intervention. Additional 3B baseline stress tests show that small models expose tradeoffs across methods: LUNAR and SimNPO-like utility-preserving methods can retain surface or internal leakage, while ReGLU-like suppressive methods can reduce target signals but at substantial utility cost. A capped robustness diagnostic further shows that strong direct suppression does not eliminate the need for adversarial and robustness-oriented probes. These results support framing the 3B setting as capacity-limited rather than as a setting where existing baselines cleanly solve entity-level representation unlearning.

## What not to overclaim

- Do not claim ERUF uniformly beats every baseline at 3B.
- Do not claim ReGLU fails to forget on Qwen 3B. It forgets strongly in the direct evaluator, but destructively.
- Do not claim the capped ReGLU robustness smoke is a final full-suite result unless the remaining rows are completed.
- Do not report Qwen 3B SimNPO as a metric result from the 5060 Ti attempt. It is only a local hardware feasibility failure.
- Do not include OPT-OUT until rerun with the Adele retain-control prompt file.
- Do not present the alpha_eff ablation as end-to-end unlearning performance. It is a mechanistic intervention sanity check.

## Next experiment priorities

| Priority | Experiment | Machine | Why |
|---:|---|---|---|
| 1 | Finish the ongoing 8B full run | 24 GB VRAM PC | Highest value because it gives full-model evidence under the main setting |
| 2 | Complete Qwen 3B ReGLU saved-suite evaluation | RTX 5060 Ti | Evaluation-only; converts the useful 80-row robustness diagnostic into a stronger full/capped result |
| 3 | Run Qwen 3B LUNAR saved-suite smoke/full evaluation | RTX 5060 Ti | Evaluation-only; gives a fairer comparison against the ReGLU robustness diagnostic |
| 4 | Llama 3B OPT-OUT with Adele retain-control file | 24 GB PC or cluster preferred | Fixes the previously failed baseline, but likely too heavy for 16 GB |
| 5 | Qwen 3B SimNPO full-parameter run | 24 GB PC or cluster | Fair SimNPO requires full-parameter training, which OOMs on 5060 Ti |

## Best use of RTX 5060 Ti next

Use the 5060 Ti for evaluation-only work, not full-parameter training. The best next local runs are robust/adversarial evaluations of the already-trained Qwen 3B LUNAR and ReGLU outputs. These should be much more feasible than SimNPO or OPT-OUT training and would give additional evidence for the reviewer concern about small-model baselines.
