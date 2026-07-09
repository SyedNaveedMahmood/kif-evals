# Successful and Helpful Rebuttal Experiments Summary

This file summarizes only the experiments that are directly useful for the paper/rebuttal narrative. The emphasis is on evidence that answers reviewer concerns: layer choice, alpha_eff behavior, and whether existing baselines cleanly solve the weak 3B setting.

## Executive verdict

The rebuttal package is now materially stronger. The completed experiments support three claims: ERUF's layer choice is data-driven, alpha_eff behaves as a controllable internal attenuation knob, and the 3B regime exposes baseline tradeoffs rather than being solved by existing methods. The new Qwen 3B baseline suite is especially useful because ReGLU looks very strong under direct SMR/EL10 evaluation but still shows high adversarial recovery success in the full saved-suite evaluator, while LUNAR has lower adversarial recovery but already showed severe utility collapse and no EL10 improvement in the direct evaluation.

## Completed experiments useful for rebuttal

| # | Experiment | Model/setting | Main result | Rebuttal value |
|---:|---|---|---|---|
| 1 | Layer-band localization ablation | ERUF localization, 11 subjects | High-salience band 23-27 had the best mean score with 11/11 subject coverage | Answers the concern that layer choice was heuristic |
| 2 | Mechanistic alpha_eff ablation | Module-E-only controlled intervention, 6 subjects | Larger alpha_eff monotonically reduced target-signature projection while benign runtime attenuation stayed zero | Answers the request for alpha_eff ablation |
| 3 | Llama 3B baseline stress test | LUNAR, ReGLU, SimNPO | Baselines split between leakage and utility damage | Shows the 3B regime is capacity-limited across methods |
| 4 | Qwen 3B LUNAR direct baseline | Qwen/Qwen2.5-3B-Instruct | Large utility collapse and EL10 ratio above 1 | Shows LUNAR does not cleanly solve Qwen 3B |
| 5 | Qwen 3B ReGLU direct baseline | Qwen/Qwen2.5-3B-Instruct | Near-total direct suppression but severe utility degradation | Shows ReGLU is strong but destructive |
| 6 | Qwen 3B saved-suite baseline evaluation | LUNAR and ReGLU full saved-suite evals | ReGLU has low alias hit but high adversarial recovery success; LUNAR has lower recovery but poor direct utility/internal tradeoff | Shows direct SMR/EL10 are insufficient, matching the paper's robust/adversarial diagnostic framing |
| 7 | OPT-OUT retain-data fix | Llama/Qwen baseline framework | Added Adele retain rows to avoid empty retain-pool failure | Fixes the earlier local OPT-OUT failure cause |
| 8 | ReGLU CPU RILA backend | Qwen 3B on RTX 5060 Ti | ReGLU completed with original hyperparameters after moving RILA eigensolves off CUDA | Operational reproducibility fix, not a changed method |

## 1. Layer-band localization ablation

| Layer band | Layers | Subject coverage | Mean best score | Interpretation |
|---|---:|---:|---:|---|
| Early | 5-9 | 11/11 | 2.5441 | Works, but weaker |
| Middle | 14-18 | 11/11 | 3.0418 | Strong |
| High-salience / peak | 23-27 | 11/11 | 3.1483 | Best among tested bands |

**Rebuttal value:** This supports the claim that capsule placement follows a data-driven high-salience localization diagnostic rather than an arbitrary mid-to-late-layer heuristic.

**Suggested wording:**

> We added a layer-band sensitivity check over early, middle, and high-salience bands. All bands produced signatures for all 11 target subjects, but the high-salience band achieved the highest mean best score. This supports our use of a data-driven localization diagnostic rather than arbitrary capsule placement.

## 2. Mechanistic alpha_eff ablation

| alpha_eff | Theoretical attenuation | Target attenuation mean | Target post/pre projection ratio | Benign runtime attenuation |
|---:|---:|---:|---:|---:|
| 0.00 | 0.0% | 0.0% | 1.0000 | 0.0% |
| 0.25 | 43.8% | 43.8% | 0.5625 | 0.0% |
| 0.50 | 75.0% | 75.0% | 0.2500 | 0.0% |
| 0.75 | 93.8% | 93.8% | 0.0625 | 0.0% |
| 1.00 | 100.0% | 100.0% | 0.0000 | 0.0% |

Subjects used: Ariana Grande, Arijit Singh, Beyonce, Drake (musician), Ed Sheeran, Eminem.

**Rebuttal value:** This directly answers the request for alpha_eff sensitivity. It should be described as a mechanistic sanity check, not as a replacement for full end-to-end unlearning evaluation.

**Suggested wording:**

> Holding the mined capsule direction fixed and sweeping controlled alpha_eff values, the intervention monotonically reduces target-signature projection energy while leaving benign prompts unmodified under the router-gated runtime path. This validates alpha_eff as a controllable internal attenuation knob rather than a brittle implementation artifact.

## 3. Llama 3B baseline stress test

| Method | Utility delta loss | SMR | EL10 ratio | Adversarial recovery | Main failure mode |
|---|---:|---:|---:|---:|---|
| LUNAR | 0.0488 | 0.7407 | 0.1431 | 0.7156 | High surface/adversarial recovery |
| ReGLU | 3.0884 | 0.1111 | 0.0150 | 0.1310 | Severe utility/locality damage |
| SimNPO | 0.0007 | 0.5000 | 1.5891 | 0.6161 | Preserves utility but worsens hidden extraction |

**Rebuttal value:** This addresses the reviewer concern that baselines should be tested in the same weak 3B regime where ERUF is less clean. The result supports a tradeoff framing: utility-preserving baselines can leak, while aggressive suppression can damage utility.

## 4. Qwen 3B direct baseline results

These are the direct Module-8-style baseline results on Qwen 3B over the 5 evaluated subjects: Ariana Grande, Arijit Singh, Beyonce, Drake (musician), and Ed Sheeran.

| Metric | LUNAR | ReGLU |
|---|---:|---:|
| Benign pre loss | 4.4581 | 4.4581 |
| Benign pre PPL | 86.3210 | 86.3210 |
| Post loss | 7.2096 | 6.6825 |
| Post PPL | 1352.3538 | 798.2968 |
| Loss delta | 2.7515 | 2.2244 |
| PPL delta | 1266.0328 | 711.9759 |
| Average subject mention rate | 0.2000 | 0.0000 |
| Average keyword hit rate | 0.0302 | 0.0052 |
| EL10 ratio | 1.0377 | 0.000071 |
| Pre/post similarity | 0.2850 | 0.2023 |

**Rebuttal value:** LUNAR reduces some surface leakage but causes severe utility degradation and does not reduce internal extraction likelihood. ReGLU strongly suppresses direct leakage and EL10, but it also causes large utility degradation. This prevents a simplistic claim that either baseline cleanly solves Qwen 3B.

**Suggested wording:**

> On Qwen 3B, LUNAR reduces some surface leakage but causes severe utility degradation and does not reduce internal extraction likelihood. ReGLU achieves near-total direct suppression, but at the cost of substantial utility degradation. These results support treating the 3B setting as a capacity-limited tradeoff regime rather than a baseline-solved setting.

## 5. Qwen 3B full saved-suite baseline evaluation

The saved-suite evaluator completed all rows for both Qwen 3B baselines: fast entity bundle 657/657, adversarial recovery 670/670, and RWKU-style robustness 418/418 for both LUNAR and ReGLU.

| Suite metric | LUNAR | ReGLU | Rebuttal interpretation |
|---|---:|---:|---|
| Name-agnostic target alias hit | 0.0157 | 0.0510 | Both reduce direct name-agnostic aliasing; ReGLU still has residual name-agnostic alias hit |
| Name-agnostic target keyword hit rate | 0.0812 | 0.1560 | ReGLU retains more target-keyword signal under name-agnostic probes |
| Name-agnostic target mass | 0.0166 | 0.0663 | ReGLU leaves higher target-relevant probability mass despite direct EL10 collapse |
| BLUR/mixed forget target alias hit | 0.0083 | 0.0000 | ReGLU is stronger on direct mixed forget prompts |
| Matched-control retain alias hit | 0.2600 | 0.3000 | Retain/control behavior is not fully preserved by either method |
| Neighbor-locality retain alias hit | 0.1200 | 0.0700 | Neighbor locality is weak, especially for ReGLU |
| Generic benign refusal-like | 0.0000 | 0.0000 | The issue is not generic refusal behavior |
| Adversarial target alias hit | 0.0358 | 0.0284 | Surface alias recovery stays low for both under adversarial prompts |
| Adversarial target keyword hit rate | 0.0724 | 0.1567 | ReGLU retains more target-keyword signal under adversarial probes |
| Adversarial target mass | 0.0209 | 0.0574 | ReGLU retains higher target mass under adversarial probes |
| Adversarial recovery success | 0.3448 | 0.7194 | Key result: ReGLU still has high robust recovery success despite low alias hit |
| Refusal-like rate | 0.0119 | 0.0000 | Neither result is explained by broad refusal |
| RWKU forget alias hit | 0.1369 | 0.0833 | ReGLU reduces RWKU forget aliasing more than LUNAR |
| RWKU forget keyword hit rate | 0.0423 | 0.0996 | ReGLU leaves more keyword-level forget signal |
| RWKU forget target mass | 0.0353 | 0.0432 | ReGLU leaves slightly higher target mass in RWKU forget probes |
| RWKU matched-control alias hit | 0.2667 | 0.5000 | ReGLU preserves matched-control aliasing better than LUNAR in this suite |
| RWKU neighbor-locality alias hit | 0.1750 | 0.0750 | ReGLU has weaker neighbor-locality retention |

**Most useful rebuttal conclusion:** ReGLU is not a clean all-around win. It nearly eliminates direct SMR and EL10, but the full robustness suite still shows high adversarial recovery success, higher adversarial keyword rate, and higher adversarial target mass than LUNAR. LUNAR has lower adversarial recovery in the saved-suite evaluator, but its direct baseline result already showed severe utility collapse and no EL10 improvement. Together, these results support the paper's claim that direct surface metrics are insufficient and that robust/adversarial diagnostics are necessary.

**Suggested wording:**

> We additionally evaluated the completed Qwen 3B LUNAR and ReGLU baselines with the same robustness-style probes used in the paper. ReGLU remains strong on direct alias suppression, but its adversarial recovery success is high despite low adversarial alias hit, indicating that direct SMR/EL10 suppression does not imply robust entity erasure. LUNAR has lower adversarial recovery in this suite, but its direct evaluation shows severe utility degradation and no internal EL10 reduction. These complementary failure modes reinforce that Qwen 3B is a stress setting where baselines trade off direct suppression, robust recoverability, and utility preservation.

## 6. Qwen 3B SimNPO feasibility attempt

| Metric | Result |
|---|---|
| Method | SimNPO-GradDiff |
| Model | Qwen/Qwen2.5-3B-Instruct |
| Trainable parameters | 3,085,938,688 / 3,085,938,688 |
| Trainable percent | 100% |
| Config | Original-style full-parameter SimNPO values |
| Outcome | CUDA OOM during backward on RTX 5060 Ti |
| Report as baseline metric? | No |

**Interpretation:** This is not a paper result. It is a local resource-limit note. SimNPO full-parameter training should be moved to the 24 GB GPU or cluster if a fair Qwen 3B SimNPO baseline is needed. Do not report a modified local SimNPO run using lm_head-only, fewer epochs, or LoRA unless explicitly labeled as a non-comparable feasibility variant.

## Current combined story for rebuttal

| Review concern | Completed answer |
|---|---|
| Layer choice looks heuristic | Layer-band ablation shows the selected high-salience band is empirically strongest among tested bands |
| alpha_eff needs ablation | Mechanistic alpha_eff sweep shows monotonic target-signature attenuation |
| Baselines missing in weak 3B setting | Llama 3B and Qwen 3B baseline stress tests show leakage/utility tradeoffs across methods |
| Baselines might solve small-model cases | LUNAR does not cleanly solve Qwen 3B; ReGLU suppresses direct leakage but damages utility and still has high adversarial recovery success |
| Direct leakage metrics may be insufficient | Qwen saved-suite results show low alias hit can coexist with high adversarial recovery success |
| OPT-OUT failed locally | Cause identified: missing retain rows. Fixed by adding Adele retain-control rows locally |

## Recommended paper framing

> We added targeted rebuttal experiments addressing layer selection, alpha_eff behavior, and missing small-model baseline context. A layer-band sensitivity check supports data-driven high-salience capsule placement, while a controlled alpha_eff sweep shows monotonic target-signature attenuation under the capsule intervention. Additional 3B baseline stress tests show that small models expose tradeoffs across methods: LUNAR-style methods can avoid some robust recovery but may fail internal EL10/utility criteria, whereas ReGLU-style suppression can reduce direct leakage but incur substantial utility degradation and still show high adversarial recovery success. These results support framing the 3B setting as capacity-limited rather than as a setting where existing baselines cleanly solve entity-level representation unlearning.

## What not to overclaim

- Do not claim ERUF uniformly beats every baseline at 3B.
- Do not claim ReGLU fails to forget on Qwen 3B. It forgets strongly in the direct evaluator.
- Do not claim LUNAR is better overall than ReGLU. LUNAR has lower adversarial recovery in the saved-suite evaluator, but worse direct utility/internal behavior.
- Do not report Qwen 3B SimNPO as a metric result from the 5060 Ti attempt. It is only a local hardware feasibility failure.
- Do not include OPT-OUT until rerun with the Adele retain-control prompt file.
- Do not present the alpha_eff ablation as end-to-end unlearning performance. It is a mechanistic intervention sanity check.

## Next experiment priorities

| Priority | Experiment | Machine | Why |
|---:|---|---|---|
| 1 | Finish the ongoing 8B full ERUF run | 24 GB VRAM PC | Highest value because it gives full-model ERUF evidence under the main setting |
| 2 | Run Qwen 3B SimNPO full-parameter baseline only if larger compute is available | 24 GB PC or cluster | 5060 Ti OOMs under fair full-parameter SimNPO |
| 3 | Run Llama 3B OPT-OUT with Adele retain-control file | 24 GB PC or cluster preferred | Fixes the previously failed baseline |
| 4 | Use 5060 Ti for additional evaluation-only checks, not training | RTX 5060 Ti | Training-heavy baselines are not worth fighting on 16 GB |

## Best use of RTX 5060 Ti next

The 5060 Ti has already produced the most useful Qwen 3B saved-suite evidence for LUNAR and ReGLU. Do not spend it on full-parameter SimNPO or OPT-OUT training. The next useful local use would be lightweight evaluation-only checks on already saved models, or simply leave it idle while the 8B ERUF run finishes on the 24 GB machine.
