# Successful and Helpful Rebuttal Experiments Summary

This file summarizes only the experiments that are directly useful for the paper/rebuttal narrative. The emphasis is on evidence that answers reviewer concerns: layer choice, alpha_eff behavior, small-model baseline context, computational-cost context, and hidden-space clustering/recoverability.

## Executive verdict

The rebuttal package is now materially stronger. The completed experiments support four claims: ERUF's layer choice is data-driven, alpha_eff behaves as a controllable internal attenuation knob, the 3B regime exposes baseline tradeoffs rather than being solved by existing methods, and the hidden-space diagnostics should be framed as representation-level attenuation rather than formal erasure. The new Weakness 5 clustering diagnostic is useful because it directly answers the reviewer concern while avoiding an overclaim: forgotten entities show reduced subject-level separability and recoverability in the upper intervention layers, but we should describe this as attenuation, not complete deletion of the full entity manifold.

The completed Llama 3B OPT-OUT run should be used carefully. It is useful rebuttal evidence, but not a simple win for ERUF. OPT-OUT preserves benign utility and lowers EL10 below 1, but it still has high direct surface leakage, high adversarial recovery, and high RWKU forget alias recovery. This strengthens the rebuttal narrative that the weak 3B setting is a tradeoff regime and that no single baseline cleanly solves surface suppression, internal attenuation, robust recovery, and utility preservation simultaneously.

## Completed experiments useful for rebuttal

| # | Experiment | Model/setting | Main result | Rebuttal value |
|---:|---|---|---|---|
| 1 | Layer-band localization ablation | ERUF localization, 11 subjects | High-salience band 23-27 had the best mean score with 11/11 subject coverage | Answers the concern that layer choice was heuristic |
| 2 | Mechanistic alpha_eff ablation | Module-E-only controlled intervention, 6 subjects | Larger alpha_eff monotonically reduced target-signature projection while benign runtime attenuation stayed zero | Answers the request for alpha_eff ablation |
| 3 | Llama 3B baseline stress test | LUNAR, ReGLU, SimNPO | Baselines split between leakage and utility damage | Shows the 3B regime is capacity-limited across methods |
| 4 | Llama 3B OPT-OUT completed run | OPT-OUT, full-parameter run with Adele retain rows | Utility is preserved and EL10 falls below 1, but SMR, adversarial recovery, and RWKU recovery remain high | Adds the missing OPT-OUT baseline and reinforces the baseline-tradeoff story |
| 5 | Qwen 3B LUNAR direct baseline | Qwen/Qwen2.5-3B-Instruct | Large utility collapse and EL10 ratio above 1 | Shows LUNAR does not cleanly solve Qwen 3B |
| 6 | Qwen 3B ReGLU direct baseline | Qwen/Qwen2.5-3B-Instruct | Near-total direct suppression but severe utility degradation | Shows ReGLU is strong but destructive |
| 7 | Qwen 3B saved-suite baseline evaluation | LUNAR and ReGLU full saved-suite evals | ReGLU has low alias hit but high adversarial recovery success; LUNAR has lower recovery but poor direct utility/internal tradeoff | Shows direct SMR/EL10 are insufficient, matching the paper's robust/adversarial diagnostic framing |
| 8 | Weakness 5 hidden-space clustering/recoverability | Llama-3.1-8B ERUF adapter, layers 23-27 | Subject-name-excluded pooling reduces kNN, silhouette, centroid distance, and linear subject-ID recoverability after ERUF | Directly addresses the reviewer's clustering request while preserving the attenuation, not erasure, framing |
| 9 | OPT-OUT retain-data fix | Llama/Qwen baseline framework | Added Adele retain rows to avoid empty retain-pool failure | Fixes the earlier local OPT-OUT failure cause |
| 10 | ReGLU CPU RILA backend | Qwen 3B on RTX 5060 Ti | ReGLU completed with original hyperparameters after moving RILA eigensolves off CUDA | Operational reproducibility fix, not a changed method |

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
| OPT-OUT | 0.0054 | 0.5000 | 0.5971 | 0.6144 | Preserves utility and lowers EL10, but surface/adversarial recovery stays high |

**Rebuttal value:** This addresses the reviewer concern that baselines should be tested in the same weak 3B regime where ERUF is less clean. The result supports a tradeoff framing: utility-preserving baselines can leak, while aggressive suppression can damage utility. OPT-OUT adds an important fourth point in this tradeoff space: it avoids the severe utility collapse seen in ReGLU and lowers EL10 below 1, but still has high direct SMR and robust recovery. Therefore it is useful evidence against the claim that the small-model case is solved by existing baselines.

## 4. Llama 3B OPT-OUT completed result

This run used `meta-llama/Llama-3.2-3B`, the original prompt file for evaluation, and `prompts_with_adele_retain.jsonl` for training so that OPT-OUT had a retain pool without adding Adele to the forget-target evaluation.

| Metric | OPT-OUT result | Interpretation |
|---|---:|---|
| Benign loss delta | 0.0054 | Utility is essentially preserved |
| Benign PPL delta | 0.6269 | Small utility drift |
| Average subject mention rate | 0.5000 | Direct surface leakage remains high |
| Average keyword hit rate | 0.0613 | Some keyword-level target signal remains |
| EL10 ratio | 0.5971 | Internal early target mass decreases below the Type-I threshold |
| Pre/post similarity | 0.1770 | Outputs shift substantially |
| Fast entity bundle completion | 1157/1157 | Full saved-suite completion |
| Adversarial recovery completion | 1206/1206 | Full saved-suite completion |
| RWKU-style completion | 747/747 | Full saved-suite completion |

Saved-suite key results:

| Suite metric | OPT-OUT |
|---|---:|
| Name-agnostic target alias hit | 0.1786 |
| Name-agnostic target keyword hit rate | 0.1567 |
| Name-agnostic target mass | 0.0228 |
| BLUR/mixed forget target alias hit | 0.0192 |
| Matched-control retain alias hit | 0.5667 |
| Neighbor-locality retain alias hit | 0.5333 |
| Generic benign refusal-like | 0.0000 |
| Adversarial target alias hit | 0.1990 |
| Adversarial target keyword hit rate | 0.1524 |
| Adversarial target mass | 0.0221 |
| Adversarial recovery success | 0.6144 |
| RWKU forget alias hit | 0.5041 |
| RWKU forget keyword hit rate | 0.1400 |
| RWKU forget target mass | 0.0431 |

**Interpretation:** OPT-OUT is not a clean win and should not be presented as ERUF being uniformly better on every individual axis. It is nevertheless useful for the rebuttal because it fills the missing Llama 3B OPT-OUT baseline and shows a different failure mode from LUNAR, ReGLU, and SimNPO. It preserves utility and reduces EL10, but fails to suppress direct surface mentions and remains highly recoverable under adversarial and RWKU-style probes. This supports the paper's central diagnostic argument: surface leakage, internal EL10, adversarial recovery, and utility must be evaluated jointly.

**Suggested wording:**

> We additionally completed a Llama 3B OPT-OUT run after fixing the retain-pool issue with Adele retain rows. OPT-OUT preserves benign utility and lowers EL10 below 1, but it does not cleanly solve the 3B setting: direct subject mention remains high, adversarial recovery success is 0.614, and RWKU-style forget alias hit is 0.504. This complements the LUNAR/ReGLU/SimNPO results by showing another baseline tradeoff rather than a baseline solution. Together, the 3B results support our use of joint surface, internal, and adversarial diagnostics rather than relying on a single forgetting metric.

## 5. Qwen 3B direct baseline results

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

## 6. Qwen 3B full saved-suite baseline evaluation

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

## 7. Weakness 5 hidden-space clustering and recoverability

**Reviewer concern:** Hidden-space diagnostics in Table 5 should include clustering evidence. The concern is that if forget entities remain well-clustered, ERUF may still retain their semantics.

**Important framing:** This result should not be used to claim formal deletion of every semantic trace. The correct claim is representation-level attenuation. The hidden-space evidence should be read together with behavioral forgetting, SMR, EL10, adversarial recovery, and name-agnostic recovery metrics.

**Experiment:** Llama-3.1-8B, ERUF LoRA adapter `outputs/global_adapters/unlearning_adapter_repaware_20260709_211224`, upper intervention layers 23-27, 24 neutral prompts per entity, subject-name tokens excluded from pooling so the result is not driven by lexical identity alone.

| Hidden-space diagnostic, layers 23-27 average | Base | ERUF | Interpretation |
|---|---:|---:|---|
| kNN subject recoverability | 0.085 | 0.066 | lower subject recoverability |
| Silhouette | -0.026 | -0.047 | weaker subject clustering |
| Between-centroid distance | 0.00256 | 0.00228 | compressed subject centroids |
| Linear subject-ID probe accuracy | 0.831 | 0.412 | lower linear decodability |

At the final intervention layer, layer 27, the same trend is visible: kNN decreases from 0.102 to 0.074, silhouette decreases from -0.023 to -0.048, centroid distance decreases from 0.00329 to 0.00286, and linear probe accuracy decreases from 0.954 to 0.415.

**Rebuttal response text:**

> We thank the reviewer for this important point. We agree that hidden-state movement in Table 5 should not, by itself, be interpreted as proof that a forgotten entity has been completely removed. Our intended claim is representation-level attenuation, not a formal guarantee of irreversible deletion.
>
> To address the reviewer's clustering concern directly, we added a forget-entity clustering and recoverability diagnostic on Llama-3.1-8B at the upper intervention layers 23-27. The analysis uses the forgotten entities, 24 neutral prompts per entity, and excludes subject-name tokens from pooling so that the result is not driven by lexical identity alone. Under this stricter setting, forgotten entities do not remain strongly clustered after ERUF. Averaged over layers 23-27, kNN subject recoverability decreases from 0.085 to 0.066, silhouette decreases from -0.026 to -0.047, mean between-centroid distance decreases from 0.00256 to 0.00228, and linear subject-ID probe accuracy decreases from 0.831 to 0.412.
>
> We will revise the Table 5 discussion to make this distinction explicit. The hidden-space evidence should be read as reduced subject-level separability and recoverability, not as proof that every semantic trace of the entity has vanished. This interpretation is also consistent with the rest of the paper: ERUF reduces surface leakage and EL10, reduces adversarial entity recovery from 63.89% to 20.15%, and reduces name-agnostic recovery metrics by 72.7-77.4%. Together, the new clustering diagnostic and the existing behavioral probes support the paper's central claim: ERUF weakens subject-linked internal activation and non-canonical recovery routes, rather than merely suppressing exact-name outputs.
>
> We will update the manuscript accordingly by: (1) adding the clustering/recoverability table to the appendix, (2) revising the main Table 5 discussion to state "attenuation" rather than "erasure," and (3) clarifying that ERUF does not claim formal irreversible deletion of the full entity manifold.

## 8. Qwen 3B SimNPO feasibility attempt

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
| Baselines might solve small-model cases | LUNAR, ReGLU, SimNPO, and OPT-OUT each expose a different tradeoff; none cleanly solves surface suppression, internal attenuation, robust recovery, and utility at once |
| Direct leakage metrics may be insufficient | Qwen saved-suite results show low alias hit can coexist with high adversarial recovery success; Llama 3B OPT-OUT shows low EL10 can coexist with high direct/adversarial recovery |
| Hidden-space clustering missing | Weakness 5 clustering/recoverability diagnostic shows reduced subject-level separability after ERUF under subject-name-excluded pooling |
| OPT-OUT failed locally | Cause identified: missing retain rows. Fixed by adding Adele retain-control rows; completed Llama 3B OPT-OUT run is now available |

## Recommended paper framing

> We added targeted rebuttal experiments addressing layer selection, alpha_eff behavior, missing small-model baseline context, and hidden-space clustering. A layer-band sensitivity check supports data-driven high-salience capsule placement, while a controlled alpha_eff sweep shows monotonic target-signature attenuation under the capsule intervention. Additional 3B baseline stress tests show that small models expose tradeoffs across methods: LUNAR and OPT-OUT preserve utility better but leave substantial recoverability, ReGLU can suppress direct leakage but damages utility, and SimNPO preserves utility while leaving internal or adversarial recovery routes. The new clustering diagnostic further supports an attenuation interpretation: forgotten entities become less subject-recoverable in the upper intervention layers, but we do not claim formal irreversible deletion of the full entity manifold.

## What not to overclaim

- Do not claim ERUF uniformly beats every baseline at 3B.
- Do not claim ReGLU fails to forget on Qwen 3B. It forgets strongly in the direct evaluator.
- Do not claim LUNAR is better overall than ReGLU. LUNAR has lower adversarial recovery in the saved-suite evaluator, but worse direct utility/internal behavior.
- Do not claim OPT-OUT fails completely. It preserves utility and lowers EL10 on Llama 3B, but retains high surface/adversarial/RWKU recoverability.
- Do not report Qwen 3B SimNPO as a metric result from the 5060 Ti attempt. It is only a local hardware feasibility failure.
- Do not present the alpha_eff ablation as end-to-end unlearning performance. It is a mechanistic intervention sanity check.
- Do not describe the Weakness 5 clustering result as formal erasure. It is evidence of reduced subject-level separability and recoverability.

## Remaining experiment priorities

| Priority | Experiment | Machine | Why |
|---:|---|---|---|
| 1 | Compile final 8B runtime/cost table from logs | 24 GB VRAM PC | Directly answers computational-cost concern |
| 2 | Run Module 8 only if full ERUF behavioral metrics are still needed | 24 GB VRAM PC | Adapter exists, but Module 8 gives the final standard evaluation table |
| 3 | Complete Qwen 3B SimNPO full-parameter baseline if it finishes on the 24 GB GPU | 24 GB PC or cluster | Adds the remaining Qwen small-model baseline evidence |
| 4 | Complete Qwen 3B OPT-OUT if SimNPO finishes or fails cleanly | 24 GB PC or cluster | Adds the remaining Qwen small-model baseline evidence |

## Best use of RTX 5060 Ti next

The 5060 Ti has already produced the most useful Qwen 3B saved-suite evidence for LUNAR and ReGLU. Do not spend it on full-parameter SimNPO or OPT-OUT training. The next useful local use would be lightweight evaluation-only checks on already saved models, or simply leave it idle while the 24 GB machine handles any remaining 8B work.
