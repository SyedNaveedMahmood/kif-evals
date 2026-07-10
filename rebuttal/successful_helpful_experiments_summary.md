# Successful and Helpful Rebuttal Experiments Summary

This file summarizes only the experiments that are directly useful for the paper/rebuttal narrative. The emphasis is on evidence that answers reviewer concerns: layer choice, alpha_eff behavior, small-model baseline context, computational-cost context, and hidden-space clustering/recoverability.

## Executive verdict

The rebuttal package is now materially stronger. The completed experiments support four claims: ERUF's layer choice is data-driven, alpha_eff behaves as a controllable internal attenuation knob, the 3B regime exposes baseline tradeoffs rather than being solved by existing methods, and the hidden-space diagnostics should be framed as representation-level attenuation rather than formal erasure. The new Weakness 5 clustering diagnostic is useful because it directly answers the reviewer concern while avoiding an overclaim: forgotten entities show reduced subject-level separability and recoverability in the upper intervention layers, but we should describe this as attenuation, not complete deletion of the full entity manifold.

The completed Llama 3B OPT-OUT run should be used carefully. It is useful rebuttal evidence, but not a simple win for ERUF. OPT-OUT preserves benign utility and lowers EL10 below 1, but it still has high direct surface leakage, high adversarial recovery, and high RWKU forget alias recovery. This strengthens the rebuttal narrative that the weak 3B setting is a tradeoff regime and that no single baseline cleanly solves surface suppression, internal attenuation, robust recovery, and utility preservation simultaneously.

The completed Qwen 3B SimNPO and OPT-OUT evaluations are especially useful for Weakness 2. SimNPO nearly eliminates EL10 while preserving benign utility reasonably well, yet fails badly on behavioral and robust forgetting: direct SMR is 77.78%, adversarial recovery success is 90.13%, and RWKU forget alias hit is 77.67%. OPT-OUT, even after the memory-feasible full-parameter PagedAdamW8bit rerun, also does not solve the Qwen 3B setting: direct SMR is 77.78%, EL10 ratio is 1.0255, adversarial recovery success is 83.33%, and RWKU forget alias hit is 64.04%. These results do not show that ERUF wins every individual scalar metric—SimNPO is substantially stronger on EL10—but they strongly support ERUF's joint advantage claim: low internal trace alone, or a utility-preserving full-parameter baseline, does not yield reliable surface suppression or robust non-recoverability.

## Completed experiments useful for rebuttal

| # | Experiment | Model/setting | Main result | Rebuttal value |
|---:|---|---|---|---|
| 1 | Layer-band localization ablation | ERUF localization, 11 subjects | High-salience band 23-27 had the best mean score with 11/11 subject coverage | Answers the concern that layer choice was heuristic |
| 2 | Mechanistic alpha_eff ablation | Module-E-only controlled intervention, 6 subjects | Larger alpha_eff monotonically reduced target-signature projection while benign runtime attenuation stayed zero | Answers the request for alpha_eff ablation |
| 3 | Llama 3B baseline stress test | LUNAR, ReGLU, SimNPO | Baselines split between leakage and utility damage | Shows the 3B regime is capacity-limited across methods |
| 4 | Llama 3B OPT-OUT completed run | OPT-OUT, full-parameter run with Adele retain rows | Utility is preserved and EL10 falls below 1, but SMR, adversarial recovery, and RWKU recovery remain high | Adds the missing OPT-OUT baseline and reinforces the baseline-tradeoff story |
| 5 | Qwen 3B LUNAR direct baseline | Qwen/Qwen2.5-3B | Large utility collapse and EL10 ratio above 1 | Shows LUNAR does not cleanly solve Qwen 3B |
| 6 | Qwen 3B ReGLU direct baseline | Qwen/Qwen2.5-3B | Near-total direct suppression but severe utility degradation | Shows ReGLU is strong but destructive |
| 7 | Qwen 3B LUNAR/ReGLU saved-suite evaluation | Qwen/Qwen2.5-3B | ReGLU has low alias hit but high adversarial recovery success; LUNAR has lower recovery but poor direct utility/internal tradeoff | Shows direct SMR/EL10 are insufficient |
| 8 | Qwen 3B SimNPO completed run | Qwen/Qwen2.5-3B, 9 subjects | EL10 collapses, but SMR=77.78%, adversarial recovery=90.13%, and RWKU forget alias hit=77.67% | Strongly answers Weakness 2: a strong baseline fails far beyond ERUF on surface and robust recovery despite low EL10 |
| 9 | Qwen 3B OPT-OUT completed run | Qwen/Qwen2.5-3B, full-parameter PagedAdamW8bit, 9 subjects | SMR=77.78%, EL10 ratio=1.0255, adversarial recovery=83.33%, and RWKU forget alias hit=64.04% | Adds OPT-OUT coverage and reinforces that Qwen 3B remains unsolved by baselines |
| 10 | Weakness 5 hidden-space clustering/recoverability | Llama-3.1-8B ERUF adapter, layers 23-27 | Subject-name-excluded pooling reduces kNN, silhouette, centroid distance, and linear subject-ID recoverability after ERUF | Directly addresses the reviewer's clustering request while preserving the attenuation, not erasure, framing |
| 11 | OPT-OUT retain-data fix | Llama/Qwen baseline framework | Added Adele retain rows to avoid empty retain-pool failure | Fixes the earlier local OPT-OUT failure cause |
| 12 | ReGLU CPU RILA backend | Qwen 3B on RTX 5060 Ti | ReGLU completed with original hyperparameters after moving RILA eigensolves off CUDA | Operational reproducibility fix, not a changed method |

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
| Adversarial target alias hit | 0.0358 | 0.0284 | Alias hit alone does not show the full picture |
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

## 8. Qwen 3B SimNPO and OPT-OUT completed results

Both full-parameter Qwen 3B baselines were evaluated on `Qwen/Qwen2.5-3B` with 9 available forgotten subjects. SimNPO used PagedAdamW32bit. OPT-OUT's standard-Adam full-parameter attempt OOMed at the first optimizer step on the 24 GB GPU, but the memory-feasible full-parameter rerun completed with PagedAdamW8bit and was evaluated with the same direct and saved-suite probes.

### Direct SMR/EL10 diagnostics

| Direct diagnostic | SimNPO | OPT-OUT | Interpretation |
|---|---:|---:|---|
| Benign pre loss | 4.0765 | 4.0765 | Same reference model/eval setup |
| Benign post loss | 4.1376 | 4.2955 | OPT-OUT changes utility more |
| Benign loss delta | 0.0611 | 0.2190 | SimNPO preserves utility better |
| Benign PPL delta | +3.7161 | +14.4334 | OPT-OUT has larger utility drift |
| Direct SMR | 0.7778 (77.78%) | 0.7778 (77.78%) | Both leak heavily on direct surface prompts |
| Direct keyword hit rate | 0.0723 | 0.0729 | Residual keyword-level target signal remains |
| EL10 ratio | 0.000556 | 1.0255 | SimNPO collapses EL10; OPT-OUT does not improve it |
| Pre/post similarity | 0.4448 | 0.3742 | Both substantially alter outputs |
| Mechanism state | Type III | Type III | Surface leakage remains high |

### Saved-suite completion

| Suite | SimNPO | OPT-OUT |
|---|---:|---:|
| Fast entity bundle | 1157/1157 | 1157/1157 |
| Adversarial recovery | 1206/1206 | 1206/1206 |
| RWKU-style robustness | 747/747 | 747/747 |

### Robustness diagnostics

| Robustness diagnostic | SimNPO | OPT-OUT |
|---|---:|---:|
| Name-agnostic target alias hit | 0.3747 | 0.2963 |
| Name-agnostic target keyword hit rate | 0.2631 | 0.2303 |
| Name-agnostic target mass | 0.0477 | 0.0274 |
| BLUR/mixed forget target alias hit | 0.0361 | 0.0264 |
| Matched-control retain alias hit | 0.8000 | 0.8333 |
| Neighbor-locality retain alias hit | 0.7944 | 0.7500 |
| Generic benign refusal-like | 0.0000 | 0.0000 |
| Adversarial target alias hit | 0.4444 | 0.3590 |
| Adversarial target keyword hit rate | 0.2705 | 0.2457 |
| Adversarial target mass | 0.0709 | 0.0377 |
| Adversarial recovery success | 0.9013 | 0.8333 |
| RWKU forget alias hit | 0.7767 | 0.6404 |
| RWKU forget keyword hit rate | 0.3458 | 0.2595 |
| RWKU forget target mass | 0.1419 | 0.0858 |

**Judgment for Weakness 2:** These are strong rebuttal results, but they must be framed carefully. SimNPO nearly eliminates EL10, yet direct SMR remains 77.78%, adversarial recovery is 90.13%, and RWKU forget alias hit is 77.67%. OPT-OUT does not solve the setting either: direct SMR remains 77.78%, EL10 ratio is 1.0255, adversarial recovery is 83.33%, and RWKU forget alias hit is 64.04%. The useful conclusion is not that ERUF dominates every scalar metric. The useful conclusion is that Qwen 3B remains a baseline-tradeoff regime: low EL10, utility preservation, and full-parameter training do not by themselves imply robust entity forgetting.

**Suggested rebuttal wording:**

> We added full-parameter SimNPO and OPT-OUT evaluations on Qwen 3B. SimNPO strongly reduces the direct EL10 ratio to 0.00056 and retains benign utility reasonably well, but it does not achieve behavioral or robust forgetting: direct SMR is 77.78%, adversarial recovery success is 90.13%, and RWKU-style forget alias hit is 77.67%. OPT-OUT also does not solve the setting: direct SMR is 77.78%, EL10 ratio is 1.0255, adversarial recovery success is 83.33%, and RWKU-style forget alias hit is 64.04%. These results show that neither low internal target mass nor utility-preserving full-parameter optimization is sufficient for robust entity unlearning. Together with LUNAR and ReGLU, the Qwen 3B baselines exhibit complementary failure modes rather than a clean solution. We therefore state our claim jointly: ERUF offers a better balance across surface suppression, internal attenuation, robust recoverability, and utility, rather than dominating every baseline on every individual metric.

**Cross-run caveat:** All Qwen 3B baseline rows should be treated as `Qwen/Qwen2.5-3B`; earlier `Instruct` labels were a logging artifact. The LUNAR/ReGLU direct rows used a smaller 5-subject direct evaluation, while the SimNPO/OPT-OUT base-checkpoint runs used the 9 available forgotten subjects. Their numerical values are useful as baseline context, but subject-set and evaluation-protocol differences should still be disclosed.

## 9. Recovered 3B training wall-clock and parameter counts

The following values were recovered from completed training logs across the RTX 5060 Ti and RTX 3090 machines. These are training runtimes only; SMR/EL10 and saved-suite evaluation time is excluded. All Qwen rows should be reported as `Qwen/Qwen2.5-3B`; earlier `Instruct` labels in some logs were a logging artifact.

| Model | Method | Hardware | Wall-clock (h) | Approx. duration | Trainable parameters | Run note |
|---|---|---|---:|---:|---:|---|
| Qwen2.5-3B | LUNAR | RTX 3090 24 GB | 0.0135 | 49 s | N/A | Clean closed-form rerun for timing |
| Qwen2.5-3B | ReGLU | RTX 5060 Ti | 0.4757 | 28 min 33 s | 59,867,136 | CPU-RILA run; explicit end-to-end wall-clock from the handover record |
| Qwen2.5-3B | SimNPO | RTX 3090 24 GB | 0.3478 | 20 min 52 s | 3,085,938,688 | Full-parameter PagedAdamW32bit run |
| Qwen2.5-3B | OPT-OUT | RTX 3090 24 GB | 0.3501 | 21 min 00 s | 3,085,938,688 | Full-parameter PagedAdamW8bit memory-feasible rerun |
| Llama-3.2-3B | LUNAR | RTX 3090 24 GB | 0.0093 | 33 s | N/A | Clean closed-form rerun for timing |
| Llama-3.2-3B | ReGLU | RTX 3090 24 GB | 0.1950 | 11 min 42 s | 48,627,712 | LoRA/ReGLU training log recovered |
| Llama-3.2-3B | SimNPO | RTX 3090 24 GB | 0.2342 | 14 min 03 s | 3,212,749,824 | Full-parameter PagedAdamW32bit run |
| Llama-3.2-3B | OPT-OUT | RTX 3090 24 GB | 1.4055 | 1 h 24 min 20 s | 3,212,749,824 | Full-parameter retain-fixed run |

**Timing caveat:** The Qwen ReGLU log parser recovered a 0.3542 h timestamp span, while the handover records an explicit end-to-end wall-clock of 28:32.69 (0.4757 h). The table uses the explicit end-to-end record. The Llama and Qwen LUNAR rows use clean RTX 3090 reruns with explicit wall-clock fields: 33.354 s for Llama and 48.688 s for Qwen. The uploaded Llama OPT-OUT `technique4_optout_train.log` is the earlier retain-pool failure and is not the completed retain-fixed OPT-OUT run used in the table.

## Current combined story for rebuttal

| Review concern | Completed answer |
|---|---|
| Layer choice looks heuristic | Layer-band ablation shows the selected high-salience band is empirically strongest among tested bands |
| alpha_eff needs ablation | Mechanistic alpha_eff sweep shows monotonic target-signature attenuation |
| Baselines missing in weak 3B setting | Llama 3B has four completed baselines; Qwen 3B now has completed LUNAR, ReGLU, SimNPO, and OPT-OUT evidence |
| Baselines might solve small-model cases | No completed baseline cleanly satisfies surface suppression, internal attenuation, robust non-recoverability, and utility simultaneously |
| Reviewer expects baselines to fail beyond ERUF in Qwen 3B | SimNPO has 77.78% direct SMR and 90.13% adversarial recovery despite extremely low EL10; OPT-OUT has 77.78% direct SMR and 83.33% adversarial recovery |
| Direct leakage or EL10 alone may be insufficient | ReGLU shows low direct alias/EL10 with high adversarial recovery; SimNPO shows extremely low EL10 with catastrophic surface and adversarial recovery |
| Hidden-space clustering missing | Weakness 5 clustering/recoverability diagnostic shows reduced subject-level separability after ERUF under subject-name-excluded pooling |
| OPT-OUT memory feasibility and quality | Llama 3B OPT-OUT completed normally; Qwen 3B standard Adam OOMed, but a full-parameter PagedAdamW8bit rerun completed and still showed high direct and robust recovery |

## Recommended paper framing

> We added targeted rebuttal experiments addressing layer selection, alpha_eff behavior, missing small-model baseline context, hidden-space clustering, and compute/runtime context. A layer-band sensitivity check supports data-driven high-salience capsule placement, while a controlled alpha_eff sweep shows monotonic target-signature attenuation under the capsule intervention. The additional 3B baseline experiments directly address the reviewer's concern that existing methods might outperform ERUF where ERUF struggles. On Llama 3B, LUNAR, ReGLU, SimNPO, and OPT-OUT expose complementary leakage-versus-utility failures. On Qwen 3B, full-parameter SimNPO nearly eliminates EL10 but still yields 77.78% direct SMR, 90.13% adversarial recovery, and 77.67% RWKU forget alias recovery. The completed Qwen OPT-OUT rerun also does not solve the setting: direct SMR remains 77.78%, EL10 ratio is 1.0255, adversarial recovery is 83.33%, and RWKU forget alias hit is 64.04%. Alongside LUNAR's utility/internal failure and ReGLU's destructive utility and robust-recovery tradeoff, these results show that the small-model regime is not cleanly solved by existing baselines. Our claim should be stated jointly: ERUF offers a stronger balance across surface suppression, internal attenuation, robust non-recoverability, and utility preservation, not universal dominance on every individual scalar metric. The new clustering diagnostic further supports an attenuation interpretation, but we do not claim formal irreversible deletion of the full entity manifold.

## What not to overclaim

- Do not claim ERUF uniformly beats every baseline on every 3B metric.
- Do not hide that Qwen 3B SimNPO achieves a much lower EL10 than ERUF; the useful result is that this does not translate into surface or robust forgetting.
- Do not claim ReGLU fails to forget on Qwen 3B. It forgets strongly in the direct evaluator, but damages utility and remains adversarially recoverable.
- Do not claim LUNAR is better overall than ReGLU. Their failure modes differ.
- Do not claim OPT-OUT fails completely. It preserves utility and lowers EL10 on Llama 3B, but retains high surface/adversarial/RWKU recoverability.
- Do not use the standard-Adam Qwen 3B OPT-OUT OOM as evidence of inferior unlearning quality. The PagedAdamW8bit rerun completed and was evaluated, but the optimizer change must be disclosed.
- Do not present LUNAR/ReGLU/SimNPO Qwen numbers as a perfectly matched ranking without noting the checkpoint and subject-set differences.
- Do not present the alpha_eff ablation as end-to-end unlearning performance.
- Do not describe the Weakness 5 clustering result as formal erasure.

## Remaining experiment priorities

| Priority | Experiment | Machine | Why |
|---:|---|---|---|
| 1 | Compile final 8B runtime/cost table from logs | 24 GB VRAM PC | Directly answers computational-cost concern |
| 2 | Run Module 8 only if full ERUF behavioral metrics are still needed | 24 GB VRAM PC | Adapter exists, but Module 8 gives the final standard evaluation table |
| 3 | DeepSeek 3B baselines only if rebuttal time and compute permit | Larger GPU/cluster | Would broaden Weakness 2 coverage, but current Llama and Qwen evidence already answers the core concern |

## Best use of RTX 5060 Ti next

The 5060 Ti has already produced the useful Qwen 3B LUNAR/ReGLU evidence, while the 24 GB RTX 3090 has completed the full-parameter Qwen 3B SimNPO and OPT-OUT evaluations. The immediate priority is the final 8B runtime/cost table and rebuttal synthesis, not more Qwen 3B baseline runs.
