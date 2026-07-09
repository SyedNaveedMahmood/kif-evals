# 3B Baseline Stress-Test Results

This note summarizes the uploaded 3B baseline outputs for LUNAR, ReGLU, and SimNPO. The main purpose is to answer the review concern that ERUF should be compared against baselines in the small/model-constrained 3B setting where ERUF itself is least favorable.

## Verdict

**Decision: useful for rebuttal, but frame as a tradeoff result rather than a pure win.**

The 3B baseline results help the rebuttal because they show that the small-model regime is difficult for all compared methods. None of the tested baselines simultaneously achieves low surface recovery, low internal extraction likelihood, and preserved utility.

The correct claim is not that ERUF is uniformly superior at 3B. The defensible claim is:

> Additional 3B baseline evaluations show that the small-model regime exposes a tradeoff across methods. Some baselines preserve utility but leave high surface or adversarial recovery, while ReGLU strongly suppresses target signals but causes severe utility degradation and locality damage. This supports treating the 3B setting as a capacity-limited stress test rather than as evidence that existing baselines solve entity-level representation unlearning.

## Methods evaluated

- LUNAR
- ReGLU
- SimNPO

OPT-OUT is not included because the run failed before evaluation. The log reports that no retain rows were found and asks for a non-forgotten retain subject such as Adele. Therefore, do not claim an OPT-OUT 3B comparison until it is rerun with a valid retain/control subject.

## Dual-metric SMR/EL10/utility summary

| Method | Utility Δ loss | PPL pre | PPL post | Avg subject mention rate | Avg keyword hit rate | EL10 pre | EL10 post | EL10 ratio | Pre/post similarity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LUNAR | 0.0488 | 114.90 | 120.66 | 0.7407 | 0.0677 | 1.514e-05 | 2.166e-06 | 0.1431 | 0.2030 |
| ReGLU | 3.0884 | 114.90 | 2521.23 | 0.1111 | 0.0799 | 1.514e-05 | 2.274e-07 | 0.0150 | 0.0098 |
| SimNPO | 0.0007 | 114.90 | 114.99 | 0.5000 | 0.0550 | 1.514e-05 | 2.406e-05 | 1.5891 | 0.2035 |

### Interpretation

LUNAR has acceptable utility drift and lowers EL10, but it leaves very high surface mention rate: **0.7407**. This is not a successful 3B entity-unlearning baseline under the paper's surface-leakage criterion.

ReGLU has the strongest apparent forgetting signal: subject mention rate **0.1111** and EL10 ratio **0.0150**. However, this comes with severe utility damage: benign loss increases by **3.0884**, perplexity jumps from **114.90** to **2521.23**, and pre/post similarity drops to **0.0098**. This should be framed as collapse/collateral damage, not clean unlearning.

SimNPO preserves utility almost perfectly: loss delta **0.0007**. However, it leaves high surface leakage and makes hidden extraction worse: EL10 ratio **1.5891**. This is useful evidence for the paper's argument that behavioral preservation alone does not imply internal target attenuation.

## Robust baseline-suite summary

| Method | Name-agnostic alias hit | Blur forget alias hit | Blur retain alias hit | Neighbor retain alias hit | Adversarial alias hit | Adversarial recovery success |
|---|---:|---:|---:|---:|---:|---:|
| LUNAR | 0.2767 | 0.0313 | 0.1923 | 0.5667 | 0.2678 | 0.7156 |
| ReGLU | 0.0109 | 0.0000 | 0.0721 | 0.2333 | 0.0191 | 0.1310 |
| SimNPO | 0.2331 | 0.0240 | 0.1563 | 0.5056 | 0.2172 | 0.6161 |

### Interpretation

The robust suite strengthens the same conclusion. LUNAR and SimNPO retain substantial adversarial recovery, with recovery success **0.7156** and **0.6161**, respectively. ReGLU reduces recovery to **0.1310**, but its utility and locality behavior are poor. Its retain-side values are much lower than the other methods, suggesting collateral damage rather than selective entity removal.

## RWKU-style robustness summary

| Method | Forget alias hit | Forget keyword hit | Forget target mass | Matched-control alias hit | Neighbor-locality alias hit |
|---|---:|---:|---:|---:|---:|
| LUNAR | 0.6125 | 0.2311 | 0.0416 | 0.6481 | 0.8056 |
| ReGLU | 0.1396 | 0.0346 | 0.0046 | 0.2407 | 0.2639 |
| SimNPO | 0.5057 | 0.1786 | 0.0451 | 0.7037 | 0.4722 |

### Interpretation

The RWKU-style results again show a tradeoff. LUNAR and SimNPO maintain control/locality behavior but also keep high forget-target alias hits. ReGLU lowers forget alias hits but also lowers matched-control and neighbor-locality alias hits, consistent with broad suppression/collateral damage.

## How this answers the review concern

This result directly addresses the review weakness asking for baseline behavior in the 3B setting. The answer should be:

1. We added 3B baseline evaluations for LUNAR, ReGLU, and SimNPO.
2. The results show that baseline methods also struggle in the 3B regime.
3. Utility-preserving methods such as SimNPO do not reliably attenuate internal target information.
4. Stronger suppressive methods such as ReGLU can reduce target signals but at the cost of severe utility degradation and locality damage.
5. Therefore, the 3B result should be described as a capacity-limited stress test, not as a setting where existing baselines solve the task.

## Rebuttal-ready wording

> We added 3B baseline evaluations for LUNAR, ReGLU, and SimNPO to address the concern that our smallest-model setting lacked sufficient baseline context. The results show that the 3B regime is difficult across methods. LUNAR and SimNPO preserve utility relatively well but retain substantial surface or adversarial recovery, and SimNPO increases hidden extraction likelihood (EL10 ratio 1.5891). ReGLU achieves stronger target suppression (EL10 ratio 0.0150 and adversarial recovery 0.1310), but this comes with severe utility degradation, with benign perplexity increasing from 114.90 to 2521.23 and pre/post similarity dropping to 0.0098. We therefore revise the discussion to frame the 3B setting as a capacity-limited stress test: existing baselines also exhibit leakage, internal retention, or collateral damage rather than clean entity-level representation unlearning.

## Paper-table candidate

| Method | Utility Δ loss | SMR | EL10 ratio | Adversarial recovery | Main failure mode |
|---|---:|---:|---:|---:|---|
| LUNAR | 0.0488 | 0.7407 | 0.1431 | 0.7156 | High surface/adversarial recovery |
| ReGLU | 3.0884 | 0.1111 | 0.0150 | 0.1310 | Severe utility/locality damage |
| SimNPO | 0.0007 | 0.5000 | 1.5891 | 0.6161 | Preserves utility but worsens hidden extraction |

Suggested caption:

> 3B baseline stress test. LUNAR and SimNPO preserve utility better but retain substantial target recovery; ReGLU suppresses target signals more strongly but causes severe utility degradation. These results support treating the 3B regime as capacity-limited rather than solved by existing baselines.

## Caveats

- The evaluated subject count is 9 in the uploaded outputs, not 11. Use the exact count when reporting.
- OPT-OUT is absent because the run failed due to missing retain rows. Do not include OPT-OUT until rerun with a valid retain/control subject.
- These are baseline results. They should be used to contextualize the 3B regime, not to overclaim ERUF superiority in the smallest-model setting.
