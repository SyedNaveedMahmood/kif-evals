# ERUF Rebuttal Handover for Next Chat

This file is a practical handover for continuing the ERUF ARR May 2026 rebuttal work in a new ChatGPT chat. It records the machines, repo paths, branch state, completed experiments, usable results, caveats, and next actions.

## Current high-level objective

We are strengthening the ARR rebuttal for ERUF, especially reviewer concerns around:

1. layer choice looking heuristic,
2. alpha_eff sensitivity,
3. missing or unfair small-model baselines,
4. computational cost,
5. hidden-space clustering and recoverability,
6. claims of erasure versus attenuation.

The safest global framing is:

> ERUF provides representation-level attenuation and reduced recoverability, not a formal guarantee of irreversible deletion of every semantic trace.

Do not overclaim erasure.

## Repository state

Main connected GitHub repo:

```text
SyedNaveedMahmood/kif-evals
branch: rebuttal-experiments
```

Important pushed files:

```text
rebuttal/successful_helpful_experiments_summary.md
rebuttal/run_qwen3b_eval_only_commands.md
rebuttal/rebuttal_handover_next_chat_20260709.md
```

Latest relevant commits:

```text
8cce963aa6251cd227be53db2306919e319ce10b
  Add Weakness 5 clustering response and update rebuttal summary

e25841d47243460f67ce5559d9099b74ccb22cf7
  Update summary with full Qwen saved-suite results

f6978a822d465e13cc94b061cac3bc70b92067b8
  Add Qwen 3B eval-only commands

872beb949ad0640fd1e4c6429b1f6c32b93f3403
  Prior successful/helpful experiment summary update
```

Pull command on the 5060 Ti WSL machine:

```bash
cd /mnt/e/eruf/kif-evals
git checkout rebuttal-experiments
git pull origin rebuttal-experiments
```

## Machine 1: RTX 5060 Ti 16 GB, WSL/Linux

Role: Qwen 3B baselines and saved-suite evaluation.

Typical path:

```text
/mnt/e/eruf/kif-evals
```

Conda environment:

```text
eruf-rebuttal
```

Important environment exports:

```bash
export EVALS=/mnt/e/eruf/kif-evals
export PYTHONPATH="$EVALS/rebuttal:$EVALS/analysis:$EVALS/framework:$EVALS:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
```

Qwen 3B model:

```text
Qwen/Qwen2.5-3B-Instruct
```

Prompt file:

```text
/mnt/e/eruf/kif-evals/framework/outputs/qwen3b/prompts_with_adele_retain.jsonl
```

Qwen baseline model paths:

```text
/mnt/e/eruf/kif-evals/framework/outputs/qwen3b_baselines/reglu/reglu/merged_model
/mnt/e/eruf/kif-evals/framework/outputs/qwen3b_baselines/lunar/lunar/unlearned_model
```

### Completed on 5060 Ti

1. Qwen 3B LUNAR direct baseline.
2. Qwen 3B ReGLU direct baseline.
3. Qwen 3B ReGLU CPU RILA backend fix.
4. Qwen 3B full saved-suite evaluation for LUNAR and ReGLU.
5. Qwen 3B SimNPO feasibility attempt, OOM during fair full-parameter backward.

### Key Qwen 3B LUNAR direct result

```text
Benign pre loss: 4.4581
Benign pre PPL: 86.3210
Post loss: 7.2096
Post PPL: 1352.3538
Loss delta: 2.7515
Average subject mention rate: 0.2000
Average keyword hit rate: 0.0302
EL10 ratio: 1.0377
Pre/post similarity: 0.2850
```

Interpretation: LUNAR reduces some surface leakage but causes severe utility degradation and does not reduce EL10.

### Key Qwen 3B ReGLU direct result

```text
Benign pre loss: 4.4581
Benign pre PPL: 86.3210
Post loss: 6.6825
Post PPL: 798.2968
Loss delta: 2.2244
Average subject mention rate: 0.0000
Average keyword hit rate: 0.0052
EL10 ratio: 0.000071
Pre/post similarity: 0.2023
Wall-clock with CPU RILA: 28:32.69
Peak host RSS: 8.27 GB
```

Interpretation: ReGLU is very strong on direct suppression, but destructive in utility.

### Qwen 3B saved-suite completion

Both LUNAR and ReGLU completed:

```text
Fast entity bundle: 657/657
Adversarial recovery: 670/670
RWKU-style robustness: 418/418
```

Most useful saved-suite comparison:

```text
LUNAR adversarial recovery success: 0.3448
ReGLU adversarial recovery success: 0.7194

LUNAR adversarial alias hit: 0.0358
ReGLU adversarial alias hit: 0.0284

LUNAR adversarial keyword hit rate: 0.0724
ReGLU adversarial keyword hit rate: 0.1567

LUNAR adversarial target mass: 0.0209
ReGLU adversarial target mass: 0.0574
```

Interpretation: ReGLU has low alias hit but high recovery success. This supports the paper's robust-diagnostic framing: direct SMR/EL10 and direct alias suppression are insufficient by themselves.

## Machine 2: 24 GB VRAM Windows PC, ERUF8B

Role: full ERUF 8B pipeline, adapter creation, Weakness 5 clustering.

Working directory:

```text
C:\Users\user4\Desktop\ERUF8B
```

PowerShell prompt:

```text
(eruf) PS C:\Users\user4\Desktop\ERUF8B>
```

Important environment setup:

```powershell
cd C:\Users\user4\Desktop\ERUF8B
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

Base model path:

```text
.\outputs\model
```

Final ERUF adapter path:

```text
.\outputs\global_adapters\unlearning_adapter_repaware_20260709_211224
```

Prompt file:

```text
.\outputs\datasets\prompts.jsonl
```

Actual subjects present in this prompt file:

```text
Ariana Grande: 143
Beyoncé: 187
Drake (musician): 131
Ed Sheeran: 147
Eminem: 135
Katy Perry: 155
Michael Jackson: 211
Queen (band): 111
Taylor Swift: 143
```

Missing despite subject.txt containing them:

```text
Kanye West
Arijit Singh
```

Do not claim the 8B prompt-file run covered Kanye West or Arijit Singh unless a regenerated prompt file proves it.

### Module runner inventory on ERUF8B

The correct runners are:

```text
llama20.modules.module0   run_module0
llama20.modules.module7   run_module7_repaware
llama20.modules.module8   run_module8_clean
llama20.modules.module_a  run_module_a
llama20.modules.module_b  run_module_b
llama20.modules.module_c  run_module_c
llama20.modules.module_d  run_module_d
llama20.modules.module_e  run_module_e_final
```

Earlier crashes were stale runner-name mismatches, not failed ERUF logic.

### Full ERUF run status on ERUF8B

Modules B, C, D, E, and 7 have run. Module 8 was intentionally not run because the immediate need was the saved LoRA adapter and Weakness 5 clustering, not the final standard evaluation table.

Adapter saved at:

```text
outputs\global_adapters\unlearning_adapter_repaware_20260709_211224
```

If Module 8 is needed later, use the true runner:

```text
module8.run_module8_clean
```

Do not call old names like `run_module_e_tight` or `run_module_7`.

## Weakness 5 clustering work

Reviewer concern: Hidden-space diagnostics in Table 5 should include clustering. If forgotten entities remain strongly clustered, the reviewer worries ERUF may retain their semantics.

Important final framing:

> Hidden-space clustering should be reported as reduced subject-level separability and recoverability, not proof of complete semantic deletion.

### First pilot clustering result

A first single-condition raw clustering probe was run. It was not ideal because it pooled representations with explicit subject-name content and did not compare forget against matched retain entities. Result was mixed and should not be used as the main response.

### Differential forget-vs-retain probe

A better script was created locally on ERUF8B:

```text
tools/run_weakness5_differential_probe.py
```

Key design:

1. Base Forget
2. ERUF Forget
3. Base Retain
4. ERUF Retain

Metrics:

```text
kNN subject recoverability
silhouette
between-centroid distance
within/between ratio
linear subject-ID probe accuracy
```

Output files created in timestamped folders:

```text
outputs\weakness5_differential_clustering_YYYYMMDD_HHMMSS\
  differential_clustering_summary.md
  differential_clustering_metrics.csv
  differential_linear_probe.csv
  raw_layer_subject_counts.csv
  config_used.json
  samples_used.jsonl
```

The script was later patched to support:

```text
--exclude_subject_name_tokens_from_pooling
--pooling last_non_subject_token
```

The final preferred Weakness 5 run used subject-name-excluded mean pooling. This avoids linear probe shortcutting from literal names like Taylor Swift or Adele.

### Final Weakness 5 numbers to use

Llama-3.1-8B, layers 23-27, 24 neutral prompts per forgotten entity, subject-name tokens excluded from pooling.

Layer average over 23-27:

```text
kNN subject recoverability: 0.085 -> 0.066
silhouette: -0.026 -> -0.047
mean between-centroid distance: 0.00256 -> 0.00228
linear subject-ID probe accuracy: 0.831 -> 0.412
```

Layer 27:

```text
kNN: 0.102 -> 0.074
silhouette: -0.023 -> -0.048
centroid distance: 0.00329 -> 0.00286
linear probe accuracy: 0.954 -> 0.415
```

Use this response for Weakness 5:

```text
We thank the reviewer for this important point. We agree that hidden-state movement in Table 5 should not, by itself, be interpreted as proof that a forgotten entity has been completely removed. Our intended claim is representation-level attenuation, not a formal guarantee of irreversible deletion.

To address the reviewer’s clustering concern directly, we added a forget-entity clustering and recoverability diagnostic on Llama-3.1-8B at the upper intervention layers 23-27. The analysis uses the forgotten entities, 24 neutral prompts per entity, and excludes subject-name tokens from pooling so that the result is not driven by lexical identity alone. Under this stricter setting, forgotten entities do not remain strongly clustered after ERUF. Averaged over layers 23-27, kNN subject recoverability decreases from 0.085 to 0.066, silhouette decreases from -0.026 to -0.047, mean between-centroid distance decreases from 0.00256 to 0.00228, and linear subject-ID probe accuracy decreases from 0.831 to 0.412.

We will revise the Table 5 discussion to make this distinction explicit. The hidden-space evidence should be read as reduced subject-level separability and recoverability, not as proof that every semantic trace of the entity has vanished. This interpretation is also consistent with the rest of the paper: ERUF reduces surface leakage and EL10, reduces adversarial entity recovery from 63.89% to 20.15%, and reduces name-agnostic recovery metrics by 72.7-77.4%. Together, the new clustering diagnostic and the existing behavioral probes support the paper’s central claim: ERUF weakens subject-linked internal activation and non-canonical recovery routes, rather than merely suppressing exact-name outputs.

We will update the manuscript accordingly by: (1) adding the clustering/recoverability table to the appendix, (2) revising the main Table 5 discussion to state “attenuation” rather than “erasure,” and (3) clarifying that ERUF does not claim formal irreversible deletion of the full entity manifold.
```

## Reviewer-by-reviewer status

### Reviewer jHsg

Most important experimental reviewer.

Done:

```text
Layer-band ablation
alpha_eff ablation
Llama 3B baselines
Qwen 3B LUNAR/ReGLU baselines
Qwen saved-suite full evaluation
Weakness 5 clustering/recoverability
```

Partially done or pending:

```text
computational cost table from final logs
DeepSeek 3B baselines, optional
Module 8 full ERUF eval, optional if final standard metrics are needed
```

### Reviewer bRsP

Mostly writing and scope concerns.

Answer with:

```text
novelty positioning
scope clarification
artifact/code correction
dataset card and subject-count transparency
RWKU overlap limitation
```

No major new experiment required.

### Reviewer PLEX

Fair and detailed.

Done:

```text
small-model baseline reruns for Qwen/Llama
Qwen saved-suite locality and adversarial robustness
Weakness 5 hidden clustering
```

Still useful:

```text
cost table
clearer novelty comparison
RWKU caveat
attenuation wording instead of erasure wording
```

## Capacity-transition explanation for reasoning models

Reviewer concern: ERUF's gains look inconsistent across size. Correct clarification:

ERUF does not fail uniformly at 8B. Standard Mistral/Llama-scale models are Type I, while the non-monotonic pattern is concentrated in reasoning-prior Qwen/DeepSeek models. The safer framing is a reasoning-prior capacity transition:

```text
small reasoning models: Type III or visible leakage
mid-sized reasoning models: Type II, surface suppression but internal recoverability
larger reasoning models: Type I, simultaneous surface and internal attenuation
```

Use the short response:

```text
We thank the reviewer for raising this point. We agree that the cross-model pattern needs clearer explanation, but we would clarify that ERUF does not fail uniformly at 8B. In our results, standard Mistral/Llama-scale models are Type I, while the non-monotonic pattern is concentrated in reasoning-prior Qwen/DeepSeek models. There, we observe a capacity transition: smaller reasoning models show Type III behavior, mid-sized reasoning models often reach surface suppression but retain internal recoverability, and larger reasoning models return to Type I.

Our interpretation is that reasoning-prior models compress factual associations, aliases, and multi-step reasoning traces more tightly at small and mid scales. Under this limited-capacity regime, suppressing a mined entity direction may remove direct surface mentions without fully attenuating alternate internal retrieval paths, producing Type II behavior. Once model capacity increases, the target representation appears more separable, allowing ERUF to suppress both surface leakage and EL10. This is consistent with recent work showing that factual knowledge storage is capacity-limited and scales with parameter count (Allen-Zhu and Li, 2024), and with reasoning-model reports showing that Qwen and DeepSeek reasoning families rely on heavy post-training/distillation across different model sizes (Yang et al., 2024; DeepSeek-AI et al., 2025). It is also consistent with model-editing literature showing that factual associations are not always cleanly isolated and that reliability/locality tradeoffs arise when editing entangled knowledge (Meng et al., 2022; Wang et al., 2024).

We will revise the discussion to describe this as a reasoning-prior capacity transition, not a general 8B failure. This also motivates our use of EL10 and adversarial diagnostics: surface forgetting alone can hide residual internal recoverability, especially in compressed reasoning models.
```

## Main rebuttal story to preserve

1. Layer selection is data-driven, not arbitrary.
2. alpha_eff is a controllable internal attenuation knob.
3. Existing baselines do not cleanly solve the weak 3B setting.
4. ReGLU can strongly suppress direct leakage, but utility and robust recovery remain problematic.
5. LUNAR can look better on some adversarial recovery metrics, but it has severe utility and EL10 issues.
6. Hidden-space clustering now supports reduced subject-level separability, but should be framed as attenuation.
7. ERUF should not be claimed as formal irreversible deletion.

## What not to overclaim

Do not say:

```text
ERUF proves complete erasure.
ERUF uniformly beats every baseline at 3B.
ReGLU fails to forget on Qwen 3B.
LUNAR is better overall than ReGLU.
The Weakness 5 clustering proves semantic deletion.
The 8B prompt file included Kanye West or Arijit Singh.
```

Do say:

```text
ERUF reduces surface leakage and internal recoverability.
ERUF attenuates subject-linked activation signatures.
Direct SMR/EL10 are useful but not sufficient by themselves.
Robust/adversarial diagnostics expose residual recoverability.
Small reasoning-prior models show a capacity transition.
Hidden-space clustering supports reduced separability, not formal erasure.
```

## Immediate next actions

1. Pull the new summary and handover files on the working machine.
2. Use `rebuttal/successful_helpful_experiments_summary.md` as the master rebuttal evidence file.
3. Add Weakness 5 clustering table to appendix or rebuttal response.
4. Update Table 5 wording from erasure to attenuation.
5. Compile final runtime/cost table from the 24 GB ERUF8B logs.
6. Only run Module 8 if the final standard evaluation table is needed.

## Useful local commands

### Check ERUF8B prompt subjects

```powershell
cd C:\Users\user4\Desktop\ERUF8B

@'
import json
from collections import Counter
from pathlib import Path
p = Path("outputs/datasets/prompts.jsonl")
c = Counter()
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    subj = row.get("subject") or row.get("target_subject") or row.get("entity") or row.get("target") or row.get("name")
    if subj:
        c[subj] += 1
for k, v in sorted(c.items()):
    print(f"{k}: {v}")
'@ | Set-Content .\debug_inventory\check_prompt_subjects.py -Encoding UTF8

python .\debug_inventory\check_prompt_subjects.py
```

### Find newest Weakness 5 output

```powershell
$latest = Get-ChildItem .\outputs -Directory |
  Where-Object { $_.Name -like "weakness5_differential_clustering_*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Write-Host "Latest output:" $latest.FullName
Get-Content "$($latest.FullName)\differential_clustering_summary.md"
```

### Pull rebuttal branch on WSL

```bash
cd /mnt/e/eruf/kif-evals
git checkout rebuttal-experiments
git pull origin rebuttal-experiments
```
