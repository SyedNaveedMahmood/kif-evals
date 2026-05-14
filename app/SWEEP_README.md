# KIF τ/k Sweep on bwUniCluster 3.0

This codebase has been organized for the reviewer-facing sweep:

- Shared once: Module 0 → Module A → Module B → Module C → Module D
- Per sweep worker: Module E → Module 7 → optional Module 8
- Each worker uses one GPU through `CUDA_VISIBLE_DEVICES`.
- Each run writes to its own folder under `outputs/sweeps/<config>/seed<seed>/`.

## Important design choice

For exactly **3 configs × 3 seeds**, each config job requests **3 H100 GPUs** and launches 3 workers. That is non-wasteful and methodologically clean.

If you want all **4 H100 GPUs** active per config job, use the provided 4-seed script instead:

```bash
bash scripts/submit_sweep_4seed.sh
```

That runs 3 configs × 4 seeds.

## One-time shared run

```bash
sbatch scripts/slurm_shared_0_to_d.sh
```

## 3 configs × 3 seeds

```bash
bash scripts/submit_sweep_3seed.sh
```

## 3 configs × 4 seeds, all 4 H100s per config job

```bash
bash scripts/submit_sweep_4seed.sh
```

## Optional environment activation

The Slurm files support either:

```bash
KIF_CONDA_ENV=my_env bash scripts/submit_sweep_3seed.sh
```

or:

```bash
KIF_VENV_ACTIVATE=/path/to/venv/bin/activate bash scripts/submit_sweep_3seed.sh
```

## Optional evaluation

By default, the sweep runs Module E and Module 7. To also run Module 8 immediately after each adapter:

```bash
RUN_EVAL=1 bash scripts/submit_sweep_3seed.sh
```

## Monitor jobs

```bash
squeue -u $USER
squeue --start -u $USER
```

## Aggregate results

```bash
python scripts/aggregate_sweep_results.py --sweep_root outputs/sweeps --out outputs/sweeps/sweep_summary.json
```
