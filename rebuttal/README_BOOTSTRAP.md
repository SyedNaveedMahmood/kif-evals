# Rebuttal experiment source bundle

This branch contains a self-contained rebuttal source bundle under `rebuttal/`.

Run this once from the repository root to materialize the Python source files:

```bash
python rebuttal/unpack_sources.py
```

Then install and run:

```bash
cd rebuttal
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_probe_suite.py --help
```

The unpacked folder includes the standalone scripts for E2, E3, E4, E5, E7, and E8 plus an intentionally empty `slurm/` folder.
