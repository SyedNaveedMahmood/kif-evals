#!/usr/bin/env python3
"""Patch the core KIF Module C positive-oversampling indexing bug.

Bug:
    indices = np.random.choice(len(positives), ...)
    oversampled_positives = [prompts[i].copy() for i in indices]

The sampled indices are local to `positives`, but the code indexes the full
`prompts` list. If controls appear before/among positives, positive oversampling
can accidentally reintroduce controls into the positive pool.

Fix:
    oversampled_positives = [positives[i].copy() for i in indices]

Run from repository root:
    python analysis/patch_core_module_c_oversampling.py
"""
from pathlib import Path

TARGETS = [
    Path("app/src/llama20/modules/module_c.py"),
    Path("elude/src/llama20/modules/module_c.py"),
]

OLD = "oversampled_positives = [prompts[i].copy() for i in indices]"
NEW = "oversampled_positives = [positives[i].copy() for i in indices]"


def patch(path: Path) -> str:
    if not path.exists():
        return f"MISSING: {path}"
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        return f"OK already patched: {path}"
    if OLD not in text:
        return f"PATTERN NOT FOUND: {path}"
    path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    return f"PATCHED: {path}"


if __name__ == "__main__":
    for p in TARGETS:
        print(patch(p))
