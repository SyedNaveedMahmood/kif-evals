#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "rebuttal_sources.tgz.b64"
ROOT = HERE.parent


def main() -> None:
    raw = base64.b64decode(BUNDLE.read_text().strip())
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(ROOT)
    print(f"Unpacked rebuttal source files into: {HERE}")
    print("Next: cd rebuttal && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt")


if __name__ == "__main__":
    main()
