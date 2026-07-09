"""
Notebook-derived pipeline modules.

Each module exposes a `run_module*` entrypoint that mirrors the original cell.
"""

import os
import sys
from pathlib import Path


def _force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8_stdio()

_ORIGINAL_WRITE_TEXT = Path.write_text


def _write_text_utf8_default(self, data, encoding=None, errors=None, newline=None):
    if encoding is None:
        encoding = "utf-8"
    return _ORIGINAL_WRITE_TEXT(
        self,
        data,
        encoding=encoding,
        errors=errors,
        newline=newline,
    )


Path.write_text = _write_text_utf8_default

from . import module0, module_a, module_b, module_c, module_d, module_e, module7, module8, module_elude, module8e

__all__ = [
    "module0",
    "module_a",
    "module_b",
    "module_c",
    "module_d",
    "module_e",
    "module7",
    "module8",
    "module_elude",
    "module8e",
]
