"""Evaluation package exports and Windows-safe text/console output."""

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

from eval.module8_eval import run_eval, run_module8_clean

__all__ = ["run_eval", "run_module8_clean"]
