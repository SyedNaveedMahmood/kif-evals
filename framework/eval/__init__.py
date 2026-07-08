"""Evaluation package exports and Windows-safe text writing."""

from pathlib import Path

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
