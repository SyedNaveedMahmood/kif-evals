"""
Notebook-derived pipeline modules.

Each module exposes a `run_module*` entrypoint that mirrors the original cell.
"""

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

from . import module0, module_a, module_b, module_c, module_d, module_e, module7, module8

__all__ = [
    "module0",
    "module_a",
    "module_b",
    "module_c",
    "module_d",
    "module_e",
    "module7",
    "module8",
]
