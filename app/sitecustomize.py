"""Windows UTF-8 compatibility for repository scripts.

Python loads this module automatically when running scripts from this
folder. Keep the patch narrow: only Path.write_text calls that omit an
encoding get UTF-8, preventing Windows cp1252 UnicodeEncodeError crashes
when JSON/text outputs contain non-ASCII characters.
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
