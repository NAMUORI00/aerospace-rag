from __future__ import annotations

import sys
from typing import TextIO


def make_console_safe(text: object, *, encoding: str | None = None) -> str:
    value = str(text)
    target = encoding or sys.stdout.encoding or "utf-8"
    return value.encode(target, errors="replace").decode(target, errors="replace")


def safe_print(text: object = "", *, stream: TextIO | None = None) -> None:
    out = stream or sys.stdout
    print(make_console_safe(text, encoding=out.encoding), file=out)
