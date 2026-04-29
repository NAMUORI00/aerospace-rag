"""Compatibility wrapper for :mod:`aerospace_rag.artifacts.importer`."""

from __future__ import annotations

from .artifacts.importer import *  # noqa: F401,F403
from .artifacts.importer import main


if __name__ == "__main__":
    raise SystemExit(main())

