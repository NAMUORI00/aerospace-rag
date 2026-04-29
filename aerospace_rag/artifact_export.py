"""Compatibility wrapper for :mod:`aerospace_rag.artifacts.export`."""

from __future__ import annotations

from .artifacts.export import *  # noqa: F401,F403
from .artifacts.export import main


if __name__ == "__main__":
    raise SystemExit(main())

