"""Compatibility wrapper for :mod:`aerospace_rag.cli.query`."""

from __future__ import annotations

from .cli.query import main


if __name__ == "__main__":
    raise SystemExit(main())

