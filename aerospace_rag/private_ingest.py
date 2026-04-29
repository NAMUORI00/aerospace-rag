"""Compatibility wrapper for :mod:`aerospace_rag.cli.private_ingest`."""

from __future__ import annotations

from .cli.private_ingest import main


if __name__ == "__main__":
    raise SystemExit(main())

