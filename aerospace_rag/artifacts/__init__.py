"""Portable index artifact import and export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_artifact_manifest(*, index_dir: Path, output_dir: Path) -> Path:
    from .export import build_artifact_manifest as impl

    return impl(index_dir=index_dir, output_dir=output_dir)


def import_artifact_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .importer import import_artifact_manifest as impl

    return impl(*args, **kwargs)


__all__ = ["build_artifact_manifest", "import_artifact_manifest"]
