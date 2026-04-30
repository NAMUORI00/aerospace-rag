from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_source_root(*, explicit: Path | None, manifest_root: str, alternate_dir: Path, label: str) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidate = Path(str(manifest_root or "")).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if alternate_dir.exists():
        return alternate_dir.resolve()
    raise FileNotFoundError(f"cannot resolve source root for {label}: {manifest_root!r}")


def _copy_entries(
    *,
    source_root: Path,
    target_root: Path,
    entries: Iterable[dict[str, object]],
    verify_sha256: bool,
) -> dict[str, int]:
    copied = 0
    copied_bytes = 0
    target_root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        rel = str(entry.get("path") or "").strip().replace("\\", "/")
        if not rel:
            continue
        source = (source_root / rel).resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"artifact file missing: {source}")
        if verify_sha256:
            expected = str(entry.get("sha256") or "").strip().lower()
            if expected:
                got = _sha256(source).lower()
                if got != expected:
                    raise ValueError(f"sha256 mismatch for {source}: expected={expected} got={got}")
        target = (target_root / rel).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target:
            shutil.copy2(source, target)
        copied += 1
        copied_bytes += int(target.stat().st_size)
    return {"copied_files": copied, "copied_bytes": copied_bytes}


def import_artifact_manifest(
    *,
    manifest_path: Path,
    index_dir: Path,
    source_overrides: dict[str, Path] | None = None,
    verify_sha256: bool = True,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts") or {}
    overrides = dict(source_overrides or {})
    summary: dict[str, object] = {
        "manifest_path": str(manifest_path),
        "index_dir": str(index_dir),
        "verify_sha256": bool(verify_sha256),
        "artifacts": {},
        "runtime": manifest.get("runtime") or {},
    }
    targets = {
        "qdrant": index_dir / "qdrant",
        "graph": index_dir / "graph",
        "bm25": index_dir,
        "chunks": index_dir,
    }
    for label, target in targets.items():
        block = artifacts.get(label) or {}
        entries = block.get("files") or []
        if not isinstance(entries, list):
            entries = []
        if not entries:
            summary["artifacts"][label] = {
                "source_root": "",
                "target_root": str(target),
                "copied_files": 0,
                "copied_bytes": 0,
            }
            continue
        source_root = _resolve_source_root(
            explicit=overrides.get(label),
            manifest_root=str(block.get("root") or ""),
            alternate_dir=manifest_path.parent / label,
            label=label,
        )
        copied = _copy_entries(
            source_root=source_root,
            target_root=target,
            entries=entries,
            verify_sha256=verify_sha256,
        )
        summary["artifacts"][label] = {
            "source_root": str(source_root),
            "target_root": str(target),
            **copied,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a portable aerospace RAG index manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--qdrant-source-dir", default="")
    parser.add_argument("--graph-source-dir", default="")
    parser.add_argument("--bm25-source-dir", default="")
    parser.add_argument("--chunks-source-dir", default="")
    parser.add_argument("--skip-sha256-check", action="store_true")
    args = parser.parse_args()

    overrides = {
        "qdrant": Path(args.qdrant_source_dir) if str(args.qdrant_source_dir).strip() else None,
        "graph": Path(args.graph_source_dir) if str(args.graph_source_dir).strip() else None,
        "bm25": Path(args.bm25_source_dir) if str(args.bm25_source_dir).strip() else None,
        "chunks": Path(args.chunks_source_dir) if str(args.chunks_source_dir).strip() else None,
    }
    summary = import_artifact_manifest(
        manifest_path=Path(args.manifest),
        index_dir=Path(args.index_dir),
        source_overrides={k: v for k, v in overrides.items() if v is not None},
        verify_sha256=not bool(args.skip_sha256_check),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
