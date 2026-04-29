from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..config import Settings
from ..stores.vector import COLLECTION_NAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def _describe_files(base: Path, files: Iterable[Path]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for path in files:
        rel = path.relative_to(base)
        out.append(
            {
                "path": str(rel).replace("\\", "/"),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return out


def build_artifact_manifest(
    *,
    index_dir: Path,
    output_dir: Path,
    settings: Settings | None = None,
) -> Path:
    resolved_settings = settings or Settings.from_env()
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = {
        "qdrant": index_dir / "qdrant",
        "falkordb": index_dir / "falkordb",
    }
    files = {
        "bm25": index_dir / "bm25.json",
        "chunks": index_dir / "chunks.jsonl",
    }
    artifacts: dict[str, object] = {}
    for label, root in blocks.items():
        block_files = _iter_files(root)
        artifacts[label] = {
            "root": str(root),
            "file_count": len(block_files),
            "files": _describe_files(root, block_files),
        }
    for label, path in files.items():
        artifacts[label] = {
            "root": str(path.parent),
            "file_count": 1 if path.exists() else 0,
            "files": _describe_files(path.parent, [path] if path.exists() else []),
        }

    manifest = {
        "schema_version": "aerospace_rag_artifact_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "runtime": {
            "embedding_model": resolved_settings.embed_model,
            "embedding_dim": resolved_settings.embed_dim,
            "qdrant_collection": COLLECTION_NAME,
            "dat_mode": resolved_settings.dat_mode,
        },
    }
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    path = output_dir / "artifact_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a portable aerospace RAG index manifest.")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--output-dir", default="data/index/export")
    args = parser.parse_args()

    out = build_artifact_manifest(index_dir=Path(args.index_dir), output_dir=Path(args.output_dir))
    print(f"[artifact-export] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

