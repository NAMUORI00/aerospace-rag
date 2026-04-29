from __future__ import annotations

import argparse

from .pipeline import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local aerospace RAG index.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    result = build_index(data_dir=args.data_dir, index_dir=args.index_dir, reset=not args.no_reset)
    print(
        f"indexed files={result.file_count} chunks={result.chunk_count} "
        f"qdrant={result.qdrant_collection} falkordb={result.falkordb_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
