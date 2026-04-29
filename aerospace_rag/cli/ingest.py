from __future__ import annotations

import argparse

from ..pipeline import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local aerospace RAG index.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--no-reset", action="store_true")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--strict-expected",
        action="store_true",
        help="require the original five release files instead of ingesting every supported file",
    )
    mode_group.add_argument(
        "--no-strict-expected",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="when using the release file contract, also ingest additional supported files",
    )
    args = parser.parse_args()

    result = build_index(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        reset=not args.no_reset,
        strict_expected=bool(args.strict_expected),
        include_extra=bool(args.include_extra),
    )
    print(
        f"indexed files={result.file_count} chunks={result.chunk_count} "
        f"qdrant={result.qdrant_collection} graph_index={result.graph_index_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
