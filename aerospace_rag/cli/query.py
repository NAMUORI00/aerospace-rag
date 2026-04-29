from __future__ import annotations

import argparse

from ..cli_utils import safe_print
from ..pipeline import ask


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a question against the local aerospace RAG index.")
    parser.add_argument("question")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--farm-id", default="default")
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument(
        "--provider",
        choices=["extractive", "ollama"],
        default=None,
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    response = ask(
        args.question,
        index_dir=args.index_dir,
        top_k=args.top_k,
        provider=args.provider,
        debug=args.debug,
        farm_id=args.farm_id,
        include_private=bool(args.include_private),
    )
    safe_print(response.answer)
    safe_print("\nSources:")
    for idx, source in enumerate(response.sources, start=1):
        loc = f"p.{source.page}" if source.page else f"{source.sheet}:{source.row}" if source.row else "table"
        safe_print(f"{idx}. {source.source_file} ({loc}) score={source.score:.3f}")
    if args.debug:
        safe_print(f"\nDiagnostics: {response.diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
