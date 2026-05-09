from __future__ import annotations

import argparse

from ..cli_utils import safe_print
from ..pipeline import ask


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a question against the local aerospace RAG index.")
    parser.add_argument("question")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--provider",
        choices=["extractive", "ollama"],
        default=None,
    )
    parser.add_argument(
        "--gpt-pro-cross-check",
        action="store_true",
        default=None,
        help="Audit the generated answer against retrieved sources with the configured OpenAI GPT Pro model.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    response = ask(
        args.question,
        index_dir=args.index_dir,
        top_k=args.top_k,
        provider=args.provider,
        debug=args.debug,
        cross_check=args.gpt_pro_cross_check,
    )
    safe_print(response.answer)
    safe_print("\nSources:")
    for idx, source in enumerate(response.sources, start=1):
        loc = f"p.{source.page}" if source.page else f"{source.sheet}:{source.row}" if source.row else "table"
        safe_print(f"{idx}. {source.source_file} ({loc}) score={source.score:.3f}")
    if args.debug:
        safe_print(f"\nDiagnostics: {response.diagnostics}")
    elif args.gpt_pro_cross_check:
        safe_print(f"\nCross-check: {response.diagnostics.get('cross_check')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
