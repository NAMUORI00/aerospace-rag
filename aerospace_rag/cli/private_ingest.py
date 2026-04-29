from __future__ import annotations

import argparse

from ..config import Settings
from ..stores.local_index import LocalIndex
from ..stores.private_overlay import PrivateOverlayStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Add private overlay knowledge to the local aerospace RAG runtime.")
    parser.add_argument("text")
    parser.add_argument("--farm-id", default="default")
    parser.add_argument("--source-type", default="memo")
    parser.add_argument("--record-id", default="")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--overlay-only", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.overlay_only:
        store = PrivateOverlayStore(settings.private_store_db_path)
        rid = store.upsert_text(
            text=args.text,
            farm_id=args.farm_id,
            source_type=args.source_type,
            record_id=args.record_id or None,
        )
    else:
        rid = LocalIndex(args.index_dir, settings=settings).upsert_private_text(
            text=args.text,
            farm_id=args.farm_id,
            source_type=args.source_type,
            record_id=args.record_id or None,
        )
    print(f"[private-ingest] record_id={rid} farm_id={args.farm_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

