from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Chunk, RetrievalHit
from .text import tokenize


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PrivateRecord:
    id: str
    farm_id: str
    source_type: str
    created_at: str
    text: str
    metadata: dict[str, Any]


class PrivateOverlayStore:
    """SQLite private knowledge overlay without the SmartFarm FastAPI layer."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS overlay_records (
                    id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,
                    farm_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_overlay_scope ON overlay_records(tier, farm_id, created_at)"
            )
            conn.commit()

    def upsert_text(
        self,
        *,
        text: str,
        farm_id: str = "default",
        source_type: str = "memo",
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> str:
        rid = str(record_id or uuid.uuid4().hex)
        created = created_at or utc_now_iso()
        datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO overlay_records(id, tier, farm_id, source_type, created_at, text, metadata_json)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  tier=excluded.tier,
                  farm_id=excluded.farm_id,
                  source_type=excluded.source_type,
                  created_at=excluded.created_at,
                  text=excluded.text,
                  metadata_json=excluded.metadata_json
                """,
                (
                    rid,
                    "private",
                    str(farm_id or "default"),
                    str(source_type or "memo"),
                    created,
                    str(text or ""),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            conn.commit()
        return rid

    def has_private_records(self, *, farm_id: str = "default") -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM overlay_records WHERE tier='private' AND farm_id=? LIMIT 1",
                (str(farm_id or "default"),),
            ).fetchone()
        return row is not None

    def purge_private(self, *, farm_id: str = "default") -> int:
        with self._conn() as conn:
            count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM overlay_records WHERE tier='private' AND farm_id=?",
                    (str(farm_id or "default"),),
                ).fetchone()[0]
            )
            conn.execute(
                "DELETE FROM overlay_records WHERE tier='private' AND farm_id=?",
                (str(farm_id or "default"),),
            )
            conn.commit()
        return count

    def _token_score(self, query: str, text: str) -> float:
        q = set(tokenize(query))
        if not q:
            return 0.0
        d = set(tokenize(text))
        if not d:
            return 0.0
        overlap = len(q & d)
        if overlap <= 0:
            return 0.0
        return float(overlap) / float(max(1, len(q)))

    def search(self, *, query: str, farm_id: str = "default", limit: int = 8) -> list[RetrievalHit]:
        with self._conn() as conn:
            rows = list(
                conn.execute(
                    """
                    SELECT id, farm_id, source_type, created_at, text, metadata_json
                    FROM overlay_records
                    WHERE tier='private' AND farm_id=?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (str(farm_id or "default"), max(100, limit * 10)),
                )
            )
        hits: list[RetrievalHit] = []
        for row in rows:
            text = str(row["text"] or "")
            score = self._token_score(query, text)
            if score <= 0:
                continue
            metadata = json.loads(str(row["metadata_json"] or "{}"))
            chunk = Chunk(
                chunk_id=str(row["id"]),
                text=text,
                source_file=f"private:{row['source_type']}",
                modality=str(metadata.get("modality") or "text"),
                metadata={
                    **metadata,
                    "tier": "private",
                    "farm_id": str(row["farm_id"]),
                    "source_type": str(row["source_type"]),
                    "created_at": str(row["created_at"]),
                },
            )
            hits.append(RetrievalHit(chunk=chunk, score=score, channels={"private_overlay": score}))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]
