from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .models import Chunk
from .text import tokenize, unique_ordered


KNOWN_ENTITIES = [
    "H3",
    "H3 8호기",
    "QZS-5",
    "미치비키",
    "NASA",
    "NOAA",
    "Momentus",
    "solar sail",
    "위성영상",
    "저장영상",
    "신규촬영",
    "K2",
    "K3",
    "K3A",
    "SAR",
    "EO",
    "JAXA",
    "ISRO",
    "KARI",
    "CNSA",
    "ESA",
    "판매대행사",
    "나라장터",
]


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def extract_entities(chunk: Chunk) -> list[str]:
    text = chunk.text
    candidates: list[str] = []
    lower = text.lower()
    for entity in KNOWN_ENTITIES:
        if entity.lower() in lower:
            candidates.append(entity)
    keywords = str(chunk.metadata.get("keywords") or "")
    if keywords:
        candidates.extend(part.strip() for part in re.split(r"[,;/]", keywords))
    token_counts = Counter(tokenize(text))
    candidates.extend(token for token, count in token_counts.most_common(12) if count >= 2 and len(token) >= 2)
    return unique_ordered(candidates)[:24]


class GraphStore:
    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.graph_dir = self.index_dir / "falkordb"
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.graph_dir / "falkordb.db"
        self.index_path = self.graph_dir / "graph_index.json"

    def build(self, chunks: list[Chunk], *, reset: bool = True) -> None:
        if reset:
            for path in (self.db_path, self.index_path):
                if path.exists():
                    path.unlink()
        entity_to_chunks: dict[str, list[str]] = defaultdict(list)
        chunk_entities: dict[str, list[str]] = {}
        for chunk in chunks:
            entities = extract_entities(chunk)
            chunk_entities[chunk.chunk_id] = entities
            for entity in entities:
                entity_to_chunks[entity].append(chunk.chunk_id)
        self._write_falkordb(chunks, chunk_entities)
        payload = {
            "entity_to_chunks": {k: unique_ordered(v) for k, v in entity_to_chunks.items()},
            "chunk_entities": chunk_entities,
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_falkordb(self, chunks: list[Chunk], chunk_entities: dict[str, list[str]]) -> None:
        graph = self._connect_graph()
        if graph is None:
            self.db_path.touch()
            return
        try:
            graph.delete()
        except Exception:
            pass
        graph = self._connect_graph()
        if graph is None:
            self.db_path.touch()
            return
        for chunk in chunks:
            doc_id = chunk.source_file
            chunk_id = chunk.chunk_id
            graph.query(
                "MERGE (d:Document {id:'%s', source_file:'%s'}) "
                "MERGE (s:SourceFile {name:'%s'}) "
                "MERGE (c:Chunk {id:'%s', modality:'%s', text:'%s'}) "
                "MERGE (d)-[:HAS_CHUNK]->(c) "
                "MERGE (c)-[:FROM_FILE]->(s)"
                % (
                    _escape(doc_id),
                    _escape(chunk.source_file),
                    _escape(chunk.source_file),
                    _escape(chunk_id),
                    _escape(chunk.modality),
                    _escape(chunk.text[:900]),
                )
            )
            for entity in chunk_entities.get(chunk_id, []):
                graph.query(
                    "MATCH (c:Chunk {id:'%s'}) "
                    "MERGE (e:Entity {name:'%s'}) "
                    "MERGE (c)-[:MENTIONS]->(e)"
                    % (_escape(chunk_id), _escape(entity))
                )

    def _connect_graph(self):
        try:
            from redislite.falkordb_client import FalkorDB

            db = FalkorDB(str(self.db_path))
            return db.select_graph("aerospace")
        except Exception:
            pass

        try:
            import os
            from falkordb import FalkorDB

            host = os.environ.get("FALKORDB_HOST")
            if not host:
                return None
            port = int(os.environ.get("FALKORDB_PORT", "6379"))
            password = os.environ.get("FALKORDB_PASSWORD") or None
            db = FalkorDB(host=host, port=port, password=password)
            return db.select_graph(os.environ.get("FALKORDB_GRAPH", "aerospace"))
        except Exception:
            return None

    def search(self, query: str, *, limit: int = 8) -> list[tuple[str, float]]:
        if not self.index_path.exists():
            return []
        payload: dict[str, Any] = json.loads(self.index_path.read_text(encoding="utf-8"))
        entity_to_chunks: dict[str, list[str]] = payload.get("entity_to_chunks") or {}
        qtokens = set(tokenize(query))
        scores: Counter[str] = Counter()
        for entity, chunk_ids in entity_to_chunks.items():
            entity_tokens = set(tokenize(entity))
            if not entity_tokens:
                continue
            overlap = len(qtokens & entity_tokens)
            substring = entity.lower() in query.lower() or any(t in entity.lower() for t in qtokens if len(t) >= 2)
            if overlap or substring:
                weight = float(overlap or 1)
                for chunk_id in chunk_ids:
                    scores[chunk_id] += weight
        return scores.most_common(limit)
