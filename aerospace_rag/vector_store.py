from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

from .config import Settings
from .embeddings import EmbeddingService
from .models import Chunk


COLLECTION_NAME = "aerospace_chunks"


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"aerospace-rag://{chunk_id}"))


class QdrantVectorStore:
    def __init__(self, index_dir: str | Path, *, settings: Settings | None = None, embeddings: EmbeddingService | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.embeddings = embeddings or EmbeddingService(self.settings)
        self.vector_size = self.embeddings.vector_size
        self.path = Path(index_dir) / "qdrant"
        self.path.mkdir(parents=True, exist_ok=True)
        self.fallback_path = self.path / "fallback_vectors.json"
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(path=str(self.path))
        except Exception:
            self.client = None

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def build(self, chunks: list[Chunk], *, reset: bool = True) -> None:
        if self.client is None:
            if reset and self.fallback_path.exists():
                self.fallback_path.unlink()
            vectors = self.embeddings.embed_texts([chunk.text for chunk in chunks])
            payload = [
                {
                    "id": _point_id(chunk.chunk_id),
                    "vector": vector,
                    "payload": chunk.to_payload(),
                }
                for chunk, vector in zip(chunks, vectors)
            ]
            self.fallback_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return

        from qdrant_client import models

        if reset and self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
            )
        vectors = self.embeddings.embed_texts([chunk.text for chunk in chunks])
        points = [
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload=chunk.to_payload(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        if points:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, query: str, *, limit: int = 8) -> list[tuple[str, float]]:
        vector = self.embeddings.embed_text(query)
        if self.client is None:
            return self._fallback_search(vector, limit=limit)
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )
        except AttributeError:
            response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=limit,
                with_payload=True,
            )
            results = response.points
        hits: list[tuple[str, float]] = []
        for point in results:
            payload = dict(point.payload or {})
            chunk_id = str(payload.get("chunk_id") or "")
            if chunk_id:
                hits.append((chunk_id, float(point.score)))
        return hits

    def _fallback_search(self, query_vector: list[float], *, limit: int) -> list[tuple[str, float]]:
        if not self.fallback_path.exists():
            return []
        rows = json.loads(self.fallback_path.read_text(encoding="utf-8"))
        qnorm = math.sqrt(sum(v * v for v in query_vector)) or 1.0
        scored: list[tuple[str, float]] = []
        for row in rows:
            vector = [float(v) for v in row.get("vector") or []]
            payload = dict(row.get("payload") or {})
            chunk_id = str(payload.get("chunk_id") or "")
            if not chunk_id or not vector:
                continue
            vnorm = math.sqrt(sum(v * v for v in vector)) or 1.0
            score = sum(a * b for a, b in zip(query_vector, vector)) / (qnorm * vnorm)
            scored.append((chunk_id, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]
