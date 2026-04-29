from __future__ import annotations

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
        from qdrant_client import QdrantClient

        self.settings = settings or Settings.from_env()
        self.embeddings = embeddings or EmbeddingService(self.settings)
        self.vector_size = self.embeddings.vector_size
        self.path = Path(index_dir) / "qdrant"
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def build(self, chunks: list[Chunk], *, reset: bool = True) -> None:
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
