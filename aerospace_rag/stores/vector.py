from __future__ import annotations

import json
import math
import hashlib
from collections import Counter
import uuid
from pathlib import Path

from ..config import Settings
from ..models import Chunk
from ..retrieval.embeddings import EmbeddingService
from ..text import tokenize


COLLECTION_NAME = "aerospace_chunks"


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"aerospace-rag://{chunk_id}"))


class QdrantVectorStore:
    def __init__(
        self,
        index_dir: str | Path,
        *,
        settings: Settings | None = None,
        embeddings: EmbeddingService | None = None,
        force_fallback: bool = False,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.embeddings = embeddings or EmbeddingService(self.settings)
        self.vector_size = self.embeddings.vector_size
        self.path = Path(index_dir) / "qdrant"
        self.path.mkdir(parents=True, exist_ok=True)
        self.fallback_path = self.path / "fallback_vectors.json"
        if force_fallback:
            self.client = None
            return
        try:
            from qdrant_client import QdrantClient

            if self.settings.qdrant_url:
                self.client = QdrantClient(url=self.settings.qdrant_url)
            elif self.settings.qdrant_host:
                self.client = QdrantClient(host=self.settings.qdrant_host, port=int(self.settings.qdrant_port))
            else:
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
            self._fallback_upsert_chunks(chunks)
            return

        from qdrant_client import models

        if reset and self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense_text": models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
                    "dense_image": models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
                },
            )
        vectors = self.embeddings.embed_texts([chunk.text for chunk in chunks])
        points = [
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector={
                    "dense_text": vector,
                    "dense_image": vector,
                    "sparse": self._to_qdrant_sparse(self._sparse_vector(chunk.text), models),
                },
                payload=chunk.to_payload(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        if points:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if self.client is None:
            self._fallback_upsert_chunks(chunks)
            return
        self.build(chunks, reset=False)

    def _sparse_vector(self, text: str) -> dict[str, list[float] | list[int]]:
        counts = Counter(tokenize(text))
        if not counts:
            return {"indices": [], "values": []}
        indices: list[int] = []
        values: list[float] = []
        for token, count in counts.items():
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % 65536
            indices.append(idx)
            values.append(1.0 + math.log1p(float(count)))
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return {"indices": indices, "values": [v / norm for v in values]}

    def _to_qdrant_sparse(self, sparse: dict[str, list[float] | list[int]], models):
        return models.SparseVector(
            indices=[int(i) for i in sparse.get("indices") or []],
            values=[float(v) for v in sparse.get("values") or []],
        )

    def _fallback_rows(self) -> list[dict]:
        if not self.fallback_path.exists():
            return []
        return list(json.loads(self.fallback_path.read_text(encoding="utf-8")))

    def _fallback_upsert_chunks(self, chunks: list[Chunk]) -> None:
        rows = self._fallback_rows()
        by_chunk = {str((row.get("payload") or {}).get("chunk_id") or ""): row for row in rows}
        vectors = self.embeddings.embed_texts([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            payload = chunk.to_payload()
            asset_ref = str(payload.get("asset_ref") or "")
            image_basis = f"{asset_ref}\n{chunk.text}".strip() if chunk.modality == "image" and asset_ref else chunk.text
            dense_image = self.embeddings.embed_text(image_basis)
            by_chunk[chunk.chunk_id] = {
                "id": _point_id(chunk.chunk_id),
                "vectors": {
                    "dense_text": vector,
                    "dense_image": dense_image,
                    "sparse": self._sparse_vector(chunk.text),
                },
                "payload": payload,
            }
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_path.write_text(json.dumps(list(by_chunk.values()), ensure_ascii=False), encoding="utf-8")

    def search(self, query: str, *, limit: int = 8) -> list[tuple[str, float]]:
        channels = self.search_channels(query, limit=limit)
        dense = channels.get("vector_dense_text") or []
        return dense[:limit]

    def search_channels(
        self,
        query: str,
        *,
        limit: int = 8,
        farm_id: str = "default",
        include_private: bool = False,
        modalities: set[str] | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        vector = self.embeddings.embed_text(query)
        if self.client is None:
            return self._fallback_search_channels(
                query=query,
                query_vector=vector,
                limit=limit,
                farm_id=farm_id,
                include_private=include_private,
                modalities=modalities,
            )
        return self._qdrant_search_channels(
            query=query,
            query_vector=vector,
            limit=limit,
            farm_id=farm_id,
            include_private=include_private,
            modalities=modalities,
        )

    def _qdrant_filter(self, farm_id: str, *, include_private: bool, models):
        if not include_private:
            return models.Filter(must=[models.FieldCondition(key="tier", match=models.MatchValue(value="public"))])
        return models.Filter(
            should=[
                models.FieldCondition(key="tier", match=models.MatchValue(value="public")),
                models.Filter(
                    must=[
                        models.FieldCondition(key="tier", match=models.MatchValue(value="private")),
                        models.FieldCondition(key="farm_id", match=models.MatchValue(value=farm_id)),
                    ]
                ),
            ]
        )

    def _qdrant_search_channels(
        self,
        *,
        query: str,
        query_vector: list[float],
        limit: int,
        farm_id: str,
        include_private: bool,
        modalities: set[str] | None,
    ) -> dict[str, list[tuple[str, float]]]:
        from qdrant_client import models

        qfilter = self._qdrant_filter(farm_id, include_private=include_private, models=models)
        out: dict[str, list[tuple[str, float]]] = {
            "vector_dense_text": [],
            "vector_sparse": [],
            "vector_image": [],
        }
        for using, channel, query_value in (
            ("dense_text", "vector_dense_text", query_vector),
            ("sparse", "vector_sparse", self._to_qdrant_sparse(self._sparse_vector(query), models)),
            ("dense_image", "vector_image", query_vector),
        ):
            if modalities and channel == "vector_image" and "image" not in modalities:
                continue
            try:
                response = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_value,
                    using=using,
                    query_filter=qfilter,
                    limit=max(limit * 3, 20),
                    with_payload=True,
                )
                points = response.points
            except Exception:
                points = []
            hits: list[tuple[str, float]] = []
            for point in points:
                payload = dict(point.payload or {})
                chunk_id = str(payload.get("chunk_id") or "")
                if chunk_id:
                    hits.append((chunk_id, float(point.score)))
            out[channel] = hits[:limit]
        return out

    def _passes_scope(
        self,
        payload: dict,
        *,
        farm_id: str,
        include_private: bool,
        modalities: set[str] | None,
    ) -> bool:
        tier = str(payload.get("tier") or "public").lower()
        if tier == "private" and (not include_private or str(payload.get("farm_id") or "") != str(farm_id)):
            return False
        if tier != "private" and tier != "public":
            return False
        if modalities and str(payload.get("modality") or "text").lower() not in modalities:
            return False
        return True

    def _sparse_score(self, query_sparse: dict, row_sparse: dict) -> float:
        q = {int(i): float(v) for i, v in zip(query_sparse.get("indices") or [], query_sparse.get("values") or [])}
        d = {int(i): float(v) for i, v in zip(row_sparse.get("indices") or [], row_sparse.get("values") or [])}
        if not q or not d:
            return 0.0
        return sum(qv * d.get(qi, 0.0) for qi, qv in q.items())

    def _fallback_search_channels(
        self,
        *,
        query: str,
        query_vector: list[float],
        limit: int,
        farm_id: str,
        include_private: bool,
        modalities: set[str] | None,
    ) -> dict[str, list[tuple[str, float]]]:
        rows = self._fallback_rows()
        query_sparse = self._sparse_vector(query)
        out: dict[str, list[tuple[str, float]]] = {
            "vector_dense_text": [],
            "vector_sparse": [],
            "vector_image": [],
        }
        for row in rows:
            payload = dict(row.get("payload") or {})
            if not self._passes_scope(payload, farm_id=farm_id, include_private=include_private, modalities=modalities):
                continue
            chunk_id = str(payload.get("chunk_id") or "")
            vectors = dict(row.get("vectors") or {})
            if not vectors and row.get("vector") is not None:
                vectors["dense_text"] = row.get("vector")
            dense_text = [float(v) for v in vectors.get("dense_text") or []]
            dense_image = [float(v) for v in vectors.get("dense_image") or dense_text]
            sparse = dict(vectors.get("sparse") or {})
            if chunk_id and dense_text:
                out["vector_dense_text"].append((chunk_id, self._cosine(query_vector, dense_text)))
            if chunk_id and sparse:
                score = self._sparse_score(query_sparse, sparse)
                if score > 0:
                    out["vector_sparse"].append((chunk_id, score))
            if chunk_id and dense_image and str(payload.get("modality") or "").lower() == "image":
                out["vector_image"].append((chunk_id, self._cosine(query_vector, dense_image)))
        for channel in out:
            out[channel].sort(key=lambda item: item[1], reverse=True)
            out[channel] = out[channel][:limit]
        return out

    def _cosine(self, left: list[float], right: list[float]) -> float:
        qnorm = math.sqrt(sum(v * v for v in left)) or 1.0
        vnorm = math.sqrt(sum(v * v for v in right)) or 1.0
        return float(sum(a * b for a, b in zip(left, right)) / (qnorm * vnorm))

    def _legacy_search(self, query: str, *, limit: int = 8) -> list[tuple[str, float]]:
        vector = self.embeddings.embed_text(query)
        if self.client is None:
            channels = self._fallback_search_channels(
                query=query,
                query_vector=vector,
                limit=limit,
                farm_id="default",
                include_private=False,
                modalities=None,
            )
            return channels.get("vector_dense_text", [])
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
