from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import Settings
from ..models import Chunk, RetrievalHit
from ..retrieval.bm25 import BM25Index
from ..retrieval.fusion import ChannelHit, resolve_enterprise_weights, weighted_rrf
from .graph import GraphStore
from .private_overlay import PrivateOverlayStore
from .vector import COLLECTION_NAME, QdrantVectorStore


def write_chunks(chunks: list[Chunk], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_payload(), ensure_ascii=False) + "\n")


def read_chunks(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    p = Path(path)
    if not p.exists():
        return chunks
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.from_payload(json.loads(line)))
    return chunks


class LocalIndex:
    def __init__(self, index_dir: str | Path, *, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.index_dir = Path(index_dir)
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.bm25_path = self.index_dir / "bm25.json"
        self.graph = GraphStore(self.index_dir)
        self.last_diagnostics: dict[str, object] = {}

    def build(self, chunks: list[Chunk], *, reset: bool = True) -> None:
        if reset and self.index_dir.exists():
            qdrant_dir = self.index_dir / "qdrant"
            if qdrant_dir.exists():
                shutil.rmtree(qdrant_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        write_chunks(chunks, self.chunks_path)
        bm25 = BM25Index.build(chunks)
        bm25.save(self.bm25_path)
        vectors = QdrantVectorStore(self.index_dir, settings=self.settings)
        try:
            vectors.build(chunks, reset=reset)
        finally:
            vectors.close()
        self.graph.build(chunks, reset=reset)

    def upsert_private_text(
        self,
        *,
        text: str,
        farm_id: str = "default",
        source_type: str = "memo",
        record_id: str | None = None,
    ) -> str:
        overlay = PrivateOverlayStore(self.settings.private_store_db_path)
        rid = overlay.upsert_text(text=text, farm_id=farm_id, source_type=source_type, record_id=record_id)
        stored = overlay.get_record(rid)
        chunk = Chunk(
            chunk_id=rid,
            text=text,
            source_file=f"private:{source_type}",
            modality=str((stored.metadata if stored else {}).get("modality") or "text"),
            metadata={
                **(stored.metadata if stored else {}),
                "tier": "private",
                "farm_id": farm_id,
                "source_type": source_type,
                "created_at": stored.created_at if stored else "",
            },
        )
        chunks = read_chunks(self.chunks_path)
        chunks = [existing for existing in chunks if existing.chunk_id != rid]
        chunks.append(chunk)
        write_chunks(chunks, self.chunks_path)
        bm25 = BM25Index.build(chunks)
        bm25.save(self.bm25_path)
        vectors = QdrantVectorStore(self.index_dir, settings=self.settings)
        try:
            vectors.upsert_chunks([chunk])
        finally:
            vectors.close()
        self.graph.upsert_private_chunk(chunk)
        return rid

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        farm_id: str = "default",
        include_private: bool = False,
    ) -> list[RetrievalHit]:
        chunks = {chunk.chunk_id: chunk for chunk in read_chunks(self.chunks_path)}
        channel_scores: dict[str, dict[str, float]] = {
            "vector_dense_text": {},
            "vector_sparse": {},
            "vector_image": {},
            "graph": {},
            "private_overlay": {},
        }
        vectors = QdrantVectorStore(self.index_dir, settings=self.settings)
        try:
            vector_channels = vectors.search_channels(
                query,
                limit=max(top_k * 3, 12),
                farm_id=farm_id,
                include_private=include_private,
            )
        finally:
            vectors.close()
        for channel in ("vector_dense_text", "vector_sparse", "vector_image"):
            channel_scores[channel].update(dict(vector_channels.get(channel) or []))
        if self.bm25_path.exists():
            for chunk_id, score in BM25Index.load(self.bm25_path).search(query, limit=max(top_k * 3, 12)):
                channel_scores["vector_sparse"][chunk_id] = max(channel_scores["vector_sparse"].get(chunk_id, 0.0), score)
        channel_scores["graph"] = dict(
            self.graph.search(
                query,
                farm_id=farm_id,
                include_private=include_private,
                limit=max(top_k * 3, 12),
            )
        )

        private_hits: list[RetrievalHit] = []
        if include_private:
            overlay = PrivateOverlayStore(self.settings.private_store_db_path)
            private_hits = overlay.search(query=query, farm_id=farm_id, limit=max(top_k * 3, 12))
            for hit in private_hits:
                chunks[hit.chunk.chunk_id] = hit.chunk
                channel_scores["private_overlay"][hit.chunk.chunk_id] = hit.score
                channel_scores["vector_sparse"][hit.chunk.chunk_id] = max(
                    channel_scores["vector_sparse"].get(hit.chunk.chunk_id, 0.0),
                    hit.score,
                )

        channel_hit_counts = {
            channel: len(scores)
            for channel, scores in channel_scores.items()
            if channel in {"vector_dense_text", "vector_sparse", "vector_image", "graph"}
        }
        weights, dat_diag = resolve_enterprise_weights(
            query,
            profile_path=self.settings.fusion_profile_path,
            profile_meta_path=self.settings.fusion_profile_meta_path,
            mode=self.settings.dat_mode,
            channel_hit_counts=channel_hit_counts,
        )
        ranked, fusion_debug = weighted_rrf(
            weights=weights,
            channel_hits={
                channel: [ChannelHit(chunk_id, score) for chunk_id, score in scores.items()]
                for channel, scores in channel_scores.items()
                if channel != "private_overlay"
            },
            limit=max(top_k * 2, top_k),
            return_debug=True,
        )
        hits: list[RetrievalHit] = []
        for ranked_hit in ranked:
            chunk = chunks.get(ranked_hit.chunk_id)
            if chunk is None:
                continue
            contributions = dict(fusion_debug.get("top_doc_channel_contributions", {}).get(ranked_hit.chunk_id, {}))
            if ranked_hit.chunk_id in channel_scores["private_overlay"]:
                contributions["private_overlay"] = channel_scores["private_overlay"][ranked_hit.chunk_id]
            hits.append(RetrievalHit(chunk=chunk, score=float(ranked_hit.score), channels=contributions))
            if len(hits) >= top_k:
                break
        if include_private and private_hits and not any(hit.chunk.metadata.get("tier") == "private" for hit in hits):
            hits = sorted(hits + private_hits, key=lambda hit: hit.score, reverse=True)[:top_k]
        self.last_diagnostics = {
            "channels": sorted(
                {
                    "qdrant" if channel == "vector_dense_text" else "bm25" if channel == "vector_sparse" else channel
                    for channel, scores in channel_scores.items()
                    if scores
                }
            ),
            "fusion": {
                "channel_weights": weights,
                "channel_enabled": {channel: bool(scores) for channel, scores in channel_scores.items()},
                **fusion_debug,
                **dat_diag,
            },
            "private": {
                "farm_id": farm_id,
                "include_private": bool(include_private),
                "private_overlay_hits": len(private_hits),
                "channels": ["private_overlay"] if private_hits else [],
            },
            "embedding_provider": getattr(vectors.embeddings, "provider_name", "unknown"),
            "embedding_model": self.settings.embed_model,
        }
        return hits


__all__ = ["COLLECTION_NAME", "LocalIndex", "read_chunks", "write_chunks"]
