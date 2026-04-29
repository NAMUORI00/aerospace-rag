from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from .bm25 import BM25Index
from .config import Settings
from .graph_store import GraphStore
from .models import Chunk, RetrievalHit
from .runtime_dat import resolve_channel_weights
from .vector_store import COLLECTION_NAME, QdrantVectorStore


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

    def search(self, query: str, *, top_k: int = 8) -> list[RetrievalHit]:
        chunks = {chunk.chunk_id: chunk for chunk in read_chunks(self.chunks_path)}
        channel_scores: dict[str, dict[str, float]] = {
            "qdrant": {},
            "bm25": {},
            "graph": {},
        }
        vectors = QdrantVectorStore(self.index_dir, settings=self.settings)
        try:
            channel_scores["qdrant"] = dict(vectors.search(query, limit=max(top_k * 3, 12)))
        finally:
            vectors.close()
        channel_scores["bm25"] = dict(BM25Index.load(self.bm25_path).search(query, limit=max(top_k * 3, 12)))
        channel_scores["graph"] = dict(self.graph.search(query, limit=max(top_k * 3, 12)))

        normalized: dict[str, dict[str, float]] = {}
        for channel, scores in channel_scores.items():
            max_score = max(scores.values(), default=0.0)
            normalized[channel] = {
                chunk_id: (score / max_score if max_score else 0.0)
                for chunk_id, score in scores.items()
            }

        weights, dat_diag = resolve_channel_weights(
            query,
            profile_path=self.settings.fusion_profile_path,
            mode=self.settings.dat_mode,
            min_weight=self.settings.dat_min_weight_per_channel,
            max_weight=self.settings.dat_max_weight_per_channel,
        )
        fused: defaultdict[str, float] = defaultdict(float)
        per_chunk_channels: defaultdict[str, dict[str, float]] = defaultdict(dict)
        for channel, scores in normalized.items():
            for chunk_id, score in scores.items():
                fused[chunk_id] += weights[channel] * score
                per_chunk_channels[chunk_id][channel] = score

        hits: list[RetrievalHit] = []
        for chunk_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True):
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            hits.append(RetrievalHit(chunk=chunk, score=float(score), channels=dict(per_chunk_channels[chunk_id])))
            if len(hits) >= top_k:
                break
        self.last_diagnostics = {
            "fusion": {
                "channel_weights": weights,
                "channel_enabled": {channel: bool(scores) for channel, scores in channel_scores.items()},
                **dat_diag,
            },
            "embedding_provider": getattr(vectors.embeddings, "provider_name", "unknown"),
            "embedding_model": self.settings.embed_model,
        }
        return hits


__all__ = ["COLLECTION_NAME", "LocalIndex", "read_chunks", "write_chunks"]
