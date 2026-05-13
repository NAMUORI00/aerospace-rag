from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import Settings
from ..models import Chunk, RetrievalHit
from ..retrieval.bm25 import BM25Index
from ..retrieval.fusion import ChannelHit, weighted_rrf
from ..retrieval.weights import FUSION_PROFILE_META_FILENAME, FUSION_WEIGHTS_FILENAME, resolve_channel_weights
from ..text import tokenize
from .graph import GraphStore
from .vector import COLLECTION_NAME, QdrantVectorStore


def _chunk_search_text(chunk: Chunk) -> str:
    metadata = chunk.metadata or {}
    return "\n".join(
        part
        for part in [
            chunk.text,
            chunk.source_file,
            str(metadata.get("title") or ""),
            str(metadata.get("category") or ""),
            str(metadata.get("keywords") or ""),
        ]
        if part
    )


def _chunk_relevance_text(chunk: Chunk) -> str:
    metadata = chunk.metadata or {}
    return "\n".join(
        part
        for part in [
            chunk.text,
            str(metadata.get("title") or ""),
            str(metadata.get("category") or ""),
            str(metadata.get("keywords") or ""),
        ]
        if part
    )


def _lexical_rerank_bonus(query: str, chunk: Chunk | None) -> float:
    if chunk is None:
        return 0.0
    query_tokens = set(tokenize(query))
    chunk_tokens = set(tokenize(_chunk_search_text(chunk)))
    if not query_tokens or not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    if not overlap:
        return 0.0
    coverage = len(overlap) / max(1, len(query_tokens))
    return min(0.008, 0.012 * coverage)


def _relevance_gate(query: str, chunk: Chunk | None) -> tuple[bool, dict[str, object]]:
    if chunk is None:
        return False, {"overlap_count": 0, "coverage": 0.0}
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return True, {"overlap_count": 0, "coverage": 1.0}
    chunk_tokens = set(tokenize(_chunk_relevance_text(chunk)))
    overlap = query_tokens & chunk_tokens
    coverage = len(overlap) / max(1, len(query_tokens))
    if len(query_tokens) <= 2:
        keep = bool(overlap)
    else:
        keep = len(overlap) >= 2 and (coverage >= 0.12 or len(overlap) >= 4)
    return keep, {
        "overlap_count": len(overlap),
        "coverage": round(coverage, 4),
        "overlap_sample": sorted(overlap)[:12],
    }


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
        self.graph = GraphStore(self.index_dir, settings=self.settings)
        self.last_diagnostics: dict[str, object] = {}
        self._last_embedding_provider = "unknown"

    def _fusion_profile_path(self) -> Path:
        if str(self.settings.fusion_profile_path).strip():
            return Path(self.settings.fusion_profile_path)
        return self.index_dir / FUSION_WEIGHTS_FILENAME

    def _fusion_profile_meta_path(self) -> Path:
        if str(self.settings.fusion_profile_meta_path).strip():
            return Path(self.settings.fusion_profile_meta_path)
        return self.index_dir / FUSION_PROFILE_META_FILENAME

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

    def collect_channel_scores(self, query: str, *, limit: int) -> dict[str, dict[str, float]]:
        channel_scores: dict[str, dict[str, float]] = {
            "vector_dense_text": {},
            "vector_sparse": {},
            "vector_image": {},
            "graph": {},
        }
        vectors = QdrantVectorStore(self.index_dir, settings=self.settings)
        try:
            self._last_embedding_provider = getattr(vectors.embeddings, "provider_name", "unknown")
            vector_channels = vectors.search_channels(
                query,
                limit=limit,
            )
        finally:
            vectors.close()
        for channel in ("vector_dense_text", "vector_sparse", "vector_image"):
            channel_scores[channel].update(dict(vector_channels.get(channel) or []))
        if self.bm25_path.exists():
            for chunk_id, score in BM25Index.load(self.bm25_path).search(query, limit=limit):
                channel_scores["vector_sparse"][chunk_id] = max(channel_scores["vector_sparse"].get(chunk_id, 0.0), score)
        channel_scores["graph"] = dict(
            self.graph.search(
                query,
                limit=limit,
            )
        )
        return channel_scores

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
    ) -> list[RetrievalHit]:
        chunks = {chunk.chunk_id: chunk for chunk in read_chunks(self.chunks_path)}
        channel_scores = self.collect_channel_scores(query, limit=max(top_k * 3, 12))

        channel_hit_counts = {
            channel: len(scores)
            for channel, scores in channel_scores.items()
            if channel in {"vector_dense_text", "vector_sparse", "vector_image", "graph"}
        }
        weights, weight_diag = resolve_channel_weights(
            query,
            profile_path=self._fusion_profile_path(),
            profile_meta_path=self._fusion_profile_meta_path(),
            mode=self.settings.fusion_mode,
            min_weight=self.settings.fusion_min_weight,
            max_weight=self.settings.fusion_max_weight,
            channel_hit_counts=channel_hit_counts,
        )
        ranked, fusion_debug = weighted_rrf(
            weights=weights,
            channel_hits={
                channel: [ChannelHit(chunk_id, score) for chunk_id, score in scores.items()]
                for channel, scores in channel_scores.items()
            },
            limit=max(top_k * 2, top_k),
            return_debug=True,
        )
        if weight_diag.get("weights_source") != "runtime_profile":
            rerank_adjustments: dict[str, float] = {}
            adjusted_ranked: list[ChannelHit] = []
            for ranked_hit in ranked:
                bonus = _lexical_rerank_bonus(query, chunks.get(ranked_hit.chunk_id))
                if bonus:
                    rerank_adjustments[ranked_hit.chunk_id] = bonus
                adjusted_ranked.append(ChannelHit(ranked_hit.chunk_id, float(ranked_hit.score) + bonus))
            if rerank_adjustments:
                ranked = sorted(adjusted_ranked, key=lambda hit: hit.score, reverse=True)
                fusion_debug["rerank_adjustments"] = {"lexical_coverage": rerank_adjustments}
        filtered_ranked: list[ChannelHit] = []
        filtered_out: dict[str, dict[str, object]] = {}
        relevance_debug: dict[str, dict[str, object]] = {}
        for ranked_hit in ranked:
            keep, details = _relevance_gate(query, chunks.get(ranked_hit.chunk_id))
            relevance_debug[ranked_hit.chunk_id] = details
            if keep:
                filtered_ranked.append(ranked_hit)
            else:
                filtered_out[ranked_hit.chunk_id] = details
        ranked = filtered_ranked
        hits: list[RetrievalHit] = []
        for ranked_hit in ranked:
            chunk = chunks.get(ranked_hit.chunk_id)
            if chunk is None:
                continue
            contributions = dict(fusion_debug.get("top_doc_channel_contributions", {}).get(ranked_hit.chunk_id, {}))
            hits.append(RetrievalHit(chunk=chunk, score=float(ranked_hit.score), channels=contributions))
            if len(hits) >= top_k:
                break
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
                **weight_diag,
            },
            "relevance_filter": {
                "kept": len(ranked),
                "removed": len(filtered_out),
                "removed_chunks": filtered_out,
                "candidates": relevance_debug,
            },
            "embedding_provider": self._last_embedding_provider,
            "embedding_model": self.settings.embed_model,
        }
        return hits


__all__ = ["COLLECTION_NAME", "LocalIndex", "read_chunks", "write_chunks"]
