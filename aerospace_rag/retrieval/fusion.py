from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CANONICAL_CHANNELS = ("vector_dense_text", "vector_sparse", "vector_image", "graph")
CHANNEL_ALIASES = {
    "qdrant": "vector_dense_text",
    "dense": "vector_dense_text",
    "dense_text": "vector_dense_text",
    "vector_dense_text": "vector_dense_text",
    "bm25": "vector_sparse",
    "sparse": "vector_sparse",
    "vector_sparse": "vector_sparse",
    "image": "vector_image",
    "dense_image": "vector_image",
    "vector_image": "vector_image",
    "graph": "graph",
}


@dataclass(frozen=True)
class ChannelHit:
    chunk_id: str
    score: float


def normalize_channel_weights(weights: dict[str, float]) -> dict[str, float]:
    core = {
        "vector_dense_text": max(0.0, float(weights.get("vector_dense_text", 0.0))),
        "vector_sparse": max(0.0, float(weights.get("vector_sparse", 0.0))),
        "vector_image": max(0.0, float(weights.get("vector_image", 0.0))),
        "graph": max(0.0, float(weights.get("graph", 0.0))),
    }
    total = core["vector_dense_text"] + core["vector_sparse"] + core["vector_image"] + core["graph"]
    if total <= 0:
        core["vector_dense_text"] = 0.5
        core["vector_sparse"] = 0.5
        core["graph"] = 0.0
        total = 1.0
    core["vector_dense_text"] /= total
    core["vector_sparse"] /= total
    core["vector_image"] /= total
    core["graph"] /= total
    out = dict(core)
    out["qdrant"] = out.get("vector_dense_text", 0.0)
    out["bm25"] = out.get("vector_sparse", 0.0)
    return out


def _rrf(rank: int, *, k: int = 60) -> float:
    return 1.0 / float(k + rank + 1)


def weighted_rrf(
    *,
    weights: dict[str, float],
    channel_hits: dict[str, list[ChannelHit]],
    limit: int,
    rrf_k: int = 60,
    return_debug: bool = False,
) -> tuple[list[ChannelHit], dict[str, Any]] | list[ChannelHit]:
    canonical_weights = normalize_channel_weights(weights)
    scores: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    for raw_channel, hits in channel_hits.items():
        channel = CHANNEL_ALIASES.get(str(raw_channel), str(raw_channel))
        weight = float(canonical_weights.get(channel, 0.0))
        if weight <= 0:
            continue
        for rank, hit in enumerate(sorted(hits, key=lambda item: item.score, reverse=True)):
            inc = weight * _rrf(rank, k=rrf_k)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + inc
            contributions.setdefault(hit.chunk_id, {})
            contributions[hit.chunk_id][channel] = contributions[hit.chunk_id].get(channel, 0.0) + inc
    ranked = [
        ChannelHit(chunk_id=chunk_id, score=float(score))
        for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]
    debug = {
        "channel_weights": canonical_weights,
        "top_doc_channel_contributions": {
            hit.chunk_id: {k: float(v) for k, v in contributions.get(hit.chunk_id, {}).items()}
            for hit in ranked
        },
    }
    if return_debug:
        return ranked, debug
    return ranked
