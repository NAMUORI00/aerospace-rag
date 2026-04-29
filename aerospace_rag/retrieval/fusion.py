from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..text import tokenize


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

DEFAULT_ENTERPRISE_WEIGHTS = {
    "entity_rich": {"vector_dense_text": 0.40, "vector_sparse": 0.25, "vector_image": 0.0, "graph": 0.35},
    "keyword_fact": {"vector_dense_text": 0.40, "vector_sparse": 0.45, "vector_image": 0.0, "graph": 0.15},
    "general": {"vector_dense_text": 0.55, "vector_sparse": 0.35, "vector_image": 0.0, "graph": 0.10},
}

ENTITY_HINTS = {
    "h3",
    "nasa",
    "momentus",
    "kari",
    "jaxa",
    "isro",
    "esa",
    "cnsa",
    "qzs",
    "k2",
    "k3",
    "k3a",
    "sar",
    "eo",
    "위성",
    "위성영상",
    "발사",
}


@dataclass(frozen=True)
class ChannelHit:
    chunk_id: str
    score: float


def classify_query(query: str) -> str:
    tokens = set(tokenize(query))
    if tokens & ENTITY_HINTS:
        return "entity_rich"
    if len(tokens) <= 6:
        return "keyword_fact"
    return "general"


def _with_compat_aliases(weights: dict[str, float]) -> dict[str, float]:
    out = dict(weights)
    out["qdrant"] = out.get("vector_dense_text", 0.0)
    out["bm25"] = out.get("vector_sparse", 0.0)
    return out


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    core = {
        "vector_dense_text": max(0.0, float(weights.get("vector_dense_text", 0.0))),
        "vector_sparse": max(0.0, float(weights.get("vector_sparse", 0.0))),
        "vector_image": max(0.0, float(weights.get("vector_image", 0.0))),
        "graph": max(0.0, float(weights.get("graph", 0.0))),
    }
    total = core["vector_dense_text"] + core["vector_sparse"] + core["graph"]
    if total <= 0:
        core["vector_dense_text"] = 0.5
        core["vector_sparse"] = 0.5
        core["graph"] = 0.0
        total = 1.0
    core["vector_dense_text"] /= total
    core["vector_sparse"] /= total
    core["graph"] /= total
    return _with_compat_aliases(core)


def _apply_evidence_adjustment(
    weights: dict[str, float],
    channel_hit_counts: dict[str, int] | None,
) -> tuple[dict[str, float], list[str]]:
    if not channel_hit_counts:
        return weights, []
    out = dict(weights)
    reasons: list[str] = []
    for channel in ("vector_dense_text", "vector_sparse", "graph"):
        count = int(channel_hit_counts.get(channel, channel_hit_counts.get({"vector_dense_text": "qdrant", "vector_sparse": "bm25"}.get(channel, channel), 0)) or 0)
        if count <= 0:
            out[channel] = 0.0
            reasons.append(f"{channel.replace('vector_', '').replace('_text', '')}_no_evidence")
        elif count < 2:
            out[channel] *= 0.5
            reasons.append(f"{channel.replace('vector_', '').replace('_text', '')}_low_evidence")
    return _normalize(out), reasons


def resolve_enterprise_weights(
    query: str,
    *,
    profile_path: str | Path,
    profile_meta_path: str | Path | None = None,
    mode: str = "hybrid",
    channel_hit_counts: dict[str, int] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    _ = (profile_path, profile_meta_path, mode)
    segment = classify_query(query)
    weights = dict(DEFAULT_ENTERPRISE_WEIGHTS[segment])
    diagnostics: dict[str, Any] = {
        "fusion_policy": "core_static",
        "query_segment": segment,
        "weights_source": "static",
        "candidate_depth": 0,
        "evidence_adjustments": [],
    }

    weights = _normalize(weights)
    weights, adjustments = _apply_evidence_adjustment(weights, channel_hit_counts)
    diagnostics["evidence_adjustments"] = adjustments
    return weights, diagnostics


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
    canonical_weights = _normalize(weights)
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
