from __future__ import annotations

from pathlib import Path
from typing import Any

from ..text import tokenize
from .fusion import normalize_channel_weights


DEFAULT_CHANNEL_WEIGHTS = {
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


def classify_query(query: str) -> str:
    tokens = set(tokenize(query))
    if tokens & ENTITY_HINTS:
        return "entity_rich"
    if len(tokens) <= 6:
        return "keyword_fact"
    return "general"


def _evidence_reason(channel: str, suffix: str) -> str:
    return f"{channel.replace('vector_', '').replace('_text', '')}_{suffix}"


def _hit_count(channel_hit_counts: dict[str, int], channel: str) -> int:
    aliases = {"vector_dense_text": "qdrant", "vector_sparse": "bm25"}
    return int(channel_hit_counts.get(channel, channel_hit_counts.get(aliases.get(channel, channel), 0)) or 0)


def _apply_evidence_adjustment(
    weights: dict[str, float],
    channel_hit_counts: dict[str, int] | None,
) -> tuple[dict[str, float], list[str]]:
    if not channel_hit_counts:
        return weights, []
    out = dict(weights)
    reasons: list[str] = []
    for channel in ("vector_dense_text", "vector_sparse", "graph"):
        count = _hit_count(channel_hit_counts, channel)
        if count <= 0:
            out[channel] = 0.0
            reasons.append(_evidence_reason(channel, "no_evidence"))
        elif count < 2:
            out[channel] *= 0.5
            reasons.append(_evidence_reason(channel, "low_evidence"))
    return normalize_channel_weights(out), reasons


def resolve_channel_weights(
    query: str,
    *,
    profile_path: str | Path,
    profile_meta_path: str | Path | None = None,
    mode: str = "hybrid",
    min_weight: float = 0.10,
    max_weight: float = 0.80,
    channel_hit_counts: dict[str, int] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    _ = (Path(profile_path), Path(profile_meta_path) if profile_meta_path else None, mode, min_weight, max_weight)
    segment = classify_query(query)
    weights = normalize_channel_weights(dict(DEFAULT_CHANNEL_WEIGHTS[segment]))
    weights, adjustments = _apply_evidence_adjustment(weights, channel_hit_counts)
    return weights, {
        "fusion_policy": "core_static",
        "query_segment": segment,
        "weights_source": "static",
        "candidate_depth": 0,
        "evidence_adjustments": adjustments,
    }
