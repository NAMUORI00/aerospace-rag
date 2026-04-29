from __future__ import annotations

import json
from pathlib import Path

from .text import tokenize


DEFAULT_WEIGHTS = {
    "entity_rich": {"qdrant": 0.40, "bm25": 0.25, "graph": 0.35},
    "keyword_fact": {"qdrant": 0.40, "bm25": 0.45, "graph": 0.15},
    "general": {"qdrant": 0.55, "bm25": 0.35, "graph": 0.10},
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


def _clamp_weights(weights: dict[str, float], *, min_weight: float, max_weight: float) -> dict[str, float]:
    clamped = {k: min(max(float(v), min_weight), max_weight) for k, v in weights.items()}
    total = sum(clamped.values()) or 1.0
    return {k: v / total for k, v in clamped.items()}


def resolve_channel_weights(
    query: str,
    *,
    profile_path: str | Path,
    mode: str = "hybrid",
    min_weight: float = 0.10,
    max_weight: float = 0.80,
) -> tuple[dict[str, float], dict[str, object]]:
    segment = classify_query(query)
    diagnostics: dict[str, object] = {
        "dat_runtime_policy": "static" if mode == "static" else "hybrid",
        "query_segment": segment,
        "fusion_profile_id": None,
        "weights_source": "default_fallback",
        "fusion_fallback_reason": "",
    }
    weights = dict(DEFAULT_WEIGHTS[segment])
    path = Path(profile_path)
    if mode != "static" and path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            profile_weights = ((payload.get("segments") or {}).get(segment) or payload.get("default") or {})
            if all(channel in profile_weights for channel in ("qdrant", "bm25", "graph")):
                weights = {channel: float(profile_weights[channel]) for channel in ("qdrant", "bm25", "graph")}
                diagnostics["weights_source"] = "profile"
                diagnostics["fusion_profile_id"] = payload.get("profile_id")
        except Exception as exc:
            diagnostics["fusion_fallback_reason"] = f"profile_load_failed:{type(exc).__name__}"
    elif mode == "static":
        diagnostics["fusion_fallback_reason"] = "dat_mode_static"
    else:
        diagnostics["fusion_fallback_reason"] = "profile_missing"
    return _clamp_weights(weights, min_weight=min_weight, max_weight=max_weight), diagnostics
