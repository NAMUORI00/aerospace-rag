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


def _coerce_runtime_weights(raw: dict[str, object]) -> dict[str, float] | None:
    aliases = {
        "qdrant": ("qdrant", "dense", "vector_dense_text", "dense_text"),
        "bm25": ("bm25", "sparse", "vector_sparse"),
        "graph": ("graph", "falkordb"),
    }
    out: dict[str, float] = {}
    for channel, keys in aliases.items():
        found = False
        for key in keys:
            if key in raw:
                out[channel] = float(raw[key])
                found = True
                break
        if not found:
            return None
    return out


def _profile_weights(payload: dict[str, object], segment: str) -> dict[str, float] | None:
    segments = payload.get("segments")
    if isinstance(segments, dict):
        segment_block = segments.get(segment)
        if isinstance(segment_block, dict):
            weights = _coerce_runtime_weights(segment_block)
            if weights is not None:
                return weights

    default_block = payload.get("default")
    if isinstance(default_block, dict):
        weights = _coerce_runtime_weights(default_block)
        if weights is not None:
            return weights

    datasets = payload.get("datasets")
    if isinstance(datasets, dict) and datasets:
        rows = []
        for value in datasets.values():
            if isinstance(value, dict):
                weights = _coerce_runtime_weights(value)
                if weights is not None:
                    rows.append(weights)
        if rows:
            return {
                channel: sum(row[channel] for row in rows) / float(len(rows))
                for channel in ("qdrant", "bm25", "graph")
            }
    return None


def _meta_allows_profile(path: Path | None) -> tuple[bool, str]:
    if path is None or not path.exists():
        return True, ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False, "profile_meta_load_failed"
    selection_run_type = str((payload or {}).get("selection_run_type") or "main").strip().lower()
    if selection_run_type and selection_run_type != "main":
        return False, f"non_main_profile:{selection_run_type}"
    return True, ""


def resolve_channel_weights(
    query: str,
    *,
    profile_path: str | Path,
    profile_meta_path: str | Path | None = None,
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
            meta_allowed, meta_reason = _meta_allows_profile(Path(profile_meta_path) if profile_meta_path else None)
            if not meta_allowed:
                diagnostics["fusion_fallback_reason"] = meta_reason
                return _clamp_weights(weights, min_weight=min_weight, max_weight=max_weight), diagnostics
            payload = json.loads(path.read_text(encoding="utf-8"))
            profile_weights = _profile_weights(payload, segment)
            if profile_weights is not None:
                weights = profile_weights
                diagnostics["weights_source"] = "profile"
                diagnostics["fusion_profile_id"] = payload.get("profile_id")
        except Exception as exc:
            diagnostics["fusion_fallback_reason"] = f"profile_load_failed:{type(exc).__name__}"
    elif mode == "static":
        diagnostics["fusion_fallback_reason"] = "dat_mode_static"
    else:
        diagnostics["fusion_fallback_reason"] = "profile_missing"
    return _clamp_weights(weights, min_weight=min_weight, max_weight=max_weight), diagnostics
