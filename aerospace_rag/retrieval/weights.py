from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..text import tokenize
from .fusion import normalize_channel_weights


FUSION_WEIGHTS_FILENAME = "fusion_weights.runtime.json"
FUSION_PROFILE_META_FILENAME = "fusion_profile_meta.runtime.json"
PROFILE_CHANNELS = ("vector_dense_text", "vector_sparse", "vector_image", "graph")

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


def _as_existing_file(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"profile JSON must be an object: {path}")
    return value


def _canonical_weight_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("profile weight block must be an object")
    aliases = {
        "qdrant": "vector_dense_text",
        "dense": "vector_dense_text",
        "dense_text": "vector_dense_text",
        "bm25": "vector_sparse",
        "sparse": "vector_sparse",
        "image": "vector_image",
        "dense_image": "vector_image",
    }
    weights: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = aliases.get(str(raw_key), str(raw_key))
        if key not in PROFILE_CHANNELS:
            continue
        weights[key] = float(raw_value)
    if not weights:
        raise ValueError("profile does not contain channel weights")
    return weights


def _select_weight_block(profile: dict[str, Any]) -> tuple[dict[str, float], str]:
    if isinstance(profile.get("default"), dict):
        return _canonical_weight_dict(profile["default"]), "default"
    if isinstance(profile.get("weights"), dict):
        return _canonical_weight_dict(profile["weights"]), "weights"
    datasets = profile.get("datasets")
    if isinstance(datasets, dict):
        for dataset_id in ("default", "global", ""):
            block = datasets.get(dataset_id)
            if isinstance(block, dict):
                if isinstance(block.get("weights"), dict):
                    return _canonical_weight_dict(block["weights"]), f"datasets.{dataset_id or '<empty>'}.weights"
                if any(key in block for key in PROFILE_CHANNELS):
                    return _canonical_weight_dict(block), f"datasets.{dataset_id or '<empty>'}"
    if any(key in profile for key in PROFILE_CHANNELS):
        return _canonical_weight_dict(profile), "root"
    raise ValueError("profile does not contain a supported weight block")


def _profile_id(profile: dict[str, Any], meta: dict[str, Any] | None = None) -> str:
    for source in (profile, meta or {}):
        for key in ("fusion_profile_id", "profile_id"):
            value = source.get(key)
            if value:
                return str(value)
    return ""


def _validate_profile_meta(profile: dict[str, Any], meta: dict[str, Any] | None) -> None:
    if not meta:
        return
    run_type = str(meta.get("selection_run_type") or "").strip().lower()
    if run_type and run_type != "main":
        raise ValueError(f"profile meta selection_run_type must be main, got {run_type!r}")
    profile_id = _profile_id(profile)
    meta_profile_id = _profile_id({}, meta)
    if profile_id and meta_profile_id and profile_id != meta_profile_id:
        raise ValueError(f"profile id mismatch: profile={profile_id!r} meta={meta_profile_id!r}")


def _candidate_depth(profile: dict[str, Any], meta: dict[str, Any] | None) -> int:
    for source in (profile, meta or {}):
        for key in ("default_candidate_depth_selected", "candidate_depth_selected", "candidate_depth"):
            value = source.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _clamp_positive_weights(weights: dict[str, float], *, min_weight: float, max_weight: float) -> dict[str, float]:
    lower = max(0.0, float(min_weight))
    upper = max(lower, float(max_weight))
    normalized = normalize_channel_weights(dict(weights))
    clamped = dict(normalized)
    for channel in PROFILE_CHANNELS:
        value = float(clamped.get(channel, 0.0))
        if value > 0.0:
            clamped[channel] = min(upper, max(lower, value))
    return normalize_channel_weights(clamped)


def _static_weights(
    segment: str,
    *,
    channel_hit_counts: dict[str, int] | None,
    fallback_reasons: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    weights = normalize_channel_weights(dict(DEFAULT_CHANNEL_WEIGHTS[segment]))
    weights, adjustments = _apply_evidence_adjustment(weights, channel_hit_counts)
    diagnostics: dict[str, Any] = {
        "fusion_policy": "core_static",
        "query_segment": segment,
        "weights_source": "static",
        "candidate_depth": 0,
        "evidence_adjustments": adjustments,
    }
    if fallback_reasons:
        diagnostics["profile_fallback_reasons"] = fallback_reasons
    return weights, diagnostics


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
    segment = classify_query(query)
    mode_name = str(mode or "hybrid").strip().lower()
    if mode_name in {"static", "off", "disabled"}:
        return _static_weights(segment, channel_hit_counts=channel_hit_counts)

    resolved_profile_path = _as_existing_file(profile_path)
    if resolved_profile_path is None:
        return _static_weights(
            segment,
            channel_hit_counts=channel_hit_counts,
            fallback_reasons=["profile_missing"],
        )

    try:
        profile = _load_json(resolved_profile_path)
        resolved_meta_path = _as_existing_file(profile_meta_path)
        meta = _load_json(resolved_meta_path) if resolved_meta_path else None
        _validate_profile_meta(profile, meta)
        selected_weights, scope = _select_weight_block(profile)
        weights = _clamp_positive_weights(selected_weights, min_weight=min_weight, max_weight=max_weight)
        weights, adjustments = _apply_evidence_adjustment(weights, channel_hit_counts)
    except Exception as exc:
        return _static_weights(
            segment,
            channel_hit_counts=channel_hit_counts,
            fallback_reasons=[f"profile_invalid:{type(exc).__name__}:{exc}"],
        )

    diagnostics: dict[str, Any] = {
        "fusion_policy": "runtime_profile_weighted_rrf",
        "query_segment": segment,
        "weights_source": "runtime_profile",
        "weights_path": str(resolved_profile_path),
        "fusion_profile_scope": scope,
        "fusion_profile_id": _profile_id(profile, meta),
        "candidate_depth": _candidate_depth(profile, meta),
        "evidence_adjustments": adjustments,
    }
    if meta:
        diagnostics["selection_run_type"] = str(meta.get("selection_run_type") or "")
    return weights, diagnostics
