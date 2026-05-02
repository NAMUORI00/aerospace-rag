from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings
from ..models import Chunk
from ..stores.local_index import LocalIndex, read_chunks
from ..text import tokenize
from .fusion import ChannelHit, normalize_channel_weights, weighted_rrf
from .weights import FUSION_PROFILE_META_FILENAME, FUSION_WEIGHTS_FILENAME


DEFAULT_WEIGHT_GRID = [
    {"vector_dense_text": 0.55, "vector_sparse": 0.45, "vector_image": 0.0, "graph": 0.0},
    {"vector_dense_text": 0.45, "vector_sparse": 0.55, "vector_image": 0.0, "graph": 0.0},
    {"vector_dense_text": 0.55, "vector_sparse": 0.35, "vector_image": 0.0, "graph": 0.10},
    {"vector_dense_text": 0.45, "vector_sparse": 0.45, "vector_image": 0.0, "graph": 0.10},
    {"vector_dense_text": 0.35, "vector_sparse": 0.55, "vector_image": 0.0, "graph": 0.10},
    {"vector_dense_text": 0.25, "vector_sparse": 0.65, "vector_image": 0.0, "graph": 0.10},
    {"vector_dense_text": 0.50, "vector_sparse": 0.25, "vector_image": 0.0, "graph": 0.25},
    {"vector_dense_text": 0.35, "vector_sparse": 0.30, "vector_image": 0.0, "graph": 0.35},
]


def _profile_id(index_dir: Path, case_count: int, weights: dict[str, float]) -> str:
    payload = json.dumps(
        {
            "index_dir": str(index_dir),
            "case_count": case_count,
            "weights": normalize_channel_weights(weights),
        },
        sort_keys=True,
    )
    return "self-calibrated-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _pseudo_query(chunk: Chunk, *, max_tokens: int = 24) -> str:
    metadata = chunk.metadata or {}
    parts = [
        str(metadata.get("title") or ""),
        Path(chunk.source_file).stem,
        " ".join(tokenize(chunk.text)[:max_tokens]),
    ]
    return " ".join(part for part in parts if part).strip() or chunk.source_file


def _calibration_cases(chunks: list[Chunk], *, max_cases: int) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        query = _pseudo_query(chunk)
        key = (chunk.source_file, query)
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            {
                "question": query,
                "expected_chunk_id": chunk.chunk_id,
                "expected_source_file": chunk.source_file,
            }
        )
        if len(cases) >= max_cases:
            break
    return cases


def _to_channel_hits(channel_scores: dict[str, dict[str, float]]) -> dict[str, list[ChannelHit]]:
    return {
        channel: [ChannelHit(chunk_id, score) for chunk_id, score in scores.items()]
        for channel, scores in channel_scores.items()
    }


def _rank_score(ranked: list[ChannelHit], chunks: dict[str, Chunk], case: dict[str, str], *, top_k: int) -> float:
    expected_chunk_id = str(case.get("expected_chunk_id") or "")
    expected_source_file = str(case.get("expected_source_file") or "")
    for idx, hit in enumerate(ranked[:top_k], start=1):
        chunk = chunks.get(hit.chunk_id)
        if hit.chunk_id == expected_chunk_id or (chunk is not None and chunk.source_file == expected_source_file):
            return 1.0 / float(idx)
    return 0.0


def write_self_calibrated_fusion_profile(
    *,
    index_dir: str | Path,
    settings: Settings | None = None,
    candidate_depth: int = 12,
    top_k: int = 5,
    max_cases: int = 32,
    weight_grid: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Write a fixed runtime fusion profile selected by offline self-calibration.

    The calibration queries are derived from indexed chunk content and evaluated
    before runtime answering. Runtime retrieval still consumes a fixed profile.
    """

    resolved_settings = settings or Settings.from_env()
    root = Path(index_dir)
    chunks_list = read_chunks(root / "chunks.jsonl")
    chunks = {chunk.chunk_id: chunk for chunk in chunks_list}
    cases = _calibration_cases(chunks_list, max_cases=max_cases)
    candidates = [normalize_channel_weights(candidate) for candidate in (weight_grid or DEFAULT_WEIGHT_GRID)]
    if not candidates:
        candidates = [normalize_channel_weights(DEFAULT_WEIGHT_GRID[0])]

    index = LocalIndex(root, settings=resolved_settings)
    channel_hits_by_case: list[tuple[dict[str, str], dict[str, list[ChannelHit]]]] = []
    for case in cases:
        channel_scores = index.collect_channel_scores(case["question"], limit=candidate_depth)
        channel_hits_by_case.append((case, _to_channel_hits(channel_scores)))

    best_weights = candidates[0]
    best_score = -1.0
    best_hit_count = -1
    candidate_summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        scores: list[float] = []
        for case, channel_hits in channel_hits_by_case:
            ranked = weighted_rrf(weights=candidate, channel_hits=channel_hits, limit=top_k)
            scores.append(_rank_score(ranked, chunks, case, top_k=top_k))
        mean_mrr = sum(scores) / len(scores) if scores else 0.0
        summary = {
            "weights": candidate,
            "mrr_at_k": mean_mrr,
            "hit_count": sum(1 for score in scores if score > 0),
        }
        candidate_summaries.append(summary)
        hit_count = int(summary["hit_count"])
        if (mean_mrr, hit_count) > (best_score, best_hit_count):
            best_score = mean_mrr
            best_hit_count = hit_count
            best_weights = candidate

    profile_id = _profile_id(root, len(cases), best_weights)
    created_at = datetime.now(timezone.utc).isoformat()
    profile_path = root / FUSION_WEIGHTS_FILENAME
    meta_path = root / FUSION_PROFILE_META_FILENAME
    profile = {
        "schema_version": "aerospace_fusion_profile_v1",
        "profile_id": profile_id,
        "default_candidate_depth_selected": candidate_depth,
        "default": best_weights,
        "evaluation": {
            "method": "self_calibrated_grid_search",
            "metric": f"mrr@{top_k}",
            "case_count": len(cases),
            "best_score": best_score if best_score >= 0 else 0.0,
            "candidates_evaluated": len(candidates),
        },
    }
    meta = {
        "schema_version": "aerospace_fusion_profile_meta_v1",
        "fusion_profile_id": profile_id,
        "selection_run_type": "main",
        "created_at": created_at,
        "profile_method": "self_calibrated_grid_search",
        "eval_case_count": len(cases),
        "candidate_depth": candidate_depth,
    }
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "profile_path": profile_path,
        "profile_meta_path": meta_path,
        "profile_id": profile_id,
        "weights": best_weights,
        "score": best_score if best_score >= 0 else 0.0,
        "case_count": len(cases),
        "candidate_summaries": candidate_summaries,
    }


__all__ = ["DEFAULT_WEIGHT_GRID", "write_self_calibrated_fusion_profile"]
