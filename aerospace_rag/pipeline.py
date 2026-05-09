from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import Settings
from .stores.local_index import COLLECTION_NAME, LocalIndex
from .ingestion import ingest_data
from .models import BuildResult, QueryResponse, SourceRef
from .generation.cross_check import run_gpt_pro_cross_check
from .generation.providers import generate_answer, route_generation_provider
from .retrieval.profile import write_self_calibrated_fusion_profile
from .text import excerpt


DEFAULT_INDEX_DIR = Path("data") / "index"


def build_index(
    *,
    data_dir: str | Path = "data",
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    reset: bool = True,
    strict_expected: bool = False,
    include_extra: bool = False,
    settings: Settings | None = None,
) -> BuildResult:
    resolved_settings = settings or Settings.from_env()
    data_path = Path(data_dir)
    index_path = Path(index_dir)
    chunks = ingest_data(data_path, strict_expected=strict_expected, include_extra=include_extra)
    index = LocalIndex(index_path, settings=resolved_settings)
    index.build(chunks, reset=reset)
    profile_summary = None
    if str(resolved_settings.fusion_mode or "hybrid").strip().lower() not in {"static", "off", "disabled"}:
        profile_summary = write_self_calibrated_fusion_profile(index_dir=index_path, settings=resolved_settings)
    return BuildResult(
        data_dir=data_path,
        index_dir=index_path,
        file_count=len({chunk.source_file for chunk in chunks}),
        chunk_count=len(chunks),
        qdrant_collection=COLLECTION_NAME,
        graph_index_path=index.graph.index_path,
        bm25_path=index.bm25_path,
        chunks_path=index.chunks_path,
        fusion_profile_path=profile_summary["profile_path"] if profile_summary else None,
        fusion_profile_meta_path=profile_summary["profile_meta_path"] if profile_summary else None,
    )


def ask(
    question: str,
    *,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    top_k: int = 8,
    provider: str | None = None,
    debug: bool = False,
    cross_check: bool | None = None,
    settings: Settings | None = None,
) -> QueryResponse:
    resolved_settings = settings or Settings.from_env()
    if cross_check is not None:
        resolved_settings = replace(resolved_settings, gpt_pro_cross_check_enabled=cross_check)
    if provider is None:
        provider = "ollama"
    index = LocalIndex(index_dir, settings=resolved_settings)
    hits = index.search(question, top_k=top_k)
    provider = route_generation_provider(provider, settings=resolved_settings)
    answer = generate_answer(question, hits, provider=provider, settings=resolved_settings)
    cross_check_result = run_gpt_pro_cross_check(
        question,
        answer,
        hits,
        settings=resolved_settings,
    )
    sources = [
        SourceRef(
            chunk_id=hit.chunk.chunk_id,
            source_file=hit.chunk.source_file,
            modality=hit.chunk.modality,
            score=hit.score,
            excerpt=excerpt(hit.chunk.text),
            page=hit.chunk.page,
            sheet=hit.chunk.sheet,
            row=hit.chunk.row,
            channels=hit.channels,
        )
        for hit in hits
    ]
    diagnostics = {}
    if debug:
        diagnostics = {
            "channels": sorted({channel for hit in hits for channel in hit.channels}),
            "source_count": len(sources),
            "provider": provider,
            **index.last_diagnostics,
        }
    if cross_check_result["status"] != "disabled":
        diagnostics["cross_check"] = cross_check_result
    return QueryResponse(
        answer=answer,
        sources=sources,
        routing={
            "provider": provider,
            "retrieval": "qdrant+bm25+graph-lite",
            "cross_check": {
                "enabled": cross_check_result["status"] != "disabled",
                "status": cross_check_result["status"],
                "provider": cross_check_result["provider"],
                "model": cross_check_result["model"],
            },
        },
        diagnostics=diagnostics,
    )
