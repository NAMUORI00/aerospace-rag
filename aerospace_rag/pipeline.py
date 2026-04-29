from __future__ import annotations

from pathlib import Path

from .config import Settings
from .stores.local_index import COLLECTION_NAME, LocalIndex
from .ingestion import ingest_data
from .models import BuildResult, QueryResponse, SourceRef
from .generation.providers import generate_answer, route_generation_provider
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
    return BuildResult(
        data_dir=data_path,
        index_dir=index_path,
        file_count=len({chunk.source_file for chunk in chunks}),
        chunk_count=len(chunks),
        qdrant_collection=COLLECTION_NAME,
        falkordb_path=index.graph.db_path,
        bm25_path=index.bm25_path,
        chunks_path=index.chunks_path,
    )


def ask(
    question: str,
    *,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    top_k: int = 8,
    provider: str | None = None,
    debug: bool = False,
    farm_id: str = "default",
    include_private: bool = False,
    settings: Settings | None = None,
) -> QueryResponse:
    resolved_settings = settings or Settings.from_env()
    if provider is None:
        provider = "ollama"
    index = LocalIndex(index_dir, settings=resolved_settings)
    hits = index.search(question, top_k=top_k, farm_id=farm_id, include_private=include_private)
    private_present = any(str(hit.chunk.metadata.get("tier") or "").lower() == "private" for hit in hits)
    provider = route_generation_provider(provider, private_present=private_present, settings=resolved_settings)
    answer = generate_answer(question, hits, provider=provider, settings=resolved_settings)
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
        diagnostics["private_present"] = private_present
    return QueryResponse(
        answer=answer,
        sources=sources,
        routing={
            "provider": provider,
            "retrieval": "qdrant+bm25+falkordb",
            "farm_id": farm_id,
            "include_private": bool(include_private),
            "private_present": private_present,
        },
        diagnostics=diagnostics,
    )
