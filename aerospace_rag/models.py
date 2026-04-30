from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    modality: str
    page: int | None = None
    sheet: str | None = None
    row: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["text"] = self.text
        payload["source_type"] = self.metadata.get("source_type", "document")
        payload["source_doc"] = self.source_file
        payload["source_doc_id"] = self.source_file
        payload["doc_id"] = self.source_file
        payload["canonical_doc_id"] = self.source_file
        payload["canonical_chunk_id"] = self.chunk_id
        payload["tier"] = self.metadata.get("tier", "public")
        payload["created_at"] = self.metadata.get("created_at", datetime.now(timezone.utc).isoformat())
        payload["asset_ref"] = self.metadata.get("asset_ref")
        payload["table_html_ref"] = self.metadata.get("table_html_ref")
        payload["image_b64_ref"] = self.metadata.get("image_b64_ref")
        payload["formula_latex_ref"] = self.metadata.get("formula_latex_ref")
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Chunk":
        metadata = dict(payload.get("metadata") or {})
        for key in (
            "tier",
            "source_type",
            "created_at",
            "asset_ref",
            "table_html_ref",
            "image_b64_ref",
            "formula_latex_ref",
            "canonical_doc_id",
            "canonical_chunk_id",
            "doc_id",
        ):
            if key in payload and payload.get(key) is not None:
                metadata[key] = payload.get(key)
        return cls(
            chunk_id=str(payload["chunk_id"]),
            text=str(payload["text"]),
            source_file=str(payload["source_file"]),
            modality=str(payload["modality"]),
            page=payload.get("page"),
            sheet=payload.get("sheet"),
            row=payload.get("row"),
            metadata=metadata,
        )


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    channels: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceRef:
    chunk_id: str
    source_file: str
    modality: str
    score: float
    excerpt: str
    page: int | None = None
    sheet: str | None = None
    row: int | None = None
    channels: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResponse:
    answer: str
    sources: list[SourceRef]
    routing: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildResult:
    data_dir: Path
    index_dir: Path
    file_count: int
    chunk_count: int
    qdrant_collection: str
    graph_index_path: Path
    bm25_path: Path
    chunks_path: Path
