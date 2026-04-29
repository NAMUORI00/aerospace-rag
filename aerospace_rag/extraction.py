from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

from .models import Chunk
from .text import tokenize, unique_ordered


KNOWN_ENTITIES = [
    "H3",
    "H3 8호기",
    "QZS-5",
    "미치비키",
    "NASA",
    "NOAA",
    "Momentus",
    "solar sail",
    "위성영상",
    "저장영상",
    "신규촬영",
    "K2",
    "K3",
    "K3A",
    "SAR",
    "EO",
    "JAXA",
    "ISRO",
    "KARI",
    "CNSA",
    "ESA",
    "판매대행사",
    "나라장터",
]


_ENTITY_TYPE_HINTS = {
    "NASA": "Agency",
    "NOAA": "Agency",
    "JAXA": "Agency",
    "ISRO": "Agency",
    "KARI": "Agency",
    "CNSA": "Agency",
    "ESA": "Agency",
    "Momentus": "Company",
    "H3": "LaunchVehicle",
    "H3 8호기": "LaunchVehicle",
    "QZS-5": "Satellite",
    "미치비키": "Satellite",
    "K2": "SatelliteMode",
    "K3": "SatelliteMode",
    "K3A": "SatelliteMode",
    "SAR": "SensorMode",
    "EO": "SensorMode",
    "위성영상": "Product",
    "저장영상": "ProductOption",
    "신규촬영": "ProductOption",
}


@dataclass(frozen=True)
class ExtractedEntity:
    canonical_id: str
    text: str
    type: str = "Concept"
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedRelation:
    source: str
    target: str
    type: str = "RELATED_TO"
    confidence: float = 0.35
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionResult:
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "relations": [relation.to_dict() for relation in self.relations],
        }


def canonical_id(value: str) -> str:
    raw = str(value or "").strip().lower()
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^a-z0-9가-힣_\-:.]+", "", raw)
    if raw:
        return raw[:120]
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]


def _entity_type(text: str) -> str:
    for key, value in _ENTITY_TYPE_HINTS.items():
        if key.lower() == str(text or "").lower():
            return value
    if re.fullmatch(r"[A-Z][A-Z0-9\-]{1,}", str(text or "")):
        return "Acronym"
    return "Concept"


def _metadata_keywords(chunk: Chunk) -> list[str]:
    keywords = str(chunk.metadata.get("keywords") or "")
    if not keywords:
        return []
    return [part.strip() for part in re.split(r"[,;/]", keywords) if part.strip()]


def extract_entity_texts(chunk: Chunk) -> list[str]:
    text = chunk.text
    lower = text.lower()
    candidates: list[str] = []
    for entity in KNOWN_ENTITIES:
        if entity.lower() in lower:
            candidates.append(entity)
    candidates.extend(_metadata_keywords(chunk))

    token_counts: dict[str, int] = {}
    for token in tokenize(text):
        if len(token) < 2:
            continue
        token_counts[token] = token_counts.get(token, 0) + 1
    repeated = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
    candidates.extend(token for token, count in repeated if count >= 2)
    return unique_ordered(candidates)[:32]


class KnowledgeExtractor:
    """Small, deterministic substitute for SmartFarm's LLM extraction stage.

    The SmartFarm workspace can call an OpenAI-compatible extractor during
    ingest.  This package keeps the deployment core self-contained, so it uses
    domain hints and co-occurrence rules while preserving the same
    entity/relation contract shape.
    """

    def extract(self, chunk: Chunk) -> ExtractionResult:
        entity_texts = extract_entity_texts(chunk)
        entities = [
            ExtractedEntity(
                canonical_id=canonical_id(text),
                text=text,
                type=_entity_type(text),
                confidence=0.75 if text in KNOWN_ENTITIES else 0.45,
            )
            for text in entity_texts
        ]
        relations = self._relations_for(chunk=chunk, entities=entities)
        return ExtractionResult(entities=entities, relations=relations)

    def _relations_for(self, *, chunk: Chunk, entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
        by_text = {entity.text.lower(): entity for entity in entities}
        relations: list[ExtractedRelation] = []

        def add(source_text: str, target_text: str, rel_type: str, confidence: float) -> None:
            source = by_text.get(source_text.lower())
            target = by_text.get(target_text.lower())
            if source is None or target is None or source.canonical_id == target.canonical_id:
                return
            relations.append(
                ExtractedRelation(
                    source=source.canonical_id,
                    target=target.canonical_id,
                    type=rel_type,
                    confidence=confidence,
                    evidence=chunk.chunk_id,
                )
            )

        text_lower = chunk.text.lower()
        if "nasa" in text_lower and "momentus" in text_lower:
            add("NASA", "Momentus", "AWARDED_CONTRACT_TO", 0.8)
        if "solar sail" in text_lower and "momentus" in text_lower:
            add("Momentus", "solar sail", "STUDIES", 0.7)
        if "h3" in text_lower and "qzs-5" in text_lower:
            add("H3", "QZS-5", "LAUNCHES", 0.75)
        if "저장영상" in chunk.text and "신규촬영" in chunk.text:
            add("저장영상", "신규촬영", "PRICE_COMPARED_WITH", 0.8)
        if "위성영상" in chunk.text and "저장영상" in chunk.text:
            add("위성영상", "저장영상", "HAS_OPTION", 0.75)
        if "위성영상" in chunk.text and "신규촬영" in chunk.text:
            add("위성영상", "신규촬영", "HAS_OPTION", 0.75)

        seen = {(relation.source, relation.target, relation.type) for relation in relations}
        for source, target in combinations(entities[:10], 2):
            key = (source.canonical_id, target.canonical_id, "RELATED_TO")
            reverse = (target.canonical_id, source.canonical_id, "RELATED_TO")
            if key in seen or reverse in seen:
                continue
            relations.append(
                ExtractedRelation(
                    source=source.canonical_id,
                    target=target.canonical_id,
                    type="RELATED_TO",
                    confidence=0.35,
                    evidence=chunk.chunk_id,
                )
            )
            seen.add(key)
        return relations[:80]

