from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any
import urllib.request

from ..config import Settings
from ..generation.transformers_backend import generate_transformers_chat
from ..models import Chunk
from ..text import tokenize, unique_ordered


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


EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities", "relations"],
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": 24,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["canonical_id", "text", "type", "confidence"],
                "properties": {
                    "canonical_id": {"type": "string", "maxLength": 120},
                    "text": {"type": "string", "maxLength": 120},
                    "type": {"type": "string", "maxLength": 64},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
        "relations": {
            "type": "array",
            "maxItems": 48,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source", "target", "type", "confidence", "evidence"],
                "properties": {
                    "source": {"type": "string", "maxLength": 120},
                    "target": {"type": "string", "maxLength": 120},
                    "type": {"type": "string", "maxLength": 64},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence": {"type": "string", "maxLength": 240},
                },
            },
        },
    },
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


def parse_llm_json_object(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be an object")
    return parsed


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
    """Knowledge extractor with Ollama/Transformers model paths and explicit local debug mode."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

    def extract(self, chunk: Chunk) -> ExtractionResult:
        provider = str(self.settings.extractor_provider or "ollama").strip().lower()
        if provider == "ollama":
            return self._extract_with_ollama(chunk)
        if provider == "transformers":
            return self._extract_with_transformers(chunk)
        if provider in {"local", "local_fallback", "debug_local"}:
            return self._extract_local_debug(chunk)
        raise ValueError("EXTRACTOR_LLM_BACKEND must be 'ollama', 'transformers', or explicit debug mode 'local_fallback'.")

    def _extract_with_ollama(self, chunk: Chunk) -> ExtractionResult:
        base_url = str(self.settings.ollama_base_url or "").strip().rstrip("/")
        model = str(self.settings.ollama_model or "").strip()
        if not base_url or not model:
            raise RuntimeError("Ollama extraction requires OLLAMA_BASE_URL and OLLAMA_MODEL.")
        max_chars = max(500, int(self.settings.ollama_extract_max_chars or 1200))
        chunk_text = chunk.text[:max_chars]
        schema_text = json.dumps(EXTRACTION_JSON_SCHEMA, ensure_ascii=False, separators=(",", ":"))
        system_prompt = (
            "You extract compact aerospace retrieval knowledge. Return only one minified JSON object. "
            "The response must validate against this JSON Schema. Do not include markdown or commentary.\n"
            f"JSON Schema: {schema_text}"
        )
        prompt = (
            "Extract entities and relations from this chunk. "
            "Use canonical_id values in relation source/target. "
            "If there is no reliable item, return empty arrays.\n\n"
            f"Document: {chunk.source_file}\nChunk: {chunk.chunk_id}\nText:\n{chunk_text}"
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": EXTRACTION_JSON_SCHEMA,
            "think": False,
            "keep_alive": str(self.settings.ollama_keep_alive or "10m"),
            "options": {
                "temperature": 0.0,
                "num_predict": max(128, int(self.settings.ollama_extract_num_predict or 768)),
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.settings.ollama_api_key}"
        timeout = max(1, int(self.settings.ollama_extract_timeout_seconds or 3600))
        attempts = max(0, int(self.settings.ollama_extract_retries or 0)) + 1
        last_error: Exception | None = None
        parsed: dict[str, Any] | None = None
        for _ in range(attempts):
            try:
                content = self._ollama_chat(base_url=base_url, payload=payload, headers=headers, timeout=timeout)
                parsed = self._parse_or_repair_json(
                    content,
                    base_url=base_url,
                    model=model,
                    headers=headers,
                    timeout=timeout,
                    system_prompt=system_prompt,
                )
                break
            except Exception as exc:
                last_error = exc
        if parsed is None:
            raise RuntimeError(
                "Ollama knowledge extraction failed. Start Ollama, pull the configured model, "
                "or increase OLLAMA_EXTRACT_TIMEOUT_SECONDS."
            ) from last_error
        return self._result_from_parsed(parsed, chunk)

    def _extract_with_transformers(self, chunk: Chunk) -> ExtractionResult:
        max_chars = max(500, int(self.settings.ollama_extract_max_chars or 1200))
        chunk_text = chunk.text[:max_chars]
        schema_text = json.dumps(EXTRACTION_JSON_SCHEMA, ensure_ascii=False, separators=(",", ":"))
        system_prompt = (
            "You extract compact aerospace retrieval knowledge. Return only one minified JSON object. "
            "The response must validate against this JSON Schema. Do not include markdown or commentary.\n"
            f"JSON Schema: {schema_text}"
        )
        prompt = (
            "Extract entities and relations from this chunk. "
            "Use canonical_id values in relation source/target. "
            "If there is no reliable item, return empty arrays.\n\n"
            f"Document: {chunk.source_file}\nChunk: {chunk.chunk_id}\nText:\n{chunk_text}"
        )
        attempts = max(0, int(self.settings.ollama_extract_retries or 0)) + 1
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                content = generate_transformers_chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    settings=self.settings,
                    max_new_tokens=max(128, int(self.settings.transformers_extract_num_predict or 768)),
                    max_time=max(1, int(self.settings.transformers_extract_timeout_seconds or 120)),
                )
                return self._result_from_parsed(parse_llm_json_object(content), chunk)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            "Transformers knowledge extraction failed. Check model loading, generation limits, "
            "or set EXTRACTOR_LLM_BACKEND='local_fallback' for no-LLM debugging."
        ) from last_error

    def _result_from_parsed(self, parsed: dict[str, Any], chunk: Chunk) -> ExtractionResult:
        entities = []
        for item in parsed.get("entities") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("canonical_id") or "").strip()
            if not text:
                continue
            entities.append(
                ExtractedEntity(
                    canonical_id=canonical_id(str(item.get("canonical_id") or text)),
                    text=text,
                    type=str(item.get("type") or "Concept"),
                    confidence=max(0.0, min(1.0, float(item.get("confidence") or 0.5))),
                )
            )
        entity_ids = {entity.canonical_id for entity in entities}
        relations = []
        for item in parsed.get("relations") or []:
            if not isinstance(item, dict):
                continue
            source = canonical_id(str(item.get("source") or ""))
            target = canonical_id(str(item.get("target") or ""))
            if source not in entity_ids or target not in entity_ids or source == target:
                continue
            relations.append(
                ExtractedRelation(
                    source=source,
                    target=target,
                    type=str(item.get("type") or "RELATED_TO"),
                    confidence=max(0.0, min(1.0, float(item.get("confidence") or 0.5))),
                    evidence=str(item.get("evidence") or chunk.chunk_id),
                )
            )
        return ExtractionResult(entities=entities[:64], relations=relations[:128])

    def _ollama_chat(
        self,
        *,
        base_url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> str:
        req = urllib.request.Request(
            base_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(((body.get("message") or {}).get("content")) or "")

    def _parse_or_repair_json(
        self,
        content: str,
        *,
        base_url: str,
        model: str,
        headers: dict[str, str],
        timeout: int,
        system_prompt: str,
    ) -> dict[str, Any]:
        try:
            return parse_llm_json_object(content)
        except Exception as exc:
            last_error: Exception = exc

        repair_retries = max(0, int(self.settings.ollama_extract_repair_retries or 0))
        for _ in range(repair_retries):
            repair_payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Repair this malformed JSON so it validates against the schema. "
                            "Return only the repaired minified JSON object.\n\n"
                            f"Malformed JSON:\n{content[:12000]}"
                        ),
                    },
                ],
                "stream": False,
                "format": EXTRACTION_JSON_SCHEMA,
                "think": False,
                "keep_alive": str(self.settings.ollama_keep_alive or "10m"),
                "options": {
                    "temperature": 0.0,
                    "num_predict": max(128, int(self.settings.ollama_extract_num_predict or 4096)),
                },
            }
            try:
                repaired = self._ollama_chat(base_url=base_url, payload=repair_payload, headers=headers, timeout=timeout)
                return parse_llm_json_object(repaired)
            except Exception as exc:
                last_error = exc
        raise last_error

    def _extract_local_debug(self, chunk: Chunk) -> ExtractionResult:
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

