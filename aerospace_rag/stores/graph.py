from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..models import Chunk
from ..retrieval.extraction import KnowledgeExtractor, canonical_id, extract_entity_texts
from ..text import tokenize, unique_ordered


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _relation_type(value: str) -> str:
    cleaned = re.sub(r"[^A-Z_]", "", str(value or "RELATED_TO").upper())
    return cleaned or "RELATED_TO"


def extract_entities(chunk: Chunk) -> list[str]:
    return extract_entity_texts(chunk)[:24]


class GraphStore:
    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.graph_dir = self.index_dir / "falkordb"
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.graph_dir / "falkordb.db"
        self.index_path = self.graph_dir / "graph_index.json"

    def build(self, chunks: list[Chunk], *, reset: bool = True) -> None:
        if reset:
            for path in (self.db_path, self.index_path):
                if path.exists():
                    path.unlink()
        entity_to_chunks: dict[str, list[str]] = defaultdict(list)
        chunk_entities: dict[str, list[str]] = {}
        entity_text: dict[str, str] = {}
        entity_types: dict[str, str] = {}
        relations: list[dict[str, Any]] = []
        extractor = KnowledgeExtractor()
        for chunk in chunks:
            extracted = extractor.extract(chunk)
            entity_ids: list[str] = []
            for entity in extracted.entities:
                entity_ids.append(entity.canonical_id)
                entity_text.setdefault(entity.canonical_id, entity.text)
                entity_types.setdefault(entity.canonical_id, entity.type)
                entity_to_chunks[entity.canonical_id].append(chunk.chunk_id)
            chunk_entities[chunk.chunk_id] = unique_ordered(entity_ids)
            for relation in extracted.relations:
                relations.append(relation.to_dict())
        self._write_falkordb(chunks, chunk_entities, entity_text, entity_types, relations)
        payload = {
            "entity_to_chunks": {k: unique_ordered(v) for k, v in entity_to_chunks.items()},
            "chunk_entities": chunk_entities,
            "entity_text": entity_text,
            "entity_types": entity_types,
            "relations": relations,
            "entity_neighbors": self._build_entity_neighbors(relations),
            "chunks": {chunk.chunk_id: chunk.to_payload() for chunk in chunks},
            "schema_version": "aerospace_graph_v2",
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert_private_chunk(self, chunk: Chunk) -> None:
        if not self.index_path.exists():
            self.build([chunk], reset=True)
            return
        payload: dict[str, Any] = json.loads(self.index_path.read_text(encoding="utf-8"))
        chunks = dict(payload.get("chunks") or {})
        chunks[chunk.chunk_id] = chunk.to_payload()
        payload["chunks"] = chunks
        extractor = KnowledgeExtractor()
        extracted = extractor.extract(chunk)
        entity_to_chunks: dict[str, list[str]] = {
            key: list(value)
            for key, value in dict(payload.get("entity_to_chunks") or {}).items()
        }
        chunk_entities = dict(payload.get("chunk_entities") or {})
        entity_text = dict(payload.get("entity_text") or {})
        entity_types = dict(payload.get("entity_types") or {})
        relations = list(payload.get("relations") or [])
        ids: list[str] = []
        for entity in extracted.entities:
            ids.append(entity.canonical_id)
            entity_text.setdefault(entity.canonical_id, entity.text)
            entity_types.setdefault(entity.canonical_id, entity.type)
            entity_to_chunks.setdefault(entity.canonical_id, [])
            if chunk.chunk_id not in entity_to_chunks[entity.canonical_id]:
                entity_to_chunks[entity.canonical_id].append(chunk.chunk_id)
        chunk_entities[chunk.chunk_id] = unique_ordered(ids)
        relations.extend(relation.to_dict() for relation in extracted.relations)
        payload["entity_to_chunks"] = {key: unique_ordered(value) for key, value in entity_to_chunks.items()}
        payload["chunk_entities"] = chunk_entities
        payload["entity_text"] = entity_text
        payload["entity_types"] = entity_types
        payload["relations"] = relations
        payload["entity_neighbors"] = self._build_entity_neighbors(relations)
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_entity_neighbors(self, relations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        neighbors: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for relation in relations:
            source = str(relation.get("source") or "")
            target = str(relation.get("target") or "")
            if not source or not target:
                continue
            edge = {
                "source": source,
                "target": target,
                "type": _relation_type(str(relation.get("type") or "RELATED_TO")),
                "confidence": float(relation.get("confidence") or 0.35),
                "evidence": str(relation.get("evidence") or ""),
            }
            neighbors[source].append(edge)
            neighbors[target].append({**edge, "source": target, "target": source})
        return {key: value[:32] for key, value in neighbors.items()}

    def _write_falkordb(
        self,
        chunks: list[Chunk],
        chunk_entities: dict[str, list[str]],
        entity_text: dict[str, str],
        entity_types: dict[str, str],
        relations: list[dict[str, Any]],
    ) -> None:
        graph = self._connect_graph()
        if graph is None:
            self.db_path.touch()
            return
        try:
            graph.delete()
        except Exception:
            pass
        graph = self._connect_graph()
        if graph is None:
            self.db_path.touch()
            return
        for chunk in chunks:
            doc_id = chunk.source_file
            chunk_id = chunk.chunk_id
            tier = str(chunk.metadata.get("tier") or "public")
            farm_id = str(chunk.metadata.get("farm_id") or "")
            source_type = str(chunk.metadata.get("source_type") or "document")
            graph.query(
                "MERGE (d:Document {id:'%s', source_file:'%s'}) "
                "MERGE (s:SourceFile {name:'%s'}) "
                "MERGE (c:Chunk {id:'%s', modality:'%s', text:'%s'}) "
                "SET c.chunk_id='%s', c.tier='%s', c.farm_id='%s', c.source_type='%s' "
                "MERGE (d)-[:HAS_CHUNK]->(c) "
                "MERGE (c)-[:FROM_FILE]->(s)"
                % (
                    _escape(doc_id),
                    _escape(chunk.source_file),
                    _escape(chunk.source_file),
                    _escape(chunk_id),
                    _escape(chunk.modality),
                    _escape(chunk.text[:900]),
                    _escape(chunk_id),
                    _escape(tier),
                    _escape(farm_id),
                    _escape(source_type),
                )
            )
            for entity in chunk_entities.get(chunk_id, []):
                graph.query(
                    "MATCH (c:Chunk {id:'%s'}) "
                    "MERGE (e:Entity {canonical_id:'%s'}) "
                    "SET e.name='%s', e.type='%s' "
                    "MERGE (c)-[:MENTIONS]->(e)"
                    % (
                        _escape(chunk_id),
                        _escape(entity),
                        _escape(entity_text.get(entity, entity)),
                        _escape(entity_types.get(entity, "Concept")),
                    )
                )
        for relation in relations:
            source = str(relation.get("source") or "")
            target = str(relation.get("target") or "")
            if not source or not target:
                continue
            graph.query(
                "MERGE (s:Entity {canonical_id:'%s'}) "
                "SET s.name='%s' "
                "MERGE (t:Entity {canonical_id:'%s'}) "
                "SET t.name='%s' "
                "MERGE (s)-[r:%s]->(t) "
                "SET r.confidence=%.4f, r.evidence='%s'"
                % (
                    _escape(source),
                    _escape(entity_text.get(source, source)),
                    _escape(target),
                    _escape(entity_text.get(target, target)),
                    _relation_type(str(relation.get("type") or "RELATED_TO")),
                    float(relation.get("confidence") or 0.35),
                    _escape(str(relation.get("evidence") or "")),
                )
            )

    def _connect_graph(self):
        try:
            from redislite.falkordb_client import FalkorDB

            db = FalkorDB(str(self.db_path))
            return db.select_graph("aerospace")
        except Exception:
            pass

        try:
            import os
            from falkordb import FalkorDB

            host = os.environ.get("FALKORDB_HOST")
            if not host:
                return None
            port = int(os.environ.get("FALKORDB_PORT", "6379"))
            password = os.environ.get("FALKORDB_PASSWORD") or None
            db = FalkorDB(host=host, port=port, password=password)
            return db.select_graph(os.environ.get("FALKORDB_GRAPH", "aerospace"))
        except Exception:
            return None

    def _chunk_allowed(
        self,
        payload: dict[str, Any],
        *,
        farm_id: str,
        include_private: bool,
    ) -> bool:
        tier = str(payload.get("tier") or "public").lower()
        if tier == "private":
            return bool(include_private and str(payload.get("farm_id") or "") == str(farm_id))
        return tier == "public"

    def _search_falkordb(
        self,
        query: str,
        *,
        farm_id: str,
        include_private: bool,
        limit: int,
    ) -> list[tuple[str, float]]:
        graph = self._connect_graph()
        if graph is None:
            return []
        terms = [t for t in tokenize(query) if len(t) >= 2][:3]
        if not terms:
            return []
        scope = "c.tier='public'"
        if include_private:
            scope = f"(c.tier='public' OR (c.tier='private' AND c.farm_id='{_escape(farm_id)}'))"
        hits: Counter[str] = Counter()
        for term in terms:
            safe = _escape(term.lower())
            for cypher in (
                "MATCH (e:Entity)-[:MENTIONS]-(c:Chunk) "
                f"WHERE {scope} AND (toLower(coalesce(e.name,'')) CONTAINS '{safe}' "
                f"OR toLower(coalesce(e.canonical_id,'')) CONTAINS '{safe}') "
                "RETURN c.id AS id, 1.0 AS score "
                f"LIMIT {max(limit * 2, 10)}",
                "MATCH p=(s:Entity)-[*1..3]-(t:Entity)-[:MENTIONS]-(c:Chunk) "
                f"WHERE {scope} AND (toLower(coalesce(s.name,'')) CONTAINS '{safe}' "
                f"OR toLower(coalesce(s.canonical_id,'')) CONTAINS '{safe}') "
                "RETURN c.id AS id, (1.0 / toFloat(length(p)+1)) AS score "
                f"LIMIT {max(limit * 2, 10)}",
            ):
                try:
                    result = graph.query(cypher)
                    rows = getattr(result, "result_set", result)
                except Exception:
                    rows = []
                for row in rows or []:
                    try:
                        chunk_id = str(row[0])
                        score = float(row[1])
                    except Exception:
                        continue
                    if chunk_id:
                        hits[chunk_id] += score
        return hits.most_common(limit)

    def search(
        self,
        query: str,
        *,
        farm_id: str = "default",
        include_private: bool = False,
        limit: int = 8,
    ) -> list[tuple[str, float]]:
        graph_hits = self._search_falkordb(query, farm_id=farm_id, include_private=include_private, limit=limit)
        if graph_hits:
            return graph_hits
        if not self.index_path.exists():
            return []
        payload: dict[str, Any] = json.loads(self.index_path.read_text(encoding="utf-8"))
        entity_to_chunks: dict[str, list[str]] = payload.get("entity_to_chunks") or {}
        entity_text: dict[str, str] = payload.get("entity_text") or {}
        entity_neighbors: dict[str, list[dict[str, Any]]] = payload.get("entity_neighbors") or {}
        chunk_payloads: dict[str, dict[str, Any]] = payload.get("chunks") or {}
        qtokens = set(tokenize(query))
        scores: Counter[str] = Counter()
        matched_entities: Counter[str] = Counter()
        for entity_id, chunk_ids in entity_to_chunks.items():
            label = str(entity_text.get(entity_id) or entity_id)
            entity_tokens = set(tokenize(label)) | set(tokenize(entity_id))
            if not entity_tokens:
                continue
            overlap = len(qtokens & entity_tokens)
            substring = label.lower() in query.lower() or any(t in label.lower() for t in qtokens if len(t) >= 2)
            if overlap or substring:
                weight = float(overlap or 1)
                matched_entities[entity_id] += weight
                for chunk_id in chunk_ids:
                    if self._chunk_allowed(chunk_payloads.get(chunk_id, {}), farm_id=farm_id, include_private=include_private):
                        scores[chunk_id] += weight

        for entity_id, base_score in matched_entities.items():
            for edge in entity_neighbors.get(entity_id, []):
                target = str(edge.get("target") or "")
                if not target:
                    continue
                confidence = float(edge.get("confidence") or 0.35)
                rel_weight = max(0.05, confidence) * float(base_score) * 0.65
                for chunk_id in entity_to_chunks.get(target, []):
                    if self._chunk_allowed(chunk_payloads.get(chunk_id, {}), farm_id=farm_id, include_private=include_private):
                        scores[chunk_id] += rel_weight

        return scores.most_common(limit)
