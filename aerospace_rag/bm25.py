from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .models import Chunk
from .text import tokenize


class BM25Index:
    def __init__(self, *, chunk_ids: list[str], documents: list[list[str]]) -> None:
        self.chunk_ids = chunk_ids
        self.documents = documents
        self.doc_count = len(documents)
        self.avgdl = sum(len(doc) for doc in documents) / max(1, len(documents))
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(doc))
        self.idf = {
            term: math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        return cls(chunk_ids=[c.chunk_id for c in chunks], documents=[tokenize(c.text) for c in chunks])

    def save(self, path: str | Path) -> None:
        payload = {
            "chunk_ids": self.chunk_ids,
            "documents": self.documents,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chunk_ids=list(payload["chunk_ids"]), documents=list(payload["documents"]))

    def search(self, query: str, *, limit: int = 8) -> list[tuple[str, float]]:
        qterms = tokenize(query)
        if not qterms:
            return []
        scores: list[tuple[str, float]] = []
        k1 = 1.5
        b = 0.75
        for chunk_id, doc in zip(self.chunk_ids, self.documents):
            tf = Counter(doc)
            dl = len(doc) or 1
            score = 0.0
            for term in qterms:
                if term not in tf:
                    continue
                numerator = tf[term] * (k1 + 1)
                denominator = tf[term] + k1 * (1 - b + b * dl / max(self.avgdl, 1e-9))
                score += self.idf.get(term, 0.0) * numerator / denominator
            if score > 0:
                scores.append((chunk_id, float(score)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:limit]
