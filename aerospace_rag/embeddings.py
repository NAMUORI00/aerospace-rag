from __future__ import annotations

import math
from typing import Iterable

from .config import Settings
from .text import hash_embedding


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [float(v) / norm for v in vector]


class EmbeddingService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.provider_name = "hash"
        self.vector_size = 384
        self._model = None
        if self.settings.embed_backend == "hash":
            self.vector_size = 384
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embed_model)
            dim = int(getattr(self._model, "get_sentence_embedding_dimension", lambda: self.settings.embed_dim)())
            self.vector_size = dim or self.settings.embed_dim
            self.provider_name = "sentence_transformers"
        except Exception:
            self._model = None
            self.vector_size = 384
            self.provider_name = "hash"

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        batch = list(texts)
        if self._model is None:
            return [hash_embedding(text, dim=self.vector_size) for text in batch]
        vectors = self._model.encode(
            batch,
            normalize_embeddings=bool(self.settings.embed_normalize),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        output = [[float(x) for x in row.tolist()] for row in vectors]
        if self.settings.embed_normalize:
            return [_normalize(row) for row in output]
        return output

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
