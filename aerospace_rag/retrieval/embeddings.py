from __future__ import annotations

import math
from typing import Iterable

from ..config import Settings
from ..text import hash_embedding


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [float(v) / norm for v in vector]


class EmbeddingService:
    _MODEL_CACHE: dict[tuple[str, str], tuple[object, int]] = {}

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        backend = str(self.settings.embed_backend or "sentence_transformers").strip().lower()
        self._model = None
        if backend == "hash":
            self.provider_name = "hash"
            self.vector_size = 384
            return
        if backend != "sentence_transformers":
            raise ValueError("AEROSPACE_EMBED_BACKEND must be 'sentence_transformers' or explicit debug mode 'hash'.")
        try:
            from sentence_transformers import SentenceTransformer

            cache_key = (backend, self.settings.embed_model)
            cached = self._MODEL_CACHE.get(cache_key)
            if cached is None:
                model = SentenceTransformer(self.settings.embed_model)
                if hasattr(model, "get_embedding_dimension"):
                    dim = int(getattr(model, "get_embedding_dimension")())
                else:
                    dim = int(getattr(model, "get_sentence_embedding_dimension", lambda: self.settings.embed_dim)())
                cached = (model, dim or self.settings.embed_dim)
                self._MODEL_CACHE[cache_key] = cached
            self._model, self.vector_size = cached
            self.provider_name = "sentence_transformers"
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers embedding is required by default. "
                "Install model dependencies with `pip install -r requirements-models.txt`, "
                "or explicitly set AEROSPACE_EMBED_BACKEND=hash for debug runs."
            ) from exc

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
