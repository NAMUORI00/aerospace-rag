from __future__ import annotations

from typing import Iterable

from ..config import Settings
from ..text import hash_embedding


class EmbeddingService:
    _MODEL_CACHE: dict[tuple[str, str], tuple[object, int]] = {}

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        backend = str(self.settings.embed_backend or "hash").strip().lower()
        if backend == "hash":
            self.provider_name = "hash"
            self.vector_size = int(self.settings.embed_dim or 384)
            return
        raise ValueError("AEROSPACE_EMBED_BACKEND must be 'hash'.")

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        batch = list(texts)
        return [hash_embedding(text, dim=self.vector_size) for text in batch]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
