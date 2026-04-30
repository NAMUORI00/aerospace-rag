from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    embed_backend: str = "sentence_transformers"
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_normalize: bool = True
    vector_backend: str = "qdrant"
    qdrant_host: str = ""
    qdrant_port: int = 6333
    qdrant_url: str = ""
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma4:e2b"
    ollama_api_key: str = ""
    ollama_keep_alive: str = "10m"
    ollama_extract_timeout_seconds: int = 3600
    ollama_generate_timeout_seconds: int = 3600
    ollama_extract_retries: int = 1
    ollama_extract_num_predict: int = 768
    ollama_answer_num_predict: int = 1024
    ollama_extract_max_chars: int = 3000
    extractor_provider: str = "ollama"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            embed_backend=os.environ.get("AEROSPACE_EMBED_BACKEND", os.environ.get("EMBED_BACKEND", "sentence_transformers")),
            embed_model=os.environ.get("AEROSPACE_EMBED_MODEL", os.environ.get("EMBED_MODEL", "BAAI/bge-m3")),
            embed_dim=int(os.environ.get("AEROSPACE_EMBED_DIM", os.environ.get("EMBED_DIM", "1024"))),
            embed_normalize=os.environ.get("AEROSPACE_EMBED_NORMALIZE", os.environ.get("EMBED_LOCAL_NORMALIZE", "true")).lower()
            in {"1", "true", "yes", "on"},
            vector_backend=os.environ.get("AEROSPACE_VECTOR_BACKEND", "qdrant"),
            qdrant_host=os.environ.get("QDRANT_HOST", ""),
            qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
            qdrant_url=os.environ.get("QDRANT_URL", ""),
            llm_provider="ollama",
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL", os.environ.get("GEMMA4_MODEL", "gemma4:e2b")),
            ollama_api_key=os.environ.get("OLLAMA_API_KEY", ""),
            ollama_keep_alive=os.environ.get("OLLAMA_KEEP_ALIVE", "10m"),
            ollama_extract_timeout_seconds=_env_int("OLLAMA_EXTRACT_TIMEOUT_SECONDS", 3600),
            ollama_generate_timeout_seconds=_env_int("OLLAMA_GENERATE_TIMEOUT_SECONDS", 3600),
            ollama_extract_retries=_env_int("OLLAMA_EXTRACT_RETRIES", 1),
            ollama_extract_num_predict=_env_int("OLLAMA_EXTRACT_NUM_PREDICT", 768),
            ollama_answer_num_predict=_env_int("OLLAMA_ANSWER_NUM_PREDICT", 1024),
            ollama_extract_max_chars=_env_int("OLLAMA_EXTRACT_MAX_CHARS", 3000),
            extractor_provider=os.environ.get("EXTRACTOR_LLM_BACKEND", os.environ.get("AEROSPACE_EXTRACTOR_PROVIDER", "ollama")),
        )
