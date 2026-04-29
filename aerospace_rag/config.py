from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    embed_backend: str = "sentence_transformers"
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_normalize: bool = True
    dat_mode: str = "hybrid"
    fusion_profile_path: Path = Path("data/artifacts/fusion_weights.runtime.json")
    fusion_profile_meta_path: Path = Path("data/artifacts/fusion_profile_meta.runtime.json")
    dat_min_weight_per_channel: float = 0.10
    dat_max_weight_per_channel: float = 0.80
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma4:e2b"
    vllm_base_url: str = ""
    vllm_api_key: str = "local-dummy"
    openai_compat_model: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            embed_backend=os.environ.get("AEROSPACE_EMBED_BACKEND", os.environ.get("EMBED_BACKEND", "sentence_transformers")),
            embed_model=os.environ.get("AEROSPACE_EMBED_MODEL", os.environ.get("EMBED_MODEL", "BAAI/bge-m3")),
            embed_dim=int(os.environ.get("AEROSPACE_EMBED_DIM", os.environ.get("EMBED_DIM", "1024"))),
            embed_normalize=os.environ.get("AEROSPACE_EMBED_NORMALIZE", os.environ.get("EMBED_LOCAL_NORMALIZE", "true")).lower()
            in {"1", "true", "yes", "on"},
            dat_mode=os.environ.get("DAT_MODE", "hybrid"),
            fusion_profile_path=Path(os.environ.get("FUSION_PROFILE_PATH", "data/artifacts/fusion_weights.runtime.json")),
            fusion_profile_meta_path=Path(os.environ.get("FUSION_PROFILE_META_PATH", "data/artifacts/fusion_profile_meta.runtime.json")),
            dat_min_weight_per_channel=float(os.environ.get("DAT_MIN_WEIGHT_PER_CHANNEL", "0.10")),
            dat_max_weight_per_channel=float(os.environ.get("DAT_MAX_WEIGHT_PER_CHANNEL", "0.80")),
            llm_provider=os.environ.get("LLM_PROVIDER", "ollama"),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL", os.environ.get("GEMMA4_MODEL", "gemma4:e2b")),
            vllm_base_url=os.environ.get("VLLM_BASE_URL", os.environ.get("OPENAI_COMPAT_BASE_URL", "")),
            vllm_api_key=os.environ.get("VLLM_API_KEY", os.environ.get("OPENAI_COMPAT_API_KEY", "local-dummy")),
            openai_compat_model=os.environ.get("OPENAI_COMPAT_MODEL", ""),
        )
