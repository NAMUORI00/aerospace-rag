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


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    embed_backend: str = "hash"
    embed_model: str = "hash"
    embed_dim: int = 384
    embed_normalize: bool = True
    vector_backend: str = "qdrant"
    qdrant_host: str = ""
    qdrant_port: int = 6333
    qdrant_url: str = ""
    llm_provider: str = "vllm"
    knowledge_extract_retries: int = 1
    knowledge_extract_repair_retries: int = 1
    knowledge_extract_max_chars: int = 1200
    llm_model: str = "ciocan/gemma-4-E4B-it-W4A16"
    vllm_dtype: str = "auto"
    vllm_quantization: str = "gptq"
    vllm_load_format: str = "auto"
    vllm_gpu_memory_utilization: float = 0.90
    vllm_max_model_len: int = 2048
    vllm_cpu_offload_gb: float = 0.0
    vllm_trust_remote_code: bool = False
    vllm_enforce_eager: bool = True
    vllm_use_v1: bool = True
    llm_answer_max_tokens: int = 1024
    llm_extract_max_tokens: int = 768
    extractor_backend: str = "vllm"
    fusion_mode: str = "hybrid"
    fusion_profile_path: str = ""
    fusion_profile_meta_path: str = ""
    fusion_min_weight: float = 0.10
    fusion_max_weight: float = 0.80

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            embed_backend=os.environ.get("AEROSPACE_EMBED_BACKEND", os.environ.get("EMBED_BACKEND", "hash")),
            embed_model=os.environ.get("AEROSPACE_EMBED_MODEL", os.environ.get("EMBED_MODEL", "hash")),
            embed_dim=int(os.environ.get("AEROSPACE_EMBED_DIM", os.environ.get("EMBED_DIM", "384"))),
            embed_normalize=os.environ.get("AEROSPACE_EMBED_NORMALIZE", os.environ.get("EMBED_LOCAL_NORMALIZE", "true")).lower()
            in {"1", "true", "yes", "on"},
            vector_backend=os.environ.get("AEROSPACE_VECTOR_BACKEND", "qdrant"),
            qdrant_host=os.environ.get("QDRANT_HOST", ""),
            qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
            qdrant_url=os.environ.get("QDRANT_URL", ""),
            llm_provider=os.environ.get("LLM_PROVIDER", "vllm").strip().lower() or "vllm",
            knowledge_extract_retries=_env_int("KNOWLEDGE_EXTRACT_RETRIES", 1),
            knowledge_extract_repair_retries=_env_int("KNOWLEDGE_EXTRACT_REPAIR_RETRIES", 1),
            knowledge_extract_max_chars=_env_int("KNOWLEDGE_EXTRACT_MAX_CHARS", 1200),
            llm_model=os.environ.get(
                "AEROSPACE_LLM_MODEL", os.environ.get("LLM_MODEL", "ciocan/gemma-4-E4B-it-W4A16")
            ),
            vllm_dtype=os.environ.get("AEROSPACE_VLLM_DTYPE", "auto"),
            vllm_quantization=os.environ.get("AEROSPACE_VLLM_QUANTIZATION", "gptq"),
            vllm_load_format=os.environ.get("AEROSPACE_VLLM_LOAD_FORMAT", "auto"),
            vllm_gpu_memory_utilization=_env_float("AEROSPACE_VLLM_GPU_MEMORY_UTILIZATION", 0.90),
            vllm_max_model_len=_env_int("AEROSPACE_VLLM_MAX_MODEL_LEN", 2048),
            vllm_cpu_offload_gb=_env_float("AEROSPACE_VLLM_CPU_OFFLOAD_GB", 0.0),
            vllm_trust_remote_code=_env_bool("AEROSPACE_VLLM_TRUST_REMOTE_CODE", False),
            vllm_enforce_eager=_env_bool("AEROSPACE_VLLM_ENFORCE_EAGER", True),
            vllm_use_v1=_env_bool("AEROSPACE_VLLM_USE_V1", True),
            llm_answer_max_tokens=_env_int("LLM_ANSWER_MAX_TOKENS", 1024),
            llm_extract_max_tokens=_env_int("LLM_EXTRACT_MAX_TOKENS", 768),
            extractor_backend=os.environ.get("EXTRACTOR_LLM_BACKEND", os.environ.get("AEROSPACE_EXTRACTOR_BACKEND", "vllm")),
            fusion_mode=os.environ.get("AEROSPACE_FUSION_MODE", "hybrid"),
            fusion_profile_path=os.environ.get("AEROSPACE_FUSION_PROFILE_PATH", ""),
            fusion_profile_meta_path=os.environ.get("AEROSPACE_FUSION_PROFILE_META_PATH", ""),
            fusion_min_weight=_env_float("AEROSPACE_FUSION_MIN_WEIGHT", 0.10),
            fusion_max_weight=_env_float("AEROSPACE_FUSION_MAX_WEIGHT", 0.80),
        )
