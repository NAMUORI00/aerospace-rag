from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from typing import Any

from ..config import Settings


_ENGINE_CACHE: dict[tuple[str, str, str, str, float, int, bool], Any] = {}


def resolve_llm_model(model: str | None) -> str:
    requested = str(model or "").strip()
    return requested or "google/gemma-4-E4B-it"


def _engine_cache_key(settings: Settings) -> tuple[str, str, str, str, float, int, bool]:
    return (
        resolve_llm_model(settings.llm_model),
        str(settings.vllm_dtype or "auto"),
        str(settings.vllm_quantization or ""),
        str(settings.vllm_load_format or "auto"),
        float(settings.vllm_gpu_memory_utilization or 0.90),
        int(settings.vllm_max_model_len or 4096),
        bool(settings.vllm_trust_remote_code),
    )


def _supports_fileno(stream: object) -> bool:
    fileno = getattr(stream, "fileno", None)
    if fileno is None:
        return False
    try:
        fileno()
    except Exception:
        return False
    return True


@contextlib.contextmanager
def _vllm_initialization_stdio() -> Iterator[None]:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    fallback_stdout = None
    fallback_stderr = None

    try:
        if not _supports_fileno(original_stdout):
            if _supports_fileno(sys.__stdout__):
                sys.stdout = sys.__stdout__
            else:
                fallback_stdout = open(os.devnull, "w", encoding="utf-8")
                sys.stdout = fallback_stdout

        if not _supports_fileno(original_stderr):
            if _supports_fileno(sys.__stderr__):
                sys.stderr = sys.__stderr__
            else:
                fallback_stderr = open(os.devnull, "w", encoding="utf-8")
                sys.stderr = fallback_stderr

        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if fallback_stdout is not None:
            fallback_stdout.close()
        if fallback_stderr is not None:
            fallback_stderr.close()


def _load_vllm_engine(settings: Settings) -> Any:
    key = _engine_cache_key(settings)
    if key in _ENGINE_CACHE:
        return _ENGINE_CACHE[key]

    try:
        from vllm import LLM
    except Exception as exc:
        raise RuntimeError(
            "vLLM generation requires vllm. Install model dependencies with "
            "`pip install -r requirements-models.txt`."
        ) from exc

    model_id, dtype, quantization, load_format, gpu_memory_utilization, max_model_len, trust_remote_code = key
    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "dtype": dtype,
        "load_format": load_format,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "trust_remote_code": trust_remote_code,
    }
    if quantization:
        llm_kwargs["quantization"] = quantization
    with _vllm_initialization_stdio():
        engine = LLM(**llm_kwargs)
    _ENGINE_CACHE[key] = engine
    return engine


def ensure_vllm_model(settings: Settings) -> dict[str, object]:
    _load_vllm_engine(settings)
    return {
        "ready": True,
        "model": resolve_llm_model(settings.llm_model),
        "dtype": settings.vllm_dtype,
        "quantization": settings.vllm_quantization,
        "load_format": settings.vllm_load_format,
        "max_model_len": settings.vllm_max_model_len,
    }


def _sampling_params(*, max_tokens: int, json_schema: dict[str, Any] | None = None) -> Any:
    try:
        from vllm import SamplingParams
    except Exception as exc:
        raise RuntimeError(
            "vLLM generation requires vllm. Install model dependencies with "
            "`pip install -r requirements-models.txt`."
        ) from exc

    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": max(1, int(max_tokens)),
    }
    if json_schema is not None:
        from vllm.sampling_params import GuidedDecodingParams

        kwargs["guided_decoding"] = GuidedDecodingParams(json=json_schema)
    return SamplingParams(**kwargs)


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip()
        content = str(message.get("content") or "").strip()
        parts.append(f"{role}:\n{content}")
    parts.append("assistant:")
    return "\n\n".join(parts)


def _first_text(outputs: Any) -> str:
    first = outputs[0]
    candidates = getattr(first, "outputs", None) or []
    if candidates:
        text = getattr(candidates[0], "text", "")
        return str(text or "").strip()
    text = getattr(first, "text", "")
    return str(text or "").strip()


def generate_vllm_chat(
    messages: list[dict[str, str]],
    *,
    settings: Settings,
    max_tokens: int,
    json_schema: dict[str, Any] | None = None,
) -> str:
    engine = _load_vllm_engine(settings)
    sampling_params = _sampling_params(max_tokens=max_tokens, json_schema=json_schema)
    try:
        outputs = engine.chat([messages], sampling_params=sampling_params, use_tqdm=False)
    except Exception:
        outputs = engine.generate([_messages_to_prompt(messages)], sampling_params=sampling_params, use_tqdm=False)
    answer = _first_text(outputs)
    if not answer:
        raise RuntimeError("vLLM model returned an empty response.")
    return answer
