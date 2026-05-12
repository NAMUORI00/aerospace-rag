from __future__ import annotations

import contextlib
import builtins
import os
import sys
from collections.abc import Iterator
from typing import Any

from ..config import Settings


_PROCESS_CACHE_NAME = "_aerospace_rag_vllm_engine_cache"
_ENGINE_CACHE: dict[tuple[str, str, str, str, float, int, float, bool, bool, bool], Any] = getattr(
    builtins, _PROCESS_CACHE_NAME, {}
)
setattr(builtins, _PROCESS_CACHE_NAME, _ENGINE_CACHE)


def resolve_llm_model(model: str | None) -> str:
    requested = str(model or "").strip()
    return requested or "ciocan/gemma-4-E4B-it-W4A16"


def _configure_vllm_engine_mode(settings: Settings) -> None:
    desired = "1" if settings.vllm_use_v1 else "0"
    loaded_envs = sys.modules.get("vllm.envs")
    loaded_use_v1 = getattr(loaded_envs, "VLLM_USE_V1", None)
    if loaded_use_v1 is not None and bool(loaded_use_v1) != settings.vllm_use_v1:
        raise RuntimeError(
            "vLLM was already imported with a different VLLM_USE_V1 mode. "
            "Restart the Python runtime, then run the notebook from the first cell."
        )
    os.environ["VLLM_USE_V1"] = desired


def _engine_cache_key(settings: Settings) -> tuple[str, str, str, str, float, int, float, bool, bool, bool]:
    return (
        resolve_llm_model(settings.llm_model),
        str(settings.vllm_dtype or "auto"),
        str(settings.vllm_quantization or ""),
        str(settings.vllm_load_format or "auto"),
        float(settings.vllm_gpu_memory_utilization or 0.90),
        int(settings.vllm_max_model_len or 2048),
        float(settings.vllm_cpu_offload_gb or 0.0),
        bool(settings.vllm_trust_remote_code),
        bool(settings.vllm_enforce_eager),
        bool(settings.vllm_use_v1),
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

    _configure_vllm_engine_mode(settings)
    try:
        from vllm import LLM
    except Exception as exc:
        raise RuntimeError(
            "vLLM generation requires vllm. Install model dependencies with "
            "`pip install -r requirements-models.txt`."
        ) from exc

    model_id, dtype, quantization, load_format, gpu_memory_utilization, max_model_len, cpu_offload_gb, trust_remote_code, enforce_eager, _use_v1 = key
    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "dtype": dtype,
        "load_format": load_format,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "cpu_offload_gb": cpu_offload_gb,
        "trust_remote_code": trust_remote_code,
        "enforce_eager": enforce_eager,
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
        "cpu_offload_gb": settings.vllm_cpu_offload_gb,
        "enforce_eager": settings.vllm_enforce_eager,
        "use_v1": settings.vllm_use_v1,
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
    # Colab T4 + vLLM can terminate the engine process when guided decoding
    # pulls in optional structured-output backends. Keep generation plain and
    # let the caller's JSON parser/repair path validate schema-shaped outputs.
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
        outputs = engine.generate([_messages_to_prompt(messages)], sampling_params=sampling_params, use_tqdm=False)
    except Exception:
        outputs = engine.chat([messages], sampling_params=sampling_params, use_tqdm=False)
    answer = _first_text(outputs)
    if not answer:
        raise RuntimeError("vLLM model returned an empty response.")
    return answer
