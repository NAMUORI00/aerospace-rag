from __future__ import annotations

from typing import Any

from ..config import Settings


_ENGINE_CACHE: dict[tuple[str, str, float, int, bool], Any] = {}


def resolve_llm_model(model: str | None) -> str:
    requested = str(model or "").strip()
    return requested or "google/gemma-4-E4B-it"


def _engine_cache_key(settings: Settings) -> tuple[str, str, float, int, bool]:
    return (
        resolve_llm_model(settings.llm_model),
        str(settings.vllm_dtype or "auto"),
        float(settings.vllm_gpu_memory_utilization or 0.90),
        int(settings.vllm_max_model_len or 4096),
        bool(settings.vllm_trust_remote_code),
    )


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

    model_id, dtype, gpu_memory_utilization, max_model_len, trust_remote_code = key
    engine = LLM(
        model=model_id,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
    )
    _ENGINE_CACHE[key] = engine
    return engine


def ensure_vllm_model(settings: Settings) -> dict[str, object]:
    _load_vllm_engine(settings)
    return {
        "ready": True,
        "model": resolve_llm_model(settings.llm_model),
        "dtype": settings.vllm_dtype,
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
