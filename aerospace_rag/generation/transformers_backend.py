from __future__ import annotations

from typing import Any

from ..config import Settings


_MODEL_CACHE: dict[tuple[str, str, str, bool], tuple[Any, Any]] = {}


def resolve_transformers_model(model: str | None) -> str:
    requested = str(model or "").strip()
    return requested or "google/gemma-4-E4B-it"


def _model_cache_key(settings: Settings) -> tuple[str, str, str, bool]:
    return (
        resolve_transformers_model(settings.transformers_model),
        str(settings.transformers_device_map or "auto"),
        str(settings.transformers_dtype or "auto"),
        bool(settings.transformers_load_in_4bit),
    )


def _load_transformers_model(settings: Settings) -> tuple[Any, Any]:
    key = _model_cache_key(settings)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Transformers generation requires torch, transformers, and accelerate. "
            "Install model dependencies with `pip install -r requirements-models.txt`."
        ) from exc

    model_id, device_map, dtype, load_in_4bit = key
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception:
        processor = AutoTokenizer.from_pretrained(model_id)
    model_kwargs: dict[str, Any] = {"device_map": device_map}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "TRANSFORMERS_LOAD_IN_4BIT=true requires bitsandbytes. "
                "Install model dependencies with `pip install -r requirements-models.txt`."
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
        )
    else:
        model_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    _MODEL_CACHE[key] = (processor, model)
    return processor, model


def ensure_transformers_model(settings: Settings) -> dict[str, object]:
    _load_transformers_model(settings)
    return {
        "ready": True,
        "model": resolve_transformers_model(settings.transformers_model),
        "device_map": settings.transformers_device_map,
        "load_in_4bit": settings.transformers_load_in_4bit,
    }


def _apply_chat_template(processor: Any, messages: list[dict[str, str]]) -> str:
    try:
        return str(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        return str(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


def _decode_response(processor: Any, raw: str) -> str:
    if hasattr(processor, "parse_response"):
        parsed = processor.parse_response(raw)
        if isinstance(parsed, str):
            return parsed.strip()
        if isinstance(parsed, dict):
            for key in ("content", "text", "answer"):
                value = parsed.get(key)
                if value:
                    return str(value).strip()
        if parsed:
            return str(parsed).strip()
    return raw.strip()


def generate_transformers_chat(
    messages: list[dict[str, str]],
    *,
    settings: Settings,
    max_new_tokens: int,
    max_time: int,
) -> str:
    processor, model = _load_transformers_model(settings)
    text = _apply_chat_template(processor, messages)
    inputs = processor(text=text, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    generate_kwargs = {
        **inputs,
        "max_new_tokens": max(1, int(max_new_tokens)),
        "do_sample": False,
    }
    if max_time:
        generate_kwargs["max_time"] = max(1, int(max_time))
    try:
        import torch

        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)
    except Exception:
        outputs = model.generate(**generate_kwargs)
    raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    answer = _decode_response(processor, raw)
    if not answer:
        raise RuntimeError("Transformers model returned an empty response.")
    return answer
