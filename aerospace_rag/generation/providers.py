from __future__ import annotations

import json
import urllib.request

from ..config import Settings
from ..models import RetrievalHit
from ..text import excerpt


class OllamaGenerationError(RuntimeError):
    """Raised when the explicit Ollama generation path cannot produce an answer."""


def route_generation_provider(
    provider: str | None,
    *,
    settings: Settings | None = None,
) -> str:
    _ = settings
    requested = str(provider or "ollama").strip().lower()
    if requested in {"ollama", "extractive"}:
        return requested
    raise ValueError("provider must be 'ollama' or explicit debug mode 'extractive'.")


def _build_prompt(question: str, hits: list[RetrievalHit]) -> str:
    context = "\n\n".join(
        f"[{idx}] {hit.chunk.source_file} {hit.chunk.page or hit.chunk.row or ''}\n{hit.chunk.text}"
        for idx, hit in enumerate(hits[:5], start=1)
    )
    return (
        "다음 근거만 사용해 한국어로 답하세요. 근거가 부족하면 부족하다고 말하세요.\n\n"
        f"질문: {question}\n\n근거:\n{context}"
    )


def _extractive_answer(question: str, hits: list[RetrievalHit]) -> str:
    if not hits:
        return "검색된 근거가 없어 답변을 생성할 수 없습니다."
    lines = ["근거 기반 답변:"]
    for idx, hit in enumerate(hits[:3], start=1):
        loc = f"p.{hit.chunk.page}" if hit.chunk.page else f"row {hit.chunk.row}" if hit.chunk.row else "table"
        lines.append(f"{idx}. [{hit.chunk.source_file} / {loc}] {excerpt(hit.chunk.text, max_chars=520)}")
    return "\n".join(lines)


def _ollama_answer(question: str, hits: list[RetrievalHit], settings: Settings) -> str:
    if not str(settings.ollama_base_url or "").strip():
        raise OllamaGenerationError("Ollama generation requires OLLAMA_BASE_URL.")
    if not str(settings.ollama_model or "").strip():
        raise OllamaGenerationError("Ollama generation requires OLLAMA_MODEL.")
    url = settings.ollama_base_url.rstrip("/") + "/api/chat"
    headers = {"Content-Type": "application/json"}
    if settings.ollama_api_key:
        headers["Authorization"] = f"Bearer {settings.ollama_api_key}"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": "다음 근거만 사용해 한국어로 간결하게 답하세요. 근거가 부족하면 부족하다고 말하세요."},
            {"role": "user", "content": _build_prompt(question, hits)},
        ],
        "stream": False,
        "think": False,
        "keep_alive": str(settings.ollama_keep_alive or "10m"),
        "options": {
            "temperature": 0.1,
            "num_predict": max(128, int(settings.ollama_answer_num_predict or 1024)),
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=max(1, int(settings.ollama_generate_timeout_seconds or 3600))) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise OllamaGenerationError(
            "Ollama generation failed. Start Ollama, pull the configured model, "
            "or call ask(..., provider='extractive') for no-LLM debugging."
        ) from exc
    answer = str(((body.get("message") or {}).get("content")) or "").strip()
    if not answer:
        raise OllamaGenerationError("Ollama returned an empty answer.")
    return answer


def generate_answer(
    question: str,
    hits: list[RetrievalHit],
    *,
    provider: str = "ollama",
    settings: Settings | None = None,
) -> str:
    resolved_settings = settings or Settings.from_env()
    provider = route_generation_provider(provider, settings=resolved_settings)
    if provider == "ollama":
        return _ollama_answer(question, hits, resolved_settings)
    return _extractive_answer(question, hits)
