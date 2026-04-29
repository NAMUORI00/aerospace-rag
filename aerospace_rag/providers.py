from __future__ import annotations

import json
import os
import urllib.request

from .config import Settings
from .models import RetrievalHit
from .text import excerpt


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


def _openai_compatible_answer(question: str, hits: list[RetrievalHit], settings: Settings) -> str:
    base_url = settings.vllm_base_url
    model = settings.openai_compat_model or settings.ollama_model
    api_key = settings.vllm_api_key
    if not base_url or not model:
        return _extractive_answer(question, hits)
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _build_prompt(question, hits)}],
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


def _ollama_answer(question: str, hits: list[RetrievalHit], settings: Settings) -> str:
    url = settings.ollama_base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": "다음 근거만 사용해 한국어로 간결하게 답하세요. 근거가 부족하면 부족하다고 말하세요."},
            {"role": "user", "content": _build_prompt(question, hits)},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(((body.get("message") or {}).get("content")) or "").strip() or _extractive_answer(question, hits)
    except Exception:
        return _extractive_answer(question, hits)


def generate_answer(
    question: str,
    hits: list[RetrievalHit],
    *,
    provider: str = "extractive",
    settings: Settings | None = None,
) -> str:
    resolved_settings = settings or Settings.from_env()
    if provider == "ollama":
        return _ollama_answer(question, hits, resolved_settings)
    if provider in {"openai_compatible", "gemma4_openai"}:
        return _openai_compatible_answer(question, hits, resolved_settings)
    return _extractive_answer(question, hits)
