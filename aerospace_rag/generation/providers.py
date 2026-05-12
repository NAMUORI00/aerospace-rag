from __future__ import annotations

from ..config import Settings
from ..models import RetrievalHit
from ..text import excerpt
from .vllm_backend import generate_vllm_chat


def route_generation_provider(
    provider: str | None,
    *,
    settings: Settings | None = None,
) -> str:
    requested = str(provider or (settings.llm_provider if settings else "vllm") or "vllm").strip().lower()
    if requested in {"extractive", "vllm"}:
        return requested
    raise ValueError("provider must be 'vllm' or explicit debug mode 'extractive'.")


def _build_prompt(question: str, hits: list[RetrievalHit]) -> str:
    has_table = any(hit.chunk.modality == "table" for hit in hits[:5])
    instructions = "다음 근거만 사용해 한국어로 답하세요. 근거가 부족하면 부족하다고 말하세요."
    if has_table:
        instructions += " 표가 있으면 열 순서를 유지하고, 같은 행의 값만 비교하며, 인접 열 값을 바꾸지 마세요."
    context = "\n\n".join(
        _format_hit_context(idx, hit)
        for idx, hit in enumerate(hits[:5], start=1)
    )
    return (
        f"{instructions}\n\n"
        f"질문: {question}\n\n근거:\n{context}"
    )


def _format_hit_context(idx: int, hit: RetrievalHit) -> str:
    location = ""
    if hit.chunk.page:
        location = f" p.{hit.chunk.page}"
    elif hit.chunk.row:
        location = f" row {hit.chunk.row}"
    header = f"[{idx}] {hit.chunk.source_file}{location}"
    if hit.chunk.modality == "table":
        return f"{header}\n표 데이터:\n{hit.chunk.text}"
    return f"{header}\n{hit.chunk.text}"


def _extractive_answer(question: str, hits: list[RetrievalHit]) -> str:
    if not hits:
        return "검색된 근거가 없어 답변을 생성할 수 없습니다."
    lines = ["근거 기반 답변:"]
    for idx, hit in enumerate(hits[:3], start=1):
        loc = f"p.{hit.chunk.page}" if hit.chunk.page else f"row {hit.chunk.row}" if hit.chunk.row else "table"
        lines.append(f"{idx}. [{hit.chunk.source_file} / {loc}] {excerpt(hit.chunk.text, max_chars=520)}")
    return "\n".join(lines)


def _vllm_answer(question: str, hits: list[RetrievalHit], settings: Settings) -> str:
    messages = [
        {"role": "system", "content": "다음 근거만 사용해 한국어로 간결하게 답하세요. 근거가 부족하면 부족하다고 말하세요."},
        {"role": "user", "content": _build_prompt(question, hits)},
    ]
    return generate_vllm_chat(
        messages,
        settings=settings,
        max_tokens=max(128, int(settings.llm_answer_max_tokens or 1024)),
    )


def generate_answer(
    question: str,
    hits: list[RetrievalHit],
    *,
    provider: str = "vllm",
    settings: Settings | None = None,
) -> str:
    resolved_settings = settings or Settings.from_env()
    provider = route_generation_provider(provider, settings=resolved_settings)
    if provider == "vllm":
        try:
            return _vllm_answer(question, hits, resolved_settings)
        except Exception:
            if not resolved_settings.vllm_fallback_on_error:
                raise
            return _extractive_answer(question, hits)
    return _extractive_answer(question, hits)
