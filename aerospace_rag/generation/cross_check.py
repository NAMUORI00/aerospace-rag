from __future__ import annotations

import json
import urllib.request
from typing import Any

from ..config import Settings
from ..models import RetrievalHit
from ..text import excerpt


CROSS_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["supported", "partially_supported", "unsupported", "insufficient_evidence"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "issues": {"type": "array", "items": {"type": "string"}},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "source_alignment": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source_file": {"type": "string"},
                    "supports_answer": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["source_file", "supports_answer", "note"],
            },
        },
        "suggested_fix": {"type": "string"},
    },
    "required": ["verdict", "confidence", "issues", "missing_evidence", "source_alignment", "suggested_fix"],
}


def _hit_location(hit: RetrievalHit) -> str:
    if hit.chunk.page is not None:
        return f"p.{hit.chunk.page}"
    if hit.chunk.sheet and hit.chunk.row is not None:
        return f"{hit.chunk.sheet}:{hit.chunk.row}"
    if hit.chunk.sheet:
        return str(hit.chunk.sheet)
    if hit.chunk.row is not None:
        return f"row {hit.chunk.row}"
    return hit.chunk.modality


def _source_pack(hits: list[RetrievalHit]) -> list[dict[str, object]]:
    return [
        {
            "rank": idx,
            "source_file": hit.chunk.source_file,
            "location": _hit_location(hit),
            "modality": hit.chunk.modality,
            "score": round(hit.score, 6),
            "channels": hit.channels,
            "excerpt": excerpt(hit.chunk.text, max_chars=1400),
        }
        for idx, hit in enumerate(hits[:8], start=1)
    ]


def _cross_check_prompt(question: str, answer: str, hits: list[RetrievalHit]) -> str:
    payload = {
        "task": "Cross-check the generated RAG answer against only the supplied retrieved sources.",
        "question": question,
        "generated_answer": answer,
        "retrieved_sources": _source_pack(hits),
        "rules": [
            "Use only retrieved_sources as evidence.",
            "Flag unsupported claims, wrong numbers, wrong source attribution, or missing evidence.",
            "Do not rewrite the full answer unless a concise suggested_fix is necessary.",
            "Return JSON that matches the provided schema.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_response_text(body: dict[str, Any]) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    pieces: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                pieces.append(text)
    return "\n".join(piece for piece in pieces if piece).strip()


def _normalize_result(parsed: dict[str, Any], *, model: str) -> dict[str, Any]:
    parsed["status"] = "ok"
    parsed["provider"] = "openai"
    parsed["model"] = model
    return parsed


def run_gpt_pro_cross_check(
    question: str,
    answer: str,
    hits: list[RetrievalHit],
    *,
    settings: Settings,
) -> dict[str, Any]:
    model = str(settings.gpt_pro_cross_check_model or "gpt-5.5-pro").strip()
    if not settings.gpt_pro_cross_check_enabled:
        return {"status": "disabled", "provider": "openai", "model": model}
    if not str(settings.openai_api_key or "").strip():
        return {
            "status": "skipped",
            "provider": "openai",
            "model": model,
            "reason": "OPENAI_API_KEY is not set.",
        }

    url = settings.openai_base_url.rstrip("/") + "/responses"
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You are a strict RAG answer auditor. Return only schema-valid JSON.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "input_text", "text": _cross_check_prompt(question, answer, hits)}]},
        ],
        "reasoning": {"effort": str(settings.gpt_pro_cross_check_reasoning_effort or "high")},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "rag_cross_check",
                "schema": CROSS_CHECK_SCHEMA,
                "strict": True,
            }
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + settings.openai_api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=max(1, int(settings.gpt_pro_cross_check_timeout_seconds or 600))) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = _extract_response_text(body)
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("cross-check response was not a JSON object")
        return _normalize_result(parsed, model=model)
    except Exception as exc:
        return {
            "status": "error",
            "provider": "openai",
            "model": model,
            "error": f"{type(exc).__name__}: {exc}",
        }
