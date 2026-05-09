from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from aerospace_rag.config import Settings
from aerospace_rag.generation.cross_check import run_gpt_pro_cross_check
from aerospace_rag.generation.providers import generate_answer
from aerospace_rag.models import Chunk, RetrievalHit


class ProviderTests(unittest.TestCase):
    def test_extractive_provider_builds_answer_from_ranked_sources(self) -> None:
        chunk = Chunk(
            chunk_id="source#1",
            text="H3 8호기는 2단 엔진 재점화 후 조기 종료되어 궤도 투입에 실패했다.",
            source_file="251222_H3 8호기 발사 경과.pdf",
            modality="text",
            page=1,
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, channels={"bm25": 1.0})

        answer = generate_answer(
            question="H3 8호기 발사 결과는?",
            hits=[hit],
            provider="extractive",
        )

        self.assertIn("2단 엔진", answer)
        self.assertIn("251222_H3 8호기 발사 경과.pdf", answer)

    def test_ollama_provider_sends_api_key_when_configured(self) -> None:
        chunk = Chunk(
            chunk_id="source#1",
            text="NASA awarded Momentus a solar sail demonstration study contract.",
            source_file="source.md",
            modality="text",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, channels={"qdrant": 1.0})
        captured_headers: dict[str, str] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"message":{"content":"cloud answer"}}'

        def fake_urlopen(request: object, timeout: int) -> FakeResponse:
            _ = timeout
            captured_headers.update(dict(getattr(request, "headers", {})))
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            answer = generate_answer(
                question="What did NASA award?",
                hits=[hit],
                provider="ollama",
                settings=Settings(
                    ollama_base_url="https://ollama.com",
                    ollama_model="gemma431bcloud",
                    ollama_api_key="test-token",
                ),
            )

        self.assertEqual(answer, "cloud answer")
        self.assertEqual(captured_headers["Authorization"], "Bearer test-token")

    def test_ollama_provider_formats_table_context_with_preserved_rows(self) -> None:
        chunk = Chunk(
            chunk_id="table#1",
            text=(
                "| 구분 | 저장영상(AO) | 신규촬영(NTO) |\n"
                "| --- | --- | --- |\n"
                "| EO/K3 | $2,048 | $4,096 |"
            ),
            source_file="위성영상가격.png",
            modality="table",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, channels={"qdrant": 1.0})
        captured_payload: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"message":{"content":"ok"}}'

        def fake_urlopen(request: object, timeout: int) -> FakeResponse:
            _ = timeout
            body = getattr(request, "data", b"")
            captured_payload.update(json.loads(body.decode("utf-8")))
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            answer = generate_answer(
                question="저장영상과 신규촬영 차이는?",
                hits=[hit],
                provider="ollama",
                settings=Settings(
                    ollama_base_url="https://ollama.com",
                    ollama_model="gemma431bcloud",
                ),
            )

        self.assertEqual(answer, "ok")
        user_prompt = str(captured_payload["messages"][1]["content"])
        self.assertIn("표 데이터:", user_prompt)
        self.assertIn("| EO/K3 | $2,048 | $4,096 |", user_prompt)
        self.assertIn("열 순서를 유지", user_prompt)

    def test_gpt_pro_cross_check_posts_responses_request_and_parses_json(self) -> None:
        chunk = Chunk(
            chunk_id="source#1",
            text="NASA awarded Momentus a solar sail demonstration study contract.",
            source_file="source.md",
            modality="text",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, channels={"qdrant": 1.0})
        observed: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                content = json.dumps(
                    {
                        "verdict": "supported",
                        "confidence": 0.94,
                        "issues": [],
                        "missing_evidence": [],
                        "source_alignment": [{"source_file": "source.md", "supports_answer": True, "note": "Directly supported."}],
                        "suggested_fix": "",
                    }
                )
                return json.dumps({"output_text": content}).encode("utf-8")

        def fake_urlopen(request: object, timeout: int) -> FakeResponse:
            observed["timeout"] = timeout
            observed["url"] = request.full_url  # type: ignore[attr-defined]
            observed["headers"] = dict(getattr(request, "headers", {}))
            observed["payload"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = run_gpt_pro_cross_check(
                "What did NASA award?",
                "NASA awarded Momentus a solar sail demonstration study contract.",
                [hit],
                settings=Settings(
                    openai_api_key="test-token",
                    gpt_pro_cross_check_enabled=True,
                    gpt_pro_cross_check_model="gpt-5.5-pro",
                    gpt_pro_cross_check_reasoning_effort="xhigh",
                    gpt_pro_cross_check_timeout_seconds=900,
                ),
            )

        payload = observed["payload"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(result["model"], "gpt-5.5-pro")
        self.assertEqual(observed["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(observed["timeout"], 900)
        self.assertEqual(observed["headers"]["Authorization"], "Bearer test-token")
        self.assertEqual(payload["model"], "gpt-5.5-pro")
        self.assertEqual(payload["reasoning"]["effort"], "xhigh")
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")

    def test_gpt_pro_cross_check_skips_without_api_key(self) -> None:
        result = run_gpt_pro_cross_check(
            "question",
            "answer",
            [],
            settings=Settings(gpt_pro_cross_check_enabled=True),
        )

        self.assertEqual(result["status"], "skipped")
        self.assertIn("OPENAI_API_KEY", result["reason"])


if __name__ == "__main__":
    unittest.main()
