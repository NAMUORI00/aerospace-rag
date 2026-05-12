from __future__ import annotations

import unittest
from unittest.mock import patch

from aerospace_rag.config import Settings
from aerospace_rag.generation import providers as provider_module
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

    def test_vllm_provider_uses_configured_gemma_model_without_http_server(self) -> None:
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
        observed: dict[str, object] = {}

        def fake_generate(
            messages: list[dict[str, str]],
            *,
            settings: Settings,
            max_tokens: int,
            json_schema: dict[str, object] | None = None,
        ) -> str:
            observed["messages"] = messages
            observed["model"] = settings.llm_model
            observed["max_tokens"] = max_tokens
            observed["json_schema"] = json_schema
            return "vllm answer"

        with patch.object(provider_module, "generate_vllm_chat", side_effect=fake_generate):
            answer = provider_module.generate_answer(
                question="저장영상과 신규촬영 차이는?",
                hits=[hit],
                provider="vllm",
                settings=Settings(
                    llm_model="google/gemma-4-E4B-it",
                    llm_answer_max_tokens=321,
                ),
            )

        self.assertEqual(answer, "vllm answer")
        self.assertEqual(observed["model"], "google/gemma-4-E4B-it")
        self.assertEqual(observed["max_tokens"], 321)
        self.assertIsNone(observed["json_schema"])
        user_prompt = str(observed["messages"][1]["content"])
        self.assertIn("표 데이터:", user_prompt)
        self.assertIn("| EO/K3 | $2,048 | $4,096 |", user_prompt)


if __name__ == "__main__":
    unittest.main()
