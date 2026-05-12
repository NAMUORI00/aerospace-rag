from __future__ import annotations

import unittest
from unittest.mock import patch

from aerospace_rag.config import Settings
from aerospace_rag.generation import vllm_backend
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

    def test_vllm_engine_initialization_uses_real_stdout_when_kernel_stdout_has_no_fileno(self) -> None:
        calls: dict[str, object] = {}

        class FakeStdout:
            def fileno(self) -> int:
                raise OSError("fileno")

        class FakeLLM:
            def __init__(self, **kwargs: object) -> None:
                import sys

                calls["kwargs"] = kwargs
                calls["stdout_is_real"] = sys.stdout is sys.__stdout__

        with patch.dict("sys.modules", {"vllm": type("FakeVllm", (), {"LLM": FakeLLM})}):
            with patch("sys.stdout", FakeStdout()):
                vllm_backend._ENGINE_CACHE.clear()
                vllm_backend._load_vllm_engine(Settings())

        self.assertTrue(calls["stdout_is_real"])
        self.assertEqual(calls["kwargs"]["model"], "google/gemma-4-E4B-it")
        self.assertNotIn("quantization", calls["kwargs"])
        self.assertEqual(calls["kwargs"]["load_format"], "auto")
        self.assertEqual(calls["kwargs"]["max_model_len"], 2048)
        self.assertEqual(calls["kwargs"]["cpu_offload_gb"], 4.0)
        self.assertEqual(calls["kwargs"]["gpu_memory_utilization"], 0.82)
        self.assertTrue(calls["kwargs"]["enforce_eager"])

    def test_vllm_engine_initialization_can_disable_quantized_loading(self) -> None:
        calls: dict[str, object] = {}

        class FakeLLM:
            def __init__(self, **kwargs: object) -> None:
                calls["kwargs"] = kwargs

        with patch.dict("sys.modules", {"vllm": type("FakeVllm", (), {"LLM": FakeLLM})}):
            vllm_backend._ENGINE_CACHE.clear()
            vllm_backend._load_vllm_engine(
                Settings(
                    vllm_quantization="",
                    vllm_load_format="auto",
                )
            )

        self.assertNotIn("quantization", calls["kwargs"])
        self.assertEqual(calls["kwargs"]["load_format"], "auto")

    def test_vllm_engine_initialization_can_use_quantized_loading(self) -> None:
        calls: dict[str, object] = {}

        class FakeLLM:
            def __init__(self, **kwargs: object) -> None:
                calls["kwargs"] = kwargs

        with patch.dict("sys.modules", {"vllm": type("FakeVllm", (), {"LLM": FakeLLM})}):
            vllm_backend._ENGINE_CACHE.clear()
            vllm_backend._load_vllm_engine(
                Settings(
                    vllm_quantization="awq",
                    vllm_load_format="auto",
                    vllm_enforce_eager=False,
                )
            )

        self.assertEqual(calls["kwargs"]["quantization"], "awq")
        self.assertEqual(calls["kwargs"]["load_format"], "auto")
        self.assertFalse(calls["kwargs"]["enforce_eager"])


if __name__ == "__main__":
    unittest.main()
