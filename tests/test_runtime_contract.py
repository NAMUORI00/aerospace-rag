from __future__ import annotations

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aerospace_rag.config import Settings
from aerospace_rag.generation.providers import generate_answer
from aerospace_rag.models import Chunk, RetrievalHit
from aerospace_rag.retrieval.embeddings import EmbeddingService
from aerospace_rag.retrieval.extraction import KnowledgeExtractor
from aerospace_rag.retrieval.weights import resolve_channel_weights


class RuntimeContractTests(unittest.TestCase):
    def test_settings_default_to_core_models_and_ollama_gemma4_e4b(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.embed_model, "BAAI/bge-m3")
        self.assertEqual(settings.vector_backend, "qdrant")
        self.assertEqual(settings.llm_provider, "ollama")
        self.assertEqual(settings.ollama_model, "gemma4:e4b")
        self.assertEqual(settings.ollama_base_url, "http://127.0.0.1:11434")
        self.assertEqual(settings.ollama_api_key, "")
        self.assertEqual(settings.ollama_extract_timeout_seconds, 3600)
        self.assertEqual(settings.ollama_generate_timeout_seconds, 3600)
        self.assertEqual(settings.ollama_keep_alive, "10m")
        self.assertEqual(settings.ollama_extract_repair_retries, 1)
        self.assertEqual(settings.ollama_extract_num_predict, 4096)
        self.assertEqual(settings.ollama_extract_max_chars, 1200)
        self.assertEqual(settings.extractor_provider, "ollama")
        self.assertFalse(hasattr(settings, "extractor_fallback_on_error"))
        self.assertFalse(hasattr(settings, "runtime_profile_mode"))

        with patch.dict(os.environ, {"LLM_PROVIDER": "extractive"}, clear=True):
            self.assertEqual(Settings.from_env().llm_provider, "extractive")

        with patch.dict(os.environ, {"OLLAMA_API_KEY": "test-token"}, clear=True):
            self.assertEqual(Settings.from_env().ollama_api_key, "test-token")

    def test_settings_read_ollama_timeout_controls_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OLLAMA_EXTRACT_TIMEOUT_SECONDS": "420",
                "OLLAMA_GENERATE_TIMEOUT_SECONDS": "360",
                "OLLAMA_KEEP_ALIVE": "15m",
                "OLLAMA_EXTRACT_RETRIES": "2",
                "OLLAMA_EXTRACT_REPAIR_RETRIES": "3",
                "OLLAMA_EXTRACT_NUM_PREDICT": "768",
                "OLLAMA_ANSWER_NUM_PREDICT": "1200",
                "OLLAMA_EXTRACT_MAX_CHARS": "2500",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.ollama_extract_timeout_seconds, 420)
        self.assertEqual(settings.ollama_generate_timeout_seconds, 360)
        self.assertEqual(settings.ollama_keep_alive, "15m")
        self.assertEqual(settings.ollama_extract_retries, 2)
        self.assertEqual(settings.ollama_extract_repair_retries, 3)
        self.assertEqual(settings.ollama_extract_num_predict, 768)
        self.assertEqual(settings.ollama_answer_num_predict, 1200)
        self.assertEqual(settings.ollama_extract_max_chars, 2500)

    def test_settings_read_transformers_provider_controls_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "transformers",
                "EXTRACTOR_LLM_BACKEND": "transformers",
                "TRANSFORMERS_MODEL": "google/gemma-4-E4B-it",
                "TRANSFORMERS_GENERATE_TIMEOUT_SECONDS": "45",
                "TRANSFORMERS_EXTRACT_TIMEOUT_SECONDS": "33",
                "TRANSFORMERS_ANSWER_NUM_PREDICT": "321",
                "TRANSFORMERS_EXTRACT_NUM_PREDICT": "222",
                "TRANSFORMERS_LOAD_IN_4BIT": "true",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.llm_provider, "transformers")
        self.assertEqual(settings.extractor_provider, "transformers")
        self.assertEqual(settings.transformers_model, "google/gemma-4-E4B-it")
        self.assertEqual(settings.transformers_generate_timeout_seconds, 45)
        self.assertEqual(settings.transformers_extract_timeout_seconds, 33)
        self.assertEqual(settings.transformers_answer_num_predict, 321)
        self.assertEqual(settings.transformers_extract_num_predict, 222)
        self.assertTrue(settings.transformers_load_in_4bit)

    def test_embedding_service_requires_sentence_transformers_by_default(self) -> None:
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-models.txt"):
                EmbeddingService(Settings(embed_backend="sentence_transformers"))

    def test_embedding_service_keeps_explicit_hash_debug_mode(self) -> None:
        settings = Settings(embed_backend="hash", embed_dim=384)
        service = EmbeddingService(settings)

        vectors = service.embed_texts(["위성영상 가격", "H3 발사 실패"])

        self.assertEqual(service.provider_name, "hash")
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 384)

    def test_weight_resolver_uses_static_core_weights_without_runtime_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-profile.json"

            weights, diagnostics = resolve_channel_weights(
                "요약해줘",
                profile_path=path,
                mode="hybrid",
            )

        self.assertLess(weights["graph"], weights["qdrant"])
        self.assertEqual(diagnostics["weights_source"], "static")
        self.assertIn("query_segment", diagnostics)

    def test_ollama_provider_raises_when_server_unavailable(self) -> None:
        chunk = Chunk(
            chunk_id="c1",
            text="Ollama gemma4:e4b는 Colab에서 임시 LLM으로 사용할 수 있다.",
            source_file="source.md",
            modality="text",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, channels={"qdrant": 1.0})

        with self.assertRaisesRegex(RuntimeError, "Ollama"):
            generate_answer(
                "Ollama 기본 모델은?",
                [hit],
                provider="ollama",
                settings=Settings(ollama_base_url="http://127.0.0.1:1", ollama_model="gemma4:e4b"),
            )

    def test_ollama_generation_uses_configured_timeout_and_generation_limits(self) -> None:
        chunk = Chunk("c1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")
        hit = RetrievalHit(chunk=chunk, score=1.0, channels={"qdrant": 1.0})
        observed: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"message": {"content": "답변"}}).encode("utf-8")

        def fake_urlopen(req: object, timeout: int) -> FakeResponse:
            observed["timeout"] = timeout
            observed["payload"] = json.loads(req.data.decode("utf-8"))  # type: ignore[attr-defined]
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            answer = generate_answer(
                "계약은?",
                [hit],
                provider="ollama",
                settings=Settings(
                    ollama_generate_timeout_seconds=420,
                    ollama_answer_num_predict=900,
                    ollama_keep_alive="15m",
                ),
            )

        payload = observed["payload"]
        self.assertEqual(answer, "답변")
        self.assertEqual(observed["timeout"], 420)
        self.assertEqual(payload["keep_alive"], "15m")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_predict"], 900)

    def test_ollama_knowledge_extractor_raises_without_explicit_local_debug_mode(self) -> None:
        chunk = Chunk("extract#1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")

        with self.assertRaisesRegex(RuntimeError, "Ollama"):
            KnowledgeExtractor(
                settings=Settings(
                    ollama_base_url="http://127.0.0.1:1",
                    ollama_model="gemma4:e4b",
                    extractor_provider="ollama",
                )
            ).extract(chunk)


if __name__ == "__main__":
    unittest.main()
