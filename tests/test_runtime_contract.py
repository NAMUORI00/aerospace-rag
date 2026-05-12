from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aerospace_rag.config import Settings
from aerospace_rag.models import Chunk
from aerospace_rag.retrieval.embeddings import EmbeddingService
from aerospace_rag.retrieval.extraction import KnowledgeExtractor
from aerospace_rag.retrieval.weights import resolve_channel_weights


class RuntimeContractTests(unittest.TestCase):
    def test_settings_default_to_core_models_and_transformers_e4b(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.embed_model, "BAAI/bge-m3")
        self.assertEqual(settings.vector_backend, "qdrant")
        self.assertEqual(settings.llm_provider, "transformers")
        self.assertEqual(settings.transformers_model, "google/gemma-4-E4B-it")
        self.assertEqual(settings.knowledge_extract_retries, 1)
        self.assertEqual(settings.knowledge_extract_max_chars, 1200)
        self.assertEqual(settings.extractor_backend, "transformers")
        self.assertFalse(hasattr(settings, "extractor_fallback_on_error"))
        self.assertFalse(hasattr(settings, "runtime_profile_mode"))

        with patch.dict(os.environ, {"LLM_PROVIDER": "extractive"}, clear=True):
            self.assertEqual(Settings.from_env().llm_provider, "extractive")

    def test_settings_read_model_runtime_controls_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EXTRACTOR_LLM_BACKEND": "transformers",
                "TRANSFORMERS_MODEL": "google/gemma-4-E4B-it",
                "TRANSFORMERS_GENERATE_TIMEOUT_SECONDS": "45",
                "TRANSFORMERS_EXTRACT_TIMEOUT_SECONDS": "33",
                "TRANSFORMERS_ANSWER_NUM_PREDICT": "321",
                "TRANSFORMERS_EXTRACT_NUM_PREDICT": "222",
                "TRANSFORMERS_LOAD_IN_4BIT": "true",
                "KNOWLEDGE_EXTRACT_RETRIES": "2",
                "KNOWLEDGE_EXTRACT_MAX_CHARS": "2500",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.llm_provider, "transformers")
        self.assertEqual(settings.extractor_backend, "transformers")
        self.assertEqual(settings.transformers_model, "google/gemma-4-E4B-it")
        self.assertEqual(settings.transformers_generate_timeout_seconds, 45)
        self.assertEqual(settings.transformers_extract_timeout_seconds, 33)
        self.assertEqual(settings.transformers_answer_num_predict, 321)
        self.assertEqual(settings.transformers_extract_num_predict, 222)
        self.assertTrue(settings.transformers_load_in_4bit)
        self.assertEqual(settings.knowledge_extract_retries, 2)
        self.assertEqual(settings.knowledge_extract_max_chars, 2500)

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

    def test_knowledge_extractor_rejects_removed_server_backend(self) -> None:
        chunk = Chunk("extract#1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")

        with self.assertRaisesRegex(ValueError, "EXTRACTOR_LLM_BACKEND"):
            KnowledgeExtractor(
                settings=Settings(
                    extractor_backend="server_api",
                )
            ).extract(chunk)


if __name__ == "__main__":
    unittest.main()
