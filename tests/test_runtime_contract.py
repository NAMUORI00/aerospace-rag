from __future__ import annotations

import os
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
    def test_settings_default_to_vllm_gemma_e4b(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.embed_backend, "hash")
        self.assertEqual(settings.embed_model, "hash")
        self.assertEqual(settings.embed_dim, 384)
        self.assertEqual(settings.vector_backend, "qdrant")
        self.assertEqual(settings.llm_provider, "vllm")
        self.assertEqual(settings.llm_model, "google/gemma-4-E4B-it")
        self.assertEqual(settings.vllm_dtype, "auto")
        self.assertEqual(settings.vllm_max_model_len, 4096)
        self.assertEqual(settings.llm_answer_max_tokens, 1024)
        self.assertEqual(settings.llm_extract_max_tokens, 768)
        self.assertEqual(settings.knowledge_extract_retries, 1)
        self.assertEqual(settings.knowledge_extract_repair_retries, 1)
        self.assertEqual(settings.knowledge_extract_max_chars, 1200)
        self.assertEqual(settings.extractor_backend, "vllm")
        self.assertFalse(hasattr(settings, "extractor_fallback_on_error"))
        self.assertFalse(hasattr(settings, "runtime_profile_mode"))

        with patch.dict(os.environ, {"LLM_PROVIDER": "extractive"}, clear=True):
            self.assertEqual(Settings.from_env().llm_provider, "extractive")

    def test_settings_read_model_runtime_controls_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EXTRACTOR_LLM_BACKEND": "vllm",
                "AEROSPACE_LLM_MODEL": "google/gemma-4-E4B-it",
                "AEROSPACE_VLLM_DTYPE": "float16",
                "AEROSPACE_VLLM_GPU_MEMORY_UTILIZATION": "0.82",
                "AEROSPACE_VLLM_MAX_MODEL_LEN": "2048",
                "AEROSPACE_VLLM_TRUST_REMOTE_CODE": "true",
                "LLM_ANSWER_MAX_TOKENS": "321",
                "LLM_EXTRACT_MAX_TOKENS": "222",
                "KNOWLEDGE_EXTRACT_RETRIES": "2",
                "KNOWLEDGE_EXTRACT_REPAIR_RETRIES": "3",
                "KNOWLEDGE_EXTRACT_MAX_CHARS": "2500",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.llm_provider, "vllm")
        self.assertEqual(settings.extractor_backend, "vllm")
        self.assertEqual(settings.llm_model, "google/gemma-4-E4B-it")
        self.assertEqual(settings.vllm_dtype, "float16")
        self.assertEqual(settings.vllm_gpu_memory_utilization, 0.82)
        self.assertEqual(settings.vllm_max_model_len, 2048)
        self.assertTrue(settings.vllm_trust_remote_code)
        self.assertEqual(settings.llm_answer_max_tokens, 321)
        self.assertEqual(settings.llm_extract_max_tokens, 222)
        self.assertEqual(settings.knowledge_extract_retries, 2)
        self.assertEqual(settings.knowledge_extract_repair_retries, 3)
        self.assertEqual(settings.knowledge_extract_max_chars, 2500)

    def test_settings_ignore_vllm_reserved_environment_names(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VLLM_MODEL": "reserved-name-should-not-drive-app-config",
                "VLLM_DTYPE": "reserved-dtype",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.12",
                "VLLM_MAX_MODEL_LEN": "128",
                "VLLM_TRUST_REMOTE_CODE": "true",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.llm_model, "google/gemma-4-E4B-it")
        self.assertEqual(settings.vllm_dtype, "auto")
        self.assertEqual(settings.vllm_gpu_memory_utilization, 0.90)
        self.assertEqual(settings.vllm_max_model_len, 4096)
        self.assertFalse(settings.vllm_trust_remote_code)

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
