from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aerospace_rag.config import Settings
from aerospace_rag.embeddings import EmbeddingService
from aerospace_rag.models import Chunk, RetrievalHit
from aerospace_rag.providers import generate_answer
from aerospace_rag.runtime_dat import resolve_channel_weights


class RuntimeContractTests(unittest.TestCase):
    def test_settings_default_to_smartfarm_like_models_and_ollama_gemma4_e2b(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.embed_model, "BAAI/bge-m3")
        self.assertEqual(settings.llm_provider, "ollama")
        self.assertEqual(settings.ollama_model, "gemma4:e2b")
        self.assertEqual(settings.ollama_base_url, "http://127.0.0.1:11434")
        self.assertEqual(settings.dat_mode, "hybrid")

    def test_embedding_service_can_fall_back_without_changing_callers(self) -> None:
        settings = Settings(embed_backend="hash", embed_dim=384)
        service = EmbeddingService(settings)

        vectors = service.embed_texts(["위성영상 가격", "H3 발사 실패"])

        self.assertEqual(service.provider_name, "hash")
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 384)

    def test_dat_resolver_downweights_graph_for_non_entity_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-profile.json"

            weights, diagnostics = resolve_channel_weights(
                "요약해줘",
                profile_path=path,
                mode="hybrid",
            )

        self.assertLess(weights["graph"], weights["qdrant"])
        self.assertEqual(diagnostics["weights_source"], "default_fallback")
        self.assertIn("query_segment", diagnostics)

    def test_ollama_provider_falls_back_to_extractive_when_server_unavailable(self) -> None:
        chunk = Chunk(
            chunk_id="c1",
            text="Ollama gemma4:e2b는 Colab에서 임시 LLM으로 사용할 수 있다.",
            source_file="source.md",
            modality="text",
        )
        hit = RetrievalHit(chunk=chunk, score=1.0, channels={"qdrant": 1.0})

        answer = generate_answer(
            "Ollama 기본 모델은?",
            [hit],
            provider="ollama",
            settings=Settings(ollama_base_url="http://127.0.0.1:1", ollama_model="gemma4:e2b"),
        )

        self.assertIn("gemma4:e2b", answer)


if __name__ == "__main__":
    unittest.main()
