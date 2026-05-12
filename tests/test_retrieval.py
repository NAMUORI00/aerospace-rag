from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

from aerospace_rag.config import Settings
from aerospace_rag.generation.providers import route_generation_provider
from aerospace_rag.models import Chunk
from aerospace_rag.retrieval.embeddings import EmbeddingService
from aerospace_rag.retrieval import extraction as extraction_module
from aerospace_rag.retrieval.extraction import KnowledgeExtractor
from aerospace_rag.retrieval.fusion import ChannelHit, weighted_rrf
from aerospace_rag.retrieval.weights import resolve_channel_weights
from aerospace_rag.stores.graph import GraphStore
from aerospace_rag.stores.vector import QdrantVectorStore


class RetrievalTests(unittest.TestCase):
    def tearDown(self) -> None:
        EmbeddingService._MODEL_CACHE.clear()

    def test_generation_provider_rejects_non_core_provider_aliases(self) -> None:
        settings = Settings(llm_provider="transformers")

        self.assertEqual(route_generation_provider(None, settings=settings), "transformers")
        self.assertEqual(route_generation_provider("extractive", settings=settings), "extractive")
        self.assertEqual(route_generation_provider("transformers", settings=settings), "transformers")
        for provider in ["local", "openai_compatible", "vllm", "remote", "server_api"]:
            with self.assertRaisesRegex(ValueError, "provider"):
                route_generation_provider(provider, settings=settings)

    def test_weighted_rrf_uses_runtime_profile_weights_and_evidence_adjustment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "fusion_weights.runtime.json"
            meta = Path(tmp) / "fusion_profile_meta.runtime.json"
            profile.write_text(
                json.dumps(
                    {
                        "profile_id": "ignored-profile",
                        "default_candidate_depth_selected": 42,
                        "default": {
                            "vector_dense_text": 0.2,
                            "vector_sparse": 0.7,
                            "vector_image": 0.0,
                            "graph": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            meta.write_text(
                json.dumps({"selection_run_type": "main", "fusion_profile_id": "ignored-profile"}),
                encoding="utf-8",
            )

            weights, diagnostics = resolve_channel_weights(
                "Momentus solar sail contract",
                profile_path=profile,
                profile_meta_path=meta,
                channel_hit_counts={"vector_dense_text": 2, "vector_sparse": 2, "graph": 0},
            )
            ranked, debug = weighted_rrf(
                weights=weights,
                channel_hits={
                    "vector_dense_text": [ChannelHit("dense-doc", 0.9), ChannelHit("shared-doc", 0.8)],
                    "vector_sparse": [ChannelHit("sparse-doc", 0.95), ChannelHit("shared-doc", 0.7)],
                    "graph": [ChannelHit("graph-doc", 1.0)],
                },
                limit=3,
                return_debug=True,
            )

        self.assertEqual(diagnostics["weights_source"], "runtime_profile")
        self.assertEqual(diagnostics["fusion_profile_id"], "ignored-profile")
        self.assertEqual(diagnostics["candidate_depth"], 42)
        self.assertIn("graph_no_evidence", diagnostics["evidence_adjustments"])
        self.assertEqual(weights["graph"], 0.0)
        self.assertEqual(ranked[0].chunk_id, "shared-doc")
        self.assertIn("top_doc_channel_contributions", debug)
        self.assertLess(
            debug["top_doc_channel_contributions"]["shared-doc"]["vector_dense_text"],
            debug["top_doc_channel_contributions"]["shared-doc"]["vector_sparse"],
        )

    def test_weight_resolver_rejects_non_main_runtime_profile_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "fusion_weights.runtime.json"
            meta = Path(tmp) / "fusion_profile_meta.runtime.json"
            profile.write_text(
                json.dumps(
                    {
                        "profile_id": "draft-profile",
                        "default": {
                            "vector_dense_text": 0.2,
                            "vector_sparse": 0.7,
                            "vector_image": 0.0,
                            "graph": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            meta.write_text(json.dumps({"selection_run_type": "draft"}), encoding="utf-8")

            weights, diagnostics = resolve_channel_weights(
                "Momentus solar sail contract",
                profile_path=profile,
                profile_meta_path=meta,
            )

        self.assertEqual(diagnostics["weights_source"], "static")
        self.assertIn("profile_invalid", diagnostics["profile_fallback_reasons"][0])
        self.assertNotEqual(weights["bm25"], 0.7)

    def test_vector_store_requires_qdrant_unless_json_debug_mode_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {"qdrant_client": None}):
                with self.assertRaisesRegex(RuntimeError, "qdrant-client"):
                    QdrantVectorStore(Path(tmp) / "index", settings=Settings(embed_backend="hash"))

    def test_vector_store_explicit_json_debug_mode_has_dense_sparse_image_channels_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(embed_backend="hash", embed_dim=384, vector_backend="json")
            store = QdrantVectorStore(Path(tmp) / "index", settings=settings)
            chunk = Chunk(
                chunk_id="doc#1",
                text="NASA Momentus solar sail demonstration",
                source_file="doc.pdf",
                modality="image",
                metadata={"asset_ref": "page:1#image:0"},
            )

            store.build([chunk])
            channels = store.search_channels("NASA solar sail", limit=5)
            payload = json.loads(store.json_path.read_text(encoding="utf-8"))[0]["payload"]

        self.assertIn("vector_dense_text", channels)
        self.assertIn("vector_sparse", channels)
        self.assertIn("vector_image", channels)
        self.assertEqual(payload["tier"], "public")
        self.assertEqual(payload["canonical_doc_id"], "doc.pdf")
        self.assertEqual(payload["canonical_chunk_id"], "doc#1")
        self.assertEqual(payload["modality"], "image")

    def test_sparse_vector_hash_collisions_keep_unique_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(embed_backend="hash", embed_dim=384, vector_backend="json")
            store = QdrantVectorStore(Path(tmp) / "index", settings=settings)

            sparse = store._sparse_vector("token76 token210")

        self.assertEqual(len(sparse["indices"]), len(set(sparse["indices"])))

    def test_embedding_service_reuses_sentence_transformer_instance(self) -> None:
        class FakeModel:
            def get_embedding_dimension(self) -> int:
                return 384

        fake_model = FakeModel()
        calls: list[str] = []

        def fake_sentence_transformer(model_name: str) -> FakeModel:
            calls.append(model_name)
            return fake_model

        fake_module = types.SimpleNamespace(SentenceTransformer=fake_sentence_transformer)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            first = EmbeddingService(Settings(embed_model="BAAI/bge-m3"))
            second = EmbeddingService(Settings(embed_model="BAAI/bge-m3"))

        self.assertIs(first._model, second._model)
        self.assertEqual(calls, ["BAAI/bge-m3"])

    def test_embedding_service_prefers_new_dimension_api_without_deprecation_call(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.deprecated_called = False

            def get_embedding_dimension(self) -> int:
                return 256

            def get_sentence_embedding_dimension(self) -> int:
                self.deprecated_called = True
                raise AssertionError("deprecated dimension API should not be used")

        fake_model = FakeModel()
        fake_module = types.SimpleNamespace(SentenceTransformer=lambda _model_name: fake_model)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            service = EmbeddingService(Settings(embed_model="custom-model"))

        self.assertEqual(service.vector_size, 256)
        self.assertFalse(fake_model.deprecated_called)

    def test_graph_store_uses_graph_lite_index_without_graph_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = GraphStore(Path(tmp) / "index", settings=Settings(extractor_backend="local_fallback"))
            graph.build(
                [
                    Chunk("public#1", "NASA awarded Momentus a solar sail contract.", "public.pdf", "text"),
                ]
            )

            hits = graph.search("NASA solar sail", limit=5)

        self.assertEqual(graph.index_path.name, "graph_index.json")
        self.assertEqual(graph.index_path.parent.name, "graph")
        self.assertFalse(graph.db_path.exists())
        self.assertIn("public#1", {chunk_id for chunk_id, _ in hits})

    def test_transformers_extractor_accepts_fenced_json_response(self) -> None:
        chunk = Chunk("extract#1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")

        content = """```json
{
  "entities": [
    {"canonical_id": "ENT_001", "text": "NASA", "type": "Agency", "confidence": 1.0},
    {"canonical_id": "ENT_002", "text": "Momentus", "type": "Company", "confidence": 1.0}
  ],
  "relations": [
    {"source": "ENT_001", "target": "ENT_002", "type": "AWARDED_CONTRACT_TO", "confidence": 1.0, "evidence": "memo"}
  ]
}
```"""

        with patch.object(extraction_module, "generate_transformers_chat", return_value=content):
            result = KnowledgeExtractor(
                settings=Settings(
                    extractor_backend="transformers",
                )
            ).extract(chunk)

        self.assertEqual([entity.text for entity in result.entities], ["NASA", "Momentus"])
        self.assertEqual(result.relations[0].type, "AWARDED_CONTRACT_TO")

    def test_transformers_extractor_raises_after_generation_failure(self) -> None:
        chunk = Chunk("extract#1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")

        with patch.object(extraction_module, "generate_transformers_chat", side_effect=TimeoutError("timed out")):
            with self.assertRaisesRegex(RuntimeError, "Transformers knowledge extraction failed"):
                KnowledgeExtractor(
                    settings=Settings(
                        extractor_backend="transformers",
                        knowledge_extract_retries=0,
                    )
                ).extract(chunk)

    def test_transformers_extractor_uses_configured_model_and_limits(self) -> None:
        chunk = Chunk("extract#1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")
        observed: dict[str, object] = {}

        def fake_generate(
            messages: list[dict[str, str]],
            *,
            settings: Settings,
            max_new_tokens: int,
            max_time: int,
        ) -> str:
            observed["messages"] = messages
            observed["model"] = settings.transformers_model
            observed["max_new_tokens"] = max_new_tokens
            observed["max_time"] = max_time
            return json.dumps(
                {
                    "entities": [{"canonical_id": "NASA", "text": "NASA", "type": "Agency"}],
                    "relations": [],
                }
            )

        with patch.object(extraction_module, "generate_transformers_chat", side_effect=fake_generate):
            result = extraction_module.KnowledgeExtractor(
                settings=Settings(
                    extractor_backend="transformers",
                    transformers_model="google/gemma-4-E4B-it",
                    transformers_extract_num_predict=222,
                    transformers_extract_timeout_seconds=33,
                )
            ).extract(chunk)

        self.assertEqual([entity.text for entity in result.entities], ["NASA"])
        self.assertEqual(observed["model"], "google/gemma-4-E4B-it")
        self.assertEqual(observed["max_new_tokens"], 222)
        self.assertEqual(observed["max_time"], 33)
        self.assertIn("JSON Schema", observed["messages"][0]["content"])

    def test_transformers_extractor_accepts_json_with_trailing_model_text(self) -> None:
        chunk = Chunk("extract#1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")
        first_json = json.dumps(
            {
                "entities": [
                    {"canonical_id": "NASA", "text": "NASA", "type": "Agency", "confidence": 1.0},
                    {"canonical_id": "Momentus", "text": "Momentus", "type": "Company", "confidence": 1.0},
                ],
                "relations": [
                    {
                        "source": "NASA",
                        "target": "Momentus",
                        "type": "AWARDED_CONTRACT_TO",
                        "confidence": 1.0,
                        "evidence": "solar sail contract",
                    }
                ],
            }
        )

        with patch.object(
            extraction_module,
            "generate_transformers_chat",
            return_value=first_json + "\n" + json.dumps({"note": "extra object"}),
        ):
            result = extraction_module.KnowledgeExtractor(
                settings=Settings(extractor_backend="transformers")
            ).extract(chunk)

        self.assertEqual([entity.text for entity in result.entities], ["NASA", "Momentus"])
        self.assertEqual(result.relations[0].type, "AWARDED_CONTRACT_TO")


if __name__ == "__main__":
    unittest.main()
