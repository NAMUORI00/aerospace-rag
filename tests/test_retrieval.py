from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from aerospace_rag.config import Settings
from aerospace_rag.generation.providers import route_generation_provider
from aerospace_rag.models import Chunk
from aerospace_rag.retrieval.embeddings import EmbeddingService
from aerospace_rag.retrieval import extraction as extraction_module
from aerospace_rag.retrieval.extraction import KnowledgeExtractor, parse_llm_json_object
from aerospace_rag.retrieval.fusion import ChannelHit, weighted_rrf
from aerospace_rag.retrieval.weights import resolve_channel_weights
from aerospace_rag.stores.graph import GraphStore
from aerospace_rag.stores.vector import QdrantVectorStore


class RetrievalTests(unittest.TestCase):
    def tearDown(self) -> None:
        EmbeddingService._MODEL_CACHE.clear()

    def test_generation_provider_rejects_non_core_provider_aliases(self) -> None:
        settings = Settings(llm_provider="vllm")

        self.assertEqual(route_generation_provider(None, settings=settings), "vllm")
        self.assertEqual(route_generation_provider("extractive", settings=settings), "extractive")
        self.assertEqual(route_generation_provider("vllm", settings=settings), "vllm")
        for provider in ["local", "openai_compatible", "remote", "server_api"]:
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

    def test_embedding_service_uses_configured_hash_dimension(self) -> None:
        service = EmbeddingService(Settings(embed_backend="hash", embed_dim=128))

        vector = service.embed_text("NASA solar sail")

        self.assertEqual(service.provider_name, "hash")
        self.assertEqual(service.vector_size, 128)
        self.assertEqual(len(vector), 128)

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

    def test_vllm_extractor_accepts_fenced_json_response(self) -> None:
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

        with patch.object(extraction_module, "generate_vllm_chat", return_value=content):
            result = KnowledgeExtractor(
                settings=Settings(
                    extractor_backend="vllm",
                )
            ).extract(chunk)

        self.assertEqual([entity.text for entity in result.entities], ["NASA", "Momentus"])
        self.assertEqual(result.relations[0].type, "AWARDED_CONTRACT_TO")

    def test_vllm_extractor_raises_after_generation_failure(self) -> None:
        chunk = Chunk("extract#1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")

        with patch.object(extraction_module, "generate_vllm_chat", side_effect=TimeoutError("timed out")):
            with self.assertRaisesRegex(RuntimeError, "vLLM knowledge extraction failed"):
                KnowledgeExtractor(
                    settings=Settings(
                        extractor_backend="vllm",
                        knowledge_extract_retries=0,
                        vllm_fallback_on_error=False,
                    )
                ).extract(chunk)

    def test_vllm_extractor_falls_back_to_local_debug_after_generation_failure(self) -> None:
        chunk = Chunk("extract#1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")

        with patch.object(extraction_module, "generate_vllm_chat", side_effect=TimeoutError("timed out")):
            result = KnowledgeExtractor(
                settings=Settings(
                    extractor_backend="vllm",
                    knowledge_extract_retries=0,
                    vllm_fallback_on_error=True,
                )
            ).extract(chunk)

        self.assertIn("NASA", [entity.text for entity in result.entities])

    def test_vllm_extractor_uses_configured_model_and_limits(self) -> None:
        chunk = Chunk("extract#1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")
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
            return json.dumps(
                {
                    "entities": [{"canonical_id": "NASA", "text": "NASA", "type": "Agency"}],
                    "relations": [],
                }
            )

        with patch.object(extraction_module, "generate_vllm_chat", side_effect=fake_generate):
            result = extraction_module.KnowledgeExtractor(
                settings=Settings(
                    extractor_backend="vllm",
                    llm_model="google/gemma-4-E4B-it",
                    llm_extract_max_tokens=222,
                )
            ).extract(chunk)

        self.assertEqual([entity.text for entity in result.entities], ["NASA"])
        self.assertEqual(observed["model"], "google/gemma-4-E4B-it")
        self.assertEqual(observed["max_tokens"], 222)
        self.assertIsNotNone(observed["json_schema"])
        self.assertIn("JSON Schema", observed["messages"][0]["content"])

    def test_vllm_extractor_accepts_json_with_trailing_model_text(self) -> None:
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
            "generate_vllm_chat",
            return_value=first_json + "\n" + json.dumps({"note": "extra object"}),
        ):
            result = extraction_module.KnowledgeExtractor(
                settings=Settings(extractor_backend="vllm")
            ).extract(chunk)

        self.assertEqual([entity.text for entity in result.entities], ["NASA", "Momentus"])
        self.assertEqual(result.relations[0].type, "AWARDED_CONTRACT_TO")

    def test_parse_llm_json_object_repairs_missing_comma_between_array_objects(self) -> None:
        parsed = parse_llm_json_object(
            '{"entities":[{"canonical_id":"NASA","text":"NASA","type":"Agency","confidence":1.0} '
            '{"canonical_id":"Momentus","text":"Momentus","type":"Company","confidence":1.0}],'
            '"relations":[]}'
        )

        self.assertEqual([item["text"] for item in parsed["entities"]], ["NASA", "Momentus"])

    def test_vllm_extractor_repairs_malformed_json_with_same_model(self) -> None:
        chunk = Chunk("extract#1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")
        malformed = '{"entities":[{"canonical_id":"NASA","text":"NASA","type":"Agency","confidence":}], "relations":[]}'
        repaired = json.dumps(
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

        with patch.object(extraction_module, "generate_vllm_chat", side_effect=[malformed, repaired]) as generate:
            result = extraction_module.KnowledgeExtractor(
                settings=Settings(
                    extractor_backend="vllm",
                    knowledge_extract_retries=0,
                    knowledge_extract_repair_retries=1,
                )
            ).extract(chunk)

        self.assertEqual(generate.call_count, 2)
        self.assertIn("Repair", generate.call_args_list[1].args[0][1]["content"])
        self.assertEqual([entity.text for entity in result.entities], ["NASA", "Momentus"])
        self.assertEqual(result.relations[0].type, "AWARDED_CONTRACT_TO")


if __name__ == "__main__":
    unittest.main()
