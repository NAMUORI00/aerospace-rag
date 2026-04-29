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
from aerospace_rag.retrieval.extraction import KnowledgeExtractor
from aerospace_rag.retrieval.fusion import ChannelHit, resolve_enterprise_weights, weighted_rrf
from aerospace_rag.stores.graph import GraphStore
from aerospace_rag.stores.vector import QdrantVectorStore


class EnterpriseRagTests(unittest.TestCase):
    def test_generation_provider_rejects_non_core_provider_aliases(self) -> None:
        settings = Settings(llm_provider="extractive")

        for provider in [None, "ollama"]:
            self.assertEqual(route_generation_provider(provider, settings=settings), "ollama")

        self.assertEqual(route_generation_provider("extractive", settings=settings), "extractive")
        for provider in ["local", "openai_compatible", "gemma4_openai", "vllm", "remote"]:
            with self.assertRaisesRegex(ValueError, "provider"):
                route_generation_provider(provider, settings=settings)

    def test_weighted_rrf_uses_static_core_weights_and_evidence_adjustment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "fusion_weights.runtime.json"
            meta = Path(tmp) / "fusion_profile_meta.runtime.json"
            profile.write_text(
                json.dumps(
                    {
                        "profile_id": "enterprise-qact",
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
            meta.write_text(json.dumps({"selection_run_type": "main"}), encoding="utf-8")

            weights, diagnostics = resolve_enterprise_weights(
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

        self.assertEqual(diagnostics["weights_source"], "static")
        self.assertNotIn("fusion_profile_id", diagnostics)
        self.assertIn("graph_no_evidence", diagnostics["evidence_adjustments"])
        self.assertEqual(weights["graph"], 0.0)
        self.assertEqual(ranked[0].chunk_id, "shared-doc")
        self.assertIn("top_doc_channel_contributions", debug)
        self.assertGreater(
            debug["top_doc_channel_contributions"]["shared-doc"]["vector_dense_text"],
            debug["top_doc_channel_contributions"]["shared-doc"]["vector_sparse"],
        )

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

    def test_graph_store_uses_graph_lite_index_without_graph_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = GraphStore(Path(tmp) / "index", settings=Settings(extractor_provider="local_fallback"))
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

    def test_ollama_extractor_accepts_fenced_json_response(self) -> None:
        chunk = Chunk("extract#1", "NASA awarded Momentus a solar sail contract.", "memo.txt", "text")

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
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
                return json.dumps({"message": {"content": content}}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = KnowledgeExtractor(
                settings=Settings(
                    ollama_base_url="https://ollama.com",
                    ollama_model="gemma4:31b",
                    ollama_api_key="test-token",
                    extractor_provider="ollama",
                )
            ).extract(chunk)

        self.assertEqual([entity.text for entity in result.entities], ["NASA", "Momentus"])
        self.assertEqual(result.relations[0].type, "AWARDED_CONTRACT_TO")


if __name__ == "__main__":
    unittest.main()
