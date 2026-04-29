from __future__ import annotations

import json
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
from aerospace_rag.stores.local_index import LocalIndex
from aerospace_rag.stores.vector import QdrantVectorStore


class EnterpriseRagTests(unittest.TestCase):
    def test_generation_provider_is_fixed_to_ollama_for_llm_aliases(self) -> None:
        settings = Settings(llm_provider="extractive")

        for provider in [None, "ollama", "local", "openai_compatible", "gemma4_openai", "vllm", "remote"]:
            self.assertEqual(route_generation_provider(provider, private_present=False, settings=settings), "ollama")
            self.assertEqual(route_generation_provider(provider, private_present=True, settings=settings), "ollama")

        self.assertEqual(route_generation_provider("extractive", private_present=False, settings=settings), "extractive")

    def test_weighted_rrf_uses_qact_profile_and_evidence_adjustment(self) -> None:
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

        self.assertEqual(diagnostics["weights_source"], "profile")
        self.assertEqual(diagnostics["candidate_depth"], 42)
        self.assertIn("graph_no_evidence", diagnostics["evidence_adjustments"])
        self.assertEqual(weights["graph"], 0.0)
        self.assertEqual(ranked[0].chunk_id, "shared-doc")
        self.assertIn("top_doc_channel_contributions", debug)
        self.assertGreater(
            debug["top_doc_channel_contributions"]["shared-doc"]["vector_sparse"],
            debug["top_doc_channel_contributions"]["shared-doc"]["vector_dense_text"],
        )

    def test_vector_store_fallback_has_dense_sparse_image_channels_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(embed_backend="hash", embed_dim=384)
            store = QdrantVectorStore(Path(tmp) / "index", settings=settings, force_fallback=True)
            chunk = Chunk(
                chunk_id="doc#1",
                text="NASA Momentus solar sail demonstration",
                source_file="doc.pdf",
                modality="image",
                metadata={"asset_ref": "page:1#image:0"},
            )

            store.build([chunk])
            channels = store.search_channels("NASA solar sail", limit=5, farm_id="tenant-a", include_private=False)
            payload = json.loads(store.fallback_path.read_text(encoding="utf-8"))[0]["payload"]

        self.assertIn("vector_dense_text", channels)
        self.assertIn("vector_sparse", channels)
        self.assertIn("vector_image", channels)
        self.assertEqual(payload["tier"], "public")
        self.assertEqual(payload["canonical_doc_id"], "doc.pdf")
        self.assertEqual(payload["canonical_chunk_id"], "doc#1")
        self.assertEqual(payload["modality"], "image")

    def test_graph_store_supports_private_scope_and_path_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = GraphStore(Path(tmp) / "index")
            graph.build(
                [
                    Chunk("public#1", "NASA awarded Momentus a solar sail contract.", "public.pdf", "text"),
                    Chunk(
                        "private#1",
                        "Tenant secret H3 launch risk memo.",
                        "private:memo",
                        "text",
                        metadata={"tier": "private", "farm_id": "tenant-a", "source_type": "memo"},
                    ),
                ]
            )

            public_hits = graph.search("H3 launch risk", farm_id="tenant-b", include_private=False, limit=5)
            private_hits = graph.search("H3 launch risk", farm_id="tenant-a", include_private=True, limit=5)

        self.assertNotIn("private#1", {chunk_id for chunk_id, _ in public_hits})
        self.assertIn("private#1", {chunk_id for chunk_id, _ in private_hits})

    def test_private_ingest_updates_overlay_vector_and_graph_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            settings = Settings(embed_backend="hash", private_store_db_path=Path(tmp) / "private.sqlite")
            index = LocalIndex(index_dir, settings=settings)
            index.build([Chunk("public#1", "NASA public contract memo", "public.md", "text")])

            record_id = index.upsert_private_text(
                text="Tenant A H3 launch customer risk memo",
                farm_id="tenant-a",
                source_type="memo",
            )
            response_hits = index.search(
                "H3 customer risk",
                farm_id="tenant-a",
                include_private=True,
                top_k=5,
            )

        self.assertTrue(record_id)
        private_ids = {hit.chunk.chunk_id for hit in response_hits if hit.chunk.metadata.get("tier") == "private"}
        self.assertIn(record_id, private_ids)
        self.assertIn("private_overlay", index.last_diagnostics["private"]["channels"])

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
