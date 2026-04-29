from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aerospace_rag.artifacts.export import build_artifact_manifest
from aerospace_rag.artifacts.importer import import_artifact_manifest
from aerospace_rag.models import Chunk
from aerospace_rag.pipeline import ask, build_index
from aerospace_rag.retrieval.weights import resolve_channel_weights
from aerospace_rag.stores.graph import GraphStore
from aerospace_rag.stores.private_overlay import PrivateOverlayStore


class SmartFarmCorePortTests(unittest.TestCase):
    def test_dat_resolver_accepts_smartfarm_runtime_profile_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights_path = root / "fusion_weights.json"
            meta_path = root / "fusion_profile_meta.json"
            weights_path.write_text(
                json.dumps(
                    {
                        "profile_id": "dat-test",
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
            meta_path.write_text(json.dumps({"selection_run_type": "main"}), encoding="utf-8")

            weights, diagnostics = resolve_channel_weights(
                "위성영상 가격",
                profile_path=weights_path,
                profile_meta_path=meta_path,
            )

        self.assertEqual(diagnostics["weights_source"], "profile")
        self.assertAlmostEqual(weights["bm25"], 0.7, places=2)
        self.assertAlmostEqual(weights["graph"], 0.1, places=2)

    def test_graph_store_uses_extracted_relations_for_neighbor_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = GraphStore(Path(tmp) / "index")
            chunks = [
                Chunk(
                    chunk_id="c1",
                    text="NASA awarded Momentus a contract for a solar sail demonstration study.",
                    source_file="nasa.pdf",
                    modality="text",
                ),
                Chunk(
                    chunk_id="c2",
                    text="Solar sail propulsion can support small satellite demonstration missions.",
                    source_file="solar.pdf",
                    modality="text",
                ),
            ]

            graph.build(chunks)
            hits = graph.search("Momentus solar sail contract", limit=4)
            graph_index_exists = (Path(tmp) / "index" / "falkordb" / "graph_index.json").exists()

        hit_ids = {chunk_id for chunk_id, _ in hits}
        self.assertIn("c1", hit_ids)
        self.assertTrue(graph_index_exists)

    def test_artifact_export_includes_all_local_index_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "index"
            (index_dir / "qdrant").mkdir(parents=True)
            (index_dir / "qdrant" / "storage.bin").write_text("q", encoding="utf-8")
            (index_dir / "falkordb").mkdir(parents=True)
            (index_dir / "falkordb" / "graph_index.json").write_text("{}", encoding="utf-8")
            (index_dir / "bm25.json").write_text("{}", encoding="utf-8")
            (index_dir / "chunks.jsonl").write_text("{}", encoding="utf-8")

            manifest_path = build_artifact_manifest(index_dir=index_dir, output_dir=root / "export")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = import_artifact_manifest(manifest_path=manifest_path, index_dir=index_dir)

        self.assertEqual(manifest["schema_version"], "aerospace_rag_artifact_v1")
        self.assertIn("qdrant", manifest["artifacts"])
        self.assertIn("falkordb", manifest["artifacts"])
        self.assertIn("bm25", manifest["artifacts"])
        self.assertIn("chunks", manifest["artifacts"])
        self.assertIn("manifest_sha256", manifest)
        self.assertEqual(summary["artifacts"]["qdrant"]["copied_files"], 1)

    def test_private_overlay_search_is_scoped_by_farm_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PrivateOverlayStore(Path(tmp) / "private.sqlite")
            store.upsert_text(text="H3 customer-specific launch memo", farm_id="tenant-a")
            store.upsert_text(text="K3 price memo", farm_id="tenant-b")

            hits_a = store.search(query="H3 launch", farm_id="tenant-a", limit=3)
            hits_b = store.search(query="H3 launch", farm_id="tenant-b", limit=3)

        self.assertEqual(len(hits_a), 1)
        self.assertEqual(hits_a[0].chunk.metadata["farm_id"], "tenant-a")
        self.assertEqual(len(hits_b), 0)

    def test_no_strict_ingest_can_build_and_query_without_qdrant_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            index = root / "index"
            data.mkdir()
            (data / "memo.txt").write_text(
                "NASA awarded Momentus a solar sail demonstration contract.",
                encoding="utf-8",
            )

            result = build_index(data_dir=data, index_dir=index, strict_expected=False)
            response = ask("Momentus solar sail contract", index_dir=index, provider="extractive", debug=True)

        self.assertEqual(result.file_count, 1)
        self.assertGreaterEqual(result.chunk_count, 1)
        self.assertIn("qdrant", response.diagnostics["channels"])
        self.assertGreaterEqual(len(response.sources), 1)


if __name__ == "__main__":
    unittest.main()
