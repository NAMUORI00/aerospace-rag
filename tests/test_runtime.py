from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aerospace_rag.artifacts.export import build_artifact_manifest
from aerospace_rag.artifacts.importer import import_artifact_manifest
from aerospace_rag.config import Settings
from aerospace_rag.models import Chunk
from aerospace_rag.pipeline import ask, build_index
from aerospace_rag.retrieval.weights import resolve_channel_weights
from aerospace_rag.stores.graph import GraphStore
from aerospace_rag.stores.local_index import LocalIndex


class RuntimeTests(unittest.TestCase):
    def test_weight_resolver_uses_runtime_profile_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights_path = root / "fusion_weights.json"
            meta_path = root / "fusion_profile_meta.json"
            weights_path.write_text(
                json.dumps(
                    {
                        "profile_id": "ignored-profile",
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

        self.assertEqual(diagnostics["weights_source"], "runtime_profile")
        self.assertEqual(diagnostics["fusion_profile_scope"], "default")
        self.assertAlmostEqual(weights["bm25"], 0.7)

    def test_graph_store_uses_extracted_relations_for_neighbor_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = GraphStore(Path(tmp) / "index", settings=Settings(extractor_provider="local_fallback"))
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
            graph_index_exists = (Path(tmp) / "index" / "graph" / "graph_index.json").exists()

        hit_ids = {chunk_id for chunk_id, _ in hits}
        self.assertIn("c1", hit_ids)
        self.assertTrue(graph_index_exists)

    def test_artifact_export_includes_all_local_index_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "index"
            (index_dir / "qdrant").mkdir(parents=True)
            (index_dir / "qdrant" / "storage.bin").write_text("q", encoding="utf-8")
            (index_dir / "graph").mkdir(parents=True)
            (index_dir / "graph" / "graph_index.json").write_text("{}", encoding="utf-8")
            (index_dir / "bm25.json").write_text("{}", encoding="utf-8")
            (index_dir / "chunks.jsonl").write_text("{}", encoding="utf-8")
            (index_dir / "fusion_weights.runtime.json").write_text(
                json.dumps({"profile_id": "test-profile", "default": {"vector_dense_text": 0.4, "vector_sparse": 0.5, "graph": 0.1}}),
                encoding="utf-8",
            )
            (index_dir / "fusion_profile_meta.runtime.json").write_text(
                json.dumps({"selection_run_type": "main", "fusion_profile_id": "test-profile"}),
                encoding="utf-8",
            )

            manifest_path = build_artifact_manifest(index_dir=index_dir, output_dir=root / "export")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = import_artifact_manifest(manifest_path=manifest_path, index_dir=index_dir)

        self.assertEqual(manifest["schema_version"], "aerospace_rag_artifact_v1")
        self.assertIn("qdrant", manifest["artifacts"])
        self.assertIn("graph", manifest["artifacts"])
        self.assertIn("bm25", manifest["artifacts"])
        self.assertIn("chunks", manifest["artifacts"])
        self.assertIn("fusion_profile", manifest["artifacts"])
        self.assertIn("fusion_profile_meta", manifest["artifacts"])
        self.assertIn("manifest_sha256", manifest)
        self.assertEqual(summary["artifacts"]["qdrant"]["copied_files"], 1)
        self.assertEqual(summary["artifacts"]["fusion_profile"]["copied_files"], 1)

    def test_no_strict_ingest_can_build_and_query_with_explicit_json_vector_debug_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            index = root / "index"
            data.mkdir()
            (data / "memo.txt").write_text(
                "NASA awarded Momentus a solar sail demonstration contract.",
                encoding="utf-8",
            )

            settings = Settings(embed_backend="hash", vector_backend="json", extractor_provider="local_fallback")
            result = build_index(data_dir=data, index_dir=index, strict_expected=False, settings=settings)
            self.assertIsNotNone(result.fusion_profile_path)
            self.assertIsNotNone(result.fusion_profile_meta_path)
            self.assertTrue(result.fusion_profile_path.exists())
            self.assertTrue(result.fusion_profile_meta_path.exists())
            response = ask(
                "Momentus solar sail contract",
                index_dir=index,
                provider="extractive",
                debug=True,
                settings=settings,
            )

        self.assertEqual(result.file_count, 1)
        self.assertGreaterEqual(result.chunk_count, 1)
        self.assertNotIn("include_private", response.routing)
        self.assertEqual(response.routing["retrieval"], "qdrant+bm25+graph-lite")
        self.assertIn("qdrant", response.diagnostics["channels"])
        self.assertEqual(response.diagnostics["fusion"]["weights_source"], "runtime_profile")
        self.assertNotIn("rerank_adjustments", response.diagnostics["fusion"])
        self.assertGreaterEqual(len(response.sources), 1)

    def test_local_index_loads_runtime_profile_from_index_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(embed_backend="hash", embed_dim=384, vector_backend="json", extractor_provider="local_fallback")
            index_dir = Path(tmp) / "index"
            index = LocalIndex(index_dir, settings=settings)
            index.build(
                [
                    Chunk("dense-doc", "NASA solar sail mission", "dense.txt", "text"),
                    Chunk("sparse-doc", "Momentus contract award", "sparse.txt", "text"),
                ]
            )
            (index_dir / "fusion_weights.runtime.json").write_text(
                json.dumps(
                    {
                        "profile_id": "runtime-test",
                        "default_candidate_depth_selected": 24,
                        "default": {"vector_dense_text": 0.2, "vector_sparse": 0.7, "vector_image": 0.0, "graph": 0.1},
                    }
                ),
                encoding="utf-8",
            )
            (index_dir / "fusion_profile_meta.runtime.json").write_text(
                json.dumps({"selection_run_type": "main", "fusion_profile_id": "runtime-test"}),
                encoding="utf-8",
            )

            hits = index.search("Momentus contract", top_k=2)

        self.assertGreaterEqual(len(hits), 1)
        fusion = index.last_diagnostics["fusion"]
        self.assertEqual(fusion["weights_source"], "runtime_profile")
        self.assertEqual(fusion["fusion_profile_id"], "runtime-test")
        self.assertEqual(fusion["candidate_depth"], 24)
        self.assertNotIn("rerank_adjustments", fusion)

    def test_lexical_rerank_promotes_matching_structured_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(embed_backend="hash", vector_backend="json", extractor_provider="local_fallback")
            index = LocalIndex(Path(tmp) / "index", settings=settings)
            index.build(
                [
                    Chunk(
                        chunk_id="qa-price",
                        text="질문: 위성영상 표준가격 결정 답변: 표준가격 10% 할인 및 최대 90% 할인",
                        source_file="인공위성_질문응답.xlsx",
                        modality="qa",
                        metadata={"keywords": "위성영상, 표준가격"},
                    ),
                    Chunk(
                        chunk_id="price-table",
                        text=(
                            "| 구분 | 위성/모드 | 저장영상(AO) | 신규촬영(NTO) |\n"
                            "| EO | K3 | $2,048, 2,867,200원 | $4,096, 5,734,400원 |"
                        ),
                        source_file="위성영상가격.png",
                        modality="table",
                        metadata={"title": "위성영상 가격표"},
                    ),
                    Chunk(
                        chunk_id="gov-table",
                        text=(
                            "| 항목 | 미국 | 한국 |\n"
                            "| 예산 투자 규모(23, 십억$) | 74.0 | 0.7 |\n"
                            "| 우주개발 기관 인력(23, 명) | NASA 18,372 | KARI 1,004 |"
                        ),
                        source_file="해외정부 우주항공 현황.png",
                        modality="table",
                        metadata={"title": "해외정부 우주항공 현황표"},
                    ),
                ]
            )

            price_hits = index.search("위성영상 가격에서 저장영상과 신규촬영은 어떻게 다른가?", top_k=2)
            gov_hits = index.search("국가별 우주항공 예산과 인력 현황은?", top_k=2)

        self.assertEqual(price_hits[0].chunk.source_file, "위성영상가격.png")
        self.assertIn("$4,096", price_hits[0].chunk.text)
        self.assertEqual(gov_hits[0].chunk.source_file, "해외정부 우주항공 현황.png")
        self.assertIn("NASA 18,372", gov_hits[0].chunk.text)


if __name__ == "__main__":
    unittest.main()
