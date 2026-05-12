from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aerospace_rag.config import Settings
from aerospace_rag.ingestion import EXPECTED_FILES
from aerospace_rag.pipeline import ask, build_index


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def has_private_dataset() -> bool:
    return all((DATA_DIR / name).exists() for name in EXPECTED_FILES)


class AerospacePipelineTests(unittest.TestCase):
    def _test_settings(self) -> Settings:
        return Settings(embed_backend="hash", embed_dim=384, vector_backend="json", extractor_backend="local_fallback")

    @unittest.skipUnless(has_private_dataset(), "private data files are not tracked in the public repo")
    def test_build_index_ingests_all_existing_dataset_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"

            result = build_index(data_dir=DATA_DIR, index_dir=index_dir, reset=True, settings=self._test_settings())

            self.assertEqual(result.file_count, 5)
            self.assertGreaterEqual(result.chunk_count, 10)
            self.assertEqual(result.qdrant_collection, "aerospace_chunks")
            self.assertTrue((index_dir / "bm25.json").exists())
            self.assertTrue((index_dir / "chunks.jsonl").exists())
            self.assertTrue(result.graph_index_path.exists())
            self.assertEqual(result.graph_index_path, index_dir / "graph" / "graph_index.json")

    @unittest.skipUnless(has_private_dataset(), "private data files are not tracked in the public repo")
    def test_query_fuses_qdrant_bm25_and_graph_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            settings = self._test_settings()
            build_index(data_dir=DATA_DIR, index_dir=index_dir, reset=True, settings=settings)

            response = ask(
                "위성영상 가격은 저장영상과 신규촬영에서 어떻게 다른가?",
                index_dir=index_dir,
                top_k=5,
                provider="extractive",
                debug=True,
                settings=settings,
            )

            self.assertIn("위성영상가격.png", {source.source_file for source in response.sources})
            self.assertIn("qdrant", response.diagnostics["channels"])
            self.assertIn("bm25", response.diagnostics["channels"])
            self.assertIn("graph", response.diagnostics["channels"])
            self.assertIn("저장영상", response.answer)
            self.assertIn("신규촬영", response.answer)


if __name__ == "__main__":
    unittest.main()
