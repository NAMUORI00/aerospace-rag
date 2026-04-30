from __future__ import annotations

import importlib
import unittest

import aerospace_rag


class PublicApiTests(unittest.TestCase):
    def test_package_exports_direct_call_api(self) -> None:
        self.assertEqual(sorted(aerospace_rag.__all__), ["ask", "build_index"])
        self.assertTrue(callable(aerospace_rag.ask))
        self.assertTrue(callable(aerospace_rag.build_index))

    def test_canonical_subpackages_import(self) -> None:
        for module_name in [
            "aerospace_rag.pipeline",
            "aerospace_rag.cli.ingest",
            "aerospace_rag.cli.query",
            "aerospace_rag.ingestion",
            "aerospace_rag.retrieval",
            "aerospace_rag.stores",
            "aerospace_rag.generation",
            "aerospace_rag.artifacts",
        ]:
            self.assertIsNotNone(importlib.import_module(module_name))

    def test_removed_root_wrappers_do_not_import(self) -> None:
        for module_name in [
            "aerospace_rag.artifact_export",
            "aerospace_rag.artifact_import",
            "aerospace_rag.bm25",
            "aerospace_rag.document_parser",
            "aerospace_rag.embeddings",
            "aerospace_rag.extraction",
            "aerospace_rag.fusion",
            "aerospace_rag.graph_store",
            "aerospace_rag.index",
            "aerospace_rag.index_store",
            "aerospace_rag.ingest",
            "aerospace_rag.providers",
            "aerospace_rag.query",
            "aerospace_rag." + "runtime_" + "dat",
            "aerospace_rag.vector_store",
        ]:
            with self.assertRaises(ModuleNotFoundError, msg=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
