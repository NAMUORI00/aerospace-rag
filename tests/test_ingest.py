from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aerospace_rag.ingestion import EXPECTED_FILES, ingest_data, iter_supported_files


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def has_private_dataset() -> bool:
    return all((DATA_DIR / name).exists() for name in EXPECTED_FILES)


class IngestionTests(unittest.TestCase):
    def test_ingest_data_defaults_to_supported_files_in_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "memo.txt").write_text("NASA awarded Momentus a solar sail contract.", encoding="utf-8")

            chunks = ingest_data(data_dir)

        self.assertEqual({chunk.source_file for chunk in chunks}, {"memo.txt"})
        self.assertTrue(any("Momentus" in chunk.text for chunk in chunks))

    def test_iter_supported_files_ignores_index_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "memo.md").write_text("source", encoding="utf-8")
            (data_dir / "index").mkdir()
            (data_dir / "index" / "old.md").write_text("generated", encoding="utf-8")

            names = [path.relative_to(data_dir).as_posix() for path in iter_supported_files(data_dir)]

        self.assertEqual(names, ["memo.md"])

    def test_docx_ingest_requires_docling_without_parser_cascade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "manual.docx").write_bytes(b"not a real docx")

            with patch.dict(sys.modules, {"docling": None}):
                with self.assertRaisesRegex(RuntimeError, "docling"):
                    ingest_data(data_dir)

    def test_ingest_data_preserves_multiline_png_table_text_in_default_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "위성영상가격.png").write_bytes(b"not-a-real-image")

            chunks = ingest_data(data_dir)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].modality, "table")
        self.assertIn("\n| EO | K3 |", chunks[0].text)
        self.assertIn("\n메모:", chunks[0].text)

    @unittest.skipUnless(has_private_dataset(), "private data files are not tracked in the public repo")
    def test_ingest_data_creates_expected_modalities_and_metadata(self) -> None:
        chunks = ingest_data(DATA_DIR)

        source_files = {chunk.source_file for chunk in chunks}
        modalities = {chunk.modality for chunk in chunks}

        self.assertEqual(len(source_files), 5)
        self.assertIn("text", modalities)
        self.assertIn("qa", modalities)
        self.assertIn("table", modalities)
        self.assertTrue(any(chunk.page == 1 for chunk in chunks if chunk.source_file.endswith(".pdf")))
        self.assertTrue(any(chunk.sheet == "Sheet1" and chunk.row == 2 for chunk in chunks))
        self.assertTrue(any("K3A" in chunk.text and "신규촬영" in chunk.text for chunk in chunks))
        self.assertTrue(any("NASA" in chunk.text and "Momentus" in chunk.text for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
