from __future__ import annotations

import unittest
from pathlib import Path

from aerospace_rag.ingestion import EXPECTED_FILES, ingest_data


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def has_private_dataset() -> bool:
    return all((DATA_DIR / name).exists() for name in EXPECTED_FILES)


class IngestionTests(unittest.TestCase):
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
