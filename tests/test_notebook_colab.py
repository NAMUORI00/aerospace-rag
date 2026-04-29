from __future__ import annotations

import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "aerospace_rag_colab_ui.ipynb"


class NotebookColabTests(unittest.TestCase):
    def test_notebook_has_colab_project_bootstrap(self) -> None:
        nb = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join(cell.source for cell in nb.cells)

        self.assertIn("PROJECT_ROOT_CANDIDATES", source)
        self.assertIn("files.upload", source)
        self.assertIn("ensure_dependencies", source)
        self.assertIn("ensure_ollama_runtime", source)
        self.assertIn("gemma4:e2b", source)
        self.assertIn("GITHUB_REPO_URL", source)
        self.assertIn("https://github.com/NAMUORI00/aerospace-rag.git", source)
        self.assertIn("git clone", source)

    def test_notebook_is_saved_without_stale_error_outputs(self) -> None:
        nb = nbformat.read(NOTEBOOK, as_version=4)

        for cell in nb.cells:
            self.assertFalse(cell.get("outputs"), "notebook should not persist runtime outputs")
            self.assertIsNone(cell.get("execution_count"), "notebook should be saved unexecuted")


if __name__ == "__main__":
    unittest.main()
