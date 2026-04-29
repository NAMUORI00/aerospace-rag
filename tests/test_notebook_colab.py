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
        colab = nb.metadata.get("colab", {})

        self.assertTrue(colab.get("include_colab_link"))
        self.assertTrue(colab.get("toc_visible"))
        self.assertIn("colab.research.google.com/github/NAMUORI00/aerospace-rag", source)
        self.assertIn("project_root_candidates", source)
        self.assertIn("ensure_valid_cwd", source)
        self.assertIn("files.upload", source)
        self.assertIn("ensure_dependencies", source)
        self.assertIn("ensure_ollama_runtime", source)
        self.assertIn("gemma4:e2b", source)
        self.assertIn("GITHUB_REPO_URL", source)
        self.assertIn("https://github.com/NAMUORI00/aerospace-rag.git", source)
        self.assertIn("git clone", source)
        self.assertIn("os.chdir(DEFAULT_COLAB_ROOT.parent)", source)
        self.assertIn("shutil.rmtree(DEFAULT_COLAB_ROOT)", source)
        self.assertIn("Google Drive는 사용하지 않으며", source)
        self.assertIn("REPO_COMMIT", source)
        self.assertIn("file_sha256", source)
        self.assertIn("DATA_MANIFEST", source)
        self.assertIn("ingest_data(DATA_DIR)", source)
        self.assertIn("LocalIndex", source)
        self.assertIn("REPRODUCIBILITY_REPORT", source)
        self.assertNotIn("/content/drive", source)
        self.assertNotIn("MyDrive", source)
        self.assertNotIn("google.colab import files, drive", source)
        self.assertNotIn("from google.colab import drive", source)
        self.assertNotIn("drive.mount", source)
        self.assertNotIn("zipfile.ZipFile", source)

    def test_notebook_sections_are_reproducibility_oriented(self) -> None:
        nb = nbformat.read(NOTEBOOK, as_version=4)
        headings = [
            cell.source.strip().splitlines()[0]
            for cell in nb.cells
            if cell.cell_type == "markdown" and cell.source.strip().startswith("## ")
        ]

        self.assertEqual(
            headings,
            [
                "## 1. 실행 환경 확인",
                "## 2. 프로젝트 소스 확보",
                "## 3. 의존성 설치와 버전 고정 확인",
                "## 4. Ollama 런타임과 모델 준비",
                "## 5. 데이터 파일 준비",
                "## 6. 실행 설정 확정",
                "## 7. 수집/파싱 단독 확인",
                "## 8. 인덱스 생성",
                "## 9. 검색 단독 검증",
                "## 10. LLM 답변 생성",
                "## 11. 근거 확인",
                "## 12. 반복 질문 예시",
                "## 13. 재현성 리포트",
                "## 14. 문제 해결 체크리스트",
            ],
        )

    def test_notebook_is_saved_without_stale_error_outputs(self) -> None:
        nb = nbformat.read(NOTEBOOK, as_version=4)

        for cell in nb.cells:
            self.assertFalse(cell.get("outputs"), "notebook should not persist runtime outputs")
            self.assertIsNone(cell.get("execution_count"), "notebook should be saved unexecuted")


if __name__ == "__main__":
    unittest.main()
