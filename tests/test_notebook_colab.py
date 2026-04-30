from __future__ import annotations

import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "aerospace_rag_colab_ui.ipynb"


class NotebookColabTests(unittest.TestCase):
    def test_notebook_uses_runtime_helper_and_keeps_user_flow(self) -> None:
        nb = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join(cell.source for cell in nb.cells)
        code_source = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        colab = nb.metadata.get("colab", {})

        self.assertTrue(colab.get("include_colab_link"))
        self.assertTrue(colab.get("toc_visible"))
        self.assertIn("colab.research.google.com/github/NAMUORI00/aerospace-rag", source)
        self.assertIn("from aerospace_rag.notebook_runtime import ensure_valid_cwd, git_output", source)
        self.assertIn("from aerospace_rag.notebook_runtime import ensure_dependencies", source)
        self.assertIn("from aerospace_rag.notebook_runtime import ensure_ollama_runtime", source)
        self.assertIn("from aerospace_rag.notebook_runtime import discover_data_files", source)
        self.assertIn("qwen2.5:7b", source)
        self.assertIn("ANSWER_PROVIDER", source)
        self.assertIn("TOP_K", source)
        self.assertIn("EXTRACTOR_LLM_BACKEND", source)
        self.assertIn('EXTRACTOR_LLM_BACKEND = "ollama"', code_source)
        self.assertNotIn("EXTRACTOR_FALLBACK_ON_ERROR", source)
        self.assertIn("OLLAMA_EXTRACT_TIMEOUT_SECONDS = 3600", code_source)
        self.assertIn("OLLAMA_GENERATE_TIMEOUT_SECONDS = 3600", code_source)
        self.assertIn("OLLAMA_EXTRACT_RETRIES = 1", code_source)
        self.assertIn("OLLAMA_EXTRACT_REPAIR_RETRIES = 1", code_source)
        self.assertIn("OLLAMA_EXTRACT_NUM_PREDICT = 4096", code_source)
        self.assertIn("OLLAMA_EXTRACT_MAX_CHARS = 1200", code_source)
        self.assertIn("GITHUB_REPO_URL", source)
        self.assertIn("DATA_MANIFEST", source)
        self.assertIn("ingest_data(DATA_DIR, strict_expected=False)", source)
        self.assertIn("LocalIndex", source)
        self.assertIn("ACTUAL_RAG_QUESTIONS", source)
        self.assertNotIn("import_google_drive_data", source)
        self.assertNotIn("USE_GOOGLE_DRIVE_DATA", source)
        self.assertNotIn("files.upload", source)
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
                "## 4. 실행 설정 확정",
                "## 5. Ollama 런타임과 모델 준비",
                "## 6. 데이터 파일 준비",
                "## 7. 수집/파싱 단독 확인",
                "## 8. 인덱스 생성",
                "## 9. 검색 단독 검증",
                "## 10. LLM 답변 생성",
                "## 11. 근거 확인",
                "## 12. 반복 질문 예시",
                "## 13. 실제 업무파일 RAG 검증",
            ],
        )

    def test_notebook_is_saved_without_runtime_outputs(self) -> None:
        nb = nbformat.read(NOTEBOOK, as_version=4)
        for cell in nb.cells:
            self.assertFalse(cell.get("outputs"), "notebook should not persist runtime outputs")
            self.assertIsNone(cell.get("execution_count"), "notebook should be saved unexecuted")


if __name__ == "__main__":
    unittest.main()
