from __future__ import annotations

import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "aerospace_rag_colab_ui.ipynb"


class NotebookColabTests(unittest.TestCase):
    def test_notebook_uses_runtime_helper_and_shows_knowledge_graph(self) -> None:
        nb = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join(cell.source for cell in nb.cells)
        code_source = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        colab = nb.metadata.get("colab", {})

        self.assertTrue(colab.get("include_colab_link"))
        self.assertTrue(colab.get("toc_visible"))
        self.assertEqual(nb.metadata.get("accelerator"), "GPU")
        self.assertEqual(nb.metadata.get("gpuClass"), "standard")
        self.assertIn("colab.research.google.com/github/NAMUORI00/aerospace-rag", source)
        self.assertIn("smartfarm-workspace", source)
        self.assertIn("Google Colab T4", source)
        self.assertIn("/content/aerospace-rag/data", source)
        self.assertIn("from aerospace_rag.notebook_runtime import ensure_valid_cwd, git_output", source)
        self.assertIn("from aerospace_rag.notebook_runtime import ensure_dependencies", source)
        self.assertIn("from aerospace_rag.notebook_runtime import ensure_ollama_runtime, ensure_transformers_runtime", source)
        self.assertIn("from aerospace_rag.notebook_runtime import discover_data_files", source)
        self.assertIn("from IPython.display import HTML, Markdown, display", source)
        self.assertNotIn("항공우주 RAG 실행 흐름", source)
        self.assertNotIn("RAG_FLOW_HTML", code_source)
        self.assertNotIn("## 1A. RAG 원리 흐름", source)
        self.assertIn("지식 데이터베이스 노드 보기", source)
        self.assertIn("KNOWLEDGE_GRAPH_HTML", code_source)
        self.assertIn("_knowledge_graph_html", code_source)
        self.assertIn("entity 노드", source)
        self.assertIn("chunk 노드", source)
        self.assertIn("relation edge", source)
        self.assertIn("graph-lite", source)
        self.assertIn("weighted RRF", source)
        self.assertIn("format_answer_markdown", source)
        self.assertIn("format_retrieval_markdown", source)
        self.assertIn("format_sources_markdown", source)
        self.assertIn("build_response_row", source)
        self.assertIn("format_results_table", source)
        self.assertIn("format_storage_visualization", source)
        self.assertIn("gemma4:e4b", source)
        self.assertIn("google/gemma-4-E4B-it", source)
        self.assertIn("ANSWER_PROVIDER", source)
        self.assertIn("TOP_K", source)
        self.assertIn("EXTRACTOR_LLM_BACKEND", source)
        self.assertIn('ANSWER_PROVIDER = "transformers"', code_source)
        self.assertIn('EXTRACTOR_LLM_BACKEND = "transformers"', code_source)
        self.assertNotIn("EXTRACTOR_FALLBACK_ON_ERROR", source)
        self.assertIn("TRANSFORMERS_LOAD_IN_4BIT = True", code_source)
        self.assertIn("TRANSFORMERS_GENERATE_TIMEOUT_SECONDS = 120", code_source)
        self.assertIn("TRANSFORMERS_EXTRACT_TIMEOUT_SECONDS = 120", code_source)
        self.assertIn("TRANSFORMERS_ANSWER_NUM_PREDICT = 1024", code_source)
        self.assertIn("TRANSFORMERS_EXTRACT_NUM_PREDICT = 768", code_source)
        self.assertIn("OLLAMA_EXTRACT_TIMEOUT_SECONDS = 3600", code_source)
        self.assertIn("OLLAMA_GENERATE_TIMEOUT_SECONDS = 3600", code_source)
        self.assertIn("OLLAMA_EXTRACT_RETRIES = 1", code_source)
        self.assertIn("OLLAMA_EXTRACT_REPAIR_RETRIES = 1", code_source)
        self.assertIn("OLLAMA_EXTRACT_NUM_PREDICT = 4096", code_source)
        self.assertIn("OLLAMA_EXTRACT_MAX_CHARS = 1200", code_source)
        self.assertIn("GITHUB_REPO_URL", source)
        self.assertIn("DATA_MANIFEST", source)
        self.assertIn("ingest_data(DATA_DIR, strict_expected=False)", source)
        self.assertIn("INDEX_ARTIFACTS", source)
        self.assertIn("Missing index artifacts", source)
        self.assertIn("LocalIndex", source)
        self.assertIn("ACTUAL_RAG_QUESTIONS", source)
        self.assertIn("display(Markdown(format_answer_markdown(response)))", source)
        self.assertIn("display(Markdown(format_sources_markdown(response.sources)))", source)
        self.assertIn("display(HTML(format_results_table(SAMPLE_RESULTS", source)
        self.assertIn("display(HTML(format_results_table(ACTUAL_RAG_RESULTS", source)
        self.assertNotIn("print(response.answer)", source)
        self.assertNotIn("print(json.dumps(response.diagnostics", source)
        self.assertNotIn("print(\"answer_preview:\"", source)
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
                "## 5. Transformers 런타임과 모델 준비",
                "## 6. 데이터 파일 준비",
                "## 7. 수집/파싱 단독 확인",
                "## 8. 인덱스 생성",
                "## 8A. 도메인 데이터베이스 저장 구조 이해",
                "## 8B. Qdrant / Graph 저장 구조 시각화",
                "## 9. 검색 단독 검증",
                "## 10. LLM 답변 생성",
                "## 11. 근거 확인",
                "## 12. 반복 질문 예시",
                "## 13. 실제 업무파일 RAG 검증",
            ],
        )

    def test_notebook_explains_domain_database_storage(self) -> None:
        nb = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join(cell.source for cell in nb.cells)
        code_source = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

        self.assertIn("## 8A. 도메인 데이터베이스 저장 구조 이해", source)
        for expected in (
            "chunks.jsonl",
            "bm25.json",
            "graph_index.json",
            "entity_to_chunks",
            "relations",
            "channel_weights",
            "query_segment",
            "top_doc_channel_contributions",
            "KNOWLEDGE_GRAPH_HTML",
            "지식 데이터베이스 노드 보기",
            "entity 노드",
            "chunk 노드",
            "relation edge",
        ):
            self.assertIn(expected, source)
        for korean_phrase in (
            "원문 조각의 표준 payload 저장소",
            "도메인 데이터베이스 저장 구조",
            "modality별 chunk 수",
            "채널별 결합 가중치",
            "질문 유형 분류",
            "한국어 표와 JSON preview",
            "노드와 엣지 형태",
        ):
            self.assertIn(korean_phrase, source)
        self.assertIn("DATABASE_PREVIEW", code_source)
        self.assertIn("payload 샘플 2개", code_source)
        self.assertIn("display(HTML(KNOWLEDGE_GRAPH_HTML))", code_source)
        self.assertIn("format_storage_visualization", code_source)

    def test_transformers_runtime_section_matches_default_provider(self) -> None:
        nb = nbformat.read(NOTEBOOK, as_version=4)
        section = next(
            cell.source
            for cell in nb.cells
            if cell.cell_type == "markdown"
            and cell.source.strip().startswith("## 5. Transformers 런타임과 모델 준비")
        )

        self.assertIn("Transformers", section)
        self.assertIn("google/gemma-4-E4B-it", section)
        self.assertIn("4-bit", section)
        self.assertNotIn("Ollama 런타임", section)
        self.assertNotIn("strict Ollama graph extraction", section)

    def test_notebook_is_clean_for_fresh_colab_execution(self) -> None:
        nb = nbformat.read(NOTEBOOK, as_version=4)
        code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
        output_cells = [cell for cell in code_cells if cell.get("outputs")]

        self.assertEqual(output_cells, [])
        for cell in code_cells:
            self.assertIsNone(cell.get("execution_count"))


if __name__ == "__main__":
    unittest.main()
