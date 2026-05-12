from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aerospace_rag import notebook_runtime
from aerospace_rag.models import QueryResponse, SourceRef


class NotebookRuntimeTests(unittest.TestCase):
    def test_is_project_root_requires_package_and_notebooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "aerospace_rag").mkdir()
            (root / "notebooks").mkdir()

            self.assertTrue(notebook_runtime.is_project_root(root))
            self.assertFalse(notebook_runtime.is_project_root(root / "aerospace_rag"))

    def test_ensure_valid_cwd_finds_local_project_and_clears_stale_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "aerospace-rag"
            root.mkdir()
            (root / "aerospace_rag").mkdir()
            (root / "notebooks").mkdir()

            previous_cwd = Path.cwd()
            previous_sys_path = list(sys.path)
            previous_modules = {name: sys.modules.get(name) for name in ["aerospace_rag", "aerospace_rag.pipeline"]}
            for name in previous_modules:
                sys.modules[name] = types.ModuleType(name)
            try:
                os.chdir(root)
                project_root = notebook_runtime.ensure_valid_cwd(Path("/content/aerospace-rag"), "https://example.invalid/repo.git", False)

                self.assertEqual(project_root, root)
                self.assertEqual(Path.cwd(), root)
                self.assertNotIn("aerospace_rag", sys.modules)
                self.assertNotIn("aerospace_rag.pipeline", sys.modules)
            finally:
                os.chdir(previous_cwd)
                sys.path[:] = previous_sys_path
                for name, module in previous_modules.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module

    def test_discover_data_files_returns_manifest_and_ignores_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            docs_dir = data_dir / "docs"
            docs_dir.mkdir(parents=True)
            (docs_dir / "memo.txt").write_text("Momentus solar sail memo", encoding="utf-8")
            (data_dir / "index").mkdir()
            (data_dir / "index" / "ignored.md").write_text("old index note", encoding="utf-8")

            manifest = notebook_runtime.discover_data_files(data_dir)

        self.assertEqual([entry["name"] for entry in manifest], ["docs/memo.txt"])
        self.assertEqual(len(str(manifest[0]["sha256"])), 64)

    def test_google_drive_data_import_is_optional_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                copied = notebook_runtime.import_google_drive_data(
                    enabled=False,
                    source_dir="/content/drive/MyDrive/aerospace-rag-data",
                    data_dir=Path(tmp) / "data",
                    in_colab=True,
                )

        self.assertEqual(copied, [])
        self.assertIn("Google Drive data import skipped", out.getvalue())

    def test_format_answer_markdown_separates_answer_and_debug_details(self) -> None:
        response = QueryResponse(
            answer="핵심 요약입니다.\n\n- 항목 A\n- 항목 B",
            sources=[
                SourceRef(
                    chunk_id="doc#1",
                    source_file="doc.pdf",
                    modality="text",
                    score=0.42,
                    excerpt="근거 문장입니다.",
                    page=3,
                    channels={"qdrant": 0.3},
                )
            ],
            routing={"provider": "vllm"},
            diagnostics={"channels": ["bm25", "qdrant"]},
        )

        markdown = notebook_runtime.format_answer_markdown(response)

        self.assertIn("### 답변", markdown)
        self.assertIn("핵심 요약입니다.", markdown)
        self.assertIn("### 상위 근거", markdown)
        self.assertIn("doc.pdf", markdown)
        self.assertIn("<details>", markdown)
        self.assertIn("```json", markdown)

    def test_build_response_row_and_html_table_compact_long_answers(self) -> None:
        response = QueryResponse(
            answer="첫 문장입니다.\n\n- 자세한 설명이 이어집니다.",
            sources=[
                SourceRef("doc#1", "alpha.pdf", "text", 0.7, "A"),
                SourceRef("doc#2", "alpha.pdf", "text", 0.6, "B"),
                SourceRef("doc#3", "beta.pdf", "text", 0.5, "C"),
            ],
            routing={"provider": "vllm"},
            diagnostics={"channels": ["bm25", "graph"]},
        )

        row = notebook_runtime.build_response_row("무슨 질문인가?", response, case=2)
        table = notebook_runtime.format_results_table([row], columns=["case", "question", "summary", "top_source", "source_files"])

        self.assertEqual(row["case"], 2)
        self.assertEqual(row["provider"], "vllm")
        self.assertEqual(row["top_source"], "alpha.pdf")
        self.assertEqual(row["source_files"], "alpha.pdf, beta.pdf")
        self.assertIn("첫 문장입니다.", row["summary"])
        self.assertNotIn("\n", row["summary"])
        self.assertIn("<table", table)
        self.assertIn("무슨 질문인가?", table)
        self.assertIn("alpha.pdf, beta.pdf", table)

    def test_format_storage_visualization_uses_index_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "index"
            graph_dir = index_dir / "graph"
            graph_dir.mkdir(parents=True)
            (index_dir / "chunks.jsonl").write_text("{}", encoding="utf-8")
            (index_dir / "bm25.json").write_text("{}", encoding="utf-8")
            (graph_dir / "graph_index.json").write_text(
                json.dumps(
                    {
                        "schema_version": "aerospace_graph_v2",
                        "entity_to_chunks": {"nasa": ["doc#1"]},
                        "chunk_entities": {"doc#1": ["nasa"]},
                        "entity_text": {"nasa": "NASA"},
                        "entity_types": {"nasa": "Organization"},
                        "relations": [],
                        "chunks": {"doc#1": {"source_file": "doc.pdf", "modality": "text"}},
                    }
                ),
                encoding="utf-8",
            )

            sections = notebook_runtime.format_storage_visualization(index_dir)

        content = "\n".join(section["content"] for section in sections)
        self.assertIn("Qdrant / Graph storage visualization", content)
        self.assertIn("graph/graph_index.json", content)
        self.assertIn("aerospace_graph_v2", content)
        self.assertIn("NASA", content)

    def test_build_response_row_prefers_clean_intro_and_first_bullet_for_table_summary(self) -> None:
        response = QueryResponse(
            answer="가격 차이는 다음과 같습니다:\n\n- **저장영상(AO)**: 기존 보유 영상 기준 가격\n- 신규촬영(NTO): 신규 촬영 요청 기준 가격",
            sources=[SourceRef("doc#1", "alpha.pdf", "text", 0.7, "A")],
            routing={"provider": "vllm"},
            diagnostics={"channels": ["bm25"]},
        )

        row = notebook_runtime.build_response_row("질문", response)
        table = notebook_runtime.format_results_table([row], columns=["summary"])

        self.assertEqual(row["summary"], "가격 차이는 다음과 같습니다: 저장영상(AO): 기존 보유 영상 기준 가격")
        self.assertNotIn("**", row["summary"])
        self.assertNotIn("- 신규촬영", row["summary"])
        self.assertIn("가격 차이는 다음과 같습니다: 저장영상(AO): 기존 보유 영상 기준 가격", table)

    def test_model_runtime_preloads_configured_model(self) -> None:
        observed: dict[str, object] = {}

        def fake_ensure(settings: object) -> dict[str, object]:
            observed["settings"] = settings
            return {"ready": True, "model": "google/gemma-4-E4B-it", "device_map": "auto"}

        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "vllm",
                "AEROSPACE_LLM_MODEL": "google/gemma-4-E4B-it",
                "AEROSPACE_VLLM_DTYPE": "float16",
                "AEROSPACE_VLLM_QUANTIZATION": "",
                "AEROSPACE_VLLM_LOAD_FORMAT": "auto",
                "AEROSPACE_VLLM_CPU_OFFLOAD_GB": "1.0",
                "AEROSPACE_VLLM_ENFORCE_EAGER": "true",
                "AEROSPACE_VLLM_USE_V1": "false",
            },
            clear=True,
        ), patch.object(notebook_runtime, "ensure_vllm_model", side_effect=fake_ensure):
            status = notebook_runtime.ensure_model_runtime(True)

        self.assertTrue(status["ready"])
        self.assertEqual(status["model"], "google/gemma-4-E4B-it")
        self.assertEqual(observed["settings"].llm_model, "google/gemma-4-E4B-it")
        self.assertEqual(observed["settings"].vllm_dtype, "float16")
        self.assertEqual(observed["settings"].vllm_quantization, "")
        self.assertEqual(observed["settings"].vllm_load_format, "auto")
        self.assertEqual(observed["settings"].vllm_cpu_offload_gb, 1.0)
        self.assertTrue(observed["settings"].vllm_enforce_eager)
        self.assertFalse(observed["settings"].vllm_use_v1)


if __name__ == "__main__":
    unittest.main()
