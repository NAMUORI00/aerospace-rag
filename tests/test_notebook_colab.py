from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess as real_subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "aerospace_rag_colab_ui.ipynb"


def project_bootstrap_source() -> str:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and "def is_project_root" in cell.source:
            return cell.source
    raise AssertionError("project bootstrap cell not found")


def ollama_runtime_source() -> str:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and "def ensure_ollama_runtime" in cell.source:
            return cell.source
    raise AssertionError("Ollama runtime cell not found")


def data_upload_source() -> str:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and "DATA_FILES = list(iter_supported_files(DATA_DIR))" in cell.source:
            return cell.source
    raise AssertionError("data upload cell not found")


def google_drive_data_source() -> str:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and "def import_google_drive_data" in cell.source:
            return cell.source
    raise AssertionError("Google Drive data import cell not found")


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
        self.assertIn("ensure_dependencies", source)
        self.assertIn("ensure_ollama_runtime", source)
        self.assertIn("gemma4:e2b", source)
        self.assertIn("ANSWER_PROVIDER", source)
        self.assertIn("TOP_K", source)
        self.assertIn("EXTRACTOR_LLM_BACKEND", source)
        self.assertIn("LLM_NEEDED", source)
        self.assertIn('EXTRACTOR_LLM_BACKEND = "ollama"', source)
        self.assertIn('EMBED_BACKEND = "sentence_transformers"', source)
        self.assertIn('VECTOR_BACKEND = "qdrant"', source)
        self.assertIn("AEROSPACE_VECTOR_BACKEND", source)
        self.assertIn("sentence_transformers", source)
        self.assertIn("docling", source)
        self.assertNotIn('EXTRACTOR_LLM_BACKEND = "deterministic"', source)
        self.assertNotIn('"deterministic"', source)
        self.assertNotIn("DAT_MODE", source)
        self.assertNotIn("falkordb", source.lower())
        self.assertNotIn("falkordb_path", source)
        self.assertNotIn("return extractive evidence if the call fails", source)
        self.assertIn("OLLAMA_API_KEY", source)
        self.assertIn("OLLAMA_API_KEY_SET", source)
        self.assertIn("is_ollama_cloud_runtime", source)
        self.assertIn("GITHUB_REPO_URL", source)
        self.assertIn("https://github.com/NAMUORI00/aerospace-rag.git", source)
        self.assertIn("git clone", source)
        self.assertIn("os.chdir(DEFAULT_COLAB_ROOT.parent)", source)
        self.assertIn("shutil.rmtree(DEFAULT_COLAB_ROOT)", source)
        self.assertIn("USE_GOOGLE_DRIVE_DATA", source)
        self.assertIn("GOOGLE_DRIVE_DATA_DIR", source)
        self.assertIn("drive.mount", source)
        self.assertIn("/content/drive/MyDrive/aerospace-rag-data", source)
        self.assertIn("REPO_COMMIT", source)
        self.assertIn("file_sha256", source)
        self.assertIn("DATA_MANIFEST", source)
        self.assertIn("iter_supported_files", source)
        self.assertIn("SUPPORTED_SUFFIXES", source)
        self.assertIn("ingest_data(DATA_DIR, strict_expected=False)", source)
        self.assertIn("strict_expected=False", source)
        self.assertIn("LocalIndex", source)
        self.assertIn("ACTUAL_RAG_QUESTIONS", source)
        self.assertIn("ACTUAL_RAG_RESULTS", source)
        self.assertNotIn("files.upload", source)
        self.assertNotIn("zipfile.ZipFile", source)
        self.assertIn("DATA_DIR.mkdir", source)

    def test_colab_fresh_clone_is_project_root_without_data_dir(self) -> None:
        source = project_bootstrap_source()

        class FakeSubprocess:
            STDOUT = real_subprocess.STDOUT

            def __init__(self, clone_root: Path) -> None:
                self.clone_root = clone_root
                self.calls: list[list[str]] = []

            def check_call(self, args: list[str]) -> None:
                self.calls.append(args)
                self.clone_root.mkdir(parents=True)
                (self.clone_root / "aerospace_rag").mkdir()
                (self.clone_root / "notebooks").mkdir()

            def check_output(
                self,
                args: list[str],
                text: bool = True,
                stderr: object | None = None,
            ) -> str:
                return "fake-git-output"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "content" / "aerospace-rag"
            root.parent.mkdir()
            source = source.replace('Path("/content/aerospace-rag")', f"Path({str(root)!r})")
            fake_subprocess = FakeSubprocess(root)

            namespace = {
                "IN_COLAB": True,
                "Path": Path,
                "os": os,
                "shutil": shutil,
                "subprocess": fake_subprocess,
                "sys": sys,
            }

            previous_cwd = Path.cwd()
            previous_sys_path = list(sys.path)
            stale_module_names = ["aerospace_rag", "aerospace_rag.pipeline"]
            previous_modules = {name: sys.modules.get(name) for name in stale_module_names}
            for name in stale_module_names:
                sys.modules[name] = types.ModuleType(name)
            try:
                exec(source, namespace)
                self.assertFalse((root / "data").exists())
                self.assertEqual(namespace["PROJECT_ROOT"], root)
                self.assertEqual(Path.cwd(), root)
                self.assertEqual(
                    fake_subprocess.calls,
                    [["git", "clone", "https://github.com/NAMUORI00/aerospace-rag.git", str(root)]],
                )
                for name in stale_module_names:
                    self.assertNotIn(name, sys.modules)
            finally:
                os.chdir(previous_cwd)
                sys.path[:] = previous_sys_path
                for name, module in previous_modules.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module

    def test_colab_ollama_install_failure_keeps_fixed_ollama_provider(self) -> None:
        source = ollama_runtime_source()

        class FakeShutil:
            @staticmethod
            def which(name: str) -> None:
                return None

        class FakeSubprocess:
            DEVNULL = real_subprocess.DEVNULL
            STDOUT = real_subprocess.STDOUT

            def __init__(self) -> None:
                self.calls: list[tuple[object, bool]] = []

            def check_call(self, args: object, shell: bool = False) -> None:
                self.calls.append((args, shell))
                if args == "curl -fsSL https://ollama.com/install.sh | sh":
                    raise real_subprocess.CalledProcessError(1, args)

            def Popen(self, *args: object, **kwargs: object) -> None:
                raise AssertionError("Ollama server should not start after install failure")

        fake_subprocess = FakeSubprocess()
        previous_env = dict(os.environ)
        try:
            os.environ.pop("LLM_PROVIDER", None)
            os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
            os.environ["OLLAMA_MODEL"] = "gemma4:e2b"
            os.environ["OLLAMA_API_KEY"] = ""
            namespace = {
                "IN_COLAB": True,
                "LLM_NEEDED": True,
                "os": os,
                "shutil": FakeShutil,
                "subprocess": fake_subprocess,
                "time": object(),
                "urllib": object(),
            }

            exec(source, namespace)

            self.assertNotIn("LLM_PROVIDER", os.environ)
            self.assertIn((["apt-get", "install", "-y", "zstd"], False), fake_subprocess.calls)
        finally:
            os.environ.clear()
            os.environ.update(previous_env)

    def test_colab_missing_data_creates_data_dir_and_prints_manual_copy_instruction(self) -> None:
        source = data_upload_source()

        class FakeFiles:
            called = False

            @staticmethod
            def upload() -> dict[str, bytes]:
                FakeFiles.called = True
                raise AssertionError("Colab upload dialog should not be used")

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "aerospace-rag"
            project_root.mkdir()
            namespace = {
                "PROJECT_ROOT": project_root,
                "DATA_DIR": project_root / "data",
                "IN_COLAB": True,
                "files": FakeFiles,
                "Path": Path,
                "hashlib": __import__("hashlib"),
            }
            out = io.StringIO()

            with contextlib.redirect_stdout(out), self.assertRaises(FileNotFoundError) as raised:
                exec(source, namespace)

            output = out.getvalue()
            self.assertTrue((project_root / "data").is_dir())
            self.assertFalse(FakeFiles.called)
            self.assertIn(str(project_root / "data"), output)
            self.assertIn("지원 형식:", output)
            self.assertIn(".pdf", output)
            self.assertIn("위 경로에 지원 문서 파일을 넣은 뒤 이 셀을 다시 실행하세요.", output)
            self.assertIn("지원되는 data 파일이 없습니다", str(raised.exception))

    def test_google_drive_data_cell_is_optional_by_default(self) -> None:
        source = google_drive_data_source()

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "aerospace-rag"
            project_root.mkdir()
            namespace = {
                "PROJECT_ROOT": project_root,
                "DATA_DIR": project_root / "data",
                "IN_COLAB": True,
                "Path": Path,
                "shutil": shutil,
            }
            out = io.StringIO()

            with contextlib.redirect_stdout(out):
                exec(source, namespace)

            output = out.getvalue()
            self.assertIn("Google Drive data import skipped", output)
            self.assertEqual(namespace["DRIVE_DATA_FILES"], [])

    def test_colab_data_cell_discovers_supported_files(self) -> None:
        source = data_upload_source()

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "aerospace-rag"
            docs_dir = project_root / "data" / "docs"
            docs_dir.mkdir(parents=True)
            (docs_dir / "memo.txt").write_text("Momentus solar sail memo", encoding="utf-8")
            (project_root / "data" / "index").mkdir()
            (project_root / "data" / "index" / "ignored.md").write_text("old index note", encoding="utf-8")
            namespace = {
                "PROJECT_ROOT": project_root,
                "DATA_DIR": project_root / "data",
                "IN_COLAB": True,
                "Path": Path,
                "hashlib": __import__("hashlib"),
            }
            out = io.StringIO()

            with contextlib.redirect_stdout(out):
                exec(source, namespace)

            output = out.getvalue()
            self.assertIn("docs/memo.txt", output)
            self.assertNotIn("index/ignored.md", output)
            self.assertEqual(namespace["DATA_MANIFEST"][0]["name"], "docs/memo.txt")

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

    def test_notebook_is_saved_without_stale_error_outputs(self) -> None:
        nb = nbformat.read(NOTEBOOK, as_version=4)

        for cell in nb.cells:
            self.assertFalse(cell.get("outputs"), "notebook should not persist runtime outputs")
            self.assertIsNone(cell.get("execution_count"), "notebook should be saved unexecuted")


if __name__ == "__main__":
    unittest.main()
