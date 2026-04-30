from __future__ import annotations

import contextlib
import io
import os
import subprocess as real_subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aerospace_rag import notebook_runtime


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

    def test_ollama_install_failure_keeps_fixed_ollama_provider(self) -> None:
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
        with patch.dict(
            os.environ,
            {
                "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
                "OLLAMA_MODEL": "gemma4:e2b",
                "OLLAMA_API_KEY": "",
            },
            clear=True,
        ), patch.object(notebook_runtime, "shutil", FakeShutil), patch.object(notebook_runtime, "subprocess", fake_subprocess):
            status = notebook_runtime.ensure_ollama_runtime(True, in_colab=True)

        self.assertFalse(status["ready"])
        self.assertIn((["apt-get", "install", "-y", "zstd"], False), fake_subprocess.calls)
        self.assertNotIn("LLM_PROVIDER", os.environ)


if __name__ == "__main__":
    unittest.main()
