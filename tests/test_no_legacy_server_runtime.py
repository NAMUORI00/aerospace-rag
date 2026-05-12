from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
IGNORED_DIRS = {".git", ".pytest_cache", "__pycache__", "data"}


class LegacyServerRuntimeRemovalTests(unittest.TestCase):
    def test_project_contains_no_legacy_server_runtime_references(self) -> None:
        forbidden = "ol" + "lama"
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(part in IGNORED_DIRS for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8")
            if forbidden in text.lower():
                offenders.append(str(path.relative_to(ROOT)))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
