from __future__ import annotations

import unittest

from aerospace_rag.cli_utils import make_console_safe


class CliTests(unittest.TestCase):
    def test_make_console_safe_replaces_unencodable_windows_console_characters(self) -> None:
        text = "NASA • Momentus"

        safe = make_console_safe(text, encoding="cp949")

        self.assertEqual(safe, "NASA ? Momentus")
        safe.encode("cp949")


if __name__ == "__main__":
    unittest.main()
