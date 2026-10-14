#!/usr/bin/env python3
"""Pins the wake-word text shapes the two ASR backends actually produce."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAKE_TEXT_DIR = PROJECT_ROOT / "src" / "audio"

HELPER_SOURCE = """
#include "wake_text.h"

#include <iostream>
#include <string>

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::cout << (isFallbackWakeText(line) ? '1' : '0');
  }
  std::cout << std::endl;
}
"""

# Transcripts captured on SpacemiT K3 from real wake-word and false-wake clips.
# Qwen3-ASR writes the wake word in CJK, so a latin-only table rejects every
# one of them; SenseVoice returns latin words whose exact spelling still varies.
CASES = (
    ("hello，B比特。", True),
    ("我有时候，觉得神经不好。", False),
    ("Hello Bibit", True),
    ("hello bibi", True),
    ("Hello", False),
    ("。", False),
    ("不要打扰我的BB疼。", False),
    ("我有时候，有时候，每天打五折。", False),
    ("你好，比特", False),
    ("hello, Bobby", False),
)


@unittest.skipUnless(shutil.which("g++"), "needs a C++ compiler")
class WakeTextMatchTests(unittest.TestCase):
    """The wake word has to stay recognisable whichever ASR wrote it down."""

    def test_recorded_transcripts_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            helper = tmp_path / "wake_text_helper.cpp"
            helper.write_text(HELPER_SOURCE)
            binary = tmp_path / "wake_text_helper"
            subprocess.run(
                ["g++", "-std=c++17", "-I", str(WAKE_TEXT_DIR), str(helper),
                 "-o", str(binary)],
                check=True,
                capture_output=True,
            )
            result = subprocess.run(
                [str(binary)],
                input="\n".join(text for text, _ in CASES),
                capture_output=True,
                text=True,
                check=True,
            )
        expected = "".join("1" if matched else "0" for _, matched in CASES)
        self.assertEqual(result.stdout.strip(), expected)


if __name__ == "__main__":
    unittest.main()
