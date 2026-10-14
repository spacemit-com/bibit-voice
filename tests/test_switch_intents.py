#!/usr/bin/env python3
"""回归：语音开关意图在构建期间的交错必须收敛到最后一次意图。

被测对象是 src/audio/switch_intents.h（开关意图账本），场景表在
tests/switch_intent_scenarios.cpp 里；这里只负责编译并执行它，保证这条回归
在任何主机上（不依赖 sherpa-onnx / PipeWire）都能跑。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "src" / "audio"
SCENARIOS = PROJECT_ROOT / "tests" / "switch_intent_scenarios.cpp"


@unittest.skipUnless(shutil.which("g++"), "needs a C++ compiler")
class SwitchIntentConvergenceTests(unittest.TestCase):
    def test_scenarios_and_interleaving_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "switch_intent_scenarios"
            subprocess.run(
                ["g++", "-std=c++17", "-Wall", "-Wextra", "-I", str(AUDIO_DIR),
                 str(SCENARIOS), "-o", str(binary)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False
            )
        self.assertEqual(
            result.returncode,
            0,
            f"场景断言失败：\n{result.stdout}{result.stderr}",
        )
        self.assertIn("ok:", result.stdout)


if __name__ == "__main__":
    unittest.main()
