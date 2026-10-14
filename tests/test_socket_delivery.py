#!/usr/bin/env python3
"""回归：写完即关的命令不能被 socket 层静默丢掉。

场景表在 tests/socket_delivery_stress.cpp 里；这里只负责编译并执行，
保证这条回归在任何主机上（不依赖 sherpa-onnx / PipeWire）都能跑。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "src" / "audio"
SCENARIOS = PROJECT_ROOT / "tests" / "socket_delivery_stress.cpp"


@unittest.skipUnless(shutil.which("g++"), "needs a C++ compiler")
class SocketDeliveryTests(unittest.TestCase):
    def test_write_and_close_commands_are_never_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "socket_delivery_stress"
            subprocess.run(
                ["g++", "-std=c++17", "-pthread", "-I", str(AUDIO_DIR),
                 str(SCENARIOS), str(AUDIO_DIR / "socket_server.cpp"),
                 "-o", str(binary)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run([str(binary), "100"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"命令被丢弃：{result.stdout}{result.stderr}")
        self.assertIn('"missing":0', result.stdout)


if __name__ == "__main__":
    unittest.main()
