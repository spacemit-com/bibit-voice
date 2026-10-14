#!/usr/bin/env python3
"""Pins the capture-queue ceiling, the overflow log throttle and the onset
pre-roll default that keeps the decode path inside realtime."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "src" / "audio"

HELPER_SOURCE = """
#include "audio_queue_guard.h"

#include <iostream>

int main() {
  // 1) ceiling in samples for a few configured values (16 samples per ms)
  std::cout << queueMaxSamples(2000) << " "
            << queueMaxSamples(30000) << " "
            << queueMaxSamples(50) << " "
            << queueMaxSamples(999999) << "\\n";

  // 2) the reporter folds drops inside one second into a single line
  QueueOverflowReporter r;
  std::cout << r.add(1000, 341) << " ";     // first drop reports
  std::cout << r.add(1100, 341) << " ";     // same second: folded
  std::cout << r.add(1999, 341) << " ";
  std::cout << r.takeDropped() << " ";      // 3 * 341
  std::cout << r.add(2000, 341) << " ";     // a full second later reports
  std::cout << r.takeDropped() << "\\n";

  // 3) per-chunk processing cost: totals accumulate, the peak holds the worst
  ProcessTimeTracker t;
  t.add(12000);
  t.add(21500);
  t.add(9000);
  std::cout << t.totalUs() << " " << t.maxUs() << " " << t.chunks() << "\\n";
  return 0;
}
"""


class AudioQueueGuardTests(unittest.TestCase):
    def _run_helper(self) -> str:
        compiler = shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("no C++ compiler available")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "helper.cpp"
            binary = Path(tmp) / "helper"
            source.write_text(HELPER_SOURCE, encoding="utf-8")
            subprocess.run(
                [compiler, "-std=c++17", "-I", str(AUDIO_DIR), str(source), "-o", str(binary)],
                check=True,
                capture_output=True,
                text=True,
            )
            return subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True
            ).stdout.strip()

    def test_ceiling_and_reporting(self) -> None:
        lines = self._run_helper().splitlines()
        # 2000 ms -> 32000 samples; 30000 ms -> 480000; 50 ms clamps up to 200;
        # 999999 ms clamps down to 30000.
        self.assertEqual(lines[0].split(), ["32000", "480000", "3200", "480000"])
        # 1 0 0 1023 1 341: the first drop reports, two more within the same
        # second fold into it (341*3), and a drop a full second later reports.
        self.assertEqual(lines[1].split(), ["1", "0", "0", "1023", "1", "341"])
        # 12000 + 21500 + 9000 us over three chunks; the peak is the 21.5 ms
        # chunk that alone exceeds the 21.3 ms a chunk represents in realtime.
        self.assertEqual(lines[2].split(), ["42500", "21500", "3"])


class OnsetPreRollDefaultTests(unittest.TestCase):
    """The onset pre-roll is 0 (scheme C) in both places that carry it.

    Re-feeding 800 ms to the onset streams on every silence-then-speech
    transition is what pushed the worker below realtime; a silent drift back
    to a non-zero built-in default would reintroduce the overflow.
    """

    def test_builtin_default_and_packaged_conf_agree_on_zero(self) -> None:
        header = (AUDIO_DIR / "config.h").read_text(encoding="utf-8")
        conf = (PROJECT_ROOT / "config" / "bibit.conf").read_text(encoding="utf-8")
        builtin = re.search(r"int kws_reset_pre_roll_ms = (\d+);", header)
        packaged = re.search(r"^kws_reset_pre_roll_ms=(\d+)$", conf, re.MULTILINE)
        self.assertIsNotNone(builtin, "config.h no longer declares kws_reset_pre_roll_ms")
        self.assertIsNotNone(packaged, "config/bibit.conf no longer sets kws_reset_pre_roll_ms")
        self.assertEqual(builtin.group(1), packaged.group(1))
        self.assertEqual("0", builtin.group(1))

    def test_silence_onset_path_stays_enabled(self) -> None:
        conf = (PROJECT_ROOT / "config" / "bibit.conf").read_text(encoding="utf-8")
        self.assertIn("kws_reset_on_speech_after_silence=true", conf)


if __name__ == "__main__":
    unittest.main()
