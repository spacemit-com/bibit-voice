#!/usr/bin/env python3
"""板上回归：识别开关的意图在异步构建期间必须收敛到最后一次意图。

背景：打开识别要异步构建后端（Qwen3 还要拉起 llama-server，冷启动常以十秒计），
关闭识别却立即生效。历史实现用「待处理布尔」记录构建期间的打开意图，于是：
构建期间关闭识别，构建结束后开关又被翻回打开；构建失败还会顺手吃掉更新的打开意图。

覆盖（对应 docs/protocol.md §4.4）：
  off-during-build    打开识别途中关闭：构建结束后必须保持关闭，打开意图拿到终态
  reopen-during-build 打开→关闭→打开：最终必须真的就绪（asr_loaded=true）
  hammer-then-off     连点后停在关闭：收敛到关闭，且不再自行变化
  stable              收敛后再观察一段稳定窗口，状态不得自己变化
  terminals           每条带 request_id 的命令恰好一个终态，不重不漏
  persisted           switches.conf 等于最终生效态（没达成的打开不写进偏好）

用法（板上，直接对 /run/bibit/audio.sock 说话）：

    python3 scripts/verify_switch_intents.py [--settle-timeout-s 120]

任一硬性检查失败退出码为 1，报告以 JSON 打到 stdout。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from typing import Any

AUDIO_SOCK = os.environ.get("AUDIO_SOCK", "/run/bibit/audio.sock")
DEFAULT_SWITCHES_CONF = os.environ.get(
    "SWITCHES_CONF", os.path.expanduser("~/.config/bibit/switches.conf")
)


def send_command(path: str, command: dict[str, Any], timeout: float = 2.0) -> None:
    """写完即关：与 GUI 侧 Reply::Sent 的连接形态一致。"""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(path)
        sock.sendall((json.dumps(command) + "\n").encode("utf-8"))


def get_status(path: str, timeout: float = 5.0) -> dict[str, Any]:
    """新开一条连接读状态快照。"""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(path)
        sock.sendall(b'{"cmd":"get_status"}\n')
        buffer = bytearray()
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                raise RuntimeError("connection closed while reading status")
            buffer.extend(chunk)
            while b"\n" in buffer:
                raw, _, rest = buffer.partition(b"\n")
                buffer[:] = rest
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "status":
                    return event


def read_persisted(path: str) -> dict[str, bool]:
    values: dict[str, bool] = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.strip().partition("=")
                if key in ("wake_enabled", "asr_enabled"):
                    values[key] = value == "true"
    except OSError:
        pass
    return values


class Monitor:
    """常驻订阅：记录服务端广播的每一个事件（带接收时刻）。"""

    def __init__(self, path: str):
        self.events: list[tuple[float, dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(1.0)
        self._sock.connect(path)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        buffer = bytearray()
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            buffer.extend(chunk)
            while b"\n" in buffer:
                raw, _, rest = buffer.partition(b"\n")
                buffer[:] = rest
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                with self._lock:
                    self.events.append((time.monotonic(), event))

    def seen(self) -> list[tuple[float, dict[str, Any]]]:
        with self._lock:
            return list(self.events)

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(2)

    def terminals(self, request_id: str) -> list[dict[str, Any]]:
        """与这条 request_id 相关的终态：switches_settled 或归属它的 error。"""
        found = []
        for _, event in self.seen():
            if event.get("request_id") != request_id:
                continue
            if event.get("event") == "switches_settled":
                found.append(event)
            elif event.get("event") == "error":
                found.append(event)
        return found

    def status_after(self, mark: float) -> list[dict[str, Any]]:
        return [
            event
            for stamp, event in self.seen()
            if stamp >= mark and event.get("event") == "status"
        ]


def wait_for_status(
    path: str, predicate, timeout: float, interval: float = 0.2
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            latest = get_status(path)
        except (OSError, RuntimeError):
            time.sleep(interval)
            continue
        if predicate(latest):
            return latest
        time.sleep(interval)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=AUDIO_SOCK, help="audio.sock 路径")
    parser.add_argument(
        "--switches-conf", default=DEFAULT_SWITCHES_CONF, help="开关持久化文件"
    )
    parser.add_argument(
        "--settle-timeout-s", type=float, default=120.0,
        help="等待构建结束（收敛）的上限",
    )
    parser.add_argument(
        "--stable-window-s", type=float, default=3.0,
        help="收敛后观察状态是否自行变化的时间窗",
    )
    parser.add_argument(
        "--keep-asr-off", action="store_true",
        help="收尾时不恢复初始开关状态（默认恢复）",
    )
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    notes: list[str] = []
    report: dict[str, Any] = {"socket": args.socket}

    # 服务重启后 initialize() 会先把后端建起来才开 accept 循环，这里等它可读。
    initial = wait_for_status(args.socket, lambda status: True, 90.0, interval=1.0)
    if initial is None:
        print(json.dumps({"socket": args.socket, "checks": {"service_reachable": False},
                          "passed": False, "notes": ["audio.sock 90 秒内没有返回状态快照"]},
                         ensure_ascii=False, indent=2))
        return 1
    report["initial"] = {"wake": initial.get("wake"), "asr": initial.get("asr")}
    if initial.get("switch_settlement") is not True:
        checks["capability"] = False
        notes.append("服务端未声明 switch_settlement：无法判定意图终态")
    else:
        checks["capability"] = True

    def settle_to(asr: bool, request_id: str, timeout: float) -> dict[str, Any] | None:
        send_command(args.socket, {"cmd": "set_switches", "asr": asr, "request_id": request_id})
        return wait_for_status(
            args.socket,
            lambda status: status.get("asr") == asr and not status.get("switching"),
            timeout,
        )

    # 明确起点：识别关闭且没有在途构建。
    if settle_to(False, "setup-off", args.settle_timeout_s) is None:
        checks["setup"] = False
        notes.append("无法把识别收敛到关闭状态")
    else:
        checks["setup"] = True

    monitor = Monitor(args.socket)
    try:
        # --- 场景一：打开识别途中关闭 ---
        send_command(
            args.socket,
            {"cmd": "set_switches", "asr": True, "request_id": "off-during-on"},
        )
        building = wait_for_status(
            args.socket,
            lambda status: isinstance(status.get("switching"), dict)
            and status["switching"].get("kind") == "asr_enable",
            15.0,
        )
        checks["build_started"] = building is not None
        if building is None:
            notes.append("打开识别没有进入构建状态，无法验证构建期间的行为")
        # 观察到构建在途就立刻关闭：间隔越短越能确定关闭落在构建窗口里。
        send_command(
            args.socket,
            {"cmd": "set_switches", "asr": False, "request_id": "off-during-off"},
        )
        off_applied = wait_for_status(
            args.socket, lambda status: status.get("asr") is False, 10.0
        )
        checks["off_applied"] = off_applied is not None
        # 关闭之后构建仍在途，才算真的覆盖了这个竞态（否则只是「开完再关」）。
        checks["off_during_build_race_exercised"] = bool(
            off_applied and off_applied.get("switching")
        )
        if not checks["off_during_build_race_exercised"]:
            notes.append("关闭到达时构建已经结束，本次没有撞上竞态（可用更慢的引擎复现）")
        off_mark = time.monotonic()
        settled = wait_for_status(
            args.socket,
            lambda status: not status.get("switching")
            and (status.get("desired") or {}).get("pending") is False,
            args.settle_timeout_s,
        )
        checks["off_during_build_settled"] = settled is not None and settled.get("asr") is False
        if settled is not None and settled.get("asr") is not False:
            notes.append(f"构建结束后开关被翻回打开：asr={settled.get('asr')}")

        # 构建结束后不得再出现 asr=true：这正是历史缺陷的现场。
        reopens = [
            status for status in monitor.status_after(off_mark) if status.get("asr") is True
        ]
        checks["off_during_build_never_reopens"] = not reopens
        if reopens:
            notes.append(f"关闭后又观察到 {len(reopens)} 次 asr=true 广播")

        on_terminals = monitor.terminals("off-during-on")
        off_terminals = monitor.terminals("off-during-off")
        checks["on_intent_terminal_once"] = len(on_terminals) == 1
        checks["off_intent_terminal_once"] = len(off_terminals) == 1
        checks["on_intent_not_reported_applied"] = bool(on_terminals) and not any(
            event.get("outcome") == "applied" for event in on_terminals
        )
        report["off_during_build"] = {
            "on_terminal": on_terminals[0] if on_terminals else None,
            "off_terminal": off_terminals[0] if off_terminals else None,
            "final_asr": (settled or {}).get("asr"),
        }

        # 构建结束后给一个稳定窗口：状态不得自行变化。
        settled_asr = (settled or {}).get("asr")
        stable_until = time.monotonic() + args.stable_window_s
        drifted = False
        while time.monotonic() < stable_until:
            latest = get_status(args.socket)
            if latest.get("asr") != settled_asr:
                drifted = True
                break
            time.sleep(0.3)
        checks["stable_after_settle"] = not drifted
        if drifted:
            notes.append("收敛后开关自行变化")

        # --- 场景二：构建期间关闭又重新打开 ---
        send_command(
            args.socket,
            {"cmd": "set_switches", "asr": True, "request_id": "reopen-on-1"},
        )
        wait_for_status(
            args.socket,
            lambda status: isinstance(status.get("switching"), dict),
            15.0,
        )
        send_command(
            args.socket,
            {"cmd": "set_switches", "asr": False, "request_id": "reopen-off"},
        )
        send_command(
            args.socket,
            {"cmd": "set_switches", "asr": True, "request_id": "reopen-on-2"},
        )
        reopened = wait_for_status(
            args.socket,
            lambda status: status.get("asr") is True and status.get("asr_loaded") is True,
            args.settle_timeout_s,
        )
        checks["reopen_converges_ready"] = reopened is not None
        if reopened is None:
            notes.append("构建期间重开识别最终没有就绪")
        reopen_terminals = {
            rid: monitor.terminals(rid) for rid in ("reopen-on-1", "reopen-off", "reopen-on-2")
        }
        checks["reopen_terminals_once"] = all(
            len(found) == 1 for found in reopen_terminals.values()
        )
        checks["reopen_last_intent_applied"] = bool(
            reopen_terminals["reopen-on-2"]
            and reopen_terminals["reopen-on-2"][0].get("outcome") == "applied"
        )
        report["reopen_during_build"] = {
            rid: (found[0] if found else None) for rid, found in reopen_terminals.items()
        }

        # --- 场景三：连点后停在关闭 ---
        for index in range(8):
            send_command(
                args.socket,
                {"cmd": "set_switches", "asr": index % 2 == 0,
                 "request_id": f"hammer-{index}"},
            )
            time.sleep(0.15)
        send_command(
            args.socket, {"cmd": "set_switches", "asr": False, "request_id": "hammer-final"}
        )
        hammered = wait_for_status(
            args.socket,
            lambda status: status.get("asr") is False
            and not status.get("switching")
            and (status.get("desired") or {}).get("pending") is False,
            args.settle_timeout_s,
        )
        checks["hammer_settles_off"] = hammered is not None
        hammer_terminals = [
            monitor.terminals(f"hammer-{index}") for index in range(8)
        ] + [monitor.terminals("hammer-final")]
        checks["hammer_terminals_once"] = all(len(found) == 1 for found in hammer_terminals)

        # 稳定窗口后仍然是关闭，且没有 asr=true 的广播。
        hammer_mark = time.monotonic()
        stable_until = time.monotonic() + args.stable_window_s
        while time.monotonic() < stable_until:
            if get_status(args.socket).get("asr") is not False:
                checks["hammer_settles_off"] = False
                notes.append("连点收尾后开关又变成打开")
                break
            time.sleep(0.3)
        checks["hammer_never_reopens"] = not [
            status for status in monitor.status_after(hammer_mark) if status.get("asr") is True
        ]

        # --- 落盘：偏好等于最终生效态 ---
        persisted = read_persisted(args.switches_conf)
        checks["persisted_matches_effective"] = persisted.get("asr_enabled") is False
        report["persisted"] = persisted
    finally:
        monitor.close()
        if not args.keep_asr_off:
            try:
                settle_to(initial.get("asr", True), "restore", args.settle_timeout_s)
            except OSError:
                pass

    failed = sorted(name for name, ok in checks.items() if not ok)
    report["checks"] = checks
    report["notes"] = notes
    report["passed"] = not failed
    report["failed_checks"] = failed
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
