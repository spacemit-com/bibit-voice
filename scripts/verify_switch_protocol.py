#!/usr/bin/env python3
"""板上回归：开关与引擎切换的协议不变量。

覆盖（对应 docs/protocol.md §4.1–§4.4）：

  capabilities   status 声明 partial_switches / queued_model_switch
  bad-command    空字段或非 boolean 字段回 BAD_COMMAND，不静默生效
  escaping       带引号/反斜杠的 request_id 仍产出合法 JSON 且原样回执
  legacy-full    旧客户端全量 payload（无 request_id）照常生效
  same-target    同一目标的两个 request_id 各自拿到终态
  superseded     被后续意图覆盖的请求拿到 model_switch_superseded 终态
  no-retry-loop  失败/不支持的请求有终态，不会反复触发切换

用法（板上，直接对 /run/bibit/audio.sock 说话）：

    python3 scripts/verify_switch_protocol.py

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

AUDIO_SOCK = os.environ.get("AUDIO_SOCK", "/run/bibit/audio.sock")
MODELS = ("sensevoice", "qwen3_asr_0.6b")
TERMINAL_CODES = (
    "ASR_MODEL_SWITCH_FAILED",
    "ASR_INIT_FAILED",
    "UNSUPPORTED_ASR_MODEL",
    "ASR_MODEL_FILES_MISSING",
)


class Link:
    """audio.sock 长连接：后台线程累积事件，命令写完即发。"""

    def __init__(self, path: str):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(0.2)
        self.sock.connect(path)
        self.events: list[dict] = []
        self.raw_lines: list[str] = []
        self.lock = threading.Lock()
        self.broken = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        buffer = bytearray()
        while True:
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                self.broken = True
                return
            if not chunk:
                self.broken = True
                return
            buffer.extend(chunk)
            while b"\n" in buffer:
                raw, _, rest = buffer.partition(b"\n")
                buffer[:] = rest
                text = raw.decode("utf-8", errors="replace")
                with self.lock:
                    self.raw_lines.append(text)
                try:
                    with self.lock:
                        self.events.append(json.loads(text))
                except json.JSONDecodeError:
                    with self.lock:
                        self.events.append({"event": "__invalid_json__", "raw": text})

    def send(self, command: dict) -> None:
        self.sock.sendall((json.dumps(command) + "\n").encode("utf-8"))

    def send_raw(self, payload: str) -> None:
        self.sock.sendall((payload + "\n").encode("utf-8"))

    def snapshot(self) -> list[dict]:
        with self.lock:
            return list(self.events)

    def raw_snapshot(self) -> list[str]:
        with self.lock:
            return list(self.raw_lines)

    def wait_for(self, predicate, timeout: float = 5.0) -> dict | None:
        deadline = time.monotonic() + timeout
        seen = 0
        while time.monotonic() < deadline:
            with self.lock:
                events = self.events[seen:]
                seen = len(self.events)
            for event in events:
                if predicate(event):
                    return event
            time.sleep(0.05)
        return None

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def status_of(link: Link, timeout: float = 3.0) -> dict:
    # 只认这次请求之后到达的 status：从事件流开头扫描会返回连接建立初期的旧快照。
    before = len(link.snapshot())
    link.send({"cmd": "get_status"})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in link.snapshot()[before:]:
            if event.get("event") == "status":
                return event
        time.sleep(0.05)
    return {}


def terminal_for(link: Link, request_id: str, timeout: float = 40.0) -> dict | None:
    """等这个 request_id 的成功/失败/被取代终态之一。"""
    def matches(event: dict) -> bool:
        if event.get("request_id") != request_id:
            return False
        kind = event.get("event")
        if kind in ("model_switched", "model_switch_superseded"):
            return True
        return kind == "error" and event.get("code") in TERMINAL_CODES

    return link.wait_for(matches, timeout)


def other(model: str) -> str:
    return MODELS[0] if model == MODELS[1] else MODELS[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=AUDIO_SOCK, help="audio.sock 路径")
    parser.add_argument("--switch-timeout", type=float, default=40.0, help="单次切换等待上限")
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    notes: dict[str, str] = {}
    link = Link(args.socket)

    # 1. 能力声明
    status = status_of(link)
    checks["capabilities"] = status.get("partial_switches") is True and status.get(
        "queued_model_switch"
    ) is True
    notes["capabilities"] = json.dumps(
        {
            "partial_switches": status.get("partial_switches"),
            "queued_model_switch": status.get("queued_model_switch"),
        }
    )
    if not checks["capabilities"]:
        # 旧服务端没有这些字段：其余用例依赖新语义，直接如实报告并退出。
        notes["note"] = "服务端未声明能力字段，跳过后续新语义用例"
        link.close()
        report = {"checks": checks, "notes": notes, "passed": False}
        print(json.dumps(report, ensure_ascii=False))
        return 1

    baseline = status.get("asr_model_type") or MODELS[0]

    # 2. 边界校验：空字段 / 非 boolean
    link.send({"cmd": "set_switches", "request_id": "probe-empty"})
    empty_reply = link.wait_for(
        lambda e: e.get("event") == "error"
        and e.get("code") == "BAD_COMMAND"
        and e.get("request_id") == "probe-empty"
    )
    link.send_raw('{"cmd":"set_switches","wake":"true","request_id":"probe-type"}')
    type_reply = link.wait_for(
        lambda e: e.get("event") == "error"
        and e.get("code") == "BAD_COMMAND"
        and e.get("request_id") == "probe-type"
    )
    checks["bad-command"] = empty_reply is not None and type_reply is not None
    notes["bad-command"] = f"empty={empty_reply is not None} type={type_reply is not None}"

    # 2b. 命令分派只看键位置的 "cmd"：值里出现 cmd 时不能再改判命令。
    #     载荷必须在真假 cmd 之间插一个键：旧实现（裸子串查找）会读到中间那个值
    #     并分派成 get_status（返回 status），只有新实现才返回 BAD_COMMAND。
    #     少了这个中间键，新旧实现都会回 BAD_COMMAND，检查就失去区分力。
    link.send_raw('{"a":"cmd","x":"get_status","cmd":"unknown_cmd"}')
    dispatch = link.wait_for(
        lambda e: e.get("event") == "error" and e.get("code") == "BAD_COMMAND", 3.0
    )
    checks["command-dispatch"] = dispatch is not None
    notes["command-dispatch"] = (
        f"bad_command_seen={dispatch is not None}"
        " (旧实现会误判成 get_status 并返回 status)"
    )

    # 3a. 字段冒充回归：request_id 的值恰好等于 wake/asr 时，不得被当成字段解析，
    #     否则会静默改掉用户没点的那个开关。
    link.send({"cmd": "set_switches", "wake": True, "asr": True, "request_id": "seed"})
    link.wait_for(lambda e: e.get("event") == "switches_applied", 3.0)
    link.send({"cmd": "set_switches", "request_id": "wake", "asr": False})
    time.sleep(0.5)
    spoofed = status_of(link)
    checks["no-field-spoofing"] = (
        spoofed.get("wake") is True and spoofed.get("asr") is False
    )
    notes["no-field-spoofing"] = json.dumps(
        {"wake": spoofed.get("wake"), "asr": spoofed.get("asr")}
    )
    link.send({"cmd": "set_switches", "asr": True, "request_id": "restore-asr"})
    link.wait_for(lambda e: e.get("event") == "switches_applied", 3.0)

    # 3. request_id 转义：含引号与反斜杠仍须产出合法 JSON 并原样回执
    tricky = 'id"with\\slash'
    link.send({"cmd": "set_switches", "wake": True, "asr": True, "request_id": tricky})
    applied = link.wait_for(
        lambda e: e.get("event") == "switches_applied" and e.get("request_id") == tricky
    )
    invalid = [line for line in link.raw_snapshot() if "__invalid_json__" in line]
    checks["escaping"] = applied is not None and not invalid
    notes["escaping"] = f"applied={applied is not None} invalid_lines={len(invalid)}"

    # 4. 旧客户端全量 payload（无 request_id）
    link.send({"cmd": "set_switches", "wake": True, "asr": True})
    legacy = link.wait_for(
        lambda e: e.get("event") == "switches_applied" and e.get("request_id") == ""
    )
    checks["legacy-full"] = legacy is not None
    notes["legacy-full"] = f"reply={legacy is not None}"

    # 5. 同一目标的两个 request_id 各自拿到终态
    target = other(baseline)
    link.send({"cmd": "set_asr_model", "model": target, "request_id": "same-a"})
    time.sleep(0.3)
    link.send({"cmd": "set_asr_model", "model": target, "request_id": "same-b"})
    first = terminal_for(link, "same-a", args.switch_timeout)
    second = terminal_for(link, "same-b", args.switch_timeout)
    checks["same-target"] = first is not None and second is not None
    notes["same-target"] = json.dumps(
        {"a": (first or {}).get("event"), "b": (second or {}).get("event")}
    )

    # 6. 被后续意图覆盖：先让切到 baseline 的任务在途，A 在途中请求 target（不会被追求），
    #    随后 B 把意图改回 baseline。A 必须拿到 superseded 终态，B 拿到成功终态。
    link.send({"cmd": "set_asr_model", "model": baseline, "request_id": "cover-pre"})
    time.sleep(0.3)
    link.send({"cmd": "set_asr_model", "model": target, "request_id": "cover-a"})
    time.sleep(0.3)
    link.send({"cmd": "set_asr_model", "model": baseline, "request_id": "cover-b"})
    cover_pre = terminal_for(link, "cover-pre", args.switch_timeout)
    cover_a = terminal_for(link, "cover-a", args.switch_timeout)
    cover_b = terminal_for(link, "cover-b", args.switch_timeout)
    checks["superseded"] = (
        cover_a is not None
        and cover_a.get("event") == "model_switch_superseded"
        and cover_b is not None
    )
    notes["superseded"] = json.dumps(
        {
            "pre": (cover_pre or {}).get("event"),
            "a": (cover_a or {}).get("event"),
            "b": (cover_b or {}).get("event"),
        }
    )

    # 7. 失败/不支持的目标必须给终态，且不会循环重试
    link.send({"cmd": "set_asr_model", "model": "not-a-model", "request_id": "fail-a"})
    fail_terminal = terminal_for(link, "fail-a", 10.0)
    switches_before = len(
        [e for e in link.snapshot() if e.get("event") == "model_switching"]
    )
    time.sleep(1.5)
    switches_after = len(
        [e for e in link.snapshot() if e.get("event") == "model_switching"]
    )
    checks["no-retry-loop"] = fail_terminal is not None and switches_after == switches_before
    notes["no-retry-loop"] = json.dumps(
        {
            "terminal": (fail_terminal or {}).get("code"),
            "switching_events": [switches_before, switches_after],
        }
    )

    # 恢复基线引擎与开关
    link.send({"cmd": "set_asr_model", "model": baseline, "request_id": "restore-model"})
    terminal_for(link, "restore-model", args.switch_timeout)
    link.send({"cmd": "set_switches", "wake": True, "asr": True, "request_id": "restore-switches"})
    link.wait_for(lambda e: e.get("event") == "switches_applied", 10.0)

    checks["connected"] = not link.broken
    notes["connected"] = f"broken={link.broken}"
    link.close()

    report = {"checks": checks, "notes": notes, "passed": all(checks.values())}
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
