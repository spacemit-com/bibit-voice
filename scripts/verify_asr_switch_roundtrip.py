#!/usr/bin/env python3
"""板上回归：连着来回切引擎时，上一只 llama-server 必须先让出 8063。

背景：切到 SenseVoice 会放下 Qwen3 后端（结束 llama-server），切回 Qwen3 时
`startServer` 会先探一次 /health，只要还有东西应答就直接失败：
`qwen3 port already has an HTTP server; stop the old llama-server first`。
实测「切到 SenseVoice 后立刻切回」会命中这条——监听 socket 比进程活得久。

用法（板上，直接对 /run/bibit/audio.sock 说话）：

    python3 scripts/verify_asr_switch_roundtrip.py --rounds 3

检查项（每轮）：
  switch-out   切到另一引擎成功（model_switched、无 error）
  switch-back  立刻切回来成功——这是本脚本的重点
  port-freed   切回 SenseVoice 生效后，8063 上不应再有东西应答 /health
  settled      收尾状态回到本轮起点（生效引擎与 asr_loaded 一致）
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
QWEN3_HOST = os.environ.get("QWEN3_HOST", "127.0.0.1")
QWEN3_PORT = int(os.environ.get("QWEN3_PORT", "8063"))
MODELS = ("sensevoice", "qwen3_asr_0.6b")


def other(model: str) -> str:
    return MODELS[0] if model == MODELS[1] else MODELS[1]


def port_answers(host: str, port: int, timeout: float = 0.5) -> bool:
    """有没有东西在监听并应答 HTTP（和客户端 startServer 的探法一致）。"""
    try:
        with socket.create_connection((host, port), timeout=timeout) as probe:
            probe.sendall(b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n")
            probe.settimeout(timeout)
            try:
                return bool(probe.recv(64))
            except OSError:
                return True  # 连上了就算有人在听
    except OSError:
        return False


class Link:
    """audio.sock 单连接：发命令，后台线程累积事件与状态快照。"""

    def __init__(self, path: str):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.sock.settimeout(0.2)
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.statuses: list[dict] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        while True:
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if not data:
                return
            self.buffer += data
            while b"\n" in self.buffer:
                line, _, rest = bytes(self.buffer).partition(b"\n")
                self.buffer = bytearray(rest)
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                with self.lock:
                    self.events.append(event)
                    if event.get("event") == "status":
                        self.statuses.append(event)

    def mark(self) -> tuple[int, int]:
        with self.lock:
            return len(self.statuses), len(self.events)

    def since(self, mark: tuple[int, int], name: str) -> list[dict]:
        with self.lock:
            statuses, events = self.statuses[mark[0]:], self.events[mark[1]:]
        if name == "status":
            return list(statuses)
        return [event for event in events if event.get("event") == name]

    def send(self, **command: object) -> None:
        self.sock.sendall((json.dumps(command) + "\n").encode())

    def snapshot(self, timeout: float = 5.0) -> dict:
        mark = self.mark()
        self.send(cmd="get_status")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event in self.since(mark, "status"):
                if event.get("reply"):
                    return event
            time.sleep(0.02)
        return {}

    def switch(self, model: str, request_id: str, timeout: float) -> dict:
        """发一次切换，返回终态事件（model_switched 或 error）。"""
        mark = self.mark()
        self.send(cmd="set_asr_model", model=model, request_id=request_id)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for name in ("model_switched", "error"):
                for event in self.since(mark, name):
                    if event.get("request_id") == request_id:
                        return event
            time.sleep(0.02)
        return {}

    def close(self) -> None:
        self.sock.close()


def connect(path: str, timeout: float) -> Link | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return Link(path)
        except OSError:
            time.sleep(0.5)
    return None


def ensure_ready(link: Link, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    status: dict = {}
    while time.monotonic() < deadline:
        status = link.snapshot()
        if status.get("asr") and status.get("asr_loaded"):
            return status
        if status and not status.get("asr"):
            link.send(cmd="set_switches", wake=True, asr=True)
        time.sleep(1.0)
    return status


def run(args: argparse.Namespace) -> dict:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "result": "pass" if ok else "fail", "detail": detail})
        return ok

    link = connect(args.socket, args.wait)
    if link is None:
        return {"checks": [{"name": "baseline", "result": "fail",
                            "detail": f"{args.socket} 在 {args.wait}s 内没起来"}],
                "passed": False}
    try:
        start = ensure_ready(link, args.wait)
        initial = start.get("asr_model_type", "")
        if not check("baseline", bool(initial) and start.get("asr_loaded") is True,
                     f"生效模型={initial} asr_loaded={start.get('asr_loaded')}"):
            return {"checks": checks, "passed": False}

        for index in range(1, args.rounds + 1):
            # 每轮都从「Qwen3 生效」一侧出发，这样两次切换都要重建 llama-server。
            if initial != MODELS[1]:
                outcome = link.switch(MODELS[1], f"rt-{index}-prep", args.wait)
                if not check(f"round{index}-prep",
                             outcome.get("event") == "model_switched",
                             f"先切到 {MODELS[1]}: {outcome or None}"):
                    break
            out = link.switch(MODELS[0], f"rt-{index}-out", args.wait)
            check(f"round{index}-switch-out", out.get("event") == "model_switched",
                  f"{MODELS[1]}→{MODELS[0]}: {out or None}")
            back = link.switch(MODELS[1], f"rt-{index}-back", args.wait)
            check(f"round{index}-switch-back", back.get("event") == "model_switched",
                  f"立刻切回 {MODELS[1]}: {back or None}")
            time.sleep(args.settle)
            snapshot = link.snapshot()
            if snapshot.get("asr_model_type") == MODELS[0]:
                check(f"round{index}-port-freed", not port_answers(args.qwen3_host, args.qwen3_port),
                      f"{MODELS[0]} 生效时 {args.qwen3_host}:{args.qwen3_port} 仍有人在听")
            check(f"round{index}-settled",
                  snapshot.get("asr_model_type") == MODELS[1]
                  and snapshot.get("asr_loaded") is True,
                  f"生效模型={snapshot.get('asr_model_type')} "
                  f"asr_loaded={snapshot.get('asr_loaded')}")
    finally:
        link.close()

    report = {"checks": checks, "passed": all(item["result"] == "pass" for item in checks)}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", default=AUDIO_SOCK)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--wait", type=float, default=90.0, help="单次等待上限（秒）")
    parser.add_argument("--settle", type=float, default=2.0, help="切换后观察窗口（秒）")
    parser.add_argument("--qwen3-host", default=QWEN3_HOST)
    parser.add_argument("--qwen3-port", type=int, default=QWEN3_PORT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in report["checks"]:
            print(f"{item['result']:4} {item['name']:22} {item['detail']}")
        print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
