#!/usr/bin/env python3
"""板上回归：切换等不到现役后端释放时，必须按失败收场并保留现役后端。

背景：切换分三步——流水线线程先放下现役后端（AsrRelease），构建线程再构建目标
后端，最后流水线线程接入。流水线线程被长解码占住时，第一步的回执会迟到；超过
`asr_release_timeout_ms` 就**不能**再往下构建：新旧后端并存会争用 SpacemiT
EP/TCM（实测 SEGV）。此时应当直接报切换失败，现役引擎保持生效。

用法（在板上跑，`/etc/bibit/bibit.conf` 的改动与 `bibit-audio` 重启由调用方负责）：

    # 默认超时（30000ms）：正常切换仍然成功
    python3 scripts/verify_asr_switch_release.py --expect success

    # 复现释放超时：把超时调到 1ms 并重启服务，再跑同一脚本
    #   cp /etc/bibit/bibit.conf /tmp/bibit.conf.bak
    #   printf '\nasr_release_timeout_ms=1\n' >> /etc/bibit/bibit.conf
    #   systemctl --user restart bibit-audio
    python3 scripts/verify_asr_switch_release.py --expect failure
    # 收尾：还原 /etc/bibit/bibit.conf 并重启服务，再跑一次 --expect success

检查项（`--expect failure`）：
  baseline   生效模型与已加载后端可读
  accepted   set_asr_model 被接受，广播 model_switching
  failed     收到 ASR_MODEL_SWITCH_FAILED，rolled_back:false，message 含 release timed out
  kept       生效模型仍是最初那个
  backend    释放命令被跳过：asr_loaded 仍为真（现役后端没被放下）
  stable     再等一段时间，asr_loaded 与生效模型都没变（迟到的释放命令不会补刀）
  restored   收尾把生效引擎切回最初观测到的那个（只记 warn：切回刚放下的 qwen3 时
             上一只 llama-server 可能还没让出端口，服务会回滚，重试即可）

`--expect success` 只检查正常路径（model_switching → model_switched、生效模型换成目标、
asr_loaded 为真、收尾切回）。任一硬性检查失败退出码为 1，报告以 JSON 打到 stdout。
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
RELEASE_ERROR = "ASR_MODEL_SWITCH_FAILED"


def other(model: str) -> str:
    return MODELS[0] if model == MODELS[1] else MODELS[1]


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

    def send(self, **command: object) -> None:
        self.sock.sendall((json.dumps(command) + "\n").encode())

    def mark(self) -> tuple[int, int]:
        with self.lock:
            return len(self.statuses), len(self.events)

    def since(self, mark: tuple[int, int], name: str) -> list[dict]:
        with self.lock:
            statuses, events = self.statuses[mark[0]:], self.events[mark[1]:]
        if name == "status":
            return list(statuses)
        return [event for event in events if event.get("event") == name]

    def snapshot(self, timeout: float = 5.0) -> dict:
        """请求一次 get_status；回执是带 reply:true 的状态快照。"""
        mark = self.mark()
        self.send(cmd="get_status")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event in self.since(mark, "status"):
                if event.get("reply"):
                    return event
            time.sleep(0.02)
        return {}

    def wait_event(self, mark: tuple[int, int], names: str | tuple[str, ...],
                   request_id: str | None = None, timeout: float = 60.0) -> dict:
        wanted = (names,) if isinstance(names, str) else names
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for name in wanted:
                for event in self.since(mark, name):
                    if request_id and event.get("request_id") != request_id:
                        continue
                    return event
            time.sleep(0.05)
        return {}

    def close(self) -> None:
        self.sock.close()


def ensure_ready(link: Link, timeout: float) -> dict:
    """等到识别开关打开且后端加载完成；开关关着时代为打开。"""
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

    def warn(name: str, ok: bool, detail: str = "") -> bool:
        """收尾卫生类检查：不通过时只记 warn，不影响这次验收的结论。"""
        checks.append({"name": name, "result": "pass" if ok else "warn", "detail": detail})
        return ok

    link = None
    deadline = time.monotonic() + args.wait
    while link is None and time.monotonic() < deadline:
        try:
            link = Link(args.socket)
        except OSError:
            time.sleep(0.5)  # 服务刚重启时 socket 还没建出来
    if link is None:
        return {"checks": [{"name": "baseline", "result": "fail",
                            "detail": f"{args.socket} 在 {args.wait}s 内没起来"}],
                "passed": False}
    try:
        start = ensure_ready(link, args.wait)
        initial = start.get("asr_model_type", "")
        target = other(initial)
        if not check("baseline", bool(initial) and start.get("asr_loaded") is True,
                     f"生效模型={initial} asr_loaded={start.get('asr_loaded')}"):
            return {"checks": checks, "passed": False, "initial": initial}

        mark = link.mark()
        link.send(cmd="set_asr_model", model=target, request_id=args.request_id)
        accepted = link.wait_event(mark, "model_switching", args.request_id, args.wait)
        check("accepted", accepted.get("to") == target, f"model_switching={accepted or None}")

        if args.expect == "failure":
            failure = link.wait_event(mark, "error", args.request_id, args.wait)
            check("failed",
                  failure.get("code") == RELEASE_ERROR
                  and failure.get("rolled_back") is False
                  and "release timed out" in str(failure.get("message", "")),
                  f"错误事件={failure or None}")
            settled = link.snapshot()
            check("kept", settled.get("asr_model_type") == initial,
                  f"生效模型={settled.get('asr_model_type')} 期望={initial}")
            check("backend", settled.get("asr_loaded") is True,
                  f"asr_loaded={settled.get('asr_loaded')}")
            time.sleep(args.settle)
            late = link.snapshot()
            check("stable",
                  late.get("asr_loaded") is True and late.get("asr_model_type") == initial
                  and late.get("switching") is None,
                  f"asr_loaded={late.get('asr_loaded')} 生效模型={late.get('asr_model_type')} "
                  f"switching={late.get('switching')}")
        else:
            switched = link.wait_event(mark, "model_switched", args.request_id, args.wait)
            check("switched", switched.get("to") == target, f"model_switched={switched or None}")
            settled = link.snapshot()
            check("loaded", settled.get("asr_model_type") == target
                  and settled.get("asr_loaded") is True,
                  f"生效模型={settled.get('asr_model_type')} asr_loaded={settled.get('asr_loaded')}")

        if args.restore:
            # 切回刚被放下的 qwen3 时，上一只 llama-server 可能还没让出端口，
            # 服务会以 ASR_MODEL_SWITCH_FAILED 回滚（既有行为，不是本次改动）。
            # 收尾只是把板还原，重试几次即可，不通过也只记 warn。
            back: dict = {}
            for attempt in range(3):
                rid = f"{args.request_id}-restore-{attempt}"
                mark = link.mark()
                link.send(cmd="set_asr_model", model=initial, request_id=rid)
                link.wait_event(mark, ("model_switched", "error"), rid, args.wait)
                back = link.snapshot()
                if back.get("asr_model_type") == initial:
                    break
                time.sleep(5)
            warn("restored", back.get("asr_model_type") == initial,
                 f"收尾生效模型={back.get('asr_model_type')} 期望={initial}")
    finally:
        link.close()

    report = {"checks": checks,
              "passed": all(item["result"] in ("pass", "warn") for item in checks)}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", default=AUDIO_SOCK)
    parser.add_argument("--expect", choices=("failure", "success"), default="failure")
    parser.add_argument("--wait", type=float, default=90.0, help="单次等待上限（秒）")
    parser.add_argument("--settle", type=float, default=5.0,
                        help="失败后观察窗口（秒），确认迟到的释放命令不会补刀")
    parser.add_argument("--request-id", default="verify-release")
    parser.add_argument("--no-restore", dest="restore", action="store_false",
                        help="收尾不切回初始引擎")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in report["checks"]:
            print(f"{item['result']:4} {item['name']:9} {item['detail']}")
        print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
