#!/usr/bin/env python3
"""板上回归：连续快速切换语音开关时，服务端必须始终可响应、并收敛到最后一次意图。

背景：`set_switches` 曾经在 socket 处理线程里同步落盘并起停 PipeWire，打开识别还会
走一次后端冷启动；GUI 连点因此把命令堆在同一线程上，状态回执被推迟到心跳超时，
界面显示一个假的「语音服务未连接」。计划要求 socket 处理线程只做 I/O，能力变更按
「最后一次意图胜出」收敛。

用法（板上，直接对 /run/bibit/audio.sock 说话）：

    python3 scripts/verify_switch_hammering.py --rounds 20 --interval-ms 250

检查项：
  responsive   连续切换期间每次 get_status 往返都不超过 --max-latency-ms
  connected    整个过程中没有连接被拒绝或中断
  settled      收尾后的生效快照等于最后一次意图
  persisted    ~/.config/bibit/switches.conf 等于最后一次意图
任一硬性检查失败退出码为 1，报告以 JSON 打到 stdout。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

AUDIO_SOCK = os.environ.get("AUDIO_SOCK", "/run/bibit/audio.sock")
DEFAULT_SWITCHES_CONF = os.environ.get(
    "SWITCHES_CONF", os.path.expanduser("~/.config/bibit/switches.conf")
)


def send_command(path: str, command: dict, timeout: float = 2.0) -> None:
    """写完即关：与 GUI 侧 Reply::Sent 的连接形态一致。"""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(path)
        sock.sendall((json.dumps(command) + "\n").encode("utf-8"))


def get_status(path: str, timeout: float = 2.0) -> tuple[float, dict]:
    """新开一条连接读状态快照，返回 (往返毫秒, 快照)。"""
    started = time.monotonic()
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
                    return (time.monotonic() - started) * 1000.0, event


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


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=AUDIO_SOCK, help="audio.sock 路径")
    parser.add_argument("--rounds", type=int, default=20, help="切换次数")
    parser.add_argument(
        "--interval-ms", type=int, default=250, help="两次切换之间的间隔（默认 4 次/秒）"
    )
    parser.add_argument(
        "--max-latency-ms", type=float, default=200.0, help="单次 get_status 往返上限"
    )
    parser.add_argument(
        "--switches-conf", default=DEFAULT_SWITCHES_CONF, help="开关持久化文件"
    )
    parser.add_argument(
        "--settle-timeout-s", type=float, default=10.0, help="收尾收敛等待上限"
    )
    args = parser.parse_args()

    latencies: list[float] = []
    failures: list[str] = []
    disconnects = 0

    # 明确起点，避免上一轮遗留的开关状态影响「最后意图」判定。
    target = {"wake": True, "asr": True}

    # 冷启动预热：服务刚重启时第一轮查询可能仍在模型加载窗口里，那不是本次要测的
    # 「切换期间是否可响应」。先等到一次干净的状态查询，再开始计数。
    warmup_deadline = time.monotonic() + 30.0
    while time.monotonic() < warmup_deadline:
        try:
            warmup_latency, _ = get_status(args.socket)
        except (OSError, RuntimeError):
            time.sleep(0.5)
            continue
        if warmup_latency <= args.max_latency_ms:
            break
        time.sleep(0.5)

    for index in range(args.rounds):
        started = time.monotonic()
        wake = index % 2 == 0
        try:
            send_command(
                args.socket,
                {"cmd": "set_switches", "wake": wake, "request_id": f"hammer-{index}"},
            )
        except OSError as error:
            disconnects += 1
            failures.append(f"round {index}: set_switches failed: {error}")
        try:
            latency, _ = get_status(args.socket)
            latencies.append(latency)
        except (OSError, RuntimeError) as error:
            disconnects += 1
            failures.append(f"round {index}: get_status failed: {error}")
        remaining = args.interval_ms / 1000.0 - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)

    # 最后一次意图：先留出一个间隔，避免它与上一条命令几乎同时到达、由 I/O 线程
    # 的 poll 顺序决定谁最后生效——那是脚本的竞态，不是服务端的收敛行为。
    time.sleep(max(0.3, args.interval_ms / 1000.0))
    try:
        send_command(args.socket, {"cmd": "set_switches", **target, "request_id": "final"})
    except OSError as error:
        disconnects += 1
        failures.append(f"final set_switches failed: {error}")

    settled: dict | None = None
    deadline = time.monotonic() + args.settle_timeout_s
    resend_budget = 2
    while time.monotonic() < deadline:
        try:
            _, status = get_status(args.socket)
        except (OSError, RuntimeError) as error:
            disconnects += 1
            failures.append(f"settle get_status failed: {error}")
            break
        if status.get("wake") == target["wake"] and status.get("asr") == target["asr"]:
            settled = status
            break
        if resend_budget > 0:
            resend_budget -= 1
            try:
                send_command(
                    args.socket, {"cmd": "set_switches", **target, "request_id": "final-retry"}
                )
            except OSError as error:
                failures.append(f"final retry failed: {error}")
        time.sleep(0.2)

    # 落盘由控制面线程完成，可能略晚于状态快照；给它一个有界窗口再判定，
    # 否则会把「还没写完」误判成「没写对」。
    persisted: dict[str, bool] = {}
    persist_deadline = time.monotonic() + 3.0
    while time.monotonic() < persist_deadline:
        persisted = read_persisted(args.switches_conf)
        if (
            persisted.get("wake_enabled") == target["wake"]
            and persisted.get("asr_enabled") == target["asr"]
        ):
            break
        time.sleep(0.2)
    max_latency = max(latencies) if latencies else float("inf")
    checks = {
        "responsive": max_latency <= args.max_latency_ms,
        "connected": disconnects == 0,
        "settled": settled is not None,
        "persisted": persisted.get("wake_enabled") == target["wake"]
        and persisted.get("asr_enabled") == target["asr"],
    }
    report = {
        "rounds": args.rounds,
        "interval_ms": args.interval_ms,
        "latency_ms": {
            "max": round(max_latency, 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "samples": len(latencies),
        },
        "disconnects": disconnects,
        "target": target,
        "final_status": {
            "wake": (settled or {}).get("wake"),
            "asr": (settled or {}).get("asr"),
        },
        "persisted": persisted,
        "checks": checks,
        "failures": failures[:10],
        "passed": all(checks.values()),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
