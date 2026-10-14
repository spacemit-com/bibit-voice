#!/usr/bin/env python3
"""板上回归：第二个实例必须绑不上就退出，且不许碰 8063。

背景：Qwen3 后端（`llama-server` 8063）与 SpacemiT EP/TCM 是**整机**独占资源，
单实例由 `/run/bibit/audio.sock` 保证。但历史实现先初始化 ASR、后绑 socket，
于是第二个实例（例如另一个用户 session 里 enabled 的同一 unit）会先把 llama-server
拉起来，绑不上再退出；退出时不干净就留下占着 8063 的孤儿，把在服务的实例卡在
`qwen3 port already has an HTTP server`。

用法（板上 root 跑；`--as-user` 指定用哪个身份起第二个实例，默认当前用户）：

    python3 scripts/verify_single_instance.py --as-user bibit-test

检查项：
  socket-held   已经有实例在服务 /run/bibit/audio.sock（前置条件）
  rejected      第二个实例非零退出，stderr 说明已有实例在服务
  port-untouched  8063 上的监听者集合没有变化（没有新 llama-server 被拉起来）
所有硬性检查通过退出码为 0，报告以 JSON 打到 stdout。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

AUDIO_SOCK = os.environ.get("AUDIO_SOCK", "/run/bibit/audio.sock")
BIBIT_AUDIO = os.environ.get("BIBIT_AUDIO", "/usr/bin/bibit-audio")
BIBIT_CONF = os.environ.get("BIBIT_CONF", "/etc/bibit/bibit.conf")


def port_listeners(port: int) -> set[str]:
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    pids: set[str] = set()
    for line in out.splitlines():
        if f":{port} " not in line:
            continue
        pids.update(re.findall(r"pid=(\d+)", line))
    return pids


def second_instance_command(as_user: str | None) -> list[str]:
    inner = f"{shlex.quote(BIBIT_AUDIO)} --config {shlex.quote(BIBIT_CONF)}"
    if as_user:
        return ["runuser", "-l", as_user, "-c", inner]
    return [BIBIT_AUDIO, "--config", BIBIT_CONF]


def run(args: argparse.Namespace) -> dict:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "result": "pass" if ok else "fail", "detail": detail})
        return ok

    socket_present = os.path.exists(args.socket)
    if not check("socket-held", socket_present, f"{args.socket} 存在={socket_present}"):
        return {"checks": checks, "passed": False}

    before = port_listeners(args.port)
    command = second_instance_command(args.as_user)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        code, stderr, stdout = completed.returncode, completed.stderr, completed.stdout
    except subprocess.TimeoutExpired as exc:
        code = None
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    time.sleep(args.settle)
    after = port_listeners(args.port)
    combined = (stderr + stdout).strip()

    # 必须看到那句话：只判「非零退出」的话，runuser 自己失败（用户不存在、
    # 没有 login shell 之类）也会被当成通过，等于什么都没测。
    check("rejected",
          code is not None and code != 0 and "another bibit-audio instance" in combined,
          f"exit={code} stderr={combined[-300:] or '(空)'}")
    check("port-untouched", before == after,
          f"8063 监听者 before={sorted(before)} after={sorted(after)}")

    report = {"checks": checks, "passed": all(item["result"] == "pass" for item in checks)}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", default=AUDIO_SOCK)
    parser.add_argument("--port", type=int, default=8063)
    parser.add_argument("--as-user", default=None,
                        help="用哪个身份起第二个实例（默认当前用户）")
    parser.add_argument("--timeout", type=float, default=60.0, help="第二个实例的运行上限（秒）")
    parser.add_argument("--settle", type=float, default=3.0, help="结束后观察窗口（秒）")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in report["checks"]:
            print(f"{item['result']:4} {item['name']:15} {item['detail']}")
        print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
