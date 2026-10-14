#!/usr/bin/env python3
"""Tool/MCP adapter for the existing Bibit audio Unix Socket protocol."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import socket
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable


DEFAULT_TIMEOUT = 30.0
MODEL_TIMEOUT = 120.0


class VoiceToolError(RuntimeError):
    def __init__(self, code: str, message: str, event: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.event = event or {}

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "event": self.event}


def runtime_socket_path(name: str) -> str:
    # 整机唯一实例：全局固定 socket 路径（/run/bibit 由 tmpfiles.d 创建）。
    return os.path.join("/run/bibit", name)


def default_audio_socket_path() -> str:
    return runtime_socket_path("audio.sock")


def default_socket_path() -> str:
    return runtime_socket_path("tool.sock")


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, **self.data}


class VoiceSocketClient:
    """One short-lived client of bibit-audio's multi-client JSONL socket."""

    def __init__(self, path: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        self.path = path or default_socket_path()
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.buffer = bytearray()
        self._capabilities_cache: dict[str, bool] | None = None

    def capabilities(self) -> dict[str, bool]:
        """探测服务端能力。旧服务端没有这些字段，一律按不支持处理。

        partial_switches    服务端接受省略 wake/asr 的字段级提交
        queued_model_switch 在途切换时提交的新目标会被记录并在提交点后收敛
        switch_settlement   服务端用 switches_settled 给出开关意图的终态
                            （switches_applied 只是受理回执，排队中的打开可能几十秒才达成）
        """
        if self._capabilities_cache is None:
            status = self.get_status().as_dict()["status"]
            self._capabilities_cache = {
                "partial_switches": status.get("partial_switches") is True,
                "queued_model_switch": status.get("queued_model_switch") is True,
                "switch_settlement": status.get("switch_settlement") is True,
            }
        return self._capabilities_cache

    def __enter__(self) -> "VoiceSocketClient":
        self.connect()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    def connect(self) -> None:
        if self.sock is not None:
            return
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(max(0.1, self.timeout))
        try:
            sock.connect(self.path)
        except OSError as exc:
            sock.close()
            raise VoiceToolError(
                "SERVICE_OFFLINE", f"无法连接 Bibit 音频服务：{self.path}: {exc}"
            ) from exc
        self.sock = sock

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def send(self, command: dict[str, Any]) -> None:
        if self.sock is None:
            self.connect()
        payload = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
        try:
            assert self.sock is not None
            self.sock.sendall(payload.encode("utf-8") + b"\n")
        except OSError as exc:
            raise VoiceToolError("IPC_SEND_FAILED", f"发送 IPC 命令失败：{exc}") from exc

    def read_event(
        self, deadline: float | None = None, wait_forever: bool = False
    ) -> dict[str, Any]:
        if self.sock is None:
            self.connect()
        if deadline is None and not wait_forever:
            deadline = time.monotonic() + self.timeout
        assert self.sock is not None
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self.buffer[:newline])
                del self.buffer[: newline + 1]
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    return event
                continue

            if wait_forever:
                self.sock.settimeout(None)
            else:
                assert deadline is not None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VoiceToolError("TIMEOUT", "等待 Bibit 音频事件超时")
                self.sock.settimeout(remaining)
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout as exc:
                raise VoiceToolError("TIMEOUT", "等待 Bibit 音频事件超时") from exc
            except OSError as exc:
                raise VoiceToolError("IPC_READ_FAILED", f"读取 IPC 事件失败：{exc}") from exc
            if not chunk:
                raise VoiceToolError("SERVICE_DISCONNECTED", "Bibit 音频服务连接已断开")
            self.buffer.extend(chunk)

    def wait_for(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        timeout: float | None = None,
        errors: bool = True,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            event = self.read_event(deadline)
            if errors and event.get("event") == "error":
                raise VoiceToolError(
                    str(event.get("code", "AUDIO_ERROR")),
                    str(event.get("message") or event.get("code") or "音频服务错误"),
                    event,
                )
            if predicate(event):
                return event

    def get_status(self, timeout: float | None = None) -> ToolResult:
        self.send({"cmd": "get_status"})
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        status: dict[str, Any] | None = None
        parameters: dict[str, Any] | None = None
        while status is None or parameters is None:
            event = self.read_event(deadline)
            kind = event.get("event")
            if kind in ("status", "ready"):
                status = event
            elif kind == "parameters":
                parameters = event
            elif kind == "error":
                raise VoiceToolError(
                    str(event.get("code", "AUDIO_ERROR")),
                    str(event.get("message") or event.get("code") or "音频服务错误"),
                    event,
                )
        return ToolResult(True, {"status": status, "parameters": parameters})

    def set_switches(
        self,
        wake: bool | None = None,
        asr: bool | None = None,
        request_id: str | None = None,
    ) -> ToolResult:
        """按字段提交开关意图。未给出的字段由服务端沿用当前生效值。"""
        if wake is None and asr is None:
            raise VoiceToolError("BAD_COMMAND", "至少需要给出 wake 或 asr 之一")
        # 先探测能力：探测要读一帧状态，放在发命令之后会把回执一起吃掉。
        capabilities = self.capabilities()
        if (wake is None or asr is None) and not capabilities["partial_switches"]:
            # 旧服务端把缺失字段按默认 true 处理，只发一个字段会把另一个开关一起改掉；
            # 这种情况下由客户端补全当前生效值，退回全量提交。
            state = self.get_status().as_dict()["status"]
            if wake is None:
                wake = bool(state.get("wake", True))
            if asr is None:
                asr = bool(state.get("asr", True))
        request_id = request_id or f"sw-{uuid.uuid4().hex[:12]}"
        command: dict[str, Any] = {"cmd": "set_switches"}
        if wake is not None:
            command["wake"] = wake
        if asr is not None:
            command["asr"] = asr
        command["request_id"] = request_id
        self.send(command)
        # 一次循环里同时收集：先等回执会把 status 吃掉，再等 status 只能超时。
        want_applied = capabilities["partial_switches"]
        # 服务端声明终态能力时，受理回执不算完成：打开识别要等后端构建完（或被取代），
        # 否则调用方会把「已受理」当成「已打开」。等待窗口按模型切换的量级给。
        want_settled = capabilities["switch_settlement"]
        status: dict[str, Any] | None = None
        applied: dict[str, Any] | None = None
        settled: dict[str, Any] | None = None
        deadline = time.monotonic() + (MODEL_TIMEOUT if want_settled else self.timeout)
        while not (
            status is not None
            and (settled is not None if want_settled else not want_applied or applied is not None)
        ):
            event = self.read_event(deadline)
            kind = event.get("event")
            event_request_id = event.get("request_id")
            if kind in ("status", "ready"):
                if status is None:
                    status = event
            elif event_request_id == request_id and kind == "switches_applied":
                # 受理回执：开关态与采集已按这次命令调整，不代表识别已经可用。
                applied = event
            elif event_request_id == request_id and kind == "switches_settled":
                if event.get("outcome") == "superseded":
                    # 意图被更新的命令取代且未生效：显式报错，别让调用方以为开关已到位。
                    raise VoiceToolError(
                        "SWITCH_INTENT_SUPERSEDED",
                        "开关意图已被更新的命令取代，未生效",
                        event,
                    )
                settled = event
            elif kind == "error":
                # 带 request_id 的错误只认自己的；广播里别人失败的事件不能算到本次调用头上。
                if event_request_id and event_request_id != request_id:
                    continue
                raise VoiceToolError(
                    str(event.get("code", "BAD_COMMAND")),
                    str(event.get("message") or event.get("code") or "开关命令被拒绝"),
                    event,
                )
        return ToolResult(
            True,
            {
                "status": status,
                "applied": applied,
                "settled": settled,
                "request_id": request_id,
            },
        )

    def set_thresholds(self, arguments: dict[str, Any]) -> ToolResult:
        current = self.get_status().as_dict()
        parameters = current["parameters"]
        command = {
            "cmd": "set_thresholds",
            "kws_threshold": arguments.get(
                "kws_threshold", parameters.get("kws_threshold", 0.12)
            ),
            "kws_score": arguments.get("kws_score", parameters.get("kws_score", 4.5)),
            "vad_threshold": arguments.get(
                "vad_threshold", parameters.get("vad_threshold", 0.30)
            ),
            "min_silence": arguments.get(
                "min_silence", parameters.get("min_silence", 0.50)
            ),
            "wake_guard_ms": arguments.get(
                "wake_guard_ms", parameters.get("wake_guard_ms", 180)
            ),
        }
        self.send(command)
        status = self.wait_for(lambda e: e.get("event") in ("status", "ready"))
        return ToolResult(True, {"status": status, "applied": command})

    def set_kws_debug(self, enabled: bool) -> ToolResult:
        self.send({"cmd": "set_kws_debug", "enabled": enabled})
        status = self.wait_for(lambda e: e.get("event") in ("status", "ready"))
        return ToolResult(True, {"status": status, "enabled": enabled})

    def set_model(self, model: str, timeout: float = MODEL_TIMEOUT) -> ToolResult:
        if model not in ("sensevoice", "qwen3_asr_0.6b"):
            raise VoiceToolError("UNSUPPORTED_ASR_MODEL", f"不支持的 ASR 模型：{model}")
        # 先探测能力再发切换：探测要读一帧状态，放在切换之后会把在途终态事件一起吃掉。
        capabilities = self.capabilities()
        # 带请求标识：并发/被合并的切换里，结果必须能对应回本次调用，而不是靠模型名猜。
        request_id = f"model-{uuid.uuid4().hex[:12]}"
        self.send({"cmd": "set_asr_model", "model": model, "request_id": request_id})
        deadline = time.monotonic() + timeout
        switching_seen = False
        while True:
            event = self.read_event(deadline)
            kind = event.get("event")
            event_request_id = event.get("request_id")
            # 带 request_id 的事件只认自己的；旧服务端不带时才按模型名兜底。
            if event_request_id and event_request_id != request_id:
                continue
            if kind == "error":
                code = str(event.get("code", "ASR_MODEL_SWITCH_FAILED"))
                if code == "ASR_MODEL_SWITCH_BUSY" and capabilities.get(
                    "queued_model_switch"
                ):
                    # 服务端已记录这次意图并会在提交点后收敛，继续等终态而不是当作失败。
                    switching_seen = True
                    continue
                raise VoiceToolError(
                    code,
                    str(event.get("message") or code or "ASR 模型切换失败"),
                    event,
                )
            if kind == "model_switch_superseded":
                # 终态之一：本次目标被更新的意图覆盖。调用方必须能看出「目标没有生效」，
                # 只回一个 ok:true 会让人以为引擎已经切过去了。
                active = str(event.get("active_model") or "")
                if active == model:
                    return ToolResult(
                        True,
                        {
                            "model": model,
                            "superseded": True,
                            "event": event,
                            "request_id": request_id,
                        },
                    )
                raise VoiceToolError(
                    "ASR_MODEL_SWITCH_SUPERSEDED",
                    f"目标 {model} 被更新的意图取代，当前生效 {active or 'unknown'}",
                    event,
                )
            if kind == "model_switching":
                switching_seen = True
            if kind == "model_switched" and (
                event.get("request_id") == request_id or event.get("model") == model
            ):
                return ToolResult(
                    True, {"model": model, "event": event, "request_id": request_id}
                )
            if kind in ("status", "ready") and event.get("asr_model_type") == model:
                if event.get("asr_loaded") or not event.get("asr", True):
                    return ToolResult(
                        True,
                        {"model": model, "status": event, "switching_seen": switching_seen},
                    )

    def cancel(self, session_id: int | None = None) -> ToolResult:
        state = self.get_status(timeout=min(self.timeout, 5.0)).as_dict()["status"]
        if state.get("mode") == "idle":
            return ToolResult(True, {"already_idle": True, "status": state})
        command = {"cmd": "cancel"}
        if session_id:
            command["session_id"] = int(session_id)
        self.send(command)
        event = self.wait_for(
            lambda e: e.get("event") == "session_cancelled"
            or (e.get("event") in ("status", "ready") and e.get("mode") == "idle")
            or (e.get("event") == "error" and e.get("code") == "SESSION_MISMATCH")
        )
        if event.get("event") == "error":
            raise VoiceToolError("SESSION_MISMATCH", "语音会话已切换，取消未执行", event)
        return ToolResult(True, {"event": event})

    def transcribe_once(self, timeout: float = DEFAULT_TIMEOUT) -> ToolResult:
        if timeout <= 0:
            raise VoiceToolError("INVALID_ARGUMENTS", "timeout_seconds 必须大于 0")
        snapshot = self.get_status(timeout=min(timeout, 5.0)).as_dict()
        state = snapshot["status"]
        if not state.get("audio_ready"):
            raise VoiceToolError("AUDIO_NOT_READY", "麦克风采集尚未就绪", state)
        if not state.get("asr"):
            raise VoiceToolError("ASR_DISABLED", "ASR 当前已关闭", state)
        if not state.get("asr_loaded"):
            raise VoiceToolError("ASR_NOT_LOADED", "ASR 模型尚未加载完成", state)
        if state.get("mode") != "idle":
            raise VoiceToolError(
                "VOICE_SESSION_BUSY", f"已有语音会话正在运行：{state.get('mode')}", state
            )

        self.send({"cmd": "start_manual_asr"})
        deadline = time.monotonic() + timeout
        session_id: int | None = None
        while True:
            event = self.read_event(deadline)
            kind = event.get("event")
            event_session = event.get("session_id")
            if kind == "error":
                # SESSION_MISMATCH 广播携带 session_id，只对发起 cancel 的客户端
                # 有意义；本会话的 error 不带该字段。
                if event_session is not None:
                    continue
                raise VoiceToolError(
                    str(event.get("code", "ASR_FAILED")),
                    str(event.get("message") or event.get("code") or "ASR 失败"),
                    event,
                )
            if kind == "listening":
                session_id = int(event_session or 0)
            if session_id is not None and event_session not in (None, session_id):
                continue
            if kind == "asr_final":
                # transcribe_once owns one utterance. Stop the automatic follow-up
                # session without changing the underlying pipeline implementation.
                self.send({"cmd": "cancel", "session_id": session_id or 0})
                followup_cancelled = False
                try:
                    self.wait_for(
                        lambda e: e.get("event") == "session_cancelled",
                        timeout=min(2.0, max(0.1, deadline - time.monotonic())),
                        errors=False,
                    )
                    followup_cancelled = True
                except VoiceToolError as exc:
                    if exc.code != "TIMEOUT":
                        raise
                return ToolResult(
                    True,
                    {
                        "session_id": session_id,
                        "text": str(event.get("text", "")),
                        "decode_ms": event.get("decode_ms"),
                        "audio_duration_ms": event.get("audio_duration_ms"),
                        "rtf": event.get("rtf"),
                        "followup_cancelled": followup_cancelled,
                    },
                )
            if kind == "session_exit":
                return ToolResult(
                    True,
                    {"session_id": session_id, "text": "", "exit_requested": True},
                )
            if kind == "session_cancelled":
                raise VoiceToolError("SESSION_CANCELLED", "语音会话已取消", event)

    def wait_for_interaction(self, timeout: float = DEFAULT_TIMEOUT) -> ToolResult:
        """Wait for a future KWS wake and the ASR final produced by that session."""
        if timeout <= 0:
            raise VoiceToolError("INVALID_ARGUMENTS", "timeout_seconds 必须大于 0")
        deadline = time.monotonic() + timeout
        session_id: int | None = None
        keyword = ""
        while True:
            event = self.read_event(deadline)
            kind = event.get("event")
            event_session = event.get("session_id")
            if kind == "wake_triggered":
                session_id = int(event_session or 0)
                keyword = str(event.get("keyword", ""))
                continue
            if session_id is None or event_session not in (None, session_id):
                continue
            if kind == "error":
                raise VoiceToolError(
                    str(event.get("code", "ASR_FAILED")),
                    str(event.get("message") or event.get("code") or "唤醒会话失败"),
                    event,
                )
            if kind == "asr_final":
                return ToolResult(
                    True,
                    {
                        "session_id": session_id,
                        "keyword": keyword,
                        "text": str(event.get("text", "")),
                        "decode_ms": event.get("decode_ms"),
                        "audio_duration_ms": event.get("audio_duration_ms"),
                        "rtf": event.get("rtf"),
                    },
                )
            if kind in ("session_cancelled", "session_exit"):
                return ToolResult(
                    True,
                    {
                        "session_id": session_id,
                        "keyword": keyword,
                        "text": "",
                        "ended_by": kind,
                    },
                )

    def events(self, timeout: float | None = None) -> Iterable[dict[str, Any]]:
        deadline = None if timeout is None or timeout <= 0 else time.monotonic() + timeout
        while True:
            yield self.read_event(deadline, wait_forever=deadline is None)


class VoiceToolProxy:
    """Multi-client tool.sock proxy for the unchanged bibit-audio socket."""

    def __init__(
        self,
        listen_path: str,
        upstream_path: str,
        retry_interval: float = 0.5,
    ):
        self.listen_path = listen_path
        self.upstream_path = upstream_path
        self.retry_interval = max(0.1, retry_interval)
        self.selector = selectors.DefaultSelector()
        self.listener: socket.socket | None = None
        self.upstream: socket.socket | None = None
        self.upstream_buffer = bytearray()
        self.clients: dict[socket.socket, bytearray] = {}
        self.running = False
        self.next_upstream_attempt = 0.0

    def _log(self, message: str) -> None:
        print(f"bibit-voice-tool: {message}", file=sys.stderr, flush=True)

    def _prepare_listener(self) -> None:
        parent = os.path.dirname(self.listen_path)
        os.makedirs(parent, mode=0o700, exist_ok=True)
        if os.path.lexists(self.listen_path):
            mode = os.lstat(self.listen_path).st_mode
            if not stat.S_ISSOCK(mode):
                raise VoiceToolError(
                    "TOOL_SOCKET_INVALID",
                    f"Tool Socket 路径已存在且不是 Socket：{self.listen_path}",
                )
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.2)
            try:
                probe.connect(self.listen_path)
            except OSError:
                os.unlink(self.listen_path)
            else:
                raise VoiceToolError(
                    "TOOL_ALREADY_RUNNING",
                    f"Voice Tool 已在监听：{self.listen_path}",
                )
            finally:
                probe.close()

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(self.listen_path)
        os.chmod(self.listen_path, 0o600)
        listener.listen(8)
        listener.setblocking(False)
        self.listener = listener
        self.selector.register(listener, selectors.EVENT_READ, "listener")

    def _send_client(self, client: socket.socket, payload: bytes) -> bool:
        try:
            written = client.send(payload)
        except (BlockingIOError, BrokenPipeError, ConnectionResetError, OSError):
            self._remove_client(client)
            return False
        if written != len(payload):
            self._remove_client(client)
            return False
        return True

    def _broadcast(self, payload: bytes) -> None:
        for client in list(self.clients):
            self._send_client(client, payload)

    def _broadcast_event(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self._broadcast(payload.encode("utf-8") + b"\n")

    def _remove_client(self, client: socket.socket) -> None:
        if client not in self.clients:
            return
        self.clients.pop(client, None)
        try:
            self.selector.unregister(client)
        except (KeyError, ValueError):
            pass
        client.close()

    def _close_clients(self) -> None:
        for client in list(self.clients):
            self._remove_client(client)

    def _disconnect_upstream(self, reason: str) -> None:
        if self.upstream is not None:
            try:
                self.selector.unregister(self.upstream)
            except (KeyError, ValueError):
                pass
            self.upstream.close()
            self.upstream = None
        self.upstream_buffer.clear()
        self.next_upstream_attempt = time.monotonic() + self.retry_interval
        if self.clients:
            self._broadcast_event(
                {
                    "event": "error",
                    "code": "SERVICE_DISCONNECTED",
                    "message": "Voice Tool 与底层音频服务连接已断开",
                }
            )
            # Qt reconnects to tool.sock and resets its health indicators.
            self._close_clients()
        self._log(f"upstream offline: {reason}")

    def _connect_upstream(self) -> None:
        if self.upstream is not None or time.monotonic() < self.next_upstream_attempt:
            return
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        upstream.settimeout(0.3)
        try:
            upstream.connect(self.upstream_path)
        except OSError as exc:
            upstream.close()
            self.next_upstream_attempt = time.monotonic() + self.retry_interval
            self._log(f"waiting for upstream {self.upstream_path}: {exc}")
            return
        upstream.setblocking(False)
        self.upstream = upstream
        self.upstream_buffer.clear()
        self.selector.register(upstream, selectors.EVENT_READ, "upstream")
        self._log(f"upstream connected: {self.upstream_path}")
        self._broadcast_event(
            {"event": "tool_state", "state": "online", "upstream": self.upstream_path}
        )
        self._send_upstream(b'{"cmd":"get_status"}\n')

    def _send_upstream(self, payload: bytes) -> bool:
        if self.upstream is None:
            return False
        try:
            written = self.upstream.send(payload)
        except (BlockingIOError, BrokenPipeError, ConnectionResetError, OSError) as exc:
            self._disconnect_upstream(str(exc))
            return False
        if written != len(payload):
            self._disconnect_upstream("partial command write")
            return False
        return True

    def _accept_clients(self) -> None:
        assert self.listener is not None
        while True:
            try:
                client, _ = self.listener.accept()
            except BlockingIOError:
                return
            client.setblocking(False)
            self.clients[client] = bytearray()
            self.selector.register(client, selectors.EVENT_READ, "client")

    def _handle_client(self, client: socket.socket) -> None:
        try:
            chunk = client.recv(65536)
        except BlockingIOError:
            return
        except OSError:
            self._remove_client(client)
            return
        if not chunk:
            self._remove_client(client)
            return

        buffer = self.clients.get(client)
        if buffer is None:
            return
        buffer.extend(chunk)
        if len(buffer) > 16384 and b"\n" not in buffer:
            self._send_client(
                client, b'{"event":"error","code":"BAD_COMMAND","message":"command too large"}\n'
            )
            buffer.clear()
            return

        while b"\n" in buffer:
            raw, _, remainder = buffer.partition(b"\n")
            buffer[:] = remainder
            if not raw:
                continue
            try:
                command = json.loads(raw)
            except json.JSONDecodeError:
                self._send_client(
                    client, b'{"event":"error","code":"BAD_COMMAND","message":"invalid JSON"}\n'
                )
                continue
            if not isinstance(command, dict):
                self._send_client(
                    client, b'{"event":"error","code":"BAD_COMMAND","message":"object required"}\n'
                )
                continue
            payload = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
            if self.upstream is None:
                self._send_client(
                    client,
                    b'{"event":"error","code":"SERVICE_OFFLINE","message":"audio service unavailable"}\n',
                )
                self._remove_client(client)
                return
            if not self._send_upstream(payload.encode("utf-8") + b"\n"):
                return

    def _handle_upstream(self) -> None:
        assert self.upstream is not None
        try:
            chunk = self.upstream.recv(65536)
        except BlockingIOError:
            return
        except OSError as exc:
            self._disconnect_upstream(str(exc))
            return
        if not chunk:
            self._disconnect_upstream("connection closed")
            return
        self.upstream_buffer.extend(chunk)
        while b"\n" in self.upstream_buffer:
            raw, _, remainder = self.upstream_buffer.partition(b"\n")
            self.upstream_buffer[:] = remainder
            if raw:
                self._broadcast(bytes(raw) + b"\n")

    def stop(self) -> None:
        self.running = False
        if self.listener is not None:
            wake = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                wake.connect(self.listen_path)
            except OSError:
                pass
            finally:
                wake.close()

    def serve(self) -> int:
        self._prepare_listener()
        self.running = True
        self._log(f"listening: {self.listen_path}")
        try:
            while self.running:
                self._connect_upstream()
                for key, _events in self.selector.select(timeout=0.5):
                    if key.data == "listener":
                        self._accept_clients()
                    elif key.data == "upstream":
                        self._handle_upstream()
                    elif key.data == "client":
                        self._handle_client(key.fileobj)
        finally:
            self._close_clients()
            if self.upstream is not None:
                try:
                    self.selector.unregister(self.upstream)
                except (KeyError, ValueError):
                    pass
                self.upstream.close()
                self.upstream = None
            if self.listener is not None:
                try:
                    self.selector.unregister(self.listener)
                except (KeyError, ValueError):
                    pass
                self.listener.close()
                self.listener = None
            self.selector.close()
            if os.path.lexists(self.listen_path):
                try:
                    if stat.S_ISSOCK(os.lstat(self.listen_path).st_mode):
                        os.unlink(self.listen_path)
                except OSError:
                    pass
        return 0


TOOLS: list[dict[str, Any]] = [
    {
        "name": "bibit_voice_status",
        "description": "读取 Bibit 麦克风、KWS、VAD、ASR 模型和当前会话状态。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "bibit_transcribe_once",
        "description": "启动一次麦克风 ASR，等待用户说完并返回 final 文本。全局同时只能有一个语音会话。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 300,
                    "default": 30,
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "bibit_wait_for_interaction",
        "description": "等待下一次 KWS 唤醒及其 ASR final，适合本地大模型消费自然唤醒事件。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 3600,
                    "default": 60,
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "bibit_cancel_voice_session",
        "description": "取消当前全局语音会话；可传 session_id 仅取消指定会话。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "bibit_set_voice_switches",
        "description": "分别开启或关闭 Bibit 的 KWS 唤醒和 ASR；可只给一个字段（服务端不支持字段级提交时会自动补全未给出的字段）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wake": {"type": "boolean"},
                "asr": {"type": "boolean"},
                "request_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "bibit_set_asr_model",
        "description": "热切换 SenseVoice 或 Qwen3-ASR，不重启音频服务。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["sensevoice", "qwen3_asr_0.6b"],
                }
            },
            "required": ["model"],
            "additionalProperties": False,
        },
    },
    {
        "name": "bibit_set_voice_thresholds",
        "description": "更新 KWS/VAD 参数；省略的字段保持当前值。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kws_threshold": {"type": "number", "minimum": 0.05, "maximum": 0.95},
                "kws_score": {"type": "number", "minimum": 0.1, "maximum": 5.0},
                "vad_threshold": {"type": "number", "minimum": 0.05, "maximum": 0.95},
                "min_silence": {"type": "number", "minimum": 0.15, "maximum": 3.0},
                "wake_guard_ms": {"type": "integer", "minimum": 0, "maximum": 600},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "bibit_set_kws_debug",
        "description": "Enable or disable saving accepted KWS wake WAV/JSON debug captures.",
        "inputSchema": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
            "required": ["enabled"],
            "additionalProperties": False,
        },
    },
]


def invoke_tool(
    name: str, arguments: dict[str, Any], socket_path: str, timeout: float
) -> ToolResult:
    with VoiceSocketClient(socket_path, timeout) as client:
        if name == "bibit_voice_status":
            return client.get_status()
        if name == "bibit_transcribe_once":
            return client.transcribe_once(float(arguments.get("timeout_seconds", timeout)))
        if name == "bibit_wait_for_interaction":
            return client.wait_for_interaction(
                float(arguments.get("timeout_seconds", max(timeout, 60.0)))
            )
        if name == "bibit_cancel_voice_session":
            session_id = arguments.get("session_id")
            if session_id is not None and (
                not isinstance(session_id, int) or isinstance(session_id, bool) or session_id < 0
            ):
                raise VoiceToolError("INVALID_ARGUMENTS", "session_id 必须是非负整数")
            return client.cancel(session_id)
        if name == "bibit_set_voice_switches":
            wake = arguments.get("wake")
            asr = arguments.get("asr")
            request_id = arguments.get("request_id")
            if wake is None and asr is None:
                raise VoiceToolError("INVALID_ARGUMENTS", "至少需要给出 wake 或 asr 之一")
            for label, value in (("wake", wake), ("asr", asr)):
                if value is not None and not isinstance(value, bool):
                    raise VoiceToolError(
                        "INVALID_ARGUMENTS", f"{label} 必须是 JSON boolean"
                    )
            if request_id is not None and not isinstance(request_id, str):
                raise VoiceToolError("INVALID_ARGUMENTS", "request_id 必须是字符串")
            return client.set_switches(wake, asr, request_id)
        if name == "bibit_set_asr_model":
            return client.set_model(str(arguments["model"]), max(timeout, MODEL_TIMEOUT))
        if name == "bibit_set_voice_thresholds":
            return client.set_thresholds(arguments)
        if name == "bibit_set_kws_debug":
            enabled = arguments["enabled"]
            if not isinstance(enabled, bool):
                raise VoiceToolError("INVALID_ARGUMENTS", "enabled must be a JSON boolean")
            return client.set_kws_debug(enabled)
    raise VoiceToolError("UNKNOWN_TOOL", f"未知工具：{name}")


def mcp_result(result: ToolResult) -> dict[str, Any]:
    data = result.as_dict()
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            }
        ],
        "structuredContent": data,
        "isError": not result.ok,
    }


def run_mcp(socket_path: str, timeout: float) -> int:
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                continue
        except json.JSONDecodeError:
            continue
        request_id = request.get("id")
        method = request.get("method")
        if request_id is None:
            continue
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        try:
            if method == "initialize":
                requested = request.get("params", {}).get("protocolVersion")
                response["result"] = {
                    "protocolVersion": requested or "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "bibit-voice-tool", "version": "1.0Beta1"},
                }
            elif method == "ping":
                response["result"] = {}
            elif method == "tools/list":
                response["result"] = {"tools": TOOLS}
            elif method == "tools/call":
                params = request.get("params") or {}
                name = str(params.get("name", ""))
                arguments = params.get("arguments") or {}
                if not isinstance(arguments, dict):
                    raise VoiceToolError("INVALID_ARGUMENTS", "tool arguments 必须是对象")
                response["result"] = mcp_result(
                    invoke_tool(name, arguments, socket_path, timeout)
                )
            else:
                response["error"] = {"code": -32601, "message": f"Method not found: {method}"}
        except VoiceToolError as exc:
            response["result"] = mcp_result(ToolResult(False, {"error": exc.as_dict()}))
        except (KeyError, TypeError, ValueError) as exc:
            response["result"] = mcp_result(
                ToolResult(
                    False,
                    {
                        "error": {
                            "code": "INVALID_ARGUMENTS",
                            "message": str(exc),
                        }
                    },
                )
            )
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "on", "yes"):
        return True
    if lowered in ("0", "false", "off", "no"):
        return False
    raise argparse.ArgumentTypeError("必须是 on/off 或 true/false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bibit-voice-tool",
        description="Bibit tool.sock 的 Qt/CLI/MCP Tool 适配器",
    )
    parser.add_argument("--socket", default=default_socket_path(), help="tool.sock 路径")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="默认超时秒数")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("schema", help="输出可提供给大模型的工具定义")
    sub.add_parser("status", help="读取音频/KWS/VAD/ASR 状态")
    transcribe = sub.add_parser("transcribe", help="启动一次手动 ASR 并等待 final")
    transcribe.add_argument("--wait", type=float, default=None, help="本次等待秒数")
    interaction = sub.add_parser("wait-interaction", help="等待下一次 KWS+ASR 交互")
    interaction.add_argument("--wait", type=float, default=None, help="本次等待秒数")
    cancel = sub.add_parser("cancel", help="取消当前语音会话")
    cancel.add_argument("--session-id", type=int, default=None, help="仅取消指定会话")
    switches = sub.add_parser("set-switches", help="设置 KWS/ASR 开关")
    switches.add_argument("--wake", required=True, type=parse_bool)
    switches.add_argument("--asr", required=True, type=parse_bool)
    model = sub.add_parser("set-model", help="热切换 ASR 模型")
    model.add_argument("model", choices=("sensevoice", "qwen3_asr_0.6b"))
    thresholds = sub.add_parser("set-thresholds", help="修改 KWS/VAD 参数")
    thresholds.add_argument("--kws-threshold", type=float)
    thresholds.add_argument("--kws-score", type=float)
    thresholds.add_argument("--vad-threshold", type=float)
    thresholds.add_argument("--min-silence", type=float)
    thresholds.add_argument("--wake-guard-ms", type=int)
    kws_debug = sub.add_parser("set-kws-debug", help="enable or disable KWS debug capture")
    kws_debug.add_argument("--enabled", required=True, type=parse_bool)
    events = sub.add_parser("events", help="原样输出 tool.sock 事件流")
    events.add_argument("--wait", type=float, default=0.0, help="0 表示持续等待")
    sub.add_parser("mcp", help="通过 stdio 运行 MCP server")
    serve = sub.add_parser("serve", help="运行 Qt/CLI/MCP 共用的 tool.sock 代理")
    serve.add_argument(
        "--upstream",
        default=default_audio_socket_path(),
        help="底层 bibit-audio Socket 路径",
    )
    serve.add_argument(
        "--retry-interval",
        type=float,
        default=0.5,
        help="底层服务断线重连间隔秒数",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if args.command == "schema":
        print_json({"tools": TOOLS})
        return 0
    if args.command == "mcp":
        return run_mcp(args.socket, args.timeout)
    if args.command == "serve":
        try:
            proxy = VoiceToolProxy(args.socket, args.upstream, args.retry_interval)
            return proxy.serve()
        except VoiceToolError as exc:
            print_json({"ok": False, "error": exc.as_dict()})
            return 7 if exc.code == "TOOL_ALREADY_RUNNING" else 1

    try:
        with VoiceSocketClient(args.socket, args.timeout) as client:
            if args.command == "status":
                result = client.get_status()
            elif args.command == "transcribe":
                result = client.transcribe_once(args.wait or args.timeout)
            elif args.command == "wait-interaction":
                result = client.wait_for_interaction(args.wait or args.timeout)
            elif args.command == "cancel":
                result = client.cancel(args.session_id)
            elif args.command == "set-switches":
                result = client.set_switches(args.wake, args.asr)
            elif args.command == "set-model":
                result = client.set_model(args.model, max(args.timeout, MODEL_TIMEOUT))
            elif args.command == "set-thresholds":
                values = {
                    key: value
                    for key, value in {
                        "kws_threshold": args.kws_threshold,
                        "kws_score": args.kws_score,
                        "vad_threshold": args.vad_threshold,
                        "min_silence": args.min_silence,
                        "wake_guard_ms": args.wake_guard_ms,
                    }.items()
                    if value is not None
                }
                result = client.set_thresholds(values)
            elif args.command == "set-kws-debug":
                result = client.set_kws_debug(args.enabled)
            elif args.command == "events":
                for event in client.events(args.wait):
                    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
                    sys.stdout.flush()
                return 0
            else:
                parser.error(f"未知命令：{args.command}")
                return 2
        print_json(result.as_dict())
        return 0
    except VoiceToolError as exc:
        print_json({"ok": False, "error": exc.as_dict()})
        return 7 if exc.code == "TOOL_ALREADY_RUNNING" else 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
