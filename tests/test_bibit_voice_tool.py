#!/usr/bin/env python3
"""Protocol tests for the external Bibit Voice Tool adapter."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "src" / "tool" / "bibit_voice_tool.py"
SPEC = importlib.util.spec_from_file_location("bibit_voice_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
voice_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = voice_tool
SPEC.loader.exec_module(voice_tool)


class FakeAudioServer:
    def __init__(self, socket_path: str, initial_mode: str = "idle"):
        self.socket_path = socket_path
        self.mode = initial_mode
        # 切换结果：ok（正常完成）/ busy（已有切换）/ fail（构建失败）
        self.switch_result = "ok"
        self.switch_request_id = "sw-1"
        # 服务端能力：默认按旧服务端（不声明能力）建模。
        self.partial_switches = False
        self.queued_model_switch = False
        self.switch_settlement = False
        # 开关终态：applied / superseded，以及可选的延迟（模拟后端构建）。
        self.settle_outcome = "applied"
        self.settle_delay = 0.0
        # 广播里混入别人的结果，验证客户端按 request_id 归属。
        self.foreign_noise = False
        self.commands: list[dict[str, Any]] = []
        self.ready = threading.Event()
        self.stop_requested = threading.Event()
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.server.bind(self.socket_path)
        self.server.listen(1)
        self.thread.start()
        self.ready.wait(2)

    def stop(self) -> None:
        self.stop_requested.set()
        wake_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            wake_socket.connect(self.socket_path)
        except OSError:
            pass
        finally:
            wake_socket.close()
        self.thread.join(2)
        self.server.close()

    def wait_for_command(self, command: str, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(item.get("cmd") == command for item in self.commands):
                return True
            time.sleep(0.01)
        return False

    def _send(self, client: socket.socket, event: dict[str, Any]) -> None:
        payload = json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            client.sendall(payload)
        except BrokenPipeError:
            pass

    def _status(self) -> dict[str, Any]:
        status = {
            "event": "status",
            "mode": self.mode,
            "audio_ready": True,
            "kws_loaded": True,
            "asr": True,
            "asr_loaded": True,
            "asr_model_type": "sensevoice",
        }
        if self.partial_switches:
            status["partial_switches"] = True
        if self.queued_model_switch:
            status["queued_model_switch"] = True
        if self.switch_settlement:
            status["switch_settlement"] = True
        return status

    def _handle(self, client: socket.socket, command: dict[str, Any]) -> None:
        name = command.get("cmd")
        if name == "get_status":
            self._send(client, self._status())
            self._send(
                client,
                {
                    "event": "parameters",
                    "kws_threshold": 0.12,
                    "kws_score": 4.5,
                    "vad_threshold": 0.30,
                    "min_silence": 0.50,
                    "wake_guard_ms": 180,
                },
            )
        elif name == "start_manual_asr":
            self.mode = "listening"
            # 广播的 SESSION_MISMATCH 来自别的客户端失败的 cancel，不应影响本会话
            self._send(
                client,
                {
                    "event": "error",
                    "code": "SESSION_MISMATCH",
                    "session_id": 99999,
                    "active_session_id": 1,
                },
            )
            self._send(client, {"event": "listening", "session_id": 7})
            self._send(
                client,
                {
                    "event": "asr_final",
                    "session_id": 7,
                    "text": "测试文本",
                    "decode_ms": 120,
                    "audio_duration_ms": 1000,
                    "rtf": 0.12,
                },
            )
        elif name == "cancel":
            requested = command.get("session_id", 0)
            if requested and requested != 7:
                self._send(
                    client,
                    {
                        "event": "error",
                        "code": "SESSION_MISMATCH",
                        "session_id": requested,
                        "active_session_id": 7,
                    },
                )
                return
            self.mode = "idle"
            self._send(client, {"event": "session_cancelled", "session_id": 7})
        elif name == "set_switches":
            self._send(client, self._status())
            if self.foreign_noise:
                self._send(
                    client,
                    {
                        "event": "error",
                        "code": "ASR_INIT_FAILED",
                        "request_id": "someone-else",
                    },
                )
                self._send(
                    client,
                    {
                        "event": "switches_settled",
                        "request_id": "someone-else",
                        "outcome": "superseded",
                        "wake": True,
                        "asr": True,
                    },
                )
            if self.partial_switches:
                # 新服务端回带 request_id 的终态回执。
                self._send(
                    client,
                    {
                        "event": "switches_applied",
                        "request_id": command.get("request_id", ""),
                        "wake": True,
                        "asr": True,
                    },
                )
            if self.switch_settlement:
                if self.settle_delay:
                    time.sleep(self.settle_delay)
                self._send(
                    client,
                    {
                        "event": "switches_settled",
                        "request_id": command.get("request_id", ""),
                        "outcome": self.settle_outcome,
                        "wake": command.get("wake", True),
                        "asr": command.get("asr", True),
                    },
                )
        elif name == "set_thresholds":
            self._send(client, self._status())
        elif name == "set_kws_debug":
            self._send(client, self._status())
        elif name == "set_asr_model":
            self._handle_switch(client, command)

    # 与 bibit-audio 的 4.3 节契约一致：回执带 request_id，
    # 切换中的快照 `asr_model_type` 仍是生效模型，目标只出现在 switching.to。
    def _handle_switch(self, client: socket.socket, command: dict[str, Any]) -> None:
        model = command["model"]
        effective = self._status()["asr_model_type"]
        # 真实服务端回显调用方给的 request_id；没有才退回内部序号。
        request_id = command.get("request_id") or self.switch_request_id
        if self.switch_result == "busy":
            self._send(
                client,
                {
                    "event": "error",
                    "code": "ASR_MODEL_SWITCH_BUSY",
                    "request_id": request_id,
                    "to": model,
                    "switching": {
                        "kind": "asr_model",
                        "from": effective,
                        "to": model,
                        "request_id": "sw-0",
                        "elapsed_ms": 1200,
                    },
                },
            )
            if self.queued_model_switch:
                # 服务端声明支持排队：同一 request_id 在提交点后收到终态。
                self._send(
                    client,
                    {
                        "event": "model_switched",
                        "model": model,
                        "request_id": request_id,
                        "from": effective,
                        "to": model,
                        "settled": True,
                    },
                )
            return
        if self.switch_result == "superseded":
            self._send(
                client,
                {
                    "event": "model_switch_superseded",
                    "request_id": request_id,
                    "to": model,
                    "active_model": effective,
                },
            )
            return
        if self.switch_result == "fail":
            self._send(
                client,
                {
                    "event": "error",
                    "code": "ASR_MODEL_SWITCH_FAILED",
                    "request_id": request_id,
                    "from": effective,
                    "to": model,
                    "active_model": effective,
                    "message": "cannot load model assets",
                },
            )
            return
        if model == effective:
            self._send(
                client,
                {
                    "event": "model_switched",
                    "model": model,
                    "request_id": request_id,
                    "from": model,
                    "to": model,
                    "duration_ms": 0,
                    "noop": True,
                },
            )
            return
        self._send(
            client,
            {
                "event": "model_switching",
                "model": model,
                "request_id": request_id,
                "from": effective,
                "to": model,
            },
        )
        mid_switch = self._status()
        mid_switch["switching"] = {
            "kind": "asr_model",
            "from": effective,
            "to": model,
            "request_id": request_id,
            "elapsed_ms": 800,
        }
        self._send(client, mid_switch)
        self._send(
            client,
            {
                "event": "model_switched",
                "model": model,
                "request_id": request_id,
                "from": effective,
                "to": model,
                "duration_ms": 4200,
            },
        )

    def _run(self) -> None:
        self.ready.set()
        while not self.stop_requested.is_set():
            try:
                client, _ = self.server.accept()
            except OSError:
                return
            with client:
                buffer = bytearray()
                while not self.stop_requested.is_set():
                    try:
                        chunk = client.recv(65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        raw, _, rest = buffer.partition(b"\n")
                        buffer = bytearray(rest)
                        if not raw:
                            continue
                        command = json.loads(raw)
                        self.commands.append(command)
                        self._handle(client, command)


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "AF_UNIX is required")
class VoiceToolProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.socket_path = str(Path(self.temp_dir.name) / "audio.sock")
        self.server = FakeAudioServer(self.socket_path)
        self.server.start()

    def tearDown(self) -> None:
        self.server.stop()
        self.temp_dir.cleanup()

    def test_get_status_collects_status_and_parameters(self) -> None:
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            result = client.get_status().as_dict()
        self.assertTrue(result["ok"])
        self.assertTrue(result["status"]["audio_ready"])
        self.assertEqual(result["parameters"]["min_silence"], 0.5)

    def test_transcribe_once_returns_metrics_and_cancels_followup(self) -> None:
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            result = client.transcribe_once(2).as_dict()
        self.assertEqual(result["text"], "测试文本")
        self.assertEqual(result["session_id"], 7)
        self.assertEqual(result["rtf"], 0.12)
        self.assertTrue(result["followup_cancelled"])
        self.assertTrue(self.server.wait_for_command("cancel"))

    def test_cancel_is_idempotent_while_idle(self) -> None:
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            result = client.cancel().as_dict()
        self.assertTrue(result["already_idle"])
        self.assertFalse(any(item.get("cmd") == "cancel" for item in self.server.commands))

    def test_cancel_matching_session_id_cancels(self) -> None:
        self.server.mode = "listening"
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            result = client.cancel(session_id=7).as_dict()
        self.assertEqual(result["event"]["event"], "session_cancelled")
        sent = [item for item in self.server.commands if item.get("cmd") == "cancel"]
        self.assertTrue(sent)
        self.assertEqual(sent[-1].get("session_id"), 7)

    def test_cancel_without_session_id_omits_field(self) -> None:
        self.server.mode = "listening"
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            client.cancel().as_dict()
        sent = [item for item in self.server.commands if item.get("cmd") == "cancel"]
        self.assertTrue(sent)
        self.assertNotIn("session_id", sent[-1])

    def test_cancel_session_mismatch_raises(self) -> None:
        self.server.mode = "listening"
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            with self.assertRaises(voice_tool.VoiceToolError) as raised:
                client.cancel(session_id=99)
        self.assertEqual(raised.exception.code, "SESSION_MISMATCH")

    def test_switch_arguments_require_json_booleans(self) -> None:
        with self.assertRaises(voice_tool.VoiceToolError) as raised:
            voice_tool.invoke_tool(
                "bibit_set_voice_switches",
                {"wake": "false", "asr": True},
                self.socket_path,
                2,
            )
        self.assertEqual(raised.exception.code, "INVALID_ARGUMENTS")

    def test_set_switches_sends_only_given_fields(self) -> None:
        self.server.partial_switches = True
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            client.set_switches(asr=False, request_id="req-1").as_dict()
        sent = [item for item in self.server.commands if item.get("cmd") == "set_switches"]
        self.assertTrue(sent)
        self.assertEqual(sent[-1].get("asr"), False)
        self.assertNotIn("wake", sent[-1])
        self.assertEqual(sent[-1].get("request_id"), "req-1")

    def test_set_switches_requires_at_least_one_field(self) -> None:
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            with self.assertRaises(voice_tool.VoiceToolError) as raised:
                client.set_switches()
        self.assertEqual(raised.exception.code, "BAD_COMMAND")

    def test_set_switches_reads_the_receipt_for_its_request_id(self) -> None:
        # 新服务端：客户端要按 request_id 认回执，而不是看到任意 status/ready 就当完成。
        self.server.partial_switches = True
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            data = client.set_switches(asr=False).as_dict()
        sent = [item for item in self.server.commands if item.get("cmd") == "set_switches"][-1]
        self.assertTrue(sent.get("request_id"))
        self.assertEqual(data["applied"]["request_id"], sent["request_id"])

    def test_mcp_switches_accept_partial_arguments(self) -> None:
        self.server.partial_switches = True
        result = voice_tool.invoke_tool(
            "bibit_set_voice_switches", {"asr": False}, self.socket_path, 2
        )
        self.assertTrue(result.as_dict()["ok"])
        sent = [item for item in self.server.commands if item.get("cmd") == "set_switches"]
        self.assertTrue(sent)
        self.assertNotIn("wake", sent[-1])

    def test_set_switches_fills_missing_field_for_legacy_service(self) -> None:
        # 旧服务端把缺失字段按默认 true 处理：客户端必须补全当前生效值，
        # 否则关掉识别会顺手把唤醒一起打开。
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            client.set_switches(asr=False).as_dict()
        sent = [item for item in self.server.commands if item.get("cmd") == "set_switches"]
        self.assertTrue(sent)
        self.assertEqual(sent[-1].get("asr"), False)
        self.assertEqual(sent[-1].get("wake"), True)

    def test_set_switches_waits_for_terminal_settlement(self) -> None:
        # 新服务端：受理回执不是终态。打开识别要等后端构建完，客户端必须等
        # switches_settled，否则会把「已受理」当成「已打开」。
        self.server.partial_switches = True
        self.server.switch_settlement = True
        self.server.settle_delay = 0.4
        started = time.monotonic()
        with voice_tool.VoiceSocketClient(self.socket_path, 5) as client:
            data = client.set_switches(asr=True).as_dict()
        self.assertGreaterEqual(time.monotonic() - started, 0.3)
        self.assertEqual(data["settled"]["outcome"], "applied")
        self.assertEqual(data["settled"]["request_id"], data["request_id"])

    def test_set_switches_reports_superseded_intent(self) -> None:
        # 意图被更新的命令取代且未生效：显式报错，别让调用方以为开关已经到位。
        self.server.partial_switches = True
        self.server.switch_settlement = True
        self.server.settle_outcome = "superseded"
        with voice_tool.VoiceSocketClient(self.socket_path, 3) as client:
            with self.assertRaises(voice_tool.VoiceToolError) as raised:
                client.set_switches(asr=True)
        self.assertEqual(raised.exception.code, "SWITCH_INTENT_SUPERSEDED")

    def test_set_switches_ignores_foreign_request_results(self) -> None:
        # 广播里别人失败的 error / 被取代的终态，不能算到本次调用头上。
        self.server.partial_switches = True
        self.server.switch_settlement = True
        self.server.foreign_noise = True
        with voice_tool.VoiceSocketClient(self.socket_path, 3) as client:
            data = client.set_switches(asr=False).as_dict()
        self.assertEqual(data["settled"]["request_id"], data["request_id"])
        self.assertEqual(data["settled"]["outcome"], "applied")

    def test_set_model_follows_queued_terminal_state(self) -> None:
        # 服务端声明支持排队：BUSY 不是终态，客户端应继续等到 model_switched。
        self.server.switch_result = "busy"
        self.server.queued_model_switch = True
        with voice_tool.VoiceSocketClient(self.socket_path, 3) as client:
            data = client.set_model("qwen3_asr_0.6b", timeout=3).as_dict()
        self.assertTrue(data["ok"])
        self.assertEqual(data["model"], "qwen3_asr_0.6b")

    def test_set_model_reports_superseded_terminal_state(self) -> None:
        self.server.switch_result = "superseded"
        with voice_tool.VoiceSocketClient(self.socket_path, 3) as client:
            with self.assertRaises(voice_tool.VoiceToolError) as raised:
                client.set_model("qwen3_asr_0.6b", timeout=3)
        # 目标被取代且未生效时必须显式报错，而不是返回 ok:true 让调用方误判。
        self.assertEqual(raised.exception.code, "ASR_MODEL_SWITCH_SUPERSEDED")

    def test_tool_schema_exposes_asr_and_kws_capabilities(self) -> None:
        names = {tool["name"] for tool in voice_tool.TOOLS}
        self.assertIn("bibit_transcribe_once", names)
        self.assertIn("bibit_wait_for_interaction", names)
        self.assertIn("bibit_set_voice_switches", names)
        self.assertIn("bibit_set_asr_model", names)
        self.assertIn("bibit_set_kws_debug", names)

    def test_kws_debug_command_is_forwarded(self) -> None:
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            result = client.set_kws_debug(True).as_dict()
        self.assertTrue(result["ok"])
        self.assertTrue(result["enabled"])
        self.assertTrue(self.server.wait_for_command("set_kws_debug"))

    def test_set_model_waits_past_mid_switch_snapshot(self) -> None:
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            result = client.set_model("qwen3_asr_0.6b").as_dict()
        # 切换中的 status 快照写的是生效模型（sensevoice），不能被当成切换完成。
        self.assertEqual(result["model"], "qwen3_asr_0.6b")
        self.assertEqual(result["event"]["duration_ms"], 4200)
        # 客户端现在自带 request_id，服务端按它回执。
        self.assertEqual(result["event"]["request_id"], result["request_id"])
        sent = [item for item in self.server.commands if item.get("cmd") == "set_asr_model"]
        self.assertEqual(sent[-1]["model"], "qwen3_asr_0.6b")

    def test_set_model_same_engine_is_success_noop(self) -> None:
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            result = client.set_model("sensevoice").as_dict()
        self.assertTrue(result["ok"])
        self.assertTrue(result["event"]["noop"])
        self.assertEqual(result["event"]["duration_ms"], 0)

    def test_set_model_busy_surfaces_switch_snapshot(self) -> None:
        self.server.switch_result = "busy"
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            with self.assertRaises(voice_tool.VoiceToolError) as raised:
                client.set_model("qwen3_asr_0.6b")
        self.assertEqual(raised.exception.code, "ASR_MODEL_SWITCH_BUSY")

    def test_set_model_failure_reports_effective_model(self) -> None:
        self.server.switch_result = "fail"
        with voice_tool.VoiceSocketClient(self.socket_path, 2) as client:
            with self.assertRaises(voice_tool.VoiceToolError) as raised:
                client.set_model("qwen3_asr_0.6b")
        self.assertEqual(raised.exception.code, "ASR_MODEL_SWITCH_FAILED")
        self.assertEqual(raised.exception.event["active_model"], "sensevoice")

    def test_proxy_forwards_commands_and_broadcasts_events(self) -> None:
        tool_socket_path = str(Path(self.temp_dir.name) / "tool.sock")
        proxy = voice_tool.VoiceToolProxy(
            tool_socket_path, self.socket_path, retry_interval=0.05
        )
        proxy_thread = threading.Thread(target=proxy.serve, daemon=True)
        proxy_thread.start()
        # socket 文件出现不等于已经 listen：真正连上一次才算就绪，否则会偶发
        # ConnectionRefused，把交付门禁变成看运气的检查。
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(tool_socket_path)
                probe.close()
                break
            except OSError:
                probe.close()
                time.sleep(0.01)

        with voice_tool.VoiceSocketClient(tool_socket_path, 2) as first:
            status = first.get_status().as_dict()
            self.assertTrue(status["status"]["audio_ready"])
            with voice_tool.VoiceSocketClient(tool_socket_path, 2) as second:
                first.send({"cmd": "start_manual_asr"})
                event = second.wait_for(
                    lambda item: item.get("event") == "listening", errors=False
                )
                self.assertEqual(event["session_id"], 7)

        proxy.stop()
        proxy_thread.join(2)
        self.assertFalse(proxy_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
