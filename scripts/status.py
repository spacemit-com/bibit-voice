#!/usr/bin/env python3
import json
import os
import socket
import sys


socket_path = os.environ.get("BIBIT_STATUS_SOCKET", "/run/bibit/tool.sock")
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.settimeout(3.0)

try:
    client.connect(socket_path)
    client.sendall(b'{"cmd":"get_status"}\n')
    buffer = b""
    seen = set()
    while seen != {"status", "parameters"}:
        chunk = client.recv(4096)
        if not chunk:
            raise RuntimeError("Voice Tool closed the socket")
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if not line:
                continue
            event = json.loads(line)
            if event.get("event") in {"status", "parameters"}:
                print(json.dumps(event, ensure_ascii=False, indent=2))
                seen.add(event["event"])
except (OSError, RuntimeError, json.JSONDecodeError) as exc:
    print(f"Bibit status failed: {exc}", file=sys.stderr)
    sys.exit(1)
finally:
    client.close()
