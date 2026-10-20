"""Link between the MCP server and a running Hertzon window.

The window (Tools > AI Assistants) listens on 127.0.0.1 and writes its port and a random token to
``~/.hertzon/links/<pid>.json``. The MCP server sends one JSON request per connection:
``{"token", "cmd", "args"}`` and reads one JSON line back: ``{"ok": true, "result": ...}``.
"""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path


def link_dir() -> Path:
    d = Path(os.environ.get("HERTZON_HOME") or (Path.home() / ".hertzon")) / "links"
    return d


def same_path(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except OSError:
        return False


class LinkError(RuntimeError):
    pass


class GuiLink:
    def __init__(self, info: dict, file: Path):
        self.port = int(info["port"])
        self.token = str(info["token"])
        self.pid = int(info.get("pid", 0))
        self.file = file

    def call(self, cmd: str, timeout: float = 120.0, **args):
        req = json.dumps({"token": self.token, "cmd": cmd, "args": args}).encode("utf-8") + b"\n"
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=timeout) as s:
                s.sendall(req)
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = s.recv(1 << 20)
                    if not chunk:
                        break
                    buf += chunk
        except OSError as e:
            raise LinkError(f"Hertzon window not reachable: {e}") from e
        if not buf:
            raise LinkError("Hertzon window closed the connection")
        resp = json.loads(buf.decode("utf-8"))
        if not resp.get("ok"):
            raise LinkError(resp.get("error", "request failed"))
        return resp.get("result")


def windows(timeout: float = 2.0) -> list[tuple[GuiLink, dict]]:
    """Running Hertzon windows with the live link on, newest first, with their 'hello' answer."""
    out = []
    d = link_dir()
    if not d.is_dir():
        return out
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        try:
            info = json.loads(f.read_text(encoding="utf-8"))
            link = GuiLink(info, f)
            hello = link.call("hello", timeout=timeout)
        except (OSError, ValueError, KeyError, LinkError):
            continue
        out.append((link, hello))
    return out
