"""Minimal Model Context Protocol server over stdio (JSON-RPC 2.0, one message per line).

Only what the tool clients need: initialize, ping, tools/list and tools/call. No third-party
package, so it runs in the Hertzon environment as it is.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    handler: Callable[[dict], Any]
    read_only: bool = False


class Image:
    """Tool result part: a PNG image (shown to the model)."""

    def __init__(self, png: bytes, text: str = ""):
        self.png = png
        self.text = text


class ToolError(Exception):
    """An error the model should read (bad arguments, missing object ...), not a crash."""


def _content(result: Any) -> list[dict]:
    import base64
    parts = result if isinstance(result, list) and any(isinstance(r, Image) for r in result) else [result]
    out = []
    for r in parts:
        if isinstance(r, Image):
            if r.text:
                out.append({"type": "text", "text": r.text})
            out.append({"type": "image", "data": base64.b64encode(r.png).decode("ascii"), "mimeType": "image/png"})
        elif isinstance(r, str):
            out.append({"type": "text", "text": r})
        else:
            out.append({"type": "text", "text": json.dumps(r, indent=1, default=_default, ensure_ascii=False)})
    return out


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


class Server:
    def __init__(self, name: str, version: str, instructions: str, tools: list[Tool]):
        self.name, self.version, self.instructions = name, version, instructions
        self.tools = {t.name: t for t in tools}
        self.out = None

    # ---- transport ----------------------------------------------------------------------------
    def serve(self) -> None:
        # Everything else (gmsh, EMerge, numba, print) must not write into the protocol stream: keep
        # a private handle on the real stdout and point fd 1 at stderr.
        self.out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        inp = sys.stdin.buffer
        while True:
            line = inp.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                continue
            for m in (msg if isinstance(msg, list) else [msg]):
                resp = self.handle(m)
                if resp is not None:
                    self._send(resp)

    def _send(self, obj: dict) -> None:
        data = json.dumps(obj, default=_default, ensure_ascii=False).encode("utf-8") + b"\n"
        self.out.write(data)
        self.out.flush()

    # ---- dispatch -----------------------------------------------------------------------------
    def handle(self, msg: dict) -> dict | None:
        mid = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}
        if mid is None:          # notification (initialized, cancelled ...): no answer
            return None
        try:
            result = self.dispatch(method, params)
        except _RpcError as e:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": e.code, "message": str(e)}}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def dispatch(self, method: str, params: dict) -> dict:
        if method == "initialize":
            want = params.get("protocolVersion", PROTOCOL_VERSIONS[0])
            return {"protocolVersion": want if want in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": self.version},
                    "instructions": self.instructions}
        if method == "ping":
            return {}
        if method == "tools/list":
            tools = []
            for t in self.tools.values():
                d = {"name": t.name, "description": t.description, "inputSchema": t.schema}
                if t.read_only:
                    d["annotations"] = {"readOnlyHint": True}
                tools.append(d)
            return {"tools": tools}
        if method == "tools/call":
            t = self.tools.get(params.get("name", ""))
            if t is None:
                raise _RpcError(-32602, f"Unknown tool {params.get('name')!r}")
            try:
                res = t.handler(dict(params.get("arguments") or {}))
                return {"content": _content(res), "isError": False}
            except ToolError as e:
                return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
            except Exception as e:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                return {"content": [{"type": "text", "text": f"Error ({type(e).__name__}): {e}"}], "isError": True}
        if method in ("resources/list", "resources/templates/list"):
            return {"resources": []} if method == "resources/list" else {"resourceTemplates": []}
        if method == "prompts/list":
            return {"prompts": []}
        raise _RpcError(-32601, f"Method not found: {method}")


class _RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
