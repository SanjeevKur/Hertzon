"""MCP server: protocol and the model tools on a project file (no FEM solve)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hertzon.mcp import clients, tools  # noqa: E402
from hertzon.mcp.protocol import Server  # noqa: E402


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("HERTZON_HOME", str(tmp_path / "home"))      # no live Hertzon window
    monkeypatch.setenv("HERTZON_MCP_MODE", "file")
    tools.S.default_path = None
    return Server("hertzon", "test", tools.INSTRUCTIONS, tools.TOOLS)


def call(srv, _tool, **args):
    r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": _tool, "arguments": args}})
    res = r["result"]
    text = "\n".join(c["text"] for c in res["content"] if c["type"] == "text")
    assert not res["isError"], text
    return json.loads(text)


def test_initialize_and_list(server):
    r = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2025-03-26", "capabilities": {}}})
    assert r["result"]["protocolVersion"] == "2025-03-26"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = {t["name"] for t in server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]}
    assert {"create_project", "add_object", "assign", "run_simulation", "get_results"} <= names


def test_build_waveguide(server, tmp_path):
    p = tmp_path / "wg.hzn"
    call(server, "create_project", path=str(p), units="mm")
    call(server, "set_design", variables={"a": "22.86mm", "b": "10.16mm", "L": "30mm"})
    r = call(server, "add_object", kind="CreateBox", name="WG", props={"XSize": "a", "YSize": "L", "ZSize": "b"})
    faces = r["geometry"]["WG"]["faces"]
    assert len(faces) == 6
    end = [f["face"] for f in faces if f["normal"][1] < -0.99][0]
    call(server, "add_object", kind="CreateCylinder", name="Post", material="pec",
         props={"CenterPosition": "a/2,L/2,0", "Radius": "1mm", "Height": "b"})
    r = call(server, "add_operation", object="WG", kind="Subtract", props={"Tools": "Post"})
    assert r["geometry"]["WG"]["faces_total"] == 7
    a = call(server, "assign", category="excitation", kind="Wave Port", faces=[{"object": "WG", "face": end}])
    assert a["port_number"] == "1"
    call(server, "set_setup", props={"Frequency": "10GHz", "MaximumPasses": 2},
         sweeps=[{"name": "S", "props": {"Start": "8GHz", "Stop": "12GHz"}}])
    assert call(server, "validate")["ok"]
    s = call(server, "get_project")
    assert [o["name"] for o in s["objects"]] == ["WG", "Post"]
    assert s["objects"][1]["consumed_by"] == "WG"
    assert s["setups"][0]["sweeps"][0]["props"]["Start"] == "8GHz"


def test_bad_property_is_reported(server, tmp_path):
    call(server, "create_project", path=str(tmp_path / "x.hzn"))
    r = server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "add_object", "arguments": {"kind": "CreateBox", "props": {"Width": "1"}}}})
    assert r["result"]["isError"] and "no property 'Width'" in r["result"]["content"][0]["text"]


def test_codex_config_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "x"\n\n[mcp_servers.other]\ncommand = "a"\n\n[mcp_servers.hertzon]\ncommand = "old"\n'
                   '[mcp_servers.hertzon.env]\nA = "1"\n\n[profiles.p]\nmodel = "y"\n', encoding="utf-8")
    clients.codex_install()
    text = cfg.read_text(encoding="utf-8")
    assert text.count("[mcp_servers.hertzon]") == 1 and '"old"' not in text and "[mcp_servers.hertzon.env]" not in text
    assert "[mcp_servers.other]" in text and "[profiles.p]" in text
    assert clients.codex_status() == "registered"
    clients.codex_remove()
    assert clients.codex_status() == "not registered"
    assert "[profiles.p]" in cfg.read_text(encoding="utf-8")


def test_claude_json_without_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(clients, "_claude_cli", lambda: None)
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps({"numStartups": 3, "mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")
    assert clients.claude_status() == "not registered"
    clients.claude_install()
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["numStartups"] == 3 and "other" in data["mcpServers"]
    assert data["mcpServers"]["hertzon"]["args"] == ["-m", "hertzon.mcp"]
    assert clients.claude_status() == "registered"
    clients.claude_remove()
    assert clients.claude_status() == "not registered" and "other" in json.loads(cfg.read_text())["mcpServers"]


def test_claude_desktop_config(tmp_path, monkeypatch):
    monkeypatch.setattr(clients, "desktop_config", lambda: tmp_path / "Claude" / "claude_desktop_config.json")
    clients.desktop_install()
    assert clients.desktop_status() == "registered"
    clients.desktop_remove()
    assert clients.desktop_status() == "not registered"


def test_auto_register(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cc"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(clients, "_claude_cli", lambda: None)
    monkeypatch.setattr(clients, "desktop_config", lambda: tmp_path / "Claude" / "claude_desktop_config.json")
    (tmp_path / "cc").mkdir()
    (tmp_path / "cc" / ".claude.json").write_text("{}", encoding="utf-8")
    assert clients.auto_register() == ["claude"]          # Claude app and Codex are not installed
    assert clients.auto_register() == []                       # already registered
    (tmp_path / "codex").mkdir()
    assert clients.auto_register(skip=("codex",)) == []
    assert clients.auto_register() == ["codex"]
