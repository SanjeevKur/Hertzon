"""Update hint at start: which release counts as newer, and silence without network."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from hertzon.gui import update_check as U  # noqa: E402


def rel(tag, draft=False, pre=False):
    return {"tag_name": tag, "draft": draft, "prerelease": pre, "html_url": f"https://example.org/{tag}"}


def test_parse_version():
    assert U.parse_version("v0.2.0") == (0, 2, 0)
    assert U.parse_version("0.10.3") == (0, 10, 3)
    assert U.parse_version("emerge-e03a085") is None
    assert U.parse_version("v0.2.0rc1") is None


def test_newest_skips_other_releases():
    # the EMerge wheel releases are published in the same repository and are often the most recent ones
    rs = [rel("emerge-e03a085"), rel("v0.3.0", draft=True), rel("v0.2.5", pre=True), rel("v0.1.25"),
          rel("v0.2.0"), rel("v0.1.9")]
    assert U.newest_release(rs)["tag_name"] == "v0.2.0"


def test_newer_release():
    rs = [rel("v0.1.25"), rel("v0.2.0"), rel("v0.10.0")]
    assert U.newer_release(rs, "0.2.0")["tag_name"] == "v0.10.0"
    assert U.newer_release(rs, "0.10.0") is None
    assert U.newer_release(rs, "0.11.0") is None
    assert U.newer_release([], "0.2.0") is None
    assert U.newer_release(rs, "0.2.0.dev3") is None


def test_offline_is_silent(monkeypatch):
    def fail(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(U, "fetch_releases", fail)
    c = U.UpdateChecker("0.2.0")
    got = []
    c.available.connect(lambda *a: got.append(a))
    c._run()
    assert got == []


def test_emits_newer(monkeypatch):
    monkeypatch.setattr(U, "fetch_releases", lambda: [rel("v0.2.0"), rel("v0.2.1")])
    c = U.UpdateChecker("0.2.0")
    got = []
    c.available.connect(lambda *a: got.append(a))
    c._run()
    assert got == [("0.2.1", "https://example.org/v0.2.1")]


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("HERTZON_NO_UPDATE_CHECK", "1")
    assert U.disabled_by_env()
    monkeypatch.setenv("HERTZON_NO_UPDATE_CHECK", "0")
    assert not U.disabled_by_env()
