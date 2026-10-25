"""Update hint: at start Hertzon asks GitHub for the newest release and, when it is newer than the running
version, says so in the Message Manager and the status bar. Runs in a background thread, never blocks the
start, and stays silent when there is no network or GitHub does not answer. Switched off in
Tools > Options > General or with the environment variable HERTZON_NO_UPDATE_CHECK=1."""
from __future__ import annotations

import json
import os
import re
import threading
import urllib.request

from PySide6.QtCore import QObject, Signal

REPO = "c8501024/Hertzon"
RELEASES_API = f"https://api.github.com/repos/{REPO}/releases?per_page=30"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
SETTING = "updates/check_on_start"
TIMEOUT = 5.0

# Hertzon releases are tagged v<major>.<minor>.<patch>; the same repository also publishes other releases
# (the EMerge wheels, tag emerge-<sha>), so GitHub's "latest release" is not necessarily a Hertzon version.
_TAG = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def parse_version(text: str) -> tuple[int, int, int] | None:
    """(major, minor, patch) of a release tag or version string, None for anything else."""
    m = _TAG.match((text or "").strip())
    return tuple(int(g) for g in m.groups()) if m else None


def newest_release(releases: list[dict]) -> dict | None:
    """The release with the highest Hertzon version among published, non-pre-releases."""
    best, best_v = None, None
    for r in releases:
        if r.get("draft") or r.get("prerelease"):
            continue
        v = parse_version(r.get("tag_name", ""))
        if v is not None and (best_v is None or v > best_v):
            best, best_v = r, v
    return best


def newer_release(releases: list[dict], current: str) -> dict | None:
    """The newest release if it is newer than the version `current`, else None."""
    r = newest_release(releases)
    cur = parse_version(current)
    if r is None or cur is None:
        return None
    return r if parse_version(r["tag_name"]) > cur else None


def fetch_releases(timeout: float = TIMEOUT) -> list[dict]:
    req = urllib.request.Request(RELEASES_API, headers={"Accept": "application/vnd.github+json",
                                                        "User-Agent": "Hertzon update check"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, list) else []


def disabled_by_env() -> bool:
    return os.environ.get("HERTZON_NO_UPDATE_CHECK", "").strip() not in ("", "0")


class UpdateChecker(QObject):
    """Checks once in a daemon thread; `available(version, url)` is emitted in the GUI thread."""
    available = Signal(str, str)

    def __init__(self, current: str, parent=None):
        super().__init__(parent)
        self.current = current

    def start(self) -> None:
        threading.Thread(target=self._run, name="hertzon-update-check", daemon=True).start()

    def _run(self) -> None:
        try:
            r = newer_release(fetch_releases(), self.current)
        except Exception:  # noqa: BLE001 - offline, proxy, rate limit, bad answer: no hint
            return
        if r is not None:
            self.available.emit(r["tag_name"].lstrip("v"), r.get("html_url") or RELEASES_PAGE)
