"""Compile the solver kernels ahead of the first solve: python -m hertzon.solver.warmup

EMerge and Hertzon compile their numba kernels on first use and keep them in the disk cache. A small
patch antenna solve (two adaptive passes with local Delaunay refinement, far field, a discrete and a
fast sweep) runs every kernel a real solve needs, so the first solve of a user starts at full speed.
Run by setup.ps1 at the end of the installation; Hertzon.exe runs it before the first start of an
installation made without it (marker .venv/hertzon_kernels.txt).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE = EXAMPLES / "PatchAntenna.hzn"
# curved elements, plane wave and RCS: kernels the patch antenna does not use (NATO sphere, two passes)
CURVED = EXAMPLES / "NATO-Sphere.hzn"


def _curved_project(dst: Path) -> tuple[str, str]:
    d = json.loads(CURVED.read_text(encoding="utf-8"))
    des = d["designs"][0]
    st = des["setups"][0]
    st["props"].update(MaximumPasses="2", MinimumPasses="2", MaximumDeltaEnergy="1e-12", SaveFields="True")
    dst.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return des["name"], st["name"]


def _project(dst: Path) -> tuple[str, str]:
    d = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    des = d["designs"][0]
    st = des["setups"][0]
    st["props"].update(MaximumPasses="2", MinimumPasses="2", MaximumDeltaS="1e-9", PercentRefinement="30",
                       RefinementMethod="Local (Delaunay)", SaveFields="True")
    sweep = {"Enabled": "True", "Distribution": "Linear Count", "Start": "2.3GHz", "Stop": "2.6GHz", "Count": "3",
             "SaveFields": "True", "ErrorTolerance": "0.5", "MaxSolutions": "20", "Parallel": "False"}
    # the discrete sweep in parallel threads: compiles the serial kernels the worker threads use
    st["sweeps"] = [{"name": "WarmDiscrete", "props": dict(sweep, SweepType="Discrete", Parallel="True")},
                    {"name": "WarmFast", "props": dict(sweep, SweepType="Fast", SaveFields="False")}]
    dst.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return des["name"], st["name"]


def run(log=print, timeout: float = 3600, background: bool = False) -> bool:
    """Solve the warm-up projects in solver processes. True when both finished."""
    if not EXAMPLE.exists():
        log(f"Warm-up example missing: {EXAMPLE}")
        return False
    t0 = time.time()
    ok = _solve(_project, "Warmup", log, timeout, background, t0)
    if CURVED.exists():
        ok = _solve(_curved_project, "WarmupCurved", log, timeout, background, t0) and ok
    log(f"Solver kernels {'ready' if ok else 'NOT ready'} after {time.time() - t0:.0f} s")
    return ok


def _solve(make, name: str, log, timeout: float, background: bool, t0: float) -> bool:
    tmp = Path(tempfile.mkdtemp(prefix="hertzon_warmup_"))
    try:
        proj = tmp / f"{name}.hzn"
        design, setup = make(proj)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        extra = {}
        if background:   # started by the GUI: a solve of the user keeps the priority
            if os.name == "nt":
                extra["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
            else:
                extra["preexec_fn"] = lambda: os.nice(10)
        p = subprocess.Popen([sys.executable, "-m", "hertzon.solver.run", str(proj), design, setup, str(tmp / "out")],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", env=env, cwd=str(Path(__file__).resolve().parents[2]), **extra)
        last = ""
        for line in p.stdout:
            if not line.startswith("@@HZN "):
                continue
            try:
                msg = json.loads(line[6:])
            except ValueError:
                continue
            if msg.get("type") == "progress" and msg.get("text") and msg["text"] != last:
                last = msg["text"]
                log(f"  [{time.time() - t0:5.0f} s] {last}")
            elif msg.get("type") == "failed":
                log(f"  failed: {msg.get('error')}")
            if time.time() - t0 > timeout:
                p.kill()
                log("  stopped: took too long")
                return False
        return p.wait() == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    from .bridge import _first_run
    if "--force" not in sys.argv and not _first_run():
        print("Solver kernels are already compiled.")
        _done("ok")
        return 0
    if os.name == "nt":
        os.system("title Hertzon setup")  # noqa: S605
    print("Hertzon setup: preparing the solver (once, a few minutes) ...", flush=True)
    ok = run(lambda s: print(s, flush=True), background="--background" in sys.argv)
    # also after a failure: Hertzon.exe must not repeat it at every start (a solve then compiles itself)
    _done("ok" if ok else "failed")
    return 0 if ok else 1


def _done(state: str) -> None:
    """Marker in the virtual environment: Hertzon.exe runs the warm-up before the first start until it exists."""
    try:
        from .bridge import KERNELS_VERSION       # first line: Hertzon.exe compares it with kernels_version.txt
        (Path(sys.prefix) / "hertzon_kernels.txt").write_text(f"{KERNELS_VERSION}\n{state}\n{time.ctime()}\n",
                                                            encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
