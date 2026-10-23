"""Speed and accuracy check of Hertzon on a given EMerge version.

    python emerge_bench.py run OUTDIR [CASE ...] [--passes N]
    python emerge_bench.py compare REFDIR NEWDIR

`run` solves each case in a fresh process (python -m ... _one) with timers around the expensive phases and
writes OUTDIR/<case>/ (the normal result folder) plus OUTDIR/<case>.json (timings, tetrahedra per pass,
EMerge version). Every case runs a fixed number of adaptive passes (MinimumPasses = MaximumPasses), so two
EMerge versions solve comparable meshes. `compare` prints the time per phase side by side and the largest
difference of the S-parameters (last adaptive pass and sweep) and of the far fields.
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EX = ROOT / "examples"
# name: (project, design, setup, passes, sweep points: None keeps them, 0 disables the sweeps); the other
# cases enable every sweep of the setup. The Rotman sweep (Fast, 11 ports) needs more than 14 GB.
CASES = {
    "microstrip": (EX / "benchmark/scaling/small_microstrip.hzn", "Microstrip50", "Setup1", 4, None),
    "hairpin": (EX / "benchmark/scaling/medium_hairpin.hzn", "Hairpin_N3", "Setup1", 4, None),
    "patch": (EX / "benchmark/scaling/large_patch.hzn", "Patch245", "Setup1", 4, None),
    "waveport": (EX / "benchmark/waveguide_post/post_pec_stable.hzn", "WR90_Post", "Setup1", 4, None),
    "rotman": (EX / "RotmanLens_24GHz.hzn", "RotmanArray", "Setup1", 4, 0),
}


def prepare(case: str, passes: int | None, tmp: Path) -> Path:
    proj, design, setup, n, pts = CASES[case]
    n = passes or n
    d = json.loads(proj.read_text(encoding="utf-8"))
    for ds in d["designs"]:
        if ds["name"] != design:
            continue
        for s in ds["setups"]:
            if s["name"] != setup:
                continue
            s["props"]["MaximumPasses"] = str(n)
            s["props"]["MinimumPasses"] = str(n)
            s["props"]["MinimumConvergedPasses"] = str(n)
            s["props"]["MaximumDeltaS"] = "1e-9"
            s["props"]["SaveFields"] = "True"   # field data is part of the measurement
            for w in s.get("sweeps", []):
                w.setdefault("props", {})["Enabled"] = "False" if pts == 0 else "True"
                if pts:
                    w["props"]["Distribution"] = "Linear Count"
                    w["props"]["Count"] = str(pts)
    p = tmp / f"{case}.hzn"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def one(case: str, project: str, out: str) -> None:
    """Child process: solve one case with timers."""
    sys.path.insert(0, str(ROOT))
    T, C, SEQ = collections.defaultdict(float), collections.defaultdict(int), []

    def wrap(obj, name, label):
        f = getattr(obj, name, None)
        if f is None:
            return

        def w(*a, **k):
            t = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                dt = time.perf_counter() - t
                T[label] += dt
                C[label] += 1
                SEQ.append((label, round(dt, 4)))
        setattr(obj, name, w)

    import emerge
    from hertzon.solver import emerge_patches
    emerge_patches.apply()
    import gmsh
    wrap(gmsh.model.mesh, "generate", "mesh.gmsh")
    from emerge._emerge.physics.microwave import microwave_3d, microwave_data
    wrap(microwave_3d.Microwave3D, "_run_adaptive_mesh", "pass.solve")
    wrap(microwave_data.MWField, "_solution_quality", "pass.error_estimate")
    from hertzon.solver import bridge, farfield_post, lazy_fields
    for n in ("save_mesh", "save_fields", "save_huygens", "compute_fields", "run_sweeps"):
        wrap(bridge.Bridge, n, n)
    wrap(farfield_post, "compute_sphere", "farfield.integral")
    wrap(lazy_fields, "fields_from_solution", "fields.from_solution")
    _, design, setup, _, _ = CASES[case]
    t0 = time.perf_counter()
    rc = bridge.run(project, design, setup, out)
    total = time.perf_counter() - t0
    import hertzon
    res = {"case": case, "rc": rc, "emerge": getattr(emerge, "__version__", "?"), "hertzon": hertzon.__file__,
           "total": total,
           "time": T, "count": C, "seq": SEQ, "cores": os.cpu_count()}
    try:
        res["profile"] = json.loads((Path(out) / "profile.json").read_text())
        conv = json.loads((Path(out) / "convergence.json").read_text())
        rows = conv["passes"] if isinstance(conv, dict) else conv
        res["tets"] = [r.get("tets") for r in rows]
    except (OSError, ValueError, KeyError):
        pass
    Path(out + ".json").write_text(json.dumps(res, indent=1))
    sys.stdout.flush()
    os._exit(0)   # same as the solver process: skip the slow/crashy library teardown


def run(outdir: str, cases: list[str], passes: int | None) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        # untimed warm-up (Hertzon's own, as after an installation): numba caches its kernels per source
        # path, so a new checkout compiles first, and the parallel error estimate is only used once compiled
        subprocess.run([sys.executable, "-m", "hertzon.solver.warmup"], cwd=ROOT, stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        for case in cases or list(CASES):
            p = prepare(case, passes, Path(td))
            t = time.time()
            with open(out / f"{case}.log", "w", encoding="utf-8") as log:
                subprocess.run([sys.executable, __file__, "_one", case, str(p), str(out / case)],
                               stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            print(f"{case}: {time.time() - t:.1f} s", flush=True)


def _load(d: Path, case: str):
    import numpy as np
    j = json.loads((d / f"{case}.json").read_text())
    data = {}
    for f in sorted((d / case).glob("sparams_*.npz")) + sorted((d / case).glob("farfield_*.npz")):
        with np.load(f, allow_pickle=False) as z:
            data[f.name] = {k: z[k] for k in z.files}
    return j, data


def compare(ref: str, new: str) -> None:
    import numpy as np
    R, N = Path(ref), Path(new)
    for jf in sorted(R.glob("*.json")):
        case = jf.stem
        if not (N / f"{case}.json").exists():
            continue
        a, da = _load(R, case)
        b, db = _load(N, case)
        print(f"\n== {case}: EMerge {a['emerge']} -> {b['emerge']}")
        print(f"   tets per pass {a.get('tets')} -> {b.get('tets')}")
        keys = sorted(set(a["time"]) | set(b["time"]))
        for k in ["total"] + keys:
            ta = a["total"] if k == "total" else a["time"].get(k, 0.0)
            tb = b["total"] if k == "total" else b["time"].get(k, 0.0)
            r = f"{100 * (tb / ta - 1):+6.1f} %" if ta > 0.05 else ""
            print(f"   {k:24s} {ta:8.2f} s  {tb:8.2f} s  {r}")
        for name in sorted(set(da) & set(db)):
            for k in sorted(set(da[name]) & set(db[name])):
                x, y = da[name][k], db[name][k]
                if not np.iscomplexobj(x) or x.shape != y.shape:
                    continue
                d = np.max(np.abs(x - y)) / max(np.max(np.abs(x)), 1e-30)
                print(f"   {name}:{k:12s} max rel. difference {d:.2e}")


if __name__ == "__main__":
    if sys.argv[1] == "_one":
        one(*sys.argv[2:5])
    elif sys.argv[1] == "run":
        args = sys.argv[3:]
        passes = None
        if "--passes" in args:
            i = args.index("--passes")
            passes = int(args[i + 1])
            del args[i:i + 2]
        run(sys.argv[2], args, passes)
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
