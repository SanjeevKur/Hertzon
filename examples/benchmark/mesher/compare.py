"""Time to solution with Hertzon's local Delaunay refinement (hzmesh) and with EMerge's AURA.

Every case is solved once per refinement method, headless, in a fresh solver process. Both methods use the
same error estimate, the same convergence criterion and the same growth per pass; only the refinement step
differs. Writes results.json and results.md into the output folder.

    python examples/benchmark/mesher/compare.py <out_dir> [--cases post_pec,microstrip] [--methods local,aura]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BENCH = ROOT / "examples" / "benchmark"
PROFRUN = BENCH / "tools" / "profrun.py"

METHODS = {"local": "Local (Delaunay)", "aura": "AURA (EMerge, experimental)"}

# name: project, design, setup, reference S11 (None: no exact reference), setup overrides
CASES = {
    "post_pec": (BENCH / "waveguide_post" / "post_pec_local.hzn", "WR90_Post", "Setup1",
                 complex(0.18037, 0.93542), {}),
    "post_dielectric": (BENCH / "waveguide_post" / "post_dielectric_local.hzn", "WR90_Post", "Setup1",
                        complex(-0.33672, 0.77887), {}),
    "microstrip": (BENCH / "scaling" / "small_microstrip.hzn", "Microstrip50", "Setup1", None, {}),
    "patch": (BENCH / "patch_antenna" / "patch_local.hzn", "Patch245", "Setup1", None, {}),
}
TARGET = 1e-3          # time to |S11 - S11,ref| <= TARGET, for cases with an exact reference


def make_project(src: Path, dst: Path, method: str, overrides: dict, passes: int | None = None) -> None:
    j = json.loads(src.read_text(encoding="utf-8"))
    for d in j["designs"]:
        for s in d.get("setups", []):
            s["props"].update(overrides, RefinementMethod=METHODS[method])
            if passes:
                s["props"]["MaximumPasses"] = str(passes)
            for sw in s.get("sweeps", []):        # adaptive passes only
                sw.setdefault("props", {})["Enabled"] = "False"
    dst.write_text(json.dumps(j, indent=1), encoding="utf-8")


def run(proj: Path, design: str, setup: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(str(out) + ".log", "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.call([sys.executable, str(PROFRUN), str(proj), design, setup, str(out), str(ROOT)],
                             stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    prof = {}
    for ln in open(str(out) + ".log", encoding="utf-8", errors="replace"):
        i = ln.find("@@PROF ")
        if i >= 0:
            prof = json.loads(ln[i + 7:])
    conv = json.loads((out / "convergence.json").read_text())
    rows = conv["passes"] if isinstance(conv, dict) else conv
    fallbacks = sum(ln.count("re-meshing the model instead") for ln in open(str(out) + ".log", errors="replace"))
    return {"rc": rc, "wall": wall, "prof": prof, "rows": rows, "fallbacks": fallbacks,
            "converged": bool(conv.get("converged")) if isinstance(conv, dict) else None}


def s11(row) -> complex:
    S = row["S"]
    return complex(*S[0][0]) if isinstance(S[0][0], list) else complex(*S[0])


def summarise(case: str, method: str, r: dict, ref) -> dict:
    rows, t = r["rows"], r["prof"].get("time", {})
    last = rows[-1]
    mesh = sum(t.get(k, 0.0) for k in ("gmsh.generate", "emerge._pre_update", "hzmesh.refine_model", "push_to_gmsh"))
    res = {"case": case, "method": method, "passes": len(rows), "tets": last["tets"],
           "converged": r["converged"], "delta_s": last.get("delta_s"), "S11": [s11(last).real, s11(last).imag],
           "total_s": r["prof"].get("total", r["wall"]), "solve_s": t.get("solve", 0.0),
           "estimate_s": t.get("error_estimate", 0.0), "mesh_s": mesh, "fallbacks": r["fallbacks"],
           "per_pass": [{"pass": x["pass"], "tets": x["tets"], "time": x.get("time"), "S11": [s11(x).real, s11(x).imag],
                         "delta_s": x.get("delta_s")} for x in rows]}
    if ref is not None:
        err = [abs(s11(x) - ref) for x in rows]
        res["error"] = err[-1]
        cum, hit = 0.0, None
        for x, e in zip(rows, err):
            cum += float(x.get("time") or 0.0)
            if e <= TARGET:
                hit = {"pass": x["pass"], "tets": x["tets"], "time_s": cum}
                break
        res["to_target"] = hit
    return res


def table(results: list[dict]) -> str:
    out = [f"| Case | Method | Passes | Tets | Converged | Final Delta S | Error vs. reference | "
           f"Time to error <= {TARGET:g} | Total | Mesh |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        err = f"{r['error']:.5f}" if "error" in r else "no reference"
        hit = r.get("to_target")
        tt = (f"{hit['time_s']:.0f} s ({hit['tets']} tets)" if hit else "not reached") if "error" in r else "-"
        ds = f"{r['delta_s']:.4f}" if r["delta_s"] is not None else "-"
        out.append(f"| {r['case']} | {r['method']} | {r['passes']} | {r['tets']} | {'yes' if r['converged'] else 'no'} "
                   f"| {ds} | {err} | {tt} | {r['total_s']:.0f} s | {r['mesh_s']:.1f} s |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--no-warmup", action="store_true", help="skip the short run that compiles the numba kernels")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases, methods = a.cases.split(","), a.methods.split(",")
    if not a.no_warmup:          # one-time numba compilation would otherwise land in the first case's time
        src, design, setup, _, ov = CASES[cases[0]]
        make_project(src, out / "warmup.hzn", methods[0], ov, passes=2)
        run(out / "warmup.hzn", design, setup, out / "warmup")
    results = []
    for case in cases:
        src, design, setup, ref, ov = CASES[case]
        for m in methods:
            proj = out / f"{case}_{m}.hzn"
            make_project(src, proj, m, ov)
            print(f"{case} / {m} ...", flush=True)
            res = summarise(case, m, run(proj, design, setup, out / f"{case}_{m}"), ref)
            results.append(res)
            print(f"  {res['passes']} passes, {res['tets']} tets, {res['total_s']:.0f} s", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "results.md").write_text(table(results) + "\n", encoding="utf-8")
    print(table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
