"""One line per run: passes, final tets, S11 at the end, time split from @@PROF."""
import json, sys, cmath
for d in sys.argv[1:]:
    j = json.load(open(d + "/convergence.json")); rows = j["passes"] if isinstance(j, dict) else j
    prof = None
    for ln in open(d + ".log", errors="replace"):
        i = ln.find("@@PROF ")
        if i >= 0:
            prof = json.loads(ln[i + 7:])
    t = prof["time"] if prof else {}
    mesh = t.get("gmsh.generate", 0) + t.get("emerge._pre_update", 0) + t.get("hzmesh.refine_model", 0) + t.get("push_to_gmsh", 0)
    r = rows[-1]; S = r["S"]; v = complex(*S[0][0]) if isinstance(S[0][0], list) else complex(*S[0])
    conv = [x["pass"] for x in rows if x.get("passed")]
    print(f"{d:14s} passes {len(rows):2d} tets {r['tets']:6d} dS {r.get('delta_s', 0):.4f} S11 {v.real:+.4f}{v.imag:+.4f}j "
          f"total {prof['total'] if prof else 0:6.1f} s  solve {t.get('solve', 0):6.1f}  err {t.get('error_estimate', 0):5.1f}  "
          f"mesh {mesh:5.1f} (gmsh {t.get('gmsh.generate', 0):.1f}, import {t.get('emerge._pre_update', 0):.1f}, hzmesh {t.get('hzmesh.refine_model', 0):.1f})  passed {conv}")
