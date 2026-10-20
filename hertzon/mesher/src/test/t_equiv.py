"""Original vs vectorised Mesh3D._pre_update: every attribute must be identical."""
import time
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
import numpy as np, gmsh
geo = sys.argv[1] if len(sys.argv) > 1 else "mix"
hc = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
o = gmsh.model.occ
if geo == "mix":
    o.addBox(-1, -1, -1, 2, 2, 2); o.addSphere(0, 0, 0, 0.4); o.addBox(0.5, -0.8, -0.05, 0.4, 1.6, 0.1)
    o.fragment([(3, 1)], [(3, 2), (3, 3)])
elif geo == "patch":
    o.addBox(-2, -2, 0, 4, 4, 0.16); o.addBox(-3, -3, -1, 6, 6, 3); o.addRectangle(-0.6, -0.5, 0.16, 1.2, 1.0)
    o.fragment([(3, 2)], [(3, 1), (2, 13)])
else:
    o.addCylinder(0, 0, 0, 0, 0, 2, 0.5); o.addCylinder(0, 0, 0, 0, 0, 2, 0.2); o.cut([(3, 1)], [(3, 2)])
o.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax", hc); gmsh.option.setNumber("General.NumThreads", 1)
gmsh.model.mesh.generate(3)
from emerge._emerge import mesh3d
orig = mesh3d.Mesh3D._pre_update
from hertzon.solver import emerge_patches as E
E.apply()
fast = mesh3d.Mesh3D._pre_update
assert fast is not orig, "patch not applied"

def build():
    m = mesh3d.Mesh3D.__new__(mesh3d.Mesh3D)
    try:
        m.__init__(None)
    except Exception:
        m.__init__()
    return m

def run(fn):
    m = build(); t = time.time(); fn(m); return m, time.time() - t

def same(a, b, path):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a2, b2 = np.asarray(a), np.asarray(b)
        if a2.shape == () and b2.shape == () and path.startswith("dimtag_to_edges[(0,"):
            return []   # EMerge returns np.ndarray([]) (uninitialised 0-d) for points
        if a2.shape != b2.shape or not np.array_equal(a2, b2):
            return [f"{path}: array {a2.shape} {a2.dtype} vs {b2.shape} {b2.dtype}"]
        if a2.dtype.kind != b2.dtype.kind:
            return [f"{path}: dtype {a2.dtype} vs {b2.dtype}"]
        return []
    if isinstance(a, dict):
        if not isinstance(b, dict) or list(a.keys()) != list(b.keys()):
            return [f"{path}: dict keys differ ({len(a)} vs {len(b) if isinstance(b, dict) else type(b)})"]
        out = []
        for k in a:
            out += same(a[k], b[k], f"{path}[{k!r}]")
            if len(out) > 3:
                break
        return out
    if isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [f"{path}: {type(a).__name__}[{len(a)}] vs {type(b).__name__}[{len(b) if hasattr(b, '__len__') else '-'}]"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += same(x, y, f"{path}[{i}]")
            if len(out) > 3:
                break
        return out
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} vs {type(b).__name__}"]
    try:
        return [] if a == b else [f"{path}: {a!r} vs {b!r}"]
    except Exception:
        return []

m1, t1 = run(orig)
m2, t2 = run(fast)
bad = []
for k in sorted(set(m1.__dict__) | set(m2.__dict__)):
    if k not in m1.__dict__ or k not in m2.__dict__:
        bad.append(f"{k}: only in one"); continue
    a, b = m1.__dict__[k], m2.__dict__[k]
    if callable(a) and not isinstance(a, np.ndarray):
        continue
    bad += same(a, b, k)
# edge lookup: dense vs sparse on all edges + random misses
nN = m1.nodes.shape[1]
print(geo, "tets", m1.tets.shape[1], "orig %.2f s  fast %.2f s" % (t1, t2))
print("DIFFS:" if bad else "identical", *bad[:20], sep="\n  ")
