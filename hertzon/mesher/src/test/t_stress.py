import time
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
import numpy as np, gmsh
from hertzon.mesher.core import GmshMesh

def geom(shape):
    o = gmsh.model.occ
    if shape == "box":
        o.addBox(0, 0, 0, 1, 1, 1); o.addBox(0.3, 0.3, 0.3, 0.4, 0.4, 0.4); o.fragment([(3, 1)], [(3, 2)])
    elif shape == "sphere":
        o.addSphere(0, 0, 0, 1); o.addBox(-0.3, -0.3, -0.3, 0.6, 0.6, 0.6); o.fragment([(3, 1)], [(3, 2)])
    elif shape == "cyl":
        o.addBox(-2, -2, -1, 4, 4, 2); o.addCylinder(0, 0, -0.2, 0, 0, 0.4, 0.7); o.fragment([(3, 1)], [(3, 2)])
    elif shape == "patch":   # thin substrate + patch sheet + air box, like a patch antenna
        o.addBox(-3, -3, 0, 6, 6, 0.16); o.addBox(-6, -6, -3, 12, 12, 9)
        o.addRectangle(-1, -1.2, 0.16, 2, 2.4)
        o.fragment([(3, 2)], [(3, 1), (2, 3)]); o.fragment(o.getEntities(3), o.getEntities(2))
    elif shape == "coax":
        o.addCylinder(0, 0, 0, 0, 0, 3, 1.0); o.addCylinder(0, 0, 0, 0, 0, 3, 0.3)
        o.cut([(3, 1)], [(3, 2)])
    o.synchronize()

def setup(shape, hc):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(shape); geom(shape)
    gmsh.option.setNumber("Mesh.MeshSizeMax", hc)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.mesh.generate(3)
    g = GmshMesh()
    E = np.concatenate([g.T[:, [i, j]] for i, j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))])
    L = np.linalg.norm(g.X[E[:, 0]] - g.X[E[:, 1]], axis=1)
    acc = np.zeros(len(g.X)); cnt = np.zeros(len(g.X))
    np.add.at(acc, E.ravel(), np.repeat(L, 2)); np.add.at(cnt, E.ravel(), 1)
    return g, acc / cnt

def tetvol(X, T):
    a, b, c, d = (X[T[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6

def one(g, hcur, seed, r, shrink=0.5, **kw):
    rng = np.random.default_rng(seed)
    pick = rng.integers(len(g.X))
    if seed % 2 == 0:   # every other run: zone on the boundary
        cand = np.where(g.vdim < 3)[0]; pick = cand[rng.integers(len(cand))]
    c = g.X[pick]
    zone = np.linalg.norm(g.X - c, axis=1) < r
    h = np.where(zone, shrink * hcur, 1.5 * hcur)
    m = g.hzm(True); m.set_size(h, zone)
    t0 = time.time(); st = m.refine(**kw); dt = time.time() - t0
    n, msg = m.check()
    a = m.arrays()
    moved = np.abs(a["X"][:len(g.X)] - g.X).max()
    q, d = m.quality()
    return dict(seed=seed, ok=n == 0, msg=msg.strip()[:120], moved=moved, nt=len(a["T"]), nt0=len(g.T), dt=dt,
                qmin=q.min(), q1=np.percentile(q, 1), dmin=d.min(), n10=int((d < 10).sum()), fail=st["fail"], ins=st["ins"])

if __name__ == "__main__":
    shape, hc, r = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    nseed = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    g, hcur = setup(shape, hc)
    q0 = g.hzm(False).quality()
    print(f"{shape}: {len(g.T)} tets, input qmin {q0[0].min():.3f} q1 {np.percentile(q0[0],1):.3f} dmin {q0[1].min():.2f}  curved {len(g.curved_entities())}")
    worst = 1
    for s in range(nseed):
        R = one(g, hcur, s, r)
        worst = min(worst, R["qmin"])
        flag = "" if R["ok"] and R["moved"] == 0 else "  <-- " + R["msg"]
        print(f" seed {s}: {R['nt0']}->{R['nt']} tets {R['dt']*1000:.0f} ms, qmin {R['qmin']:.3f} q1 {R['q1']:.3f} dmin {R['dmin']:.2f} n<10deg {R['n10']} fail {R['fail']:.0f}{flag}")
    gmsh.finalize()
