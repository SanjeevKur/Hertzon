"""Curved element kernel: with zero displacement it must equal EMerge's affine element matrices;
with displacement, the curved volume integral must match the quadratic map."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
import numpy as np
from hertzon.solver import curvilinear as CV
from emerge._emerge.physics.microwave.assembly import curlcurl as CC
from emerge._emerge.compiled.ccbf import parse_dofcode
from emerge._emerge.mth.optimized import matinv
from emerge._emerge.elements.dofsets import ElementSpace
rng = np.random.default_rng(1)
tv = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float) * 1e-3 + rng.normal(0, 2e-4, (3, 4))
tets = np.array([[0], [1], [2], [3]], np.int64)
edges = np.array([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]], np.int64)
tris = np.array([[0, 0, 0, 1], [1, 1, 2, 2], [2, 3, 3, 3]], np.int64)
t2e = np.arange(6, dtype=np.int64).reshape(6, 1)
t2t = np.arange(4, dtype=np.int64).reshape(4, 1)
lem = CC.local_tet_to_edgeid(t2e, tets, edges, 0)
ltm = CC.local_tet_to_triid(t2t, tets, tris, 0)
ur = np.eye(3, dtype=complex) * (1.3 - 0.1j)
er = np.array([[2.2, 0.1, 0], [0.1, 2.0, 0], [0, 0, 1.7]], complex)
tet_matrices, build = CV.kernels()
for sp in ElementSpace:
    try:
        ds = sp.value if not callable(getattr(sp, "get", None)) else None
    except Exception:
        ds = None
print([s for s in ElementSpace])
QP = CV._QP
Ms = matinv(ur)
for sp in ElementSpace:
    codes = np.array(sp.get_set().set3d.codes, dtype=np.int64)
    typ, idx = parse_dofcode(codes)
    K0, M0 = CC.ned2_tet_stiff_mass(tv, lem, ltm, Ms, er, codes, typ, idx)
    K1, M1 = tet_matrices(tv, np.zeros((6, 3)), lem, ltm, Ms, er, codes, typ, idx, QP)
    eK = np.abs(K1 - K0).max() / np.abs(K0).max(); eM = np.abs(M1 - M0).max() / np.abs(M0).max()
    # displaced midpoints: a smooth bulge; the matrices must stay symmetric and close to the straight ones
    d = rng.normal(0, 3e-5, (6, 3))
    K2, M2 = tet_matrices(tv, d, lem, ltm, Ms, er, codes, typ, idx, QP)
    print(f"{sp.name:24s} ndof {len(codes):2d}  d=0: rel.diff K {eK:.2e} M {eM:.2e}   d!=0: sym K {np.abs(K2-K2.T).max():.1e} "
          f"change K {np.abs(K2-K0).max()/np.abs(K0).max():.2e} M {np.abs(M2-M0).max()/np.abs(M0).max():.2e}")
# volume of the quadratic map: integrate det D with the rule, compare with a fine Monte-Carlo-free check
g, V = CV._grads(tv.T[None])
d = rng.normal(0, 3e-5, (1, 6, 3))
v_rule = abs(V[0]) * (CV._detD(g, d, QP[1:].T) @ QP[0])[0]
# exact: det D is a cubic polynomial in barycentrics -> a degree-7 rule is exact; compare with n=6 rule
QP6 = CV._rule(6)
v_6 = abs(V[0]) * (CV._detD(g, d, QP6[1:].T) @ QP6[0])[0]
print(f"volume straight {abs(V[0]):.6e} curved (n=4) {v_rule:.10e} (n=6) {v_6:.10e}")
