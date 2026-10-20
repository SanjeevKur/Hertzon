"""Adaptive refinement of an EMerge model with hzmesh (local Delaunay refinement, local).

Per adaptive pass:
  1. the element size at every node is the mean size of its tetrahedra (as in Remesh (stable));
  2. at the nodes of the tetrahedra with the largest error the target size is halved, graded towards
     the rest of the mesh with the growth rate; nodes whose target is below their current size form
     the refinement zone;
  3. hzmesh inserts nodes in the zone only (every old node and every element outside the zone stays
     exactly as it is), so consecutive meshes are nested and Delta S has no re-meshing noise;
  4. how many elements are marked is chosen so that the mesh grows by the refinement percentage
     (the C++ refinement takes milliseconds, so a few trial runs on copies cost nothing);
  5. the mesh goes back to gmsh and EMerge imports it as usual (boundaries, ports, materials).
"""
from __future__ import annotations

import os
import time

import numpy as np

from .core import GmshMesh, available  # noqa: F401  (available re-exported)

SHRINK = 0.7        # target size of the marked elements relative to their current size (0.5 converged slower)
CAP = 1.5           # target size outside the zone (relative): never triggers refinement there
_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def graded_size(X, T, h0, hcap, growth: float, sweeps: int = 64):
    """min(hcap, min_z h0[z] + (growth - 1) * graph distance to z): h0 is inf except in the zone."""
    E = np.unique(np.sort(np.concatenate([T[:, list(e)] for e in _EDGES]), axis=1), axis=0)
    L = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) * max(growth - 1.0, 0.05)
    h = np.minimum(h0, hcap)
    a, b = E[:, 0], E[:, 1]
    for _ in range(sweeps):
        old = h.copy()
        np.minimum.at(h, a, h[b] + L)
        np.minimum.at(h, b, h[a] + L)
        if np.array_equal(old, h):
            break
    return h


def node_sizes(nn: int, tets: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Mean element size of the tetrahedra around every node (EMerge numbering, tets (4, nt))."""
    acc = np.zeros(nn)
    cnt = np.zeros(nn)
    for k in range(4):
        np.add.at(acc, tets[k], lengths)
        np.add.at(cnt, tets[k], 1)
    return acc / np.maximum(cnt, 1)


def size_field(g, hcur, gi, tets, order, k: int, growth: float, shrink: float = SHRINK, grade: bool = True,
               hzm=None):
    """Target size and zone flag per hzmesh node for the k worst tetrahedra (EMerge numbering).
    ``hzm``: the unrefined hzmesh mesh, which grades the field in C++ (else numpy)."""
    h0 = np.full(len(g.X), np.inf)
    zn = gi[np.unique(tets[:, order[:k]])]
    h0[zn] = shrink * hcur[zn]
    if grade:
        slope = max(growth - 1.0, 0.05)
        h = hzm.grade(h0, CAP * hcur, slope) if hzm is not None else graded_size(g.X, g.T, h0, CAP * hcur, growth)
        zone = h < hcur * 0.999
    else:
        h = np.minimum(h0, CAP * hcur)
        zone = np.isfinite(h0)
    return h, zone


FAIL_DIR: str | None = None     # set by the solver: the results folder of the setup


def save_failure(g, h, zone, params: dict, msg: str) -> str | None:
    """Everything needed to repeat a failed refinement offline (python -m hertzon.mesher.replay FILE):
    the input mesh with its CAD classification, the size field, the parameters and every projection
    onto the CAD that the refinement asked for."""
    try:
        import json
        folder = os.environ.get("HZMESH_FAIL_DIR") or FAIL_DIR or os.path.join(os.path.expanduser("~"), ".hertzon")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"hzmesh_failure_{time.strftime('%Y%m%d_%H%M%S')}.npz")
        log = g.proj_log or {}
        keys = np.array(list(log.keys()), float).reshape(-1, 5)
        vals = np.array(list(log.values()), float).reshape(-1, 3)
        np.savez_compressed(path, X=g.X, vdim=g.vdim, vtag=g.vtag, T=g.T, tvol=g.tvol, tri=g.tri, ftag=g.ftag,
                            seg=g.seg, stag=g.stag, curved=np.array(g.curved_entities(), np.int64).reshape(-1, 2),
                            h=np.asarray(h, float), zone=np.asarray(zone, bool), params=json.dumps(params),
                            proj_in=keys, proj_out=vals, msg=msg)
        return path
    except Exception:  # noqa: BLE001
        return None


def refine_model(m, error, lengths, pct: float, growth: float, frac0: float = 0.0, log=print,
                 max_attempts: int = 6, first: bool = False, shrink: float = SHRINK, grade: bool = True, **params):
    """Refine EMerge model ``m`` in place (the gmsh mesh is replaced and EMerge re-imports it).
    Returns (growth in %, fraction of marked elements, mesh arrays for resuming, stats)."""
    from ..solver import local_refine as LR
    t0 = time.time()
    shrink = float(os.environ.get("HZMESH_SHRINK", shrink))         # overrides for experiments
    grade = os.environ.get("HZMESH_GRADE", "1" if grade else "0") == "1"
    if os.environ.get("HZMESH_SAG"):
        params.setdefault("sag", float(os.environ["HZMESH_SAG"]))
    g = GmshMesh()
    g.proj_log = {}
    nodes, tets = np.asarray(m.mesh.nodes), np.asarray(m.mesh.tets)
    nn = nodes.shape[1]
    em_tags = np.fromiter((m.mesh.n_i2t[i] for i in range(nn)), np.int64, nn)
    gi = g.imap[em_tags]                         # EMerge node index -> hzmesh node index
    if (gi < 0).any() or len(g.T) != tets.shape[1]:
        raise RuntimeError("EMerge and gmsh meshes differ")
    hcur = np.zeros(len(g.X))
    hcur[gi] = node_sizes(nn, tets, np.asarray(lengths, float))
    base = g.hzm(project=True)
    t_build = time.time() - t0
    dump = os.environ.get("HZMESH_DUMP")
    if dump:   # input of this pass for offline experiments
        import gmsh
        os.makedirs(dump, exist_ok=True)
        n = len([f for f in os.listdir(dump) if f.endswith(".msh")]) + 1
        gmsh.write(os.path.join(dump, f"pass{n:02d}.msh"))
        np.savez(os.path.join(dump, f"pass{n:02d}.npz"), hcur=hcur, gi=gi, tets=tets, error=np.asarray(error, float),
                 lengths=np.asarray(lengths, float), pct=pct, growth=growth)
    order = np.argsort(-np.asarray(error, float), kind="stable")
    nt0 = len(g.T)
    pct = max(float(pct), 1.0)                     # 0 or negative would divide by zero below
    frac = frac0 or pct / 100.0 / 5.0
    params.setdefault("budget", max(1000, 3.0 * pct / 100.0 * nt0))
    if first:            # mesh straight from gmsh: remove its slivers once (flips only, nodes stay)
        params.setdefault("improve_all", 1)
    best = None
    t_ref = 0.0
    for attempt in range(max_attempts):
        k = int(np.clip(round(frac * nt0), 1, nt0))
        h, zone = size_field(g, hcur, gi, tets, order, k, growth, shrink, grade, base)
        mm = base.clone()
        mm.set_size(h, zone)
        t1 = time.time()
        st = mm.refine(**params)
        t_ref += time.time() - t1
        grow = 100.0 * st["newtets"] / nt0
        log(f"hzmesh: {k} elements marked -> +{grow:.1f}% ({st['ins']:.0f} nodes, {st['fail']:.0f} failed, "
            f"{st['t_refine'] + st['t_improve']:.0f} ms)")
        if best is None or abs(np.log(max(grow, 1e-3) / pct)) < abs(np.log(max(best[0], 1e-3) / pct)):
            best = (grow, frac, mm, st, h, zone)
        if 0.8 * pct <= grow <= 1.25 * pct:
            break
        # new elements grow roughly linearly with the number of marked ones
        frac = float(np.clip(frac * pct / max(grow, 0.5), 1e-5, 0.95))
        if k >= nt0 and grow < pct:
            break
    grow, frac, mm, st, h, zone = best
    n, msg = mm.check()
    if os.environ.get("HZMESH_FAIL_TEST") == "1":      # tests of the failure dump
        n, msg = 1, "test failure"
    if n:
        where = save_failure(g, h, zone, params, msg)
        raise RuntimeError(f"hzmesh produced an invalid mesh: {msg.strip()}"
                           + (f" (input saved to {where})" if where else ""))
    arr = g.to_gmsh_arrays(mm.arrays())
    t2 = time.time()
    LR.load_mesh(m, arr, what="hzmesh refined mesh")
    t_load = time.time() - t2
    st = dict(st, t_build=t_build, t_trials=t_ref, t_load=t_load, t_total=time.time() - t0)
    return grow, frac, arr, st
