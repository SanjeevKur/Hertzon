"""Adaptive mesh refinement with Mmg (mmg3d, LGPL, https://www.mmgtools.org).

Mmg adapts an existing tetrahedral mesh to a target element size per node with local operations
(edge split/collapse, edge and face swaps, node relocation) while keeping surfaces, feature curves
and material interfaces. This is the closest open-source equivalent of commercial local adaptive meshing:
only where the error estimate asks for smaller elements does the mesh change.

The current gmsh mesh is written in Medit format (entity tags as references), mmg3d is run on it
with a per-node size field and the adapted mesh is handed back to EMerge through gmsh.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .local_refine import LocalMesh, load_mesh


def mmg_binary() -> Path | None:
    """Path of the mmg3d executable shipped by the ``pymmg`` package (None when not installed)."""
    try:
        import mmg3d
    except Exception:  # noqa: BLE001
        return None
    d = Path(mmg3d.__file__).parent
    for name in ("mmg3d_O3.exe", "mmg3d_O3", "mmg3d.exe", "mmg3d"):
        if (d / name).is_file():
            return d / name
    return None


def available() -> bool:
    return mmg_binary() is not None


# ---------------------------------------------------------------------------------------------
# Medit I/O
# ---------------------------------------------------------------------------------------------
def _write_mesh(path: Path, X, ref_v, edges, ref_e, tris, ref_f, tets, ref_t, corners) -> None:
    with open(path, "w") as f:
        f.write("MeshVersionFormatted 2\nDimension 3\n")
        f.write(f"Vertices\n{len(X)}\n")
        np.savetxt(f, np.column_stack([X, ref_v]), fmt="%.17g %.17g %.17g %d")
        if len(edges):
            f.write(f"Edges\n{len(edges)}\n")
            np.savetxt(f, np.column_stack([edges + 1, ref_e]), fmt="%d")
            # feature curves: Mmg may split them but keeps them as sharp edges
            f.write(f"Ridges\n{len(edges)}\n")
            np.savetxt(f, np.arange(1, len(edges) + 1), fmt="%d")
        f.write(f"Triangles\n{len(tris)}\n")
        np.savetxt(f, np.column_stack([tris + 1, ref_f]), fmt="%d")
        f.write(f"Tetrahedra\n{len(tets)}\n")
        np.savetxt(f, np.column_stack([tets + 1, ref_t]), fmt="%d")
        if len(corners):
            f.write(f"Corners\n{len(corners)}\n")
            np.savetxt(f, corners + 1, fmt="%d")
            f.write(f"RequiredVertices\n{len(corners)}\n")
            np.savetxt(f, corners + 1, fmt="%d")
        f.write("End\n")


def _write_sol(path: Path, h: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(f"MeshVersionFormatted 2\nDimension 3\nSolAtVertices\n{len(h)}\n1 1\n")
        np.savetxt(f, h, fmt="%.17g")
        f.write("End\n")


def _read_mesh(path: Path) -> dict[str, np.ndarray]:
    tok = path.read_text().split()
    out: dict[str, np.ndarray] = {}
    widths = {"Vertices": 4, "Edges": 3, "Triangles": 4, "Tetrahedra": 5}
    skip = {"Corners": 1, "Ridges": 1, "RequiredVertices": 1, "RequiredEdges": 1, "RequiredTriangles": 1,
            "RequiredTetrahedra": 1, "Normals": 3, "Tangents": 3, "NormalAtVertices": 2, "TangentAtVertices": 2,
            "NormalAtTriangleVertices": 3, "Quadrilaterals": 5, "Prisms": 7, "Hexahedra": 9}
    i = 0
    while i < len(tok):
        k = tok[i]
        if k in widths:
            n = int(tok[i + 1])
            w = widths[k]
            out[k] = np.array(tok[i + 2:i + 2 + n * w], float).reshape(n, w)
            i += 2 + n * w
        elif k in ("MeshVersionFormatted", "Dimension"):
            i += 2
        elif k == "End":
            break
        elif k in skip:
            i += 2 + int(tok[i + 1]) * skip[k]
        else:
            raise RuntimeError(f"unexpected section '{k}' in the Mmg output")
    return out


# ---------------------------------------------------------------------------------------------
# Adaptation
# ---------------------------------------------------------------------------------------------
class _Export:
    """The current gmsh mesh as Medit arrays (node indices 0..n-1, references = gmsh entity tags)."""

    def __init__(self) -> None:
        lm = LocalMesh()
        tags = np.fromiter(lm.xyz.keys(), np.int64, len(lm.xyz))
        row = np.zeros(int(tags.max()) + 1, np.int64)
        row[tags] = np.arange(len(tags))
        self.X = np.array([lm.xyz[t] for t in tags.tolist()], float)
        ent = np.array([lm.ent[t] for t in tags.tolist()], np.int64)
        self.ref_v = np.zeros(len(tags), np.int64)
        self.corners = np.flatnonzero(ent[:, 0] == 0)
        self.point_tag = {int(i): int(ent[i, 1]) for i in self.corners}
        e, re, f, rf = [], [], [], []
        for (dim, tag), conn in lm.elems.items():
            if dim == 1:
                e.extend(conn)
                re.extend([tag] * len(conn))
            elif dim == 2:
                f.extend(conn)
                rf.extend([tag] * len(conn))
        self.edges = row[np.asarray(e, np.int64).reshape(-1, 2)] if e else np.zeros((0, 2), np.int64)
        self.ref_e = np.asarray(re, np.int64)
        self.tris = row[np.asarray(f, np.int64).reshape(-1, 3)]
        self.ref_f = np.asarray(rf, np.int64)
        self.tets = row[np.asarray(lm.tets, np.int64)]
        self.ref_t = np.array([o[1] for o in lm.tet_owner], np.int64)

    def node_lengths(self) -> np.ndarray:
        """Mean length of the tet edges at every node (the current local mesh size)."""
        s = np.zeros(len(self.X))
        c = np.zeros(len(self.X))
        T = self.tets
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            ln = np.linalg.norm(self.X[T[:, a]] - self.X[T[:, b]], axis=1)
            np.add.at(s, T[:, a], ln)
            np.add.at(s, T[:, b], ln)
            np.add.at(c, T[:, a], 1)
            np.add.at(c, T[:, b], 1)
        return s / np.maximum(c, 1)


def _run_mmg(ex: _Export, h: np.ndarray, work: Path, hgrad: float, hausd: float) -> dict[str, np.ndarray]:
    src, sol, dst = work / "in.mesh", work / "in.sol", work / "out.mesh"
    _write_mesh(src, ex.X, ex.ref_v, ex.edges, ex.ref_e, ex.tris, ex.ref_f, ex.tets, ex.ref_t, ex.corners)
    _write_sol(sol, h)
    if dst.exists():
        dst.unlink()
    cmd = [str(mmg_binary()), "-in", str(src), "-sol", str(sol), "-out", str(dst), "-v", "-1",
           "-hgrad", "-1", "-hgradreq", "-1", "-hausd", f"{hausd:g}", "-opnbdy"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, creationflags=flags)
    if r.returncode != 0 or not dst.exists():
        raise RuntimeError(f"mmg3d failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
    return _read_mesh(dst)


def _to_arrays(ex: _Export, out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mmg output -> mesh arrays for gmsh (nodes classified on the lowest-dimensional entity)."""
    import gmsh
    V = out["Vertices"][:, :3].copy()
    n = len(V)
    E = out.get("Edges", np.zeros((0, 3)))
    F = out["Triangles"]
    T = out["Tetrahedra"]
    edges, re = E[:, :2].astype(np.int64) - 1, E[:, 2].astype(np.int64)
    tris, rf = F[:, :3].astype(np.int64) - 1, F[:, 3].astype(np.int64)
    tets, rt = T[:, :4].astype(np.int64) - 1, T[:, 4].astype(np.int64)
    # keep only edges/triangles that belong to gmsh entities (Mmg may add reference-0 boundary items)
    curves = {t for d, t in gmsh.model.getEntities(1)}
    surfs = {t for d, t in gmsh.model.getEntities(2)}
    ke = np.isin(re, list(curves))
    edges, re = edges[ke], re[ke]
    kf = np.isin(rf, list(surfs))
    tris, rf = tris[kf], rf[kf]
    # classification: volume < surface < curve < point
    ent = np.zeros((n, 2), np.int64)
    ent[tets.ravel(), 0] = 3
    ent[tets.ravel(), 1] = np.repeat(rt, 4)
    ent[tris.ravel(), 0] = 2
    ent[tris.ravel(), 1] = np.repeat(rf, 3)
    ent[edges.ravel(), 0] = 1
    ent[edges.ravel(), 1] = np.repeat(re, 2)
    from scipy.spatial import cKDTree
    if len(ex.corners):
        d, idx = cKDTree(V).query(ex.X[ex.corners])
        for i, j in zip(ex.corners.tolist(), idx.tolist()):
            ent[j] = (0, ex.point_tag[i])
    # keep the geometry exact: put every node on a curve/surface back onto its CAD entity
    # (Mmg only knows the faceted surface; this undoes any drift: nodes stay on the model)
    V0 = V.copy()
    for j in np.flatnonzero((ent[:, 0] == 1) | (ent[:, 0] == 2)).tolist():
        try:
            p, _ = gmsh.model.getClosestPoint(int(ent[j, 0]), int(ent[j, 1]), V[j].tolist())
            V[j] = p
        except Exception:  # noqa: BLE001
            pass

    def vol(P):
        a, b, c, d = (P[tets[:, k]] for k in range(4))
        return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
    v0 = vol(V0)
    for _ in range(3):                  # undo a snap that would flip or flatten a tetrahedron
        bad = np.flatnonzero((vol(V) * np.sign(v0)) < 0.05 * np.abs(v0))
        if not len(bad):
            break
        nodes = np.unique(tets[bad].ravel())
        V[nodes] = V0[nodes]
    used = np.zeros(n, bool)
    used[tets.ravel()] = True
    tags = np.arange(1, n + 1, dtype=np.int64)
    keys, conn = [], []
    pts = np.flatnonzero(ent[:, 0] == 0)
    for j in pts.tolist():
        keys.append((0, int(ent[j, 1]), 1))
        conn.append(np.array([j + 1]))
    for dim, items, refs in ((1, edges, re), (2, tris, rf), (3, tets, rt)):
        for tag in np.unique(refs).tolist():
            sel = items[refs == tag]
            keys.append((dim, int(tag), len(sel)))
            conn.append((sel + 1).ravel())
    return {"node_tags": tags[used], "node_xyz": V[used], "node_ent": ent[used],
            "elem_keys": np.array(keys, np.int64).reshape(-1, 3),
            "elem_conn": np.concatenate(conn).astype(np.int64)}


def refine_with_mmg(m, size_for, pct: float, growth: float) -> dict[str, np.ndarray]:
    """Adapt the mesh of ``m`` with Mmg, reload it into EMerge and return it.

    ``size_for(frac)`` returns the target size field when the worst ``frac`` of the tetrahedra are
    refined; ``frac`` is adjusted so that the mesh grows by about ``pct`` percent."""
    ex = _Export()
    n0 = len(ex.tets)
    target = n0 * (1 + pct / 100.0)
    h_now = ex.node_lengths()
    span = np.ptp(ex.X, axis=0).max()
    # Mmg does not know the CAD surfaces; a tight Hausdorff tolerance makes it refine flat sheets
    # (open boundaries) needlessly, so only gross deviations are corrected
    hausd = 1e-2 * span
    # start from the fraction that hit the target in the previous pass
    frac, lo, hi, best = getattr(m, "_mmg_frac", None) or 1e-3, 0.0, 1.0, None
    with tempfile.TemporaryDirectory(prefix="hertzon_mmg_") as tmp:
        work = Path(os.environ.get("HERTZON_MMG_DIR") or tmp)
        for _ in range(5):
            # never coarsen (local): the target size is at most the current local size
            h = np.minimum(size_for(frac)(ex.X.T), h_now)
            t0 = time.time()
            out = _run_mmg(ex, h, work, growth, hausd)
            n = len(out["Tetrahedra"])
            if os.environ.get("HERTZON_MMG_DEBUG"):
                print(f"MMG run frac={frac:.4f} tets {n0}->{n} ({time.time() - t0:.2f}s)", flush=True)
            if best is None or abs(n - target) < abs(best[0] - target):
                best = (n, out)
            grow = 100.0 * (n / n0 - 1)
            if 0.8 * pct <= grow <= 1.25 * pct:
                m._mmg_frac = frac
                break
            if grow < pct:
                lo = frac
            else:
                hi = frac
            # added elements grow about linearly with the refined fraction
            guess = frac * pct / max(grow, 0.5)
            frac = guess if lo < guess < hi else 0.5 * (lo + hi)
    arr = _to_arrays(ex, best[1])
    load_mesh(m, arr)
    return arr


# ---------------------------------------------------------------------------------------------
# Zone re-mesh (local local adaptive refinement)
# ---------------------------------------------------------------------------------------------
ZONE_BUFFER = 3          # layers of neighbour tets added around the marked ones (room for grading)
ZONE_HGRAD = 1.3         # max size ratio between neighbouring edges inside the zone


def _append_required(path: Path, tets: np.ndarray, tris: np.ndarray, verts: np.ndarray) -> None:
    """Freeze elements: Mmg must not touch required tetrahedra, triangles and vertices."""
    txt = path.read_text()
    assert txt.rstrip().endswith("End")
    with open(path, "w") as f:
        f.write(txt.rstrip()[:-3])
        for key, ids in (("RequiredTetrahedra", tets), ("RequiredTriangles", tris), ("RequiredVertices", verts)):
            if len(ids):
                f.write(f"{key}\n{len(ids)}\n")
                np.savetxt(f, np.asarray(ids) + 1, fmt="%d")
        f.write("End\n")


def zone_refine(m, error, pct: float, log=None) -> float:
    """Zone re-mesh: only a zone around the tets with the largest error is re-meshed by Mmg with half
    the element size; every other element is frozen (required) and stays exactly the same, so
    consecutive meshes are identical outside the zone (no re-meshing noise in Delta S) while the zone
    gets a fresh high-quality mesh (also on surfaces). Returns the growth in percent."""
    ex = _Export()
    X, T = ex.X, ex.tets
    n0 = len(T)
    # EMerge tet order -> exported tet order (both via gmsh node tags)
    lm_tags = np.fromiter(LocalMesh().xyz.keys(), np.int64)
    row = {int(t): i for i, t in enumerate(lm_tags.tolist())}
    key = {tuple(sorted(t)): k for k, t in enumerate(T.tolist())}
    i2t = m.mesh.n_i2t
    E = np.asarray(m.mesh.tets)
    emap = np.array([key[tuple(sorted(row[i2t[int(a)]] for a in E[:, i]))] for i in range(E.shape[1])])
    order = emap[np.argsort(-np.asarray(error))]
    h_now = ex.node_lengths()
    hausd = 1e-2 * np.ptp(X, axis=0).max()
    target = n0 * (1 + pct / 100.0)
    frac, lo, hi, best = getattr(m, "_zone_frac", None) or pct / 100.0 / 40.0, 0.0, 1.0, None
    with tempfile.TemporaryDirectory(prefix="hertzon_zone_") as tmp:
        work = Path(os.environ.get("HERTZON_MMG_DIR") or tmp)
        for attempt in range(4):
            k = int(np.clip(round(frac * n0), 1, n0))
            marked = np.zeros(n0, bool)
            marked[order[:k]] = True
            zone = marked.copy()
            for _ in range(ZONE_BUFFER):
                zn = np.zeros(len(X), bool)
                zn[T[zone].ravel()] = True
                zone |= zn[T].any(1)
            frozen_n = np.zeros(len(X), bool)
            frozen_n[T[~zone].ravel()] = True
            h = h_now.copy()
            mk_n = np.zeros(len(X), bool)
            mk_n[T[marked].ravel()] = True
            h[mk_n & ~frozen_n] *= 0.5
            src, sol, dst = work / "in.mesh", work / "in.sol", work / "out.mesh"
            _write_mesh(src, X, ex.ref_v, ex.edges, ex.ref_e, ex.tris, ex.ref_f, T, ex.ref_t, ex.corners)
            _append_required(src, np.flatnonzero(~zone), np.flatnonzero(frozen_n[ex.tris].all(1)),
                             np.flatnonzero(frozen_n))
            _write_sol(sol, h)
            if dst.exists():
                dst.unlink()
            cmd = [str(mmg_binary()), "-in", str(src), "-sol", str(sol), "-out", str(dst), "-v", "-1",
                   "-hgrad", f"{ZONE_HGRAD:g}", "-hgradreq", "-1", "-hausd", f"{hausd:g}", "-opnbdy"]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
            t0 = time.time()
            r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, creationflags=flags)
            if r.returncode != 0 or not dst.exists():
                raise RuntimeError(f"mmg3d failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
            out = _read_mesh(dst)
            n = len(out["Tetrahedra"])
            grow = 100.0 * (n / n0 - 1)
            if log:
                log(f"zone re-mesh: {k} marked, zone {int(zone.sum())} tets -> {n} tetrahedra "
                    f"(+{grow:.1f}%, {time.time() - t0:.1f}s)")
            if best is None or abs(n - target) < abs(best[0] - target):
                best = (n, out, frac)
            if 0.7 * pct <= grow <= 1.4 * pct:
                break
            if grow < pct:
                lo = frac
            else:
                hi = frac
            guess = frac * pct / max(grow, 0.5)
            frac = guess if lo < guess < hi else 0.5 * (lo + hi)
    n, out, m._zone_frac = best
    load_mesh(m, _to_arrays(ex, out))
    return 100.0 * (n / n0 - 1)
