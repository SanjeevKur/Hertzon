"""Curvilinear (second-order geometry) tetrahedra for EMerge's curl-curl assembly.

EMerge maps every tetrahedron affinely, so a curved CAD surface is replaced by flat facets. On a strongly
reflecting curved object (a post in a waveguide, a coax inner conductor) this faceting error dominates the
S-parameters long before the field is resolved, and the error estimator cannot see it.

Here the tetrahedra next to curved PEC or material surfaces get quadratic edges: the midpoint of every mesh
edge on a curved CAD surface or curve is moved onto the true geometry (gmsh.getClosestPoint), giving the
usual 10-node isoparametric geometry x(y) = y + sum_e d_e phi_e(y) with phi_e = 4 l_i l_j on the straight
tetrahedron y. With D = dx/dy, the covariant Piola map turns this into the straight tetrahedron with
point-wise materials

    curl-curl:  mu^-1  ->  D^T mu^-1 D / det D
    mass:       eps    ->  det D * D^-1 eps D^-T

so EMerge's own basis functions are reused unchanged and only the element matrices of the curved
tetrahedra are recomputed (64-point conical product rule, degree 7). Faces without curved edges keep
their exact tangential traces, so curved and straight elements stay conforming, and nothing changes on
ports and other boundaries with surface integrals (edges on them are never curved).
"""
from __future__ import annotations

import os

import numpy as np
from numba import njit, prange

_FLAT = {"Plane", "Line", "Point"}


def _curved_type(dim: int, tag: int) -> bool:
    """CAD entity that may be curved. gmsh calls swept ellipses, splines, pipes and lofts "Unknown", so only
    planes and lines count as flat (a flat face that is not recognised just gets zero displacements).
    Discrete entities (a mesh loaded from file) have no geometry to project onto."""
    import gmsh
    t = gmsh.model.getType(dim, tag)
    return t not in _FLAT and "iscrete" not in t
_PAIRS = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
_CACHE: dict = {}
CTX: dict = {"exclude": frozenset()}       # face tags with surface integrals (set by the assembler wrapper)
STATS: dict = {}


def enabled() -> bool:
    env = os.environ.get("HERTZON_CURVED")          # override for experiments
    return env != "0" if env is not None else bool(CTX.get("enabled", True))


def curvable(bc) -> bool:
    """Boundaries whose faces may be curved: PEC, PMC (no surface integral) and metal walls
    (SurfaceImpedance from a material: the impedance term is tiny next to the field)."""
    from emerge._emerge.physics.microwave.assembly import assembler
    try:
        from emerge._emerge.physics.microwave.bcs.boundary_conditions import PMC
    except Exception:  # noqa: BLE001
        PMC = assembler.PEC
    if isinstance(bc, (assembler.PEC, PMC)):
        return True
    return isinstance(bc, assembler.SurfaceImpedance) and getattr(bc, "_Zf", None) is None


def exclude_tags(bcs) -> frozenset:
    """Face tags with surface integrals (ports, absorbers, lumped elements, sheets): they stay straight."""
    return frozenset(int(t) for bc in bcs if not curvable(bc) for t in (getattr(bc, "tags", None) or []))


def curved_map(X, d, lam):
    """Positions (nt, np, 3) and Jacobians D (nt, np, 3, 3) of the curved map at barycentric points
    lam (np, 4), for straight tets X (nt, 4, 3) with edge displacements d (nt, 6, 3) in _PAIRS order.
    Fields of the straight element map with the covariant Piola transform: E = D^-T E_s, B = D B_s / det D."""
    g, _ = _grads(X)
    pos = np.einsum("pk,tkd->tpd", lam, X)
    D = np.broadcast_to(np.eye(3), (len(X), len(lam), 3, 3)).copy()
    for e, (i, j) in enumerate(_PAIRS):
        pos += 4 * (lam[:, i] * lam[:, j])[None, :, None] * d[:, None, e, :]
        gp = 4 * (lam[None, :, i, None] * g[:, None, j] + lam[None, :, j, None] * g[:, None, i])   # (nt, np, 3)
        D += d[:, None, e, :, None] * gp[:, :, None, :]
    return pos, D


def _rule(n: int = 4) -> np.ndarray:
    """Conical product rule on the tetrahedron, (5, n^3): weights (sum 1) and 4 barycentric rows."""
    from scipy.special import roots_jacobi
    tu, wu = roots_jacobi(n, 2.0, 0.0)
    tv, wv = roots_jacobi(n, 1.0, 0.0)
    tw, ww = roots_jacobi(n, 0.0, 0.0)
    u, v, w = (1 + tu) / 2, (1 + tv) / 2, (1 + tw) / 2
    U, V, W = np.meshgrid(u, v, w, indexing="ij")
    WU, WV, WW = np.meshgrid(wu, wv, ww, indexing="ij")
    x = U
    y = V * (1 - U)
    z = W * (1 - U) * (1 - V)
    wt = (WU * WV * WW).ravel()
    out = np.empty((5, wt.size))
    out[0] = wt / wt.sum()
    out[2], out[3], out[4] = x.ravel(), y.ravel(), z.ravel()
    out[1] = 1 - out[2] - out[3] - out[4]
    return out


_QP = _rule(4)


# ------------------------------------------------------------------------------------------------
# geometry: which edges are curved and by how much
# ------------------------------------------------------------------------------------------------

def _edge_keys(e0, e1, n):
    a, b = np.minimum(e0, e1).astype(np.int64), np.maximum(e0, e1).astype(np.int64)
    return a * np.int64(n) + b


def _closest(dim: int, tag: int, pts: np.ndarray) -> np.ndarray:
    import gmsh
    if not len(pts):
        return pts
    q, _ = gmsh.model.getClosestPoint(dim, tag, np.ascontiguousarray(pts, float).ravel())
    q = np.asarray(q, float).reshape(-1, 3)
    return q if q.shape == pts.shape else pts


def _grads(X):
    """Barycentric gradients (nt, 4, 3) and volumes of tetrahedra X (nt, 4, 3)."""
    J = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=1)   # rows = edges
    det = np.linalg.det(J)
    Ji = np.linalg.inv(J)                     # columns: gradients of l1, l2, l3
    g = np.empty((len(X), 4, 3))
    g[:, 1:] = np.transpose(Ji, (0, 2, 1))
    g[:, 0] = -g[:, 1:].sum(1)
    return g, det / 6


def _detD(g, d, lam):
    """det(D) at barycentric points lam (np, 4) for tets with gradients g (nt,4,3), displacements d (nt,6,3)."""
    out = np.empty((g.shape[0], lam.shape[0]))
    for p, l in enumerate(lam):
        D = np.broadcast_to(np.eye(3), (g.shape[0], 3, 3)).copy()
        for e, (i, j) in enumerate(_PAIRS):
            gp = 4 * (l[i] * g[:, j] + l[j] * g[:, i])            # grad phi_e (nt, 3)
            D += d[:, e, :, None] * gp[:, None, :]
        out[:, p] = np.linalg.det(D)
    return out


def _check_points():
    pts = [np.eye(4)[k] for k in range(4)]
    pts += [(np.eye(4)[i] + np.eye(4)[j]) / 2 for i, j in _PAIRS]
    pts += [np.full(4, 0.25)]
    pts += [(np.ones(4) - np.eye(4)[k]) / 3 for k in range(4)]
    return np.vstack([np.array(pts), _QP[1:].T])


_CHK = _check_points()


def curved_data(mesh, exclude=frozenset()):
    """Curved tets of an EMerge mesh: (tet ids, displacement (nc, 6, 3) of the edges in _PAIRS order).
    None when nothing is curved."""
    import gmsh
    nodes = np.asarray(mesh.nodes, float)
    tets = np.asarray(mesh.tets, np.int64)
    edges = np.asarray(mesh.edges, np.int64)
    nn, nt, ne = nodes.shape[1], tets.shape[1], edges.shape[1]
    fp = (id(mesh), nn, nt, ne, float(nodes[:, :: max(1, nn // 97)].sum()), int(tets[:, :: max(1, nt // 97)].sum()),
          tuple(sorted(exclude)))
    if fp in _CACHE:
        return _CACHE[fp]
    _CACHE.clear()
    disp = np.zeros((ne, 3))
    done = np.zeros(ne, bool)
    # edges on excluded faces stay straight (ports, absorbing and impedance boundaries, periodic faces)
    for t in exclude:
        ids = mesh.ftag_to_edge.get(t, [])
        done[np.asarray(ids, np.int64)] = True
    try:
        curves = [t for _, t in gmsh.model.getEntities(1) if _curved_type(1, t)]
        faces = [t for _, t in gmsh.model.getEntities(2) if _curved_type(2, t)]
        # A curved surface that meets a port (or another face with surface integrals) stays straight as a
        # whole: the port mode is solved on the straight facets, and a curved line behind a straight port
        # would be an impedance step (coax, circular waveguide). Curves on such faces stay straight too.
        rim = set()
        for t in exclude:
            rim.update(abs(c) for _, c in gmsh.model.getBoundary([(2, int(t))], False, False, False))
        bnd = {f: {abs(c) for _, c in gmsh.model.getBoundary([(2, f)], False, False, False)} for f in faces}
        changed = True
        while changed:                   # connected curved faces (a conductor made of several faces)
            changed = False
            for f in list(bnd):
                if bnd[f] & rim:
                    rim.update(bnd.pop(f))
                    changed = True
        faces = sorted(bnd)
        curves = [c for c in curves if c not in rim]
    except Exception:  # noqa: BLE001
        curves, faces = [], []
    mid = 0.5 * (nodes[:, edges[0]] + nodes[:, edges[1]]).T
    for dim, tags, emap in ((1, curves, mesh.etag_to_edge), (2, faces, mesh.ftag_to_edge)):
        for t in tags:
            ids = np.asarray(emap.get(t, []), np.int64)
            if not len(ids):
                continue
            ids = ids[~done[ids]]
            if not len(ids):
                continue
            q = _closest(dim, t, mid[ids])
            disp[ids] = q - mid[ids]
            done[ids] = True
    L = np.linalg.norm(nodes[:, edges[0]] - nodes[:, edges[1]], axis=0)
    dn = np.linalg.norm(disp, axis=1)
    disp[dn > 0.25 * L] = 0.0        # projection went elsewhere (seam, wrong patch): keep straight
    disp[dn < 1e-12 * np.maximum(L, 1e-300)] = 0.0
    ek = _edge_keys(edges[0], edges[1], nn)
    order = np.argsort(ek)
    ek_s = ek[order]
    # local edges of every tet
    a, b = tets[_PAIRS[:, 0]], tets[_PAIRS[:, 1]]                  # (6, nt)
    pos = np.searchsorted(ek_s, _edge_keys(a, b, nn))
    eid = order[np.minimum(pos, len(order) - 1)]                    # (6, nt)
    curved_edge = np.linalg.norm(disp, axis=1) > 0
    tc = np.where(curved_edge[eid].any(0))[0]
    if not len(tc):
        _CACHE[fp] = None
        return None
    X = np.transpose(nodes[:, tets[:, tc]], (2, 1, 0))             # (nc, 4, 3)
    g, _ = _grads(X)
    # valid mapping: det D well above 0 everywhere; else shrink the offending edges (globally, so that
    # neighbouring tets keep the same face shape)
    scale = np.ones(ne)
    for _ in range(6):
        d = disp[eid[:, tc].T] * scale[eid[:, tc].T][..., None]      # (nc, 6, 3)
        bad = _detD(g, d, _CHK).min(1) < 0.3
        if not bad.any():
            break
        e_bad = np.unique(eid[:, tc[bad]].ravel())
        e_bad = e_bad[curved_edge[e_bad]]
        scale[e_bad] *= 0.5
    else:
        # still invalid: straighten those tets completely, and repeat for neighbours that lose a
        # compensating edge (ends at the latest when all tets are straight, det D = 1)
        while True:
            d = disp[eid[:, tc].T] * scale[eid[:, tc].T][..., None]
            bad = _detD(g, d, _CHK).min(1) < 0.3
            if not bad.any():
                break
            scale[np.unique(eid[:, tc[bad]].ravel())] = 0.0
    keep = np.abs(d).sum((1, 2)) > 0
    res = (tc[keep], np.ascontiguousarray(d[keep]))
    STATS.update(curved_tets=int(keep.sum()), curved_edges=int((scale * curved_edge > 0).sum()),
                 shrunk_edges=int(((scale < 1) & curved_edge).sum()))
    _CACHE[fp] = res
    return res


# ------------------------------------------------------------------------------------------------
# element matrices
# ------------------------------------------------------------------------------------------------

# module-level kernels so that numba can cache the compiled code next to this file (first run only)
from .ned2_kernels import (_eval_f_3d, _eval_curl_f_3d, parse_dofcode, local_tet_to_edgeid,  # noqa: E402
                           local_tet_to_triid, matinv, tet_coefficients)


@njit(cache=True, nogil=True)
def tet_matrices(tv, d, lem, ltm, Ms, Mm, dofcodes, typ, idx, QP):
    ndof = dofcodes.shape[0]
    txs = tv[0, :]
    tys = tv[1, :]
    tzs = tv[2, :]
    aas, bbs, ccs, dds, V = tet_coefficients(txs, tys, tzs)
    coeff = np.empty((4, 4), dtype=np.float64)
    coeff[0, :] = aas / (6 * V)
    coeff[1, :] = bbs / (6 * V)
    coeff[2, :] = ccs / (6 * V)
    coeff[3, :] = dds / (6 * V)
    # tet_coefficients divides by |V|: for a negatively oriented tet its rows are minus the barycentric
    # gradients, which the basis functions do not mind but the geometry map does
    e1x, e1y, e1z = txs[1] - txs[0], tys[1] - tys[0], tzs[1] - tzs[0]
    e2x, e2y, e2z = txs[2] - txs[0], tys[2] - tys[0], tzs[2] - tzs[0]
    e3x, e3y, e3z = txs[3] - txs[0], tys[3] - tys[0], tzs[3] - tzs[0]
    orient = e1x * (e2y * e3z - e2z * e3y) - e1y * (e2x * e3z - e2z * e3x) + e1z * (e2x * e3y - e2y * e3x)
    sg = 1.0 if orient > 0 else -1.0
    nq = QP.shape[1]
    coords = np.empty((3, nq), dtype=np.float64)
    for q in range(nq):
        for c in range(3):
            coords[c, q] = tv[c, 0] * QP[1, q] + tv[c, 1] * QP[2, q] + tv[c, 2] * QP[3, q] + tv[c, 3] * QP[4, q]
    # point-wise effective materials
    MsE = np.empty((nq, 3, 3), dtype=np.complex128)
    MmE = np.empty((nq, 3, 3), dtype=np.complex128)
    Dm = np.empty((3, 3))
    Di = np.empty((3, 3))
    for q in range(nq):
        for r in range(3):
            for s in range(3):
                Dm[r, s] = 1.0 if r == s else 0.0
        for e in range(6):
            i = 0 if e < 3 else (1 if e < 5 else 2)
            j = (1, 2, 3, 2, 3, 3)[e]
            li = QP[1 + i, q]
            lj = QP[1 + j, q]
            for s in range(3):
                gp = sg * 4.0 * (li * coeff[1 + s, j] + lj * coeff[1 + s, i])
                for r in range(3):
                    Dm[r, s] += d[e, r] * gp
        det = (Dm[0, 0] * (Dm[1, 1] * Dm[2, 2] - Dm[1, 2] * Dm[2, 1])
               - Dm[0, 1] * (Dm[1, 0] * Dm[2, 2] - Dm[1, 2] * Dm[2, 0])
               + Dm[0, 2] * (Dm[1, 0] * Dm[2, 1] - Dm[1, 1] * Dm[2, 0]))
        Di[0, 0] = (Dm[1, 1] * Dm[2, 2] - Dm[1, 2] * Dm[2, 1]) / det
        Di[0, 1] = (Dm[0, 2] * Dm[2, 1] - Dm[0, 1] * Dm[2, 2]) / det
        Di[0, 2] = (Dm[0, 1] * Dm[1, 2] - Dm[0, 2] * Dm[1, 1]) / det
        Di[1, 0] = (Dm[1, 2] * Dm[2, 0] - Dm[1, 0] * Dm[2, 2]) / det
        Di[1, 1] = (Dm[0, 0] * Dm[2, 2] - Dm[0, 2] * Dm[2, 0]) / det
        Di[1, 2] = (Dm[0, 2] * Dm[1, 0] - Dm[0, 0] * Dm[1, 2]) / det
        Di[2, 0] = (Dm[1, 0] * Dm[2, 1] - Dm[1, 1] * Dm[2, 0]) / det
        Di[2, 1] = (Dm[0, 1] * Dm[2, 0] - Dm[0, 0] * Dm[2, 1]) / det
        Di[2, 2] = (Dm[0, 0] * Dm[1, 1] - Dm[0, 1] * Dm[1, 0]) / det
        for r in range(3):
            for s in range(3):
                a1 = 0.0 + 0.0j
                a2 = 0.0 + 0.0j
                for k in range(3):
                    for m in range(3):
                        a1 += Dm[k, r] * Ms[k, m] * Dm[m, s]
                        a2 += Di[r, k] * Mm[k, m] * Di[s, m]
                MsE[q, r, s] = a1 / det
                MmE[q, r, s] = a2 * det
    f = np.empty((ndof, 3, nq), dtype=np.complex128)
    cf = np.empty((ndof, 3, nq), dtype=np.complex128)
    for idof in range(ndof):
        if typ[idof] == 0:
            i1 = lem[0, idx[idof]]
            j1 = lem[1, idx[idof]]
            k1 = 0
        else:
            i1 = ltm[0, idx[idof]]
            j1 = ltm[1, idx[idof]]
            k1 = ltm[2, idx[idof]]
        _eval_f_3d(coeff, coords, i1, j1, k1, dofcodes[idof], f[idof])
        _eval_curl_f_3d(coeff, coords, i1, j1, k1, dofcodes[idof], cf[idof])
    K = np.empty((ndof, ndof), dtype=np.complex128)
    M = np.empty((ndof, ndof), dtype=np.complex128)
    for i in range(ndof):
        for j in range(i, ndof):
            aS = 0.0 + 0.0j
            aM = 0.0 + 0.0j
            for q in range(nq):
                w = QP[0, q]
                for r in range(3):
                    tS = 0.0 + 0.0j
                    tM = 0.0 + 0.0j
                    for s in range(3):
                        tS += MsE[q, r, s] * cf[i, s, q]
                        tM += MmE[q, r, s] * f[i, s, q]
                    aS += w * tS * cf[j, r, q]
                    aM += w * tM * f[j, r, q]
            K[i, j] = aS * V
            M[i, j] = aM * V
            K[j, i] = aS * V
            M[j, i] = aM * V
    return K, M

@njit(cache=True, nogil=True, parallel=True)
def build(nodes, tets, tris, edges, tet_to_edge, tet_to_tri, ur, er, ctets, cpos, disp, dofcodes, QP,
          dataE, dataB):
    ndof = dofcodes.shape[0]
    typ, idx = parse_dofcode(dofcodes)
    for c in prange(ctets.shape[0]):
        itet = ctets[c]
        p = cpos[c] * ndof * ndof
        ltm = local_tet_to_triid(tet_to_tri, tets, tris, itet)
        lem = local_tet_to_edgeid(tet_to_edge, tets, edges, itet)
        K, M = tet_matrices(nodes[:, tets[:, itet]], disp[c], lem, ltm, matinv(ur[:, :, itet]), er[:, :, itet],
                            dofcodes, typ, idx, QP)
        k = 0
        for ii in range(ndof):
            for jj in range(ndof):
                dataE[p + k] = K[ii, jj]
                dataB[p + k] = M[ii, jj]
                k += 1


def kernels():
    return tet_matrices, build


def apply_curved(field, er, ur, conductor_tets, dataE, dataB) -> int:
    """Overwrite the element matrices of the curved tets in EMerge's assembled data; returns their number."""
    mesh = field.mesh
    cd = curved_data(mesh, CTX.get("exclude", frozenset()))
    if cd is None:
        return 0
    ctets, disp = cd
    nt = mesh.tets.shape[1]
    pos = np.full(nt, -1, np.int64)
    assy = np.arange(nt)
    if len(conductor_tets):
        assy = np.delete(assy, np.asarray(conductor_tets, np.int64))
    pos[assy] = np.arange(len(assy))
    sel = pos[ctets] >= 0
    if not sel.any():
        return 0
    _, build = kernels()
    # edge displacements follow EMerge's vertex order of each tet (the same order the pairs refer to)
    build(np.asarray(mesh.nodes, np.float64), np.asarray(mesh.tets, np.int64), np.asarray(mesh.tris, np.int64),
          np.asarray(mesh.edges, np.int64), np.asarray(mesh.tet_to_edge, np.int64),
          np.asarray(mesh.tet_to_tri, np.int64), ur, er, ctets[sel], pos[ctets[sel]], disp[sel],
          field.dofcodes3d, _QP, dataE, dataB)
    return int(sel.sum())


def curved_volume(mesh, exclude=frozenset()) -> tuple[float, float]:
    """(straight, curved) volume of all tets: a check of the geometry mapping."""
    X = np.transpose(np.asarray(mesh.nodes, float)[:, np.asarray(mesh.tets)], (2, 1, 0))
    g, V = _grads(X)
    V = np.abs(V)
    tot = V.sum()
    cd = curved_data(mesh, exclude)
    if cd is None:
        return tot, tot
    tc, d = cd
    dets = _detD(g[tc], d, _QP[1:].T)
    corr = (V[tc] * (dets @ _QP[0] - 1.0)).sum()
    return tot, tot + corr
