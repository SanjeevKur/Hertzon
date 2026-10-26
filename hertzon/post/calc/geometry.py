"""Sample points of the Fields Calculator: Gauss points with weights on tetrahedra, triangles and line
segments, each with the tetrahedron it lies in and its barycentric coordinates there.

The fields are stored per tetrahedron (4 corners or 10 nodes, discontinuous across materials), so every
sample is evaluated with the shape functions of its own element: the interpolation is exact for the stored
field and an integral sums element by element, without averaging across material boundaries."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

TET10_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]   # VTK quadratic tetra mid-edge order


def node_lambdas(npt: int) -> np.ndarray:
    """Barycentric coordinates (npt, 4) of the field nodes of a tetrahedron (4 corners, + 6 mid-edges)."""
    lam = np.eye(4)
    if npt == 10:
        lam = np.vstack([lam] + [(lam[i] + lam[j]) / 2 for i, j in TET10_EDGES])
    return lam


def shape(lam: np.ndarray, npt: int) -> np.ndarray:
    """Shape functions (n, npt) at barycentric coordinates lam (n, 4): linear or quadratic Lagrange."""
    if npt == 4:
        return lam.copy()
    out = np.empty((len(lam), 10))
    out[:, :4] = lam * (2 * lam - 1)
    for k, (i, j) in enumerate(TET10_EDGES):
        out[:, 4 + k] = 4 * lam[:, i] * lam[:, j]
    return out


def dshape(lam: np.ndarray, npt: int) -> np.ndarray:
    """Derivatives (n, npt, 4) of the shape functions by the four barycentric coordinates."""
    n = len(lam)
    if npt == 4:
        return np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    out = np.zeros((n, 10, 4))
    for i in range(4):
        out[:, i, i] = 4 * lam[:, i] - 1
    for k, (i, j) in enumerate(TET10_EDGES):
        out[:, 4 + k, i] = 4 * lam[:, j]
        out[:, 4 + k, j] = 4 * lam[:, i]
    return out


def tet_geometry(nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gradients of the barycentric coordinates (M, 4, 3), inverse Jacobians (M, 3, 3), volumes (M,)) of
    the straight tetrahedra."""
    X = nodes[tets]                                       # (M, 4, 3)
    J = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=2)   # columns
    det = np.linalg.det(J)
    ok = np.abs(det) > 1e-300
    Ji = np.zeros_like(J)
    Ji[ok] = np.linalg.inv(J[ok])
    G = np.empty((len(tets), 4, 3))
    G[:, 1:] = Ji                                         # rows: grad lambda_1..3
    G[:, 0] = -Ji.sum(axis=1)
    return G, Ji, np.abs(det) / 6


def barycentric(nodes, tets, Ji, tet_ids, P) -> np.ndarray:
    """Barycentric coordinates (n, 4) of points P (n, 3) in the tetrahedra tet_ids."""
    x0 = nodes[tets[tet_ids, 0]]
    l3 = np.einsum("nij,nj->ni", Ji[tet_ids], P - x0)
    return np.column_stack([1 - l3.sum(axis=1), l3])


# ------------------------------------------------------------------------------------------------
# quadrature rules (EMerge's Keast and Strang-Fix rules)
# ------------------------------------------------------------------------------------------------
_RULES: dict = {}


def tet_rule() -> tuple[np.ndarray, np.ndarray]:
    """(barycentric coordinates (q, 4), weights (q,) summing to 1): exact to degree 5, positive weights."""
    if "tet" not in _RULES:
        from emerge._emerge.mth.optimized import gaus_quad_tet
        r = np.asarray(gaus_quad_tet(6), float)
        _RULES["tet"] = (np.ascontiguousarray(r[1:5].T), r[0].copy())
    return _RULES["tet"]


def tri_rule() -> tuple[np.ndarray, np.ndarray]:
    """(barycentric coordinates (q, 3), weights (q,) summing to 1): exact to degree 5."""
    if "tri" not in _RULES:
        from emerge._emerge.mth.optimized import gaus_quad_tri
        r = np.asarray(gaus_quad_tri(7), float)
        _RULES["tri"] = (np.ascontiguousarray(r[1:4].T), r[0].copy())
    return _RULES["tri"]


_GL3 = (np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5 / 9, 8 / 9, 5 / 9]))


# ------------------------------------------------------------------------------------------------
# sample sets
# ------------------------------------------------------------------------------------------------
@dataclass
class Samples:
    """Points of a geometry with quadrature weights (m^dim; 1 for points) and their elements.

    tets / lam: the element and barycentric coordinates of every point (tets -1: outside the solution).
    tets2 / lam2: on a face between two elements (a sheet inside the solution) the element on the other
    side, else -1; the field there is the mean of both sides and Js the jump n x (H - H2).
    N: unit normals (faces), T: unit tangents (lines)."""
    P: np.ndarray
    W: np.ndarray
    tets: np.ndarray
    lam: np.ndarray
    dim: int
    N: np.ndarray | None = None
    T: np.ndarray | None = None
    tets2: np.ndarray | None = None
    lam2: np.ndarray | None = None
    label: str = ""

    def __len__(self) -> int:
        return len(self.P)

    def take(self, sl) -> "Samples":
        g = lambda a: None if a is None else a[sl]  # noqa: E731
        return Samples(self.P[sl], self.W[sl], self.tets[sl], self.lam[sl], self.dim, g(self.N), g(self.T),
                       g(self.tets2), g(self.lam2), self.label)

    def flipped(self) -> "Samples":
        """The reversed geometry: normals and tangents turned around, the sides of a face swapped."""
        out = replace(self, N=None if self.N is None else -self.N, T=None if self.T is None else -self.T)
        if self.tets2 is not None:
            two = self.tets2 >= 0
            t1, t2, l1, l2 = self.tets.copy(), self.tets2.copy(), self.lam.copy(), self.lam2.copy()
            t1[two], t2[two], l1[two], l2[two] = self.tets2[two], self.tets[two], self.lam2[two], self.lam[two]
            out = replace(out, tets=t1, tets2=t2, lam=l1, lam2=l2)
        return out

    def chunks(self, size: int = 200_000):
        for i in range(0, max(len(self), 1), size):
            yield self.take(slice(i, i + size))

    @staticmethod
    def concat(parts: list["Samples"]) -> "Samples":
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("empty")
        if len(parts) == 1:
            return parts[0]

        def cat(name):
            arrs = [getattr(p, name) for p in parts]
            if all(a is None for a in arrs):
                return None
            if name in ("tets2",):
                arrs = [a if a is not None else np.full(len(p), -1) for a, p in zip(arrs, parts)]
            elif name == "lam2":
                arrs = [a if a is not None else np.zeros((len(p), 4)) for a, p in zip(arrs, parts)]
            elif any(a is None for a in arrs):
                return None
            return np.concatenate(arrs)
        return Samples(cat("P"), cat("W"), cat("tets"), cat("lam"), parts[0].dim, cat("N"), cat("T"), cat("tets2"),
                       cat("lam2"), parts[0].label)


class Mesh:
    """The solution mesh with the element geometry the samples need (straight tetrahedra)."""

    def __init__(self, nodes: np.ndarray, tets: np.ndarray, curved=None):
        """curved: (row of every element in disp or -1, mid-edge displacements (nc, 6, 3) in the
        TET10_EDGES order) of the curved elements at curved surfaces, or None."""
        self.nodes = np.asarray(nodes, float)
        self.tets = np.asarray(tets, np.int64)
        self.G, self.Ji, self.vol = tet_geometry(self.nodes, self.tets)
        self._tree = None
        self._h = None
        self.cpos = self.cX = None
        if curved is not None and len(curved[1]):
            pos, disp = np.asarray(curved[0], np.int64), np.asarray(curved[1], float)
            ct = np.where(pos >= 0)[0]
            X = np.empty((len(disp), 10, 3))
            C = self.nodes[self.tets[ct]]
            X[pos[ct], :4] = C
            for k, (i, j) in enumerate(TET10_EDGES):
                X[pos[ct], 4 + k] = (C[:, i] + C[:, j]) / 2 + disp[pos[ct], k]
            self.cpos, self.cX = pos, X

    # ---- curved elements: x(lambda) = sum N_k(lambda) X_k with the 10 geometry nodes -------------
    def _curved(self, t: np.ndarray) -> np.ndarray:
        return np.zeros(len(t), bool) if self.cpos is None else self.cpos[t] >= 0

    def _map(self, t: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Position (n, 3) and dx/dlambda (n, 3, 4) in curved elements t."""
        X = self.cX[self.cpos[t]]
        P = np.einsum("nk,nkd->nd", shape(lam, 10), X)
        D = np.einsum("nki,nkd->ndi", dshape(lam, 10), X)
        return P, D

    @staticmethod
    def _jac(D: np.ndarray) -> np.ndarray:
        return D[:, :, 1:] - D[:, :, :1]                          # dx / d(lambda_1..3)

    def grad_lambda(self, t: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """Gradients of the barycentric coordinates (n, 4, 3) at the samples (curved elements included)."""
        G = self.G[t].copy()
        c = self._curved(t)
        if c.any():
            _, D = self._map(t[c], lam[c])
            Ji = np.linalg.inv(self._jac(D))
            G[c, 1:] = Ji
            G[c, 0] = -Ji.sum(axis=1)
        return G

    def _refine(self, t: np.ndarray, P: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """Barycentric coordinates of points P in curved elements t (Newton from the straight guess)."""
        lam = lam.copy()
        c = self._curved(t)
        if not c.any():
            return lam
        tc, Pc, L = t[c], P[c], lam[c]
        for _ in range(6):
            x, D = self._map(tc, L)
            dxi = np.linalg.solve(self._jac(D), (Pc - x)[:, :, None])[:, :, 0]
            L = L + np.column_stack([-dxi.sum(axis=1), dxi])
            if np.abs(dxi).max(initial=0) < 1e-12:
                break
        lam[c] = L
        return lam

    @property
    def h(self) -> float:
        """Median edge length."""
        if self._h is None:
            X = self.nodes[self.tets]
            self._h = float(np.median(np.linalg.norm(X[:, 1] - X[:, 0], axis=1))) if len(X) else 1.0
        return self._h

    # ---- volumes --------------------------------------------------------------------------------
    def volume_samples(self, tet_ids, label: str = "") -> Samples:
        lam_q, w_q = tet_rule()
        t = np.asarray(tet_ids, np.int64)
        q = len(w_q)
        tets = np.repeat(t, q)
        lam = np.tile(lam_q, (len(t), 1))
        W = (self.vol[t][:, None] * w_q[None, :]).ravel()
        P = np.einsum("ni,nid->nd", lam, self.nodes[self.tets[tets]])
        c = self._curved(tets)
        if c.any():
            P[c], D = self._map(tets[c], lam[c])
            W[c] = np.tile(w_q, len(t))[c] * np.abs(np.linalg.det(self._jac(D))) / 6
        return Samples(P, W, tets, lam, 3, label=label)

    # ---- faces ----------------------------------------------------------------------------------
    def triangle_samples(self, tri: np.ndarray, owner: np.ndarray, normals: np.ndarray | None = None,
                         owner2: np.ndarray | None = None, label: str = "", faces: bool = False) -> Samples:
        """Gauss points on triangles tri (k, 3, 3) lying in (or on a face of) the elements owner; normals
        (k, 3) give the orientation (the triangle normal is turned to agree); owner2: the element on the
        other side (-1: none). faces: the triangles are element faces (of owner), which follow the curved
        surface where the element is curved; other triangles (cuts) are flat."""
        tri = np.asarray(tri, float)
        owner = np.asarray(owner, np.int64)
        lam_q, w_q = tri_rule()
        q = len(w_q)
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(cr, axis=1) / 2
        n = cr / np.maximum(2 * area, 1e-300)[:, None]
        if normals is not None:
            n = np.where((np.einsum("kd,kd->k", n, normals) < 0)[:, None], -n, n)
        P = np.einsum("qi,kid->kqd", lam_q, tri).reshape(-1, 3)
        W = (area[:, None] * w_q[None, :]).ravel()
        tets = np.repeat(owner, q)
        lam = barycentric(self.nodes, self.tets, self.Ji, tets, P)
        N = np.repeat(n, q, axis=0)
        c = self._curved(tets)
        if c.any() and faces:
            # on the curved face: the same lambda, the position and area element of the curved map
            corner = np.rint(barycentric(self.nodes, self.tets, self.Ji, np.repeat(owner, 3),
                                         tri.reshape(-1, 3)).reshape(-1, 3, 4))
            E1 = np.repeat(corner[:, 1] - corner[:, 0], q, axis=0)[c]
            E2 = np.repeat(corner[:, 2] - corner[:, 0], q, axis=0)[c]
            Pc, D = self._map(tets[c], lam[c])
            nc = np.cross(np.einsum("ndi,ni->nd", D, E1), np.einsum("ndi,ni->nd", D, E2))
            a = np.linalg.norm(nc, axis=1)
            P[c] = Pc
            W[c] = np.tile(w_q, len(tri))[c] * a / 2
            nc /= np.maximum(a, 1e-300)[:, None]
            N[c] = np.where((np.einsum("nd,nd->n", nc, N[c]) < 0)[:, None], -nc, nc)
        elif c.any():
            lam = self._refine(tets, P, lam)
        t2 = l2 = None
        if owner2 is not None:
            t2 = np.repeat(np.asarray(owner2, np.int64), q)
            l2 = np.zeros_like(lam)
            two = t2 >= 0
            if two.any():
                l2[two] = barycentric(self.nodes, self.tets, self.Ji, t2[two], P[two])
                if not faces:
                    l2[two] = self._refine(t2[two], P[two], l2[two])
        return Samples(P, W, tets, np.clip(lam, 0, 1), 2, N=N, tets2=t2, lam2=None if l2 is None else np.clip(l2, 0, 1),
                       label=label)

    def face_triangles(self, owner: np.ndarray, local: np.ndarray) -> np.ndarray:
        """Corner coordinates (k, 3, 3) of element faces given by owner and local corner indices (k, 3)."""
        g = np.take_along_axis(self.tets[owner], local, axis=1)
        return self.nodes[g]

    def boundary_faces(self, tet_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Faces on the boundary of a set of elements: (owner, local corners (k, 3), outward normals)."""
        t = np.asarray(tet_ids, np.int64)
        F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        g = np.sort(self.tets[t][:, F].reshape(-1, 3), axis=1)
        key = (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        one = cnt[inv.ravel()] == 1
        owner = np.repeat(t, 4)[one]
        local = np.tile(F, (len(t), 1))[one]
        tri = self.face_triangles(owner, local)
        c = self.nodes[self.tets[owner]].mean(axis=1)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n = np.where((np.einsum("kd,kd->k", n, tri.mean(axis=1) - c) < 0)[:, None], -n, n)
        return owner, local, n / np.linalg.norm(n, axis=1, keepdims=True)

    # ---- cut planes ------------------------------------------------------------------------------
    def plane_triangles(self, normal, d: float, tet_ids=None):
        """The plane normal . x = d cut out of the elements: triangles (k, 3, 3), their elements and the
        element on the other side (-1: none; for element faces lying in the plane)."""
        nrm = np.asarray(normal, float) / np.linalg.norm(normal)
        t = np.arange(len(self.tets)) if tet_ids is None else np.asarray(tet_ids, np.int64)
        X = self.nodes[self.tets[t]]                               # (m, 4, 3)
        s = X @ nrm - d
        eps = 1e-6 * max(self.h, 1e-12)
        on, pos, neg = np.abs(s) <= eps, s > eps, s < -eps
        tris, owners, others = [], [], []
        # element faces in the plane: one or two elements share them
        face3 = on.sum(axis=1) == 3
        if face3.any():
            ft = t[face3]
            local = np.argsort(~on[face3], axis=1, kind="stable")[:, :3]
            g = np.sort(np.take_along_axis(self.tets[ft], local, axis=1), axis=1)
            key = (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]
            order = np.argsort(key, kind="stable")
            key, ft, local = key[order], ft[order], local[order]
            first = np.r_[True, key[1:] != key[:-1]]
            idx = np.where(first)[0]
            nxt = np.minimum(idx + 1, len(key) - 1)
            two = (idx + 1 < len(key)) & (key[nxt] == key[idx])
            o1, o2 = ft[idx], np.where(two, ft[nxt], -1)
            # the element on the side the normal points to comes first
            c1 = self.nodes[self.tets[o1]].mean(axis=1) @ nrm - d
            swap = two & (c1 < 0)
            o1, o2 = np.where(swap, o2, o1), np.where(swap, o1, o2)
            tris.append(self.face_triangles(ft[idx], local[idx]))
            owners.append(o1)
            others.append(o2)
        cut = pos.any(axis=1) & neg.any(axis=1)
        if cut.any():
            tc, Xc, sc, onc, posc, negc = t[cut], X[cut], s[cut], on[cut], pos[cut], neg[cut]
            pts = np.full((len(tc), 4, 3), np.nan)
            cnt = np.zeros(len(tc), np.int64)
            for i in range(4):
                x = onc[:, i]
                pts[x, cnt[x]] = Xc[x, i]
                cnt[x] += 1
            for i, j in TET10_EDGES:
                x = (posc[:, i] & negc[:, j]) | (negc[:, i] & posc[:, j])
                if not x.any():
                    continue
                a = (sc[x, i] / (sc[x, i] - sc[x, j]))[:, None]
                pts[x, cnt[x]] = Xc[x, i] + a * (Xc[x, j] - Xc[x, i])
                cnt[x] += 1
            # order the 3 or 4 points around their centre in the plane
            u = np.cross(nrm, [1.0, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1.0, 0])
            u /= np.linalg.norm(u)
            v = np.cross(nrm, u)
            c = np.nanmean(pts, axis=1)
            rel = pts - c[:, None]
            ang = np.arctan2(np.nan_to_num(rel @ v), np.nan_to_num(rel @ u))
            ang[np.isnan(pts[:, :, 0])] = np.inf
            pts = np.take_along_axis(pts, np.argsort(ang, axis=1)[:, :, None], axis=1)
            ok3 = cnt >= 3
            tris.append(pts[ok3][:, [0, 1, 2]])
            owners.append(tc[ok3])
            others.append(np.full(int(ok3.sum()), -1))
            quad = cnt == 4
            if quad.any():
                tris.append(pts[quad][:, [0, 2, 3]])
                owners.append(tc[quad])
                others.append(np.full(int(quad.sum()), -1))
        if not tris:
            return np.zeros((0, 3, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(tris), np.concatenate(owners), np.concatenate(others)

    def plane_samples(self, normal, d: float, label: str = "") -> Samples:
        tri, owner, other = self.plane_triangles(normal, d)
        if not len(tri):
            raise ValueError(f"{label or 'The plane'} does not cut the solution volume")
        return self.triangle_samples(tri, owner, np.broadcast_to(np.asarray(normal, float), (len(tri), 3)),
                                     owner2=other if (other >= 0).any() else None, label=label)

    # ---- points and lines -------------------------------------------------------------------------
    def locate(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Element (-1: outside) and barycentric coordinates of points P (n, 3)."""
        from scipy.spatial import cKDTree
        P = np.atleast_2d(np.asarray(P, float))
        if self._tree is None:
            self._tree = cKDTree(self.nodes[self.tets].mean(axis=1))
        tets = np.full(len(P), -1, np.int64)
        lam = np.zeros((len(P), 4))
        todo = np.arange(len(P))
        for k in (8, 32, 128):
            if not len(todo):
                break
            k = min(k, len(self.tets))
            _, cand = self._tree.query(P[todo], k=k)
            cand = np.asarray(cand).reshape(len(todo), -1)
            found = np.zeros(len(todo), bool)
            for c in range(cand.shape[1]):
                rest = ~found
                if not rest.any():
                    break
                ids = cand[rest, c]
                L = barycentric(self.nodes, self.tets, self.Ji, ids, P[todo[rest]])
                inside = L.min(axis=1) > -1e-9
                sel = np.where(rest)[0][inside]
                tets[todo[sel]] = ids[inside]
                lam[todo[sel]] = np.clip(L[inside], 0, 1)
                found[sel] = True
            todo = todo[~found]
        ok = tets >= 0
        if ok.any():
            lam[ok] = np.clip(self._refine(tets[ok], P[ok], lam[ok]), 0, 1)
        return tets, lam

    def point_samples(self, xyz, label: str = "") -> Samples:
        P = np.asarray(xyz, float).reshape(-1, 3)
        tets, lam = self.locate(P)
        return Samples(P, np.ones(len(P)), tets, lam, 0, label=label)

    def line_samples(self, polylines: list[np.ndarray], label: str = "") -> Samples:
        """Gauss points along polylines (each an (n, 3) array in its direction), about three per element."""
        xi, w = _GL3
        P, W, T = [], [], []
        h = self.h / 3
        for Q in polylines:
            Q = np.asarray(Q, float)
            for a, b in zip(Q[:-1], Q[1:]):
                L = float(np.linalg.norm(b - a))
                if L <= 0:
                    continue
                k = max(1, int(np.ceil(L / h)))
                s = (np.arange(k)[:, None] + (xi[None, :] + 1) / 2).ravel() / k
                P.append(a + s[:, None] * (b - a))
                W.append(np.tile(w / 2 * L / k, k))
                T.append(np.broadcast_to((b - a) / L, (len(s), 3)))
        if not P:
            raise ValueError(f"{label or 'The line'} has no length")
        P, W, T = np.vstack(P), np.concatenate(W), np.vstack(T)
        tets, lam = self.locate(P)
        return Samples(P, W, tets, lam, 1, T=T, label=label)

    def node_samples(self, tet_ids: np.ndarray, npt: int) -> Samples:
        """The field nodes of the elements tet_ids (npt per element), as samples (for derivatives)."""
        t = np.asarray(tet_ids, np.int64)
        lam1 = node_lambdas(npt)
        tets = np.repeat(t, npt)
        lam = np.tile(lam1, (len(t), 1))
        P = np.einsum("ni,nid->nd", lam, self.nodes[self.tets[tets]])
        c = self._curved(tets)
        if c.any():
            P[c] = self._map(tets[c], lam[c])[0]
        return Samples(P, np.ones(len(P)), tets, lam, 3)
