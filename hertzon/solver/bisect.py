"""Local, nested mesh refinement by conforming longest-edge bisection (local adaptive passes).

The current gmsh mesh is read (tets with their volume, boundary triangles with their surface,
curve segments with their curve). Marked tets are bisected at their longest edge; the closure
bisects neighbours until the mesh is conforming again. Old elements stay, so consecutive
adaptive passes are nested and the change of S only reflects the refinement, not remeshing noise.
New edge midpoints on CAD curves/surfaces are projected onto the true geometry. The result is
written back into the gmsh model, so EMerge reads it like a freshly generated mesh.
"""
from __future__ import annotations

import os

import numpy as np

# faces of a tet opposite each local vertex
SHORT_EDGE = 0.85       # pending edges shorter than this x the longest edge: split the longest first
_OPP = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


class TetMesh:
    def __init__(self, X: np.ndarray, tets: np.ndarray, vtag: np.ndarray, ftag: np.ndarray,
                 segs: np.ndarray, stag: np.ndarray, ntag: dict | None = None):
        self.X = [tuple(map(float, x)) for x in X]          # node coordinates
        self.tets = [list(map(int, t)) for t in tets]       # (4,) node ids
        self.vtag = [int(v) for v in vtag]                  # volume entity per tet
        self.ftag = [list(map(int, f)) for f in ftag]       # surface entity of face opposite vertex i (0 = interior)
        self.segs = [tuple(map(int, s)) for s in segs]      # curve segments
        self.stag = [int(s) for s in stag]
        self.mid: dict[tuple[int, int], int] = {}
        # entity (dim, tag) a node lies on (for projection of midpoints); missing = volume interior
        self.ntag = ntag or {}
        self.lin: dict[int, tuple] = {}     # straight midpoint of projected nodes (to undo inversions)

    # -- geometry ---------------------------------------------------------------------------------
    def _len2(self, a: int, b: int) -> float:
        p, q = self.X[a], self.X[b]
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2

    def longest(self, t: list[int]) -> tuple[int, int]:
        best, key = None, None
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = (t[i], t[j]) if t[i] < t[j] else (t[j], t[i])
                k = (self._len2(a, b), -a, -b)        # deterministic tie-break
                if key is None or k > key:
                    key, best = k, (a, b)
        return best

    def midpoint(self, e: tuple[int, int], project=None) -> int:
        m = self.mid.get(e)
        if m is None:
            p, q = self.X[e[0]], self.X[e[1]]
            x = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2)
            ent = self._edge_entity(e)
            m = len(self.X)
            if ent is not None and project is not None:
                self.lin[m] = x
                x = project(ent, x)
            self.X.append(tuple(map(float, x)))
            if ent is not None:
                self.ntag[m] = ent
            self.mid[e] = m
        return m

    def _edge_entity(self, e):
        return self._edge_ent.get(e)

    # -- refinement --------------------------------------------------------------------------------
    def _edge_map(self):
        em: dict[tuple[int, int], list[int]] = {}
        for k, t in enumerate(self.tets):
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = t[i], t[j]
                    em.setdefault((a, b) if a < b else (b, a), []).append(k)
        return em

    def _classify_edges(self):
        """CAD entity of every mesh edge on a curve or a surface (for midpoint projection)."""
        ent = {}
        for k, t in enumerate(self.tets):
            for i in range(4):
                s = self.ftag[k][i]
                if s:
                    f = [t[j] for j in _OPP[i]]
                    for a, b in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2])):
                        ent[(a, b) if a < b else (b, a)] = (2, s)
        for (a, b), c in zip(self.segs, self.stag):
            ent[(a, b) if a < b else (b, a)] = (1, c)
        self._edge_ent = ent

    def refine(self, marked, project=None) -> int:
        """Bisect the marked tets (indices) and close the mesh; returns the number of new tets."""
        n0 = len(self.tets)
        self._x0 = len(self.X)
        self._classify_edges()
        pending = {self.longest(self.tets[k]) for k in marked}
        em = self._edge_map()
        # closure on the current mesh (fixed before any tet is cut, so every face is split in the
        # same order from both sides: longest marked edge first): a tet whose marked edges are all
        # much shorter than its longest edge would become a sliver, so its longest edge is marked too
        stack = list(pending)
        while stack:
            e = stack.pop()
            for k in em.get(e, ()):
                t = self.tets[k]
                LL = self.longest(t)
                if LL in pending:
                    continue
                lp = max(self._len2(t[i], t[j]) for i in range(4) for j in range(i + 1, 4)
                         if ((t[i], t[j]) if t[i] < t[j] else (t[j], t[i])) in pending)
                if lp < SHORT_EDGE ** 2 * self._len2(*LL):
                    pending.add(LL)
                    stack.append(LL)
        # every tet touching marked edges is cut recursively at its longest marked edge
        new_t, new_v, new_f = [], [], []
        work = [(t, self.vtag[k], self.ftag[k]) for k, t in enumerate(self.tets)]
        while work:
            t, v, ft = work.pop()
            L, Lk = None, None
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = (t[i], t[j]) if t[i] < t[j] else (t[j], t[i])
                    if (a, b) in pending:
                        k2 = (self._len2(a, b), -a, -b)        # deterministic tie-break
                        if L is None or k2 > Lk:
                            L, Lk = (a, b), k2
            if L is None:
                new_t.append(t); new_v.append(v); new_f.append(ft)
                continue
            m = self.midpoint(L, project)
            ia, ib = t.index(L[0]), t.index(L[1])
            c1, c2 = list(t), list(t)
            c1[ib] = m
            c2[ia] = m
            f1, f2 = list(ft), list(ft)
            f1[ia] = 0          # the new interior face (m and the two other vertices)
            f2[ib] = 0
            work += [(c1, v, f1), (c2, v, f2)]
        self.tets, self.vtag, self.ftag = new_t, new_v, new_f
        self._fix_inverted()
        # split curve segments at the midpoints that exist
        segs, stag = [], []
        work = list(zip(self.segs, self.stag))
        while work:
            (a, b), c = work.pop()
            m = self.mid.get((a, b) if a < b else (b, a))
            if m is None:
                segs.append((a, b)); stag.append(c)
            else:
                work += [((a, m), c), ((m, b), c)]
        self.segs, self.stag = segs, stag
        return len(self.tets) - n0

    def _fix_inverted(self) -> None:
        """Projection onto curved CAD surfaces can flip thin tets: pull those nodes back toward
        the straight midpoint (halving the move, finally fully straight)."""
        for it in range(6):
            v = self.volumes()
            bad = np.nonzero(v <= 0)[0]
            if not len(bad):
                return
            nodes = {a for k in bad for a in self.tets[k] if a in self.lin and a >= self._x0}
            if not nodes:
                return
            for a in nodes:
                p, q = self.X[a], self.lin[a]
                w = 0.0 if it == 5 else 0.5
                self.X[a] = tuple(w * p[i] + (1 - w) * q[i] for i in range(3))

    # -- derived data -----------------------------------------------------------------------------
    def boundary_tris(self):
        """Tagged triangles (each once): list of (surface, (a, b, c))."""
        seen, out = set(), []
        for k, t in enumerate(self.tets):
            for i in range(4):
                s = self.ftag[k][i]
                if s:
                    f = tuple(t[j] for j in _OPP[i])
                    key = (s,) + tuple(sorted(f))
                    if key not in seen:
                        seen.add(key)
                        out.append((s, f))
        return out

    def volumes(self) -> np.ndarray:
        X = np.asarray(self.X)
        T = np.asarray(self.tets)
        a, b, c, d = (X[T[:, i]] for i in range(4))
        return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6

    def quality(self) -> np.ndarray:
        """Normalised mean-ratio-like quality: 1 = regular tet."""
        X = np.asarray(self.X)
        T = np.asarray(self.tets)
        V = np.abs(self.volumes())
        L2 = sum(((X[T[:, i]] - X[T[:, j]]) ** 2).sum(1) for i in range(4) for j in range(i + 1, 4))
        return 12 * (3 * V) ** (2 / 3) / L2


# -- gmsh interface -----------------------------------------------------------------------------------
def from_gmsh() -> TetMesh:
    import gmsh
    tags, coords, _ = gmsh.model.mesh.getNodes()
    idx = {int(t): i for i, t in enumerate(tags)}
    gtags = np.asarray(tags, np.int64)
    X = coords.reshape(-1, 3)
    ntag = {}
    for dim in (0, 1, 2):
        for _, e in gmsh.model.getEntities(dim):
            nt, _, _ = gmsh.model.mesh.getNodes(dim, e)
            for t in nt:
                ntag.setdefault(idx[int(t)], (dim, e))
    tets, vtag = [], []
    for _, v in gmsh.model.getEntities(3):
        _, _, nodes = gmsh.model.mesh.getElements(3, v)
        if len(nodes):
            T = np.array([idx[int(t)] for t in nodes[0]]).reshape(-1, 4)
            tets.append(T)
            vtag += [v] * len(T)
    tets = np.vstack(tets)
    tri_tag = {}
    for _, s in gmsh.model.getEntities(2):
        _, _, nodes = gmsh.model.mesh.getElements(2, s)
        if len(nodes):
            for tri in np.array([idx[int(t)] for t in nodes[0]]).reshape(-1, 3):
                tri_tag[tuple(sorted(tri))] = s
    ftag = np.zeros((len(tets), 4), int)
    for k, t in enumerate(tets):
        for i in range(4):
            ftag[k, i] = tri_tag.get(tuple(sorted(t[list(_OPP[i])])), 0)
    segs, stag = [], []
    for _, c in gmsh.model.getEntities(1):
        _, _, nodes = gmsh.model.mesh.getElements(1, c)
        if len(nodes):
            for s in np.array([idx[int(t)] for t in nodes[0]]).reshape(-1, 2):
                segs.append(s); stag.append(c)
    M = TetMesh(X, tets, vtag, ftag, np.array(segs, int).reshape(-1, 2), stag, ntag)
    M.gtags = gtags
    return M


def gmsh_projector():
    """Projects a midpoint onto its CAD curve/surface (dim, tag)."""
    import gmsh

    def project(ent, x):
        try:
            p = gmsh.model.getClosestPoint(ent[0], ent[1], list(x))[0]
            return tuple(p)
        except Exception:  # noqa: BLE001
            return x
    return project


def to_gmsh(M: TetMesh) -> None:
    """Replace the gmsh mesh by M (nodes on their entities, elements per entity)."""
    import gmsh
    gmsh.model.mesh.clear()
    n = len(M.X)
    X = np.asarray(M.X)
    # node -> entity: geometry vertices/curves/surfaces as classified, the rest in its volume
    ent = dict(M.ntag)
    for t, v in zip(M.tets, M.vtag):
        for a in t:
            ent.setdefault(a, (3, v))
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        groups.setdefault(ent.get(i, (3, M.vtag[0])), []).append(i)
    for (d, e), ids in groups.items():
        gmsh.model.mesh.addNodes(d, e, [i + 1 for i in ids], X[ids].ravel().tolist())
    by_curve: dict[int, list[int]] = {}
    for (a, b), c in zip(M.segs, M.stag):
        by_curve.setdefault(c, []).extend([a + 1, b + 1])
    for c, nodes in by_curve.items():
        gmsh.model.mesh.addElementsByType(c, 1, [], nodes)
    by_surf: dict[int, list[int]] = {}
    for s, f in M.boundary_tris():
        by_surf.setdefault(s, []).extend([f[0] + 1, f[1] + 1, f[2] + 1])
    for s, nodes in by_surf.items():
        gmsh.model.mesh.addElementsByType(s, 2, [], nodes)
    by_vol: dict[int, list[int]] = {}
    for t, v in zip(M.tets, M.vtag):
        by_vol.setdefault(v, []).extend([a + 1 for a in t])
    for v, nodes in by_vol.items():
        gmsh.model.mesh.addElementsByType(v, 4, [], nodes)
    # geometry points (dim 0) keep a point element
    for _, p in gmsh.model.getEntities(0):
        ids = [i for i, e in M.ntag.items() if e == (0, p)]
        if ids:
            gmsh.model.mesh.addElementsByType(p, 15, [], [ids[0] + 1])


def copy(M: TetMesh) -> TetMesh:
    import copy as _c
    return _c.deepcopy(M)


def _load_and_repair(M: TetMesh) -> None:
    """Write M into gmsh and repair its bad tets locally (edge/face swaps and relocation of the
    interior nodes of tets below gmsh's quality threshold; boundary nodes and all good tets stay)."""
    import gmsh
    to_gmsh(M)
    try:
        gmsh.model.mesh.optimize("")
    except Exception:  # noqa: BLE001
        pass


def load_into_emerge(m, M: TetMesh) -> None:
    """Install M into EMerge model m through its normal mesh import (generate() is replaced)."""
    import gmsh
    orig = gmsh.model.mesh.generate
    m._reset_mesh()
    gmsh.model.mesh.generate = lambda dim=3: _load_and_repair(M)
    try:
        m.generate_mesh(True)
    finally:
        gmsh.model.mesh.generate = orig


def refine_emerge(m, error, pct: float, frac: float, log=None):
    """Bisect the worst tets of EMerge model m so the mesh grows by about pct percent.
    Returns (growth %, fraction of tets marked) – the fraction is a good start for the next pass."""
    M = from_gmsh()
    # EMerge tet i -> index in M (both from gmsh node tags)
    key = {tuple(sorted(M.gtags[t])): k for k, t in enumerate(M.tets)}
    i2t = m.mesh.n_i2t
    T = np.asarray(m.mesh.tets)
    emap = np.array([key[tuple(sorted(i2t[int(a)] for a in T[:, i]))] for i in range(T.shape[1])])
    order = emap[np.argsort(-np.asarray(error))]
    n0 = len(M.tets)
    proj = None if os.environ.get("HZN_NOPROJ") else gmsh_projector()
    best = None
    lo, hi = 0.0, 1.0
    for attempt in range(6):
        k = int(np.clip(round(frac * n0), 1, n0))
        W = copy(M)
        W.refine(order[:k], proj)
        grow = (len(W.tets) / n0 - 1) * 100
        if log:
            log(f"bisection: {k} marked -> {len(W.tets)} tetrahedra (+{grow:.1f}%)")
        if best is None or abs(np.log(max(grow, 0.1) / pct)) < abs(np.log(max(best[0], 0.1) / pct)):
            best = (grow, W, frac)
            best_k = k
        if pct * 0.7 <= grow <= pct * 1.4:
            break
        if grow < pct:
            lo = frac
            frac = min(frac * 2, (frac + hi) / 2) if hi < 1 else frac * 2
        else:
            hi = frac
            frac = (lo + frac) / 2 if lo > 0 else frac / 2
        frac = float(np.clip(frac, 1.0 / n0, 1.0))
    grow, W, frac = best
    if log and os.environ.get("HZN_BISCHK"):
        def areas(M):
            X = np.asarray(M.X); A = {}
            for sf, f in M.boundary_tris():
                a = 0.5 * np.linalg.norm(np.cross(X[f[1]] - X[f[0]], X[f[2]] - X[f[0]]))
                A[sf] = A.get(sf, 0) + a
            return A
        A0, A1 = areas(M), areas(W)
        bad = {k: (A0.get(k, 0), A1.get(k, 0)) for k in set(A0) | set(A1)
               if abs(A0.get(k, 0) - A1.get(k, 0)) > 1e-9 * max(1e-30, A0.get(k, 0))}
        log(f"CHK area changes {bad} vol {abs(M.volumes()).sum():.6e} -> {abs(W.volumes()).sum():.6e} neg {(W.volumes() <= 0).sum()}")
    load_into_emerge(m, W)
    return grow, frac


def mesh_quality(m) -> np.ndarray:
    """Quality (1 = regular) of every tet of EMerge model m."""
    X = np.asarray(m.mesh.nodes).T
    T = np.asarray(m.mesh.tets).T
    a, b, c, d = (X[T[:, i]] for i in range(4))
    V = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6
    L2 = sum(((X[T[:, i]] - X[T[:, j]]) ** 2).sum(1) for i in range(4) for j in range(i + 1, 4))
    return 12 * (3 * V) ** (2 / 3) / L2
