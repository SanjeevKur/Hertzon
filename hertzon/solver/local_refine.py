"""Local adaptive mesh refinement (style).

Instead of re-meshing the whole model after every adaptive pass, the tetrahedra with the largest
error estimate are split in place by longest-edge bisection (Rivara's LEPP algorithm). Every
tetrahedron, boundary triangle and boundary line that shares a bisected edge is split at the same
midpoint, so the mesh stays conforming. Midpoints of edges on curved curves/surfaces are projected
back onto the CAD geometry. The refined mesh is handed to EMerge through gmsh, so EMerge's own
mesh import (boundary tags, ports, materials) is reused unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

_ELEM = {0: 15, 1: 1, 2: 2, 3: 4}           # gmsh element type per entity dimension (point, line, tri, tet)
_NPE = {0: 1, 1: 2, 2: 3, 3: 4}             # nodes per element
BOUNDARY_WEIGHT = 2.0


class LocalMesh:
    """Editable copy of the current gmsh mesh, classified on the geometric entities."""

    def __init__(self) -> None:
        import gmsh
        self.xyz: dict[int, np.ndarray] = {}
        self.ent: dict[int, tuple[int, int]] = {}
        for dim, tag in gmsh.model.getEntities():
            tags, coords, _ = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=False, returnParametricCoord=False)
            coords = np.asarray(coords, float).reshape(-1, 3)
            for t, c in zip(tags, coords):
                t = int(t)
                if t not in self.ent:
                    self.xyz[t] = c
                    self.ent[t] = (dim, tag)
        self.next_node = max(self.xyz) + 1 if self.xyz else 1
        # elements per entity: lists of node-tag lists (kept as lists so they can be edited in place)
        self.elems: dict[tuple[int, int], list[list[int]]] = {}
        self.tet_owner: list[tuple[int, int]] = []
        self.tets: list[list[int]] = []
        self.tet_tag_to_idx: dict[int, int] = {}
        for dim, tag in gmsh.model.getEntities():
            types, etags, enodes = gmsh.model.mesh.getElements(dim, tag)
            for ty, et, en in zip(types, etags, enodes):
                if ty != _ELEM[dim]:
                    continue
                conn = np.asarray(en, np.int64).reshape(-1, _NPE[dim]).tolist()
                if dim == 3:
                    for t, c in zip(et, conn):
                        self.tet_tag_to_idx[int(t)] = len(self.tets)
                        self.tets.append(c)
                        self.tet_owner.append((dim, tag))
                else:
                    self.elems.setdefault((dim, tag), []).extend(conn)
        # edge -> tets, edge -> (surface, triangle index), edge -> (curve, line index)
        self.e2t: dict[tuple[int, int], set[int]] = {}
        for i, t in enumerate(self.tets):
            self._link_tet(i, t)
        self.e2f: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {}
        self.e2l: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {}
        for key, conn in self.elems.items():
            if key[0] == 2:
                for j, f in enumerate(conn):
                    for e in _tri_edges(f):
                        self.e2f.setdefault(e, []).append((key, j))
            elif key[0] == 1:
                for j, ln in enumerate(conn):
                    self.e2l.setdefault(_ek(ln[0], ln[1]), []).append((key, j))
        self._len: dict[tuple[int, int], float] = {}

    # -- topology helpers -----------------------------------------------------------------------
    def _link_tet(self, i: int, t: list[int]) -> None:
        for e in _tet_edges(t):
            self.e2t.setdefault(e, set()).add(i)

    def _unlink_tet(self, i: int, t: list[int]) -> None:
        for e in _tet_edges(t):
            s = self.e2t.get(e)
            if s is not None:
                s.discard(i)
                if not s:
                    del self.e2t[e]

    def length(self, e: tuple[int, int]) -> float:
        v = self._len.get(e)
        if v is None:
            d = self.xyz[e[0]] - self.xyz[e[1]]
            v = float(d @ d)
            self._len[e] = v
        return v

    def longest(self, t: list[int]) -> tuple[int, int]:
        # ties are broken by the node tags so that neighbours agree on the same edge
        return max(_tet_edges(t), key=lambda e: (self.length(e), e))

    def _vol(self, t: list[int]) -> float:
        a, b, c, d = (self.xyz[n] for n in t)
        return float(np.dot(np.cross(b - a, c - a), d - a))

    # -- refinement -----------------------------------------------------------------------------
    def _terminal_edge(self, e: tuple[int, int]) -> tuple[int, int]:
        """Follow the longest-edge propagation path until an edge that is the longest of all its tets."""
        while True:
            best = e
            for i in self.e2t.get(e, ()):
                le = self.longest(self.tets[i])
                if (self.length(le), le) > (self.length(best), best):
                    best = le
            if best == e:
                return e
            e = best

    def _midpoint(self, e: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
        import gmsh
        mid = 0.5 * (self.xyz[e[0]] + self.xyz[e[1]])
        on = None
        if e in self.e2l:
            on = self.e2l[e][0][0]
        elif e in self.e2f:
            on = self.e2f[e][0][0]
        if on is None:
            tets = self.e2t.get(e)
            return mid, self.tet_owner[next(iter(tets))]
        try:
            p, _ = gmsh.model.getClosestPoint(on[0], on[1], mid.tolist())
            p = np.asarray(p, float)
            # keep the projection only if it is a small move (protects against seams/odd parametrisations)
            if np.linalg.norm(p - mid) < 0.25 * np.sqrt(self.length(e)):
                return p, on
        except Exception:  # noqa: BLE001
            pass
        return mid, on

    def bisect(self, e: tuple[int, int]) -> None:
        a, b = e
        p, on = self._midpoint(e)
        tets = list(self.e2t.get(e, ()))
        # reject a projected midpoint that would invert or flatten one of the child tetrahedra
        mid = 0.5 * (self.xyz[a] + self.xyz[b])
        if not np.array_equal(p, mid):
            self.xyz[-1] = p
            for i in tets:
                t = self.tets[i]
                v0 = self._vol(t)
                for x in (a, b):
                    c = [(-1 if n == x else n) for n in t]
                    vc = self._vol(c)
                    if vc * v0 <= 0 or abs(vc) < 0.1 * abs(v0):
                        p = mid
                        break
                if p is mid:
                    break
            del self.xyz[-1]
        m = self.next_node
        self.next_node += 1
        self.xyz[m] = p
        self.ent[m] = on if on is not None else self.tet_owner[tets[0]]
        for i in tets:
            t = self.tets[i]
            self._unlink_tet(i, t)
            t2 = [(m if n == a else n) for n in t]
            t1 = [(m if n == b else n) for n in t]
            self.tets[i] = t1
            self._link_tet(i, t1)
            j = len(self.tets)
            self.tets.append(t2)
            self.tet_owner.append(self.tet_owner[i])
            self._link_tet(j, t2)
        for key, j in self.e2f.pop(e, []):
            conn = self.elems[key]
            f = conn[j]
            for ed in _tri_edges(f):
                lst = self.e2f.get(ed)
                if lst is not None and ed != e:
                    lst.remove((key, j))
            f1 = [(m if n == b else n) for n in f]
            f2 = [(m if n == a else n) for n in f]
            conn[j] = f1
            k = len(conn)
            conn.append(f2)
            for ed in _tri_edges(f1):
                self.e2f.setdefault(ed, []).append((key, j))
            for ed in _tri_edges(f2):
                self.e2f.setdefault(ed, []).append((key, k))
        for key, j in self.e2l.pop(e, []):
            conn = self.elems[key]
            ln = conn[j]
            l1 = [(m if n == b else n) for n in ln]
            l2 = [(m if n == a else n) for n in ln]
            conn[j] = l1
            k = len(conn)
            conn.append(l2)
            self.e2l.setdefault(_ek(*l1), []).append((key, j))
            self.e2l.setdefault(_ek(*l2), []).append((key, k))
        self._len.pop(e, None)

    def _refine_tet(self, i: int) -> None:
        """Bisect tet ``i`` across its longest edge (Rivara LEPP: longer neighbouring edges first)."""
        t0 = list(self.tets[i])
        e = self.longest(t0)
        for _ in range(64):
            term = self._terminal_edge(e)
            self.bisect(term)
            if term == e:
                return
            if self.tets[i] != t0:
                return
            e = self.longest(self.tets[i])

    def refine(self, order: list[int], target: int) -> int:
        """Bisect the tets in ``order`` (indices into ``self.tets``, worst first) until there are ``target`` tets."""
        done = 0
        for i in order:
            if len(self.tets) >= target:
                break
            self._refine_tet(i)
            done += 1
        return done

    def _tet_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        tags = np.fromiter(self.xyz.keys(), np.int64, len(self.xyz))
        row = np.zeros(int(tags.max()) + 1, np.int64)
        row[tags] = np.arange(len(tags))
        X = np.array([self.xyz[t] for t in tags.tolist()], float)
        T = row[np.asarray(self.tets, np.int64)]
        P = X[T]                                                   # (n, 4, 3)
        L = np.zeros(len(T))
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            L = np.maximum(L, np.linalg.norm(P[:, a] - P[:, b], axis=1))
        return P.mean(axis=1), L

    def refine_to_size(self, size_at, target: int, max_rounds: int = 16) -> None:
        """Bisect every tet whose longest edge exceeds the target size, largest excess first, until
        the mesh has ``target`` tets. The size field is tightened when it is met before the target."""
        scale = 1.0
        for _ in range(max_rounds):
            if len(self.tets) >= target:
                return
            cen, L = self._tet_geometry()
            r = L / (np.maximum(size_at(cen.T), 1e-300) * scale)
            cand = np.flatnonzero(r > 1.0)
            if len(cand) == 0:
                scale *= 0.8
                continue
            for i in cand[np.argsort(-r[cand])].tolist():
                if len(self.tets) >= target:
                    return
                self._refine_tet(i)

    def refine_edges(self, size_at, target: int, max_rounds: int = 16) -> None:
        """Bisect every edge (volume, surface and curve edges alike) that is longer than the target size
        at its midpoint, largest excess first, until the mesh has ``target`` tets. Bisections follow
        the longest-edge propagation path, so element quality stays bounded. The size field is
        tightened when it is met before the target."""
        scale = 1.0
        for _ in range(max_rounds):
            if len(self.tets) >= target:
                return
            edges = list(self.e2t.keys())
            ea = np.asarray(edges, np.int64)
            tags = np.fromiter(self.xyz.keys(), np.int64, len(self.xyz))
            row = np.zeros(int(tags.max()) + 1, np.int64)
            row[tags] = np.arange(len(tags))
            X = np.array([self.xyz[t] for t in tags.tolist()], float)
            A, B = X[row[ea[:, 0]]], X[row[ea[:, 1]]]
            L = np.linalg.norm(A - B, axis=1)
            r = L / (np.maximum(size_at((0.5 * (A + B)).T), 1e-300) * scale)
            # edges on surfaces/curves (conductors, ports, interfaces) carry the surface mesh, which a
            # fresh gmsh mesh resolves first; give them priority (and a stricter size) does
            on_bnd = np.fromiter(((e in self.e2f) or (e in self.e2l) for e in edges), bool, len(edges))
            r[on_bnd] *= BOUNDARY_WEIGHT
            cand = np.flatnonzero(r > 1.0)
            if len(cand) == 0:
                scale *= 0.8
                continue
            for k in cand[np.argsort(-r[cand])].tolist():
                if len(self.tets) >= target:
                    return
                e = edges[k]
                for _ in range(64):
                    if e not in self.e2t:
                        break
                    term = self._terminal_edge(e)
                    self.bisect(term)
                    if term == e:
                        break

    # -- write back -----------------------------------------------------------------------------
    def to_arrays(self) -> dict[str, np.ndarray]:
        """Flat arrays describing the mesh (used for gmsh and for saving/restoring)."""
        tags = np.fromiter(self.xyz.keys(), np.int64, len(self.xyz))
        out = {"node_tags": tags,
               "node_xyz": np.array([self.xyz[t] for t in tags.tolist()], float).reshape(-1, 3),
               "node_ent": np.array([self.ent[t] for t in tags.tolist()], np.int64).reshape(-1, 2)}
        ek, ec = [], []
        for (dim, tag), conn in self.elems.items():
            ek.append((dim, tag, len(conn)))
            ec.extend(n for el in conn for n in el)
        tets_by = {}
        for t, own in zip(self.tets, self.tet_owner):
            tets_by.setdefault(own, []).append(t)
        for (dim, tag), conn in tets_by.items():
            ek.append((dim, tag, len(conn)))
            ec.extend(n for el in conn for n in el)
        out["elem_keys"] = np.array(ek, np.int64).reshape(-1, 3)
        out["elem_conn"] = np.array(ec, np.int64)
        return out


def push_to_gmsh(arr: dict[str, np.ndarray]) -> None:
    """Replace the gmsh mesh with the given mesh arrays (same entities as the current model)."""
    import gmsh
    gmsh.model.mesh.clear()
    tags, xyz, ent = arr["node_tags"], arr["node_xyz"], arr["node_ent"]
    keys = [tuple(int(v) for v in k) for k in ent]
    order = np.lexsort((ent[:, 1], ent[:, 0]))
    start = 0
    while start < len(order):
        k = keys[order[start]]
        stop = start
        while stop < len(order) and keys[order[stop]] == k:
            stop += 1
        sel = order[start:stop]
        gmsh.model.mesh.addNodes(k[0], k[1], tags[sel].tolist(), xyz[sel].ravel().tolist())
        start = stop
    next_el = 1
    pos = 0
    conn = arr["elem_conn"]
    for dim, tag, n in arr["elem_keys"].tolist():
        npe = _NPE[dim]
        nodes = conn[pos:pos + n * npe]
        pos += n * npe
        gmsh.model.mesh.addElementsByType(tag, _ELEM[dim], list(range(next_el, next_el + n)), nodes.tolist())
        next_el += n


def optimize_gmsh_mesh() -> None:
    """Edge/face swaps and node smoothing of the tetrahedra (gmsh's own optimizers; boundaries are kept)."""
    import gmsh
    gmsh.model.mesh.optimize("")
    try:
        gmsh.model.mesh.optimize("Netgen")
    except Exception:  # noqa: BLE001
        pass


# what EMerge's generate_mesh() really does while it is replaced (its log still says "Calling GMSH
# mesher"); the solver's log forwarding reports this instead
LOADING: list[str] = []


@contextmanager
def _generate_replaced(arr, optimize: bool = False, what: str = "mesh"):
    """Make EMerge's generate_mesh() load ``arr`` instead of running the gmsh mesher."""
    import gmsh
    orig = gmsh.model.mesh.generate

    def fake(dim=3):
        push_to_gmsh(arr)
        if optimize:
            optimize_gmsh_mesh()
    gmsh.model.mesh.generate = fake
    LOADING.append(what)
    try:
        yield
    finally:
        LOADING.pop()
        gmsh.model.mesh.generate = orig


def load_mesh(m, arr, optimize: bool = False, what: str = "refined mesh") -> None:
    """Install a stored/refined mesh into EMerge model ``m`` through its normal mesh import (gmsh only
    carries the arrays, it does not mesh)."""
    m._reset_mesh()
    with _generate_replaced(arr, optimize, what):
        m.generate_mesh(True)


def refine_to_size(m, size_at, pct: float) -> dict[str, np.ndarray]:
    """Bisect tetrahedra of ``m`` towards the size field ``size_at(points)`` until the mesh has grown by
    ``pct`` percent, then reload it into EMerge. Returns the mesh arrays (to be stored for resuming)."""
    lm = LocalMesh()
    lm.refine_edges(size_at, int(round(len(lm.tets) * (1 + pct / 100.0))))
    load_mesh(m, lm.to_arrays(), optimize=True)
    return LocalMesh().to_arrays()          # the optimised mesh, for resuming


def supported(m) -> bool:
    """Local refinement needs a plain (non-periodic) linear tetrahedral mesh."""
    try:
        return not m.mesher._get_periodic_bcs()
    except Exception:  # noqa: BLE001
        return True


def _ek(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _tet_edges(t):
    a, b, c, d = t
    return (_ek(a, b), _ek(a, c), _ek(a, d), _ek(b, c), _ek(b, d), _ek(c, d))


def _tri_edges(f):
    a, b, c = f
    return (_ek(a, b), _ek(b, c), _ek(a, c))
