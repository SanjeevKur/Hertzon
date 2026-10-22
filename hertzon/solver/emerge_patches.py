"""Run-time fixes for EMerge 2.8.9 and 3.0 (applied once in the solver worker). Every fix checks the
EMerge source it changes and leaves other versions alone; EMerge 3 already has most of the speed-ups.

Mesh3D._pre_update builds a dense (nodes x nodes) int32 edge lookup table. With 342k nodes that is 436 GiB
(MemoryError for large imported models). It is replaced by a sorted-key lookup with the same indexing."""
from __future__ import annotations

import inspect
import os
import textwrap

import numpy as np

_DONE = False


class _SparseEdgeLookup:
    """Drop-in for ``mat[a, b]`` on the symmetric edge table: O(edges) memory instead of O(nodes^2)."""

    def __init__(self, edges, n_nodes: int, missing: int):
        e = np.asarray(edges, dtype=np.int64)
        lo, hi = np.minimum(e[0], e[1]), np.maximum(e[0], e[1])
        keys = lo * int(n_nodes) + hi
        order = np.argsort(keys)
        self.keys = keys[order]
        self.ids = order.astype(np.int64)
        self.n = int(n_nodes)
        self.missing = missing

    def __getitem__(self, ab):
        a, b = (np.asarray(x, dtype=np.int64) for x in ab)
        k = np.minimum(a, b) * self.n + np.maximum(a, b)
        pos = np.clip(np.searchsorted(self.keys, k), 0, max(len(self.keys) - 1, 0))
        out = self.ids[pos] if len(self.keys) else np.full(k.shape, self.missing, np.int64)
        hit = self.keys[pos] == k if len(self.keys) else np.zeros(k.shape, bool)
        return np.where(hit, out, self.missing)


# -- vectorised mesh import --------------------------------------------------------------------------
# _pre_update builds edges, triangles and all index maps in per-tetrahedron Python loops (about 10 s
# for 176k tetrahedra, more than the local mesh refinement by far). The loops below are replaced by
# numpy sorting/searching with identical results (same arrays, same dicts, same order); every block
# is only swapped when the EMerge source matches exactly.

def _tag_map(d: dict) -> np.ndarray:
    """gmsh tag -> index array for the dict ``d`` (-1 = missing)."""
    k = np.fromiter(d.keys(), np.int64, len(d))
    v = np.fromiter(d.values(), np.int64, len(d))
    out = np.full(int(k.max()) + 1 if len(k) else 1, -1, np.int64)
    out[k] = v
    return out


def _map_tags(tm: np.ndarray, tags) -> np.ndarray:
    t = np.asarray(tags, dtype=np.int64)
    if t.size and (t.min() < 0 or t.max() >= len(tm)):
        raise KeyError(int(t[(t < 0) | (t >= len(tm))][0]))
    r = tm[t]
    if r.size and r.min() < 0:
        raise KeyError(int(t[r < 0][0]))
    return r


def _key2(a, b, n: int) -> np.ndarray:
    return np.asarray(a, np.int64) * np.int64(n) + np.asarray(b, np.int64)


def _key3(abc, n: int) -> np.ndarray:
    if n >= 2_000_000:
        raise OverflowError("too many nodes for int64 triangle keys")
    abc = np.asarray(abc, np.int64)
    return (abc[0] * np.int64(n) + abc[1]) * np.int64(n) + abc[2]


def _find(keys: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Positions of q in the sorted keys (KeyError if one is missing)."""
    if not len(keys):
        if len(q):
            raise KeyError("empty")
        return np.zeros(0, np.int64)
    pos = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    bad = keys[pos] != q
    if bad.any():
        raise KeyError(int(q[bad][0]))
    return pos


def _edges_tris(tets):
    """Unique sorted node pairs and triples of the tetrahedra, columns in lexicographic order
    (= np.array(sorted(set_of_tuples)).T of the original loop)."""
    st = np.sort(np.asarray(tets, np.int64), axis=0)
    if not st.size:
        return np.zeros((2, 0), int), np.zeros((3, 0), int)
    n = np.int64(int(st.max()) + 1)
    k = np.unique(np.concatenate([st[a] * n + st[b] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]))
    edges = np.stack([k // n, k % n]).astype(int)
    trip = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    if n < 2_000_000:
        k = np.unique(np.concatenate([(st[a] * n + st[b]) * n + st[c] for a, b, c in trip]))
        tris = np.stack([k // (n * n), (k // n) % n, k % n]).astype(int)
    else:
        tris = np.unique(np.concatenate([st[list(t)] for t in trip], axis=1), axis=1).astype(int)
    return edges, tris


_VEC_BLOCKS = [
    ("    tet_node_tags = [self.n_t2i[int(t)] for t in tet_node_tags[0]]\n",
     "    tet_node_tags = _map_tags(_tag_map(self.n_t2i), tet_node_tags[0]).tolist()\n"),
    ("""    edgeset = set()
    triset = set()
""", """    self.edges, self.tris = _edges_tris(self.tets)
    edgeset, triset = self.edges.T, self.tris.T
"""),
    ("""    for itet in range(self.tets.shape[1]):
        i1, i2, i3, i4 = sorted([int(ind) for ind in self.tets[:, itet]])
        edgeset.add((i1, i2))
        edgeset.add((i1, i3))
        edgeset.add((i1, i4))
        edgeset.add((i2, i3))
        edgeset.add((i2, i4))
        edgeset.add((i3, i4))
        triset.add((i1, i2, i3))
        triset.add((i1, i2, i4))
        triset.add((i1, i3, i4))
        triset.add((i2, i3, i4))
""", ""),
    ("""    self.edges = np.array(sorted(list(edgeset))).T
    self.tris = np.array(sorted(list(triset))).T
""", ""),
    ("""    self.inv_edges = {tuple(_sorted_edges[:, i].tolist()): i for i in range(nE)}
    self.inv_tris = {tuple(_sorted_tris[:, i].tolist()): i for i in range(nR)}
    self.inv_tets = {tuple(_sorted_tets[:, i].tolist()): i for i in range(nT)}
""", """    self.inv_edges = dict(zip(map(tuple, _sorted_edges.T.tolist()), range(nE)))
    self.inv_tris = dict(zip(map(tuple, _sorted_tris.T.tolist()), range(nR)))
    self.inv_tets = dict(zip(map(tuple, _sorted_tets.T.tolist()), range(nT)))
"""),
    ("""    for itet in range(self.tets.shape[1]):
        t = self.tets[:, itet]
        self.tet_to_edge[:, itet] = [""", """    try:
        _ek = _key2(_sorted_edges[0], _sorted_edges[1], nN)
        for _c, (_i, _j) in enumerate(_idset1):
            _a, _b = self.tets[_i - 1], self.tets[_j - 1]
            self.tet_to_edge[_c] = _find(_ek, _key2(np.minimum(_a, _b), np.maximum(_a, _b), nN))
        _rk = _key3(_sorted_tris, nN)
        for _c, (_i, _j, _k) in enumerate(_idset2):
            self.tet_to_tri[_c] = _find(_rk, _key3(np.sort(self.tets[[_i - 1, _j - 1, _k - 1]], axis=0), nN))
        _vec_t2e = True
    except (KeyError, OverflowError):
        _vec_t2e = False
    for itet in range(0 if _vec_t2e else self.tets.shape[1]):
        t = self.tets[:, itet]
        self.tet_to_edge[:, itet] = ["""),
    ("""    tri_nodes = np.array(
        [self.n_t2i[int(t)] for t in tri_node_tags[0]], dtype=int
    ).reshape(-1, 3)
""", """    tri_nodes = np.array(
        _map_tags(_tag_map(self.n_t2i), tri_node_tags[0]), dtype=int
    ).reshape(-1, 3)
"""),
    ("""        for k, tag in enumerate(tri_tags):
            tri_id = self.inv_tris[tuple(_sorted_tri_nodes[k, :].tolist())]
            self.tri_i2t[tri_id] = int(tag)
""", """        try:
            _ids = _find(_key3(_sorted_tris, nN), _key3(_sorted_tri_nodes.T, nN))
            self.tri_i2t = dict(zip(_ids.tolist(), tri_tags.tolist()))
            _loop_tags = ()
        except (KeyError, OverflowError):
            self.tri_i2t = {}
            _loop_tags = tri_tags
        for k, tag in enumerate(_loop_tags):
            tri_id = self.inv_tris[tuple(_sorted_tri_nodes[k, :].tolist())]
            self.tri_i2t[tri_id] = int(tag)
"""),
    ("""    for _d, t in face_dimtags:
        node_tags = _face_entity_map[t]
        node_tags = [self.n_t2i[int(t)] for t in node_tags[0]]
        self.ftag_to_node[t] = node_tags
        node_tags = np.squeeze(np.array(node_tags)).reshape(-1, 3).T
        self.ftag_to_tri[t] = [
            self.get_tri(node_tags[0, i], node_tags[1, i], node_tags[2, i])
            for i in range(node_tags.shape[1])
        ]
""", """    _tmf = _tag_map(self.n_t2i)
    try:
        _rkf = _key3(_sorted_tris, nN)
    except OverflowError:
        _rkf = None
    for _d, t in face_dimtags:
        node_tags = _face_entity_map[t]
        node_tags = _map_tags(_tmf, node_tags[0]).tolist()
        self.ftag_to_node[t] = node_tags
        node_tags = np.squeeze(np.array(node_tags)).reshape(-1, 3).T
        try:
            if _rkf is None:
                raise KeyError("fallback")
            self.ftag_to_tri[t] = _find(_rkf, _key3(np.sort(node_tags, axis=0), nN)).tolist()
        except KeyError:
            self.ftag_to_tri[t] = [
                self.get_tri(node_tags[0, i], node_tags[1, i], node_tags[2, i])
                for i in range(node_tags.shape[1])
            ]
"""),
    ("""        self.vtag_to_tet[t] = [
            self.inv_tets[tuple(row)] for row in sorted_nodes.tolist()
        ]
""", """        self.vtag_to_tet[t] = list(map(self.inv_tets.__getitem__, map(tuple, sorted_nodes.tolist())))
"""),
]


def _vectorize(src: str) -> tuple[str, int]:
    n = 0
    for old, new in _VEC_BLOCKS:
        if src.count(old) == 1:
            src = src.replace(old, new)
            n += 1
    return src, n


def apply() -> None:
    global _DONE
    if _DONE:
        return
    _DONE = True
    try:      # before apply_curved: the curved-element wrapper wraps the faster assembly
        from . import fast_assembly
        fast_assembly.apply()
    except Exception as ex:  # noqa: BLE001
        from loguru import logger
        logger.warning(f"Fast matrix assembly not available ({ex}); using EMerge's assembly")
    try:      # after fast_assembly: wraps its PARDISO solve
        from . import mt_pardiso
        mt_pardiso.apply()
    except Exception as ex:  # noqa: BLE001
        from loguru import logger
        logger.warning(f"Parallel sweeps with PARDISO not available ({ex})")
    try:      # after fast_assembly: wraps its SolveRoutine.solve
        from . import hybrid_sweep
        hybrid_sweep.apply()
    except Exception as ex:  # noqa: BLE001
        from loguru import logger
        logger.warning(f"Reduced basis for the interpolating sweep not available ({ex})")
    apply_amr()
    apply_curved()
    apply_mesh_fallback()
    apply_eigs_retry()
    apply_thin_conductor()
    try:
        from emerge._emerge import mesh3d
        fn = mesh3d.Mesh3D._pre_update
        src0 = textwrap.dedent(inspect.getsource(fn))
        variants = [src0]
        if os.environ.get("HERTZON_SLOW_IMPORT") != "1":
            vec, nb = _vectorize(src0)
            if nb == len(_VEC_BLOCKS):  # all blocks or none: they depend on each other
                variants.insert(0, vec)
        for src in variants:            # vectorised first, the plain sparse-lookup fix as fallback
            out = _sparse_lookup(src)
            if out is None:
                return                  # different EMerge version: leave it alone
            ns = dict(mesh3d.__dict__)
            ns["_SparseEdgeLookup"] = _SparseEdgeLookup
            ns.update(_edges_tris=_edges_tris, _tag_map=_tag_map, _map_tags=_map_tags, _key2=_key2, _key3=_key3,
                      _find=_find)
            try:
                exec(compile(out, inspect.getsourcefile(fn), "exec"), ns)  # noqa: S102
            except Exception:  # noqa: BLE001
                continue
            mesh3d.Mesh3D._pre_update = ns["_pre_update"]
            return
    except Exception:  # noqa: BLE001
        pass


def _sparse_lookup(src: str):
    """Swap the dense (nodes x nodes) edge table for _SparseEdgeLookup (None if the source differs)."""
    new = []
    found = False
    for ln in src.split("\n"):
        st = ln.strip()
        if st == "_edge_lookup_mat = np.full((nN, nN), _MISSING_ID, dtype=np.int32)":
            ind = ln[: len(ln) - len(ln.lstrip())]
            new.append(ind + "_edge_lookup_mat = _SparseEdgeLookup(self.edges, nN, _MISSING_ID)")
            found = True
        elif st.startswith("_edge_lookup_mat[self.edges["):
            continue
        else:
            new.append(ln)
    return "\n".join(new) if found else None


# ------------------------------------------------------------------------------------------------
# Adaptive refinement size field
# ------------------------------------------------------------------------------------------------
# EMerge evaluates  size(x) = min(lc, min_i max(s_i, A_i + c*|x - p_i|))  over *all* refinement points
# (they accumulate pass after pass) for every gmsh size query, and recompiles a numba closure each pass.
# On pass 5+ that made "estimating solution error / refining" slower than the FEM solve itself.
# Same function, exact, but with a uniform grid over the points and a pruned shell search; the kernel is
# compiled once (cached on disk).

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        from numba import njit

        @njit(cache=True, nogil=True, fastmath=True)
        def query(x, y, z, lc, px, py, pz, ps, pa, c, amin, lo, h, dims, start, order, ring0):
            best = lc
            nx, ny, nz = dims[0], dims[1], dims[2]
            ix = min(max(int((x - lo[0]) / h), 0), nx - 1)
            iy = min(max(int((y - lo[1]) / h), 0), ny - 1)
            iz = min(max(int((z - lo[2]) / h), 0), nz - 1)
            # distance from the query to the grid box, for points outside it
            ox = max(lo[0] - x, 0.0, x - (lo[0] + nx * h))
            oy = max(lo[1] - y, 0.0, y - (lo[1] + ny * h))
            oz = max(lo[2] - z, 0.0, z - (lo[2] + nz * h))
            off = (ox * ox + oy * oy + oz * oz) ** 0.5
            rmax = max(nx, ny, nz)
            # rings closer than ring0 contain no points (chessboard distance transform)
            for r in range(ring0[(ix * ny + iy) * nz + iz], rmax + 1):
                g = max(r - 1, 0) * h
                bound = amin + c * (off * off + g * g) ** 0.5
                if bound >= best:
                    break
                for jx in range(max(ix - r, 0), min(ix + r, nx - 1) + 1):
                    xface = abs(jx - ix) == r
                    for jy in range(max(iy - r, 0), min(iy + r, ny - 1) + 1):
                        if xface or abs(jy - iy) == r:
                            z0 = max(iz - r, 0)
                            z1 = min(iz + r, nz - 1)
                            zs = 1
                        else:
                            # interior column of the shell: only its two end caps
                            z0 = iz - r
                            z1 = iz + r
                            zs = 2 * r if r > 0 else 1
                        for jz in range(z0, z1 + 1, zs):
                            if jz < 0 or jz >= nz:
                                continue
                            cell = (jx * ny + jy) * nz + jz
                            for k in range(start[cell], start[cell + 1]):
                                i = order[k]
                                dx = x - px[i]
                                dy = y - py[i]
                                dz = z - pz[i]
                                v = pa[i] + c * (dx * dx + dy * dy + dz * dz) ** 0.5
                                if v < ps[i]:
                                    v = ps[i]
                                if v < best:
                                    best = v
            return best
        _KERNEL = query
    return _KERNEL


def _size_args(amr, gr: float, _qf: float):
    """Grid acceleration structure for EMerge's AMR size function (None when it does not apply)."""
    if amr._amr_coords is None:
        return None
    P = np.ascontiguousarray(amr._amr_coords, dtype=np.float64)
    ns = np.ascontiguousarray(amr._amr_ratios * amr._amr_sizes, dtype=np.float64)
    A = ns / gr
    c = float(-(1 - gr) / gr * _qf)
    if P.shape[1] == 0 or c <= 0:
        return None
    lo = P.min(axis=1)
    span = np.maximum(P.max(axis=1) - lo, 1e-12)
    n = P.shape[1]
    h = float(max(span.max() / 128.0, (np.prod(span) / max(n / 4.0, 1.0)) ** (1 / 3), 1e-12))
    dims = np.maximum(np.ceil(span / h).astype(np.int64), 1)
    ijk = np.minimum(((P - lo[:, None]) / h).astype(np.int64), (dims - 1)[:, None])
    cell = (ijk[0] * dims[1] + ijk[1]) * dims[2] + ijk[2]
    order = np.argsort(cell, kind="stable").astype(np.int64)
    counts = np.bincount(cell, minlength=int(np.prod(dims)))
    from scipy.ndimage import distance_transform_cdt
    ring0 = distance_transform_cdt((counts == 0).reshape(tuple(dims)), metric="chessboard")
    ring0 = np.ascontiguousarray(ring0.ravel(), dtype=np.int64)
    start = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=start[1:])
    px, py, pz = (np.ascontiguousarray(P[i]) for i in range(3))
    return (px, py, pz, ns, np.ascontiguousarray(A), c, float(A.min()), np.ascontiguousarray(lo), h, dims,
            start, order, ring0)


def amr_size_at(amr, gr: float, _qf: float, pts: np.ndarray, lc: float = 1e300) -> np.ndarray:
    """EMerge's AMR target element size evaluated at the points ``pts`` (3, n)."""
    args = _size_args(amr, gr, _qf)
    pts = np.asarray(pts, float)
    if args is None:
        return np.full(pts.shape[1], lc)
    q = _kernel()
    return np.array([q(x, y, z, lc, *args) for x, y, z in pts.T.tolist()])


def _fast_set_refinement_function(self, gr: float = 1.5, _qf: float = 1.0):
    import gmsh
    args = _size_args(self, gr, _qf)
    if args is None:
        return _ORIG_SET_REF(self, gr, _qf)
    q = _kernel()

    def func(dim, tag, x, y, z, lc):
        return q(x, y, z, lc, *args)
    self._fast_cb = func              # keep a reference while gmsh uses it
    # gmsh worker threads would all call back into Python and serialise on the GIL,
    # which made adaptive remeshing 3-6x slower than meshing on a single thread.
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.mesh.setSizeCallback(func)


_ORIG_SET_REF = None


def _robust_eigs(eigs):
    """scipy eigs for the port mode solve: ARPACK starts from a random vector and now and then stops
    without convergence (seen in sweeps of the waveguide example, with EMerge's own code too), which
    ended the whole solve. Retried with a fixed start vector and a larger Krylov space, then densely
    for small port problems."""
    def robust(A, k=6, M=None, sigma=None, which="LM", **kw):
        from scipy.sparse.linalg import ArpackError, ArpackNoConvergence
        try:
            return eigs(A, k=k, M=M, sigma=sigma, which=which, **kw)
        except (ArpackNoConvergence, ArpackError) as ex:
            first = ex
        n = A.shape[0]
        for ncv in (min(n - 1, max(2 * k + 1, 40)), min(n - 1, max(4 * k + 1, 120))):
            try:
                return eigs(A, k=k, M=M, sigma=sigma, which=which, ncv=ncv, maxiter=20 * n,
                            v0=np.ones(n, dtype=complex) / np.sqrt(n))
            except (ArpackNoConvergence, ArpackError):
                pass
        if sigma is None or n > 4000:
            raise first
        import scipy.linalg as sl
        Ad = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
        Md = np.eye(n) if M is None else (M.toarray() if hasattr(M, "toarray") else np.asarray(M))
        w, v = sl.eig(Ad, Md)
        ok = np.isfinite(w)
        w, v = w[ok], v[:, ok]
        idx = np.argsort(np.abs(w - sigma))[:k]      # shift-invert "LM" = closest to sigma
        return w[idx], v[:, idx]
    robust._hertzon_robust = True
    return robust


def apply_eigs_retry() -> None:
    try:
        from emerge._emerge import solver
        if not getattr(solver.eigs, "_hertzon_robust", False):
            solver.eigs = _robust_eigs(solver.eigs)
    except Exception:  # noqa: BLE001
        pass


def apply_thin_conductor() -> None:
    """ThinConductor (finite conductivity with thickness on a sheet inside the model) splits the DoFs of the
    sheet in Nedelec2.partition_dof and remaps them in tet_to_field, but only for one tetrahedron per sheet
    triangle and on a side picked from an unoriented normal, so about half of the tetrahedra at the sheet
    use the copies. The matrix assembly (curl-curl and mass through get_tet_to_field) uses the original
    numbering, so the solution is that of a one-sided impedance sheet and the copies only carry the second
    Robin term. Field plots, far fields and the error estimate read tet_to_field and got noise at every
    conductor sheet (Jsurf, refinement chasing it). The original numbering is restored after the split,
    so everything reads the DoFs the solution lives on; the S-parameters do not change."""
    try:
        from emerge._emerge.elements.nedelec2 import Nedelec2
    except Exception:  # noqa: BLE001
        return
    orig = Nedelec2.partition_dof
    if getattr(orig, "_hertzon_consistent", False):
        return

    def partition_dof(self, list_tags):
        orig(self, list_tags)
        if getattr(self, "_tet_to_field_orig", None) is not None:
            self.tet_to_field = self._tet_to_field_orig.copy()

    partition_dof._hertzon_consistent = True
    Nedelec2.partition_dof = partition_dof
    _thin_conductor_new_mesh()


def _thin_conductor_new_mesh() -> None:
    """EMerge 3 splits the sheet DoFs only once, when the boundary data is first initialized; the basis of
    a new mesh (next adaptive pass) then has no second side and the assembly fails (tri_to_field_os is
    None). Every new basis is split here as well."""
    try:
        from emerge._emerge.physics.microwave import microwave_3d as M
        from emerge._emerge.physics.microwave.bcs.boundary_conditions import ThinConductor
    except Exception:  # noqa: BLE001
        return
    orig = M.Microwave3D._initialize_field
    if getattr(orig, "_hertzon_split", False):
        return

    def _initialize_field(self):
        new = self.basis is None
        orig(self)
        if new and getattr(self, "_bc_data_initialized", False):
            tc = self.bc.oftype(ThinConductor)
            if tc:
                self.basis.partition_dof([x.tags for x in tc])
                self.basis._partitioned = True

    _initialize_field._hertzon_split = True
    M.Microwave3D._initialize_field = _initialize_field


def apply_mesh_fallback() -> None:
    """Trimmed B-spline faces of CAD exports (degenerate poles, edges of a few micrometres) can break
    gmsh's surface mesher when the element size follows the curvature ("Impossible case in triangle
    search"). The mesh is then generated again without curvature based sizing; the size fields
    (wavelength, mesh operations, refinement) stay in force, so only curved faces start coarser."""
    import gmsh
    orig = gmsh.model.mesh.generate
    if getattr(orig, "_hertzon_fallback", False):
        return

    def generate(dim=3):
        try:
            return orig(dim)
        except Exception as e:  # noqa: BLE001
            curv = gmsh.option.getNumber("Mesh.MeshSizeFromCurvature")
            if dim < 2 or curv == 0:
                raise
            from .bridge import warn
            warn(f"Mesh failed ({e}); meshing again without curvature based element sizes")
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            try:
                out = orig(dim)
            finally:
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curv)
            # after a failed 3D pass gmsh can report success with an empty volume mesh
            if dim >= 3 and not any(len(t) for t in gmsh.model.mesh.getElements(3)[1]):
                raise RuntimeError(f"{e}. The geometry has overlapping or nearly touching surfaces (often sliver "
                                   "faces of a CAD export); repair them in CAD or import a cleaned STL") from e
            return out
    generate._hertzon_fallback = True
    gmsh.model.mesh.generate = generate


def apply_curved() -> None:
    """Curvilinear tets at curved PEC/material surfaces (see curvilinear.py): the element matrices of the
    tets next to curved CAD surfaces are recomputed after EMerge's assembly. Any error leaves EMerge's
    own (straight) matrices in place."""
    try:
        from emerge._emerge.physics.microwave.assembly import assembler
        try:
            from emerge._emerge.physics.microwave.assembly import curlcurl
        except ImportError:     # EMerge 3: the assembler reads it from emr_microwave at call time
            from emerge._emerge.physics.microwave.assembly import emr_microwave as curlcurl
        from . import curvilinear as CV
    except Exception:  # noqa: BLE001
        return
    orig = curlcurl.tet_mass_stiffness_matrices
    if getattr(orig, "_hertzon_curved", False):
        return

    def tet_mass_stiffness_matrices(field, er, ur, conductor_tets, cscmap=None):
        dataE, dataB, cscmap = orig(field, er, ur, conductor_tets, cscmap)
        if CV.enabled():
            try:
                n = CV.apply_curved(field, er, ur, conductor_tets, dataE, dataB)
                if n:
                    from loguru import logger
                    logger.info(f"Curved elements: {n} tetrahedra on curved surfaces "
                                f"({CV.STATS.get('curved_edges', 0)} curved edges)")
                    if os.environ.get("HERTZON_CURVED_CHECK") == "1":
                        v0, v1 = CV.curved_volume(field.mesh, CV.CTX.get("exclude", frozenset()))
                        logger.info(f"Curved elements: volume straight {v0:.9e}, curved {v1:.9e}")
            except Exception as e:  # noqa: BLE001
                from loguru import logger
                logger.warning(f"Curved elements failed ({e}); using straight elements")
        return dataE, dataB, cscmap

    tet_mass_stiffness_matrices._hertzon_curved = True
    curlcurl.tet_mass_stiffness_matrices = tet_mass_stiffness_matrices

    def ctx(fn):
        def wrapped(self, field, materials, bcs, *a, **k):
            try:      # faces with surface integrals (ports, absorbers, lumped elements, sheets) stay straight
                CV.CTX["exclude"] = CV.exclude_tags(bcs)
            except Exception:  # noqa: BLE001
                CV.CTX["exclude"] = frozenset()
            return fn(self, field, materials, bcs, *a, **k)
        wrapped.__wrapped__ = fn
        return wrapped

    for name in ("assemble_freq_matrix", "assemble_scattering_matrix", "assemble_eig_matrix"):
        fn = getattr(assembler.Assembler, name, None)
        if fn is not None and not hasattr(fn, "__wrapped__"):
            setattr(assembler.Assembler, name, ctx(fn))


def apply_amr() -> None:
    global _ORIG_SET_REF
    try:
        from emerge._emerge import mesher
        if _ORIG_SET_REF is None:
            _ORIG_SET_REF = mesher.AMRPoints.set_refinement_function
            mesher.AMRPoints.set_refinement_function = _fast_set_refinement_function
    except Exception:  # noqa: BLE001
        pass
