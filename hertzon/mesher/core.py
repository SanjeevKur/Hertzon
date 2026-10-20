"""ctypes interface of hzmesh, the local constrained Delaunay refinement written in C++.

The mesh is refined in place: all existing nodes stay, new nodes are inserted by local cavity
re-triangulation bounded by the CAD surfaces and curves, so only the neighbourhood of every new node
changes (nested meshes, no re-meshing noise in the adaptive loop).
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

PARAMS = ("alpha", "btet", "bface", "beta", "budget", "maxins", "improve", "sliver", "smooth", "verbose",
          "qmin_new", "improve_all", "sag")
STATS = ("ins", "seg", "face", "tet", "enc", "fail", "f23", "f32", "smooth", "newtets", "t_refine",
         "t_improve", "qmin", "dmin", "skipped", "f44", "fr_edge", "fr_loc2", "fr_loc3", "fr_cavity", "fr_core",
         "fr_tags", "fr_volume", "fr_topo", "fr_ext2", "fr_int3", "fr_extunused", "fr_hole", "fr_iface", "fr_subgone",
         "fr_14", "fr_15")
_DEFAULT = -12345.0

_ELEM = {0: 15, 1: 1, 2: 2, 3: 4}           # gmsh element type per entity dimension
_NPE = {0: 1, 1: 2, 2: 3, 3: 4}
_FLAT = {"Plane", "Line"}                   # gmsh/OCC entity types that need no projection

PROJ_CB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                           ctypes.POINTER(ctypes.c_double))

_LIB = None


def _libname() -> str:
    if sys.platform == "win32":
        return "hzmesh.dll"
    if sys.platform == "darwin":
        return "libhzmesh.dylib"
    return "libhzmesh.so"


def lib():
    """Load the shared library (HZMESH_LIB overrides the bundled one)."""
    global _LIB
    if _LIB is not None:
        return _LIB
    path = os.environ.get("HZMESH_LIB") or os.path.join(os.path.dirname(__file__), "_bin", _libname())
    L = ctypes.CDLL(path)
    P, I, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    pd, pi, pu = ctypes.POINTER(D), ctypes.POINTER(I), ctypes.POINTER(ctypes.c_ubyte)
    L.hzm_create.restype = P
    L.hzm_create.argtypes = [I, pd, pi, pi, I, pi, pi, I, pi, pi, I, pi, pi]
    L.hzm_clone.restype = P
    L.hzm_clone.argtypes = [P]
    L.hzm_free.argtypes = [P]
    L.hzm_last_error.restype = ctypes.c_char_p
    L.hzm_set_curved.argtypes = [P, I, I, I]
    L.hzm_set_projector.argtypes = [P, PROJ_CB]
    L.hzm_set_size.argtypes = [P, pd, pu]
    L.hzm_refine.argtypes = [P, pd, pd]
    L.hzm_check.argtypes = [P]
    L.hzm_grade.argtypes = [P, pd, pd, D, pd]
    L.hzm_sizes.argtypes = [P, pi]
    L.hzm_get.argtypes = [P, pd, pi, pi, pi, pi, pi, pi, pi, pi]
    L.hzm_quality.argtypes = [P, pd, pd]
    _LIB = L
    return L


def available() -> bool:
    try:
        lib()
        return True
    except (OSError, AttributeError):     # missing or outdated library
        return False


def _ptr(a, ct):
    return a.ctypes.data_as(ctypes.POINTER(ct))


def _i(a):
    return np.ascontiguousarray(a, dtype=np.int32)


def _d(a):
    return np.ascontiguousarray(a, dtype=np.float64)


class Mesh:
    """A tetrahedral mesh with its CAD classification, owned by the C++ library.

    X (nv, 3); vdim/vtag: entity of every node; T (nt, 4) with volume tags; tri (nf, 3) surface
    triangles with surface tags; seg (ns, 2) curve segments with curve tags. Indices are 0-based."""

    def __init__(self, X, vdim, vtag, T, tvol, tri, ftag, seg, stag, _handle=None):
        self._L = lib()
        self._proj = None
        if _handle is not None:
            self._h = _handle
            return
        X, vdim, vtag = _d(X).reshape(-1, 3), _i(vdim), _i(vtag)
        T, tvol, tri, ftag, seg, stag = (_i(T).reshape(-1, 4), _i(tvol), _i(tri).reshape(-1, 3), _i(ftag),
                                         _i(seg).reshape(-1, 2), _i(stag))
        I = ctypes.c_int
        self._h = self._L.hzm_create(len(X), _ptr(X, ctypes.c_double), _ptr(vdim, I), _ptr(vtag, I),
                                    len(T), _ptr(T, I), _ptr(tvol, I), len(tri), _ptr(tri, I), _ptr(ftag, I),
                                    len(seg), _ptr(seg, I), _ptr(stag, I))
        if not self._h:
            raise RuntimeError("hzmesh: " + self._L.hzm_last_error().decode())
        self.warning = self._L.hzm_last_error().decode()

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._L.hzm_free(h)
            self._h = None

    def clone(self) -> "Mesh":
        c = Mesh(None, None, None, None, None, None, None, None, None, _handle=self._L.hzm_clone(self._h))
        c._proj = self._proj
        return c

    # -- setup ------------------------------------------------------------------------------------
    def set_curved(self, dim: int, tag: int, curved: bool = True) -> None:
        self._L.hzm_set_curved(self._h, int(dim), int(tag), int(bool(curved)))

    def set_projector(self, fn) -> None:
        """fn(dim, tag, xyz) -> xyz on the entity (called for new nodes on curved entities)."""
        def cb(dim, tag, pin, pout):
            try:
                q = fn(dim, tag, (pin[0], pin[1], pin[2]))
                pout[0], pout[1], pout[2] = float(q[0]), float(q[1]), float(q[2])
            except Exception:  # noqa: BLE001 - keep the unprojected point
                pout[0], pout[1], pout[2] = pin[0], pin[1], pin[2]
        self._proj = PROJ_CB(cb)
        self._L.hzm_set_projector(self._h, self._proj)

    def set_size(self, h, zone) -> None:
        n = self.sizes()[0]
        h = _d(h)
        z = np.ascontiguousarray(np.asarray(zone).astype(np.uint8))
        if len(h) != n or len(z) != n:
            raise ValueError("size/zone arrays must have one entry per node")
        self._L.hzm_set_size(self._h, _ptr(h, ctypes.c_double), _ptr(z, ctypes.c_ubyte))

    def grade(self, h0, hcap, slope: float) -> np.ndarray:
        """min(hcap, min_z h0[z] + slope * edge-path distance to z) per node (multi-source Dijkstra)."""
        h0, hcap = _d(h0), _d(hcap)
        out = np.zeros(len(h0))
        self._L.hzm_grade(self._h, _ptr(h0, ctypes.c_double), _ptr(hcap, ctypes.c_double), float(slope),
                         _ptr(out, ctypes.c_double))
        return out

    # -- work -------------------------------------------------------------------------------------
    def refine(self, **params) -> dict:
        p = np.full(len(PARAMS), _DEFAULT)
        for k, v in params.items():
            p[PARAMS.index(k)] = float(v)
        st = np.zeros(len(STATS))
        self._L.hzm_refine(self._h, _ptr(p, ctypes.c_double), _ptr(st, ctypes.c_double))
        return dict(zip(STATS, st.tolist()))

    def check(self) -> tuple[int, str]:
        n = self._L.hzm_check(self._h)
        return n, self._L.hzm_last_error().decode()

    def sizes(self) -> tuple[int, int, int, int]:
        out = np.zeros(4, np.int32)
        self._L.hzm_sizes(self._h, _ptr(out, ctypes.c_int))
        return tuple(int(v) for v in out)

    def arrays(self) -> dict[str, np.ndarray]:
        nv, nt, nf, ns = self.sizes()
        a = {"X": np.zeros((nv, 3)), "vdim": np.zeros(nv, np.int32), "vtag": np.zeros(nv, np.int32),
             "T": np.zeros((nt, 4), np.int32), "tvol": np.zeros(nt, np.int32), "tri": np.zeros((nf, 3), np.int32),
             "ftag": np.zeros(nf, np.int32), "seg": np.zeros((ns, 2), np.int32), "stag": np.zeros(ns, np.int32)}
        I = ctypes.c_int
        self._L.hzm_get(self._h, _ptr(a["X"], ctypes.c_double), _ptr(a["vdim"], I), _ptr(a["vtag"], I),
                       _ptr(a["T"], I), _ptr(a["tvol"], I), _ptr(a["tri"], I), _ptr(a["ftag"], I),
                       _ptr(a["seg"], I), _ptr(a["stag"], I))
        return a

    def quality(self) -> tuple[np.ndarray, np.ndarray]:
        nt = self.sizes()[1]
        q, d = np.zeros(nt), np.zeros(nt)
        self._L.hzm_quality(self._h, _ptr(q, ctypes.c_double), _ptr(d, ctypes.c_double))
        return q, d


# -- gmsh exchange ----------------------------------------------------------------------------------
class GmshMesh:
    """The current gmsh mesh as hzmesh input, plus what is needed to write it back."""

    def __init__(self):
        import gmsh
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        tags = np.asarray(tags, np.int64)
        self.node_tags = tags
        self.X = np.asarray(xyz, float).reshape(-1, 3)
        imap = np.full(int(tags.max()) + 1, -1, np.int64)
        imap[tags] = np.arange(len(tags))
        self.imap = imap
        nv = len(tags)
        self.vdim = np.full(nv, -1, np.int32)
        self.vtag = np.zeros(nv, np.int32)
        for dim in (0, 1, 2, 3):
            for _, tag in gmsh.model.getEntities(dim):
                nt, _, _ = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=False, returnParametricCoord=False)
                if len(nt) == 0:
                    continue
                idx = imap[np.asarray(nt, np.int64)]
                free = self.vdim[idx] < 0
                self.vdim[idx[free]] = dim
                self.vtag[idx[free]] = tag
        if (self.vdim < 0).any():
            raise RuntimeError("hzmesh: mesh nodes without geometric entity")
        conn = {d: [] for d in (0, 1, 2, 3)}
        own = {d: [] for d in (0, 1, 2, 3)}
        for dim in (0, 1, 2, 3):
            for _, tag in gmsh.model.getEntities(dim):
                _, en = gmsh.model.mesh.getElementsByType(_ELEM[dim], tag)
                if len(en) == 0:
                    continue
                c = imap[np.asarray(en, np.int64)].reshape(-1, _NPE[dim])
                conn[dim].append(c)
                own[dim].append(np.full(len(c), tag, np.int32))
        cat = lambda d: (np.concatenate(conn[d]) if conn[d] else np.zeros((0, _NPE[d]), np.int64),  # noqa: E731
                         np.concatenate(own[d]) if own[d] else np.zeros(0, np.int32))
        self.pts, self.ptag = cat(0)
        self.seg, self.stag = cat(1)
        self.tri, self.ftag = cat(2)
        self.T, self.tvol = cat(3)
        self.proj_log: dict | None = None
        self.types = {}
        for dim in (1, 2):
            for _, tag in gmsh.model.getEntities(dim):
                try:
                    self.types[(dim, tag)] = gmsh.model.getType(dim, tag)
                except Exception:  # noqa: BLE001
                    self.types[(dim, tag)] = "Unknown"

    def curved_entities(self) -> list[tuple[int, int]]:
        """Entities with a CAD shape that new nodes must be projected on (not planes, lines or
        discrete/STL entities, which keep their facets)."""
        return [k for k, t in self.types.items() if t not in _FLAT and "iscrete" not in t and t != "Unknown"]

    def hzm(self, project: bool = True) -> Mesh:
        m = Mesh(self.X, self.vdim, self.vtag, self.T, self.tvol, self.tri, self.ftag, self.seg, self.stag)
        if project:
            curved = self.curved_entities()
            for dim, tag in curved:
                m.set_curved(dim, tag)
            if curved:
                import gmsh

                def proj(dim, tag, p):
                    q, _ = gmsh.model.getClosestPoint(dim, tag, list(p))
                    if self.proj_log is not None:     # for replaying a failed refinement without the CAD
                        self.proj_log[(dim, tag, *map(float, p))] = tuple(map(float, q[:3]))
                    return q
                m.set_projector(proj)
        return m

    def to_gmsh_arrays(self, a: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Arrays for local_refine.push_to_gmsh: old nodes keep their gmsh tags, new ones are appended."""
        nv0 = len(self.node_tags)
        nv = len(a["X"])
        tags = np.concatenate([self.node_tags, int(self.node_tags.max()) + 1 + np.arange(nv - nv0, dtype=np.int64)])
        out = {"node_tags": tags, "node_xyz": a["X"].copy(),
               "node_ent": np.stack([a["vdim"], a["vtag"]], 1).astype(np.int64)}
        keys, conns = [], []
        for dim, c, tg in ((0, self.pts, self.ptag), (1, a["seg"], a["stag"]), (2, a["tri"], a["ftag"]),
                           (3, a["T"], a["tvol"])):
            if len(tg) == 0:
                continue
            order = np.argsort(tg, kind="stable")
            tg_s = tg[order]
            c_s = tags[np.asarray(c, np.int64)[order]]
            ut, start = np.unique(tg_s, return_index=True)
            stop = np.append(start[1:], len(tg_s))
            for t, s0, s1 in zip(ut.tolist(), start.tolist(), stop.tolist()):
                keys.append((dim, t, s1 - s0))
                conns.append(c_s[s0:s1].ravel())
        out["elem_keys"] = np.array(keys, np.int64).reshape(-1, 3)
        out["elem_conn"] = np.concatenate(conns) if conns else np.zeros(0, np.int64)
        return out
