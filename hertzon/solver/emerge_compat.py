"""Differences between the supported EMerge versions (2.8.9 and 3.0).

EMerge 3 moved its numba kernels to a compiled Rust library (emr), replaced the gmsh re-mesh refinement
(AMRPoints size field) by its own local refinement and dropped a few helpers Hertzon used. The pieces Hertzon
still needs are provided here with the 2.8.9 behaviour (EMerge is GPL-2.0-or-later, like Hertzon):

- select_refinement_indices: Doerfler marking of the error estimate;
- tet_to_node: smallest element size per node of the marked tetrahedra;
- AMRPoints: the size field of the re-mesh refinement ("Remesh" methods), evaluated with Hertzon's grid
  accelerated kernel (emerge_patches.amr_size_at).
"""
from __future__ import annotations

import numpy as np


def emerge_version() -> tuple[int, ...]:
    try:
        from importlib.metadata import version
        v = version("emerge")
    except Exception:  # noqa: BLE001
        import emerge
        v = getattr(emerge, "__version__", "0")
    out = []
    for part in v.split("."):
        num = "".join(ch for ch in part if ch.isdigit()) if part[:1].isdigit() else ""
        if not num:
            break
        out.append(int(num))
        if not part.isdigit():          # 0a25 and the like: pre-release, stop here
            break
    return tuple(out) or (0,)


EMERGE3 = emerge_version() >= (3,)


def select_refinement_indices(errors: np.ndarray, refine: float) -> np.ndarray:
    """Doerfler marking: the fewest elements whose squared errors add up to ``refine`` of the total, sorted
    by decreasing error (same result as EMerge 2.8.9)."""
    ind = np.abs(np.asarray(errors, float)) ** 2
    order = np.argsort(ind)[::-1]
    if not len(order):
        return np.zeros(0, np.int64)
    csum = np.cumsum(ind[order])
    k = int(np.searchsorted(csum, refine * float(np.sum(ind)), side="left")) + 1
    return order[:min(k, len(order))]


def tet_to_node(nodes, tets, sizes, included):
    """Coordinates (3, K) and smallest size (K,) of the nodes of the included tetrahedra, in node order."""
    nodes = np.asarray(nodes, float)
    tets = np.asarray(tets, np.int64)
    sizes = np.asarray(sizes, float)
    inc = np.asarray(included, bool)
    node_sizes = np.full(nodes.shape[1], np.inf)
    t = tets[:, inc]
    np.minimum.at(node_sizes, t.ravel(), np.broadcast_to(sizes[inc], t.shape).ravel())
    hit = np.isfinite(node_sizes)
    return np.ascontiguousarray(nodes[:, hit]), node_sizes[hit]


class AMRPoints:
    """Refinement points and the gmsh size callback of the re-mesh refinement (EMerge 2.8.9's AMRPoints)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._amr_coords = None
        self._amr_sizes = None
        self._amr_ratios = None
        self._amr_new = None

    @property
    def npts(self) -> int:
        return 0 if self._amr_coords is None else self._amr_coords.shape[1]

    def reduce_set(self, ids: np.ndarray) -> None:
        self._amr_coords = self._amr_coords[:, ids]
        self._amr_sizes = self._amr_sizes[ids]
        self._amr_ratios = self._amr_ratios[ids]
        self._amr_new = self._amr_new[ids]

    def add_refinement_points(self, coords: np.ndarray, sizes: np.ndarray, ratios: np.ndarray) -> None:
        if self._amr_coords is None:
            self._amr_coords, self._amr_sizes, self._amr_ratios = coords, sizes, ratios
            self._amr_new = np.ones_like(sizes)
            return
        self._amr_coords = np.hstack((self._amr_coords, coords))
        self._amr_sizes = np.hstack((self._amr_sizes, sizes))
        self._amr_ratios = np.hstack((self._amr_ratios, ratios))
        self._amr_new = np.hstack((0.0 * self._amr_new, np.ones_like(sizes)))

    def set_refinement_function(self, gr: float = 1.5, _qf: float = 1.0) -> None:
        import gmsh
        from .emerge_patches import _kernel, _size_args
        args = _size_args(self, gr, _qf)
        if args is None:
            gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: lc)
            return
        q = _kernel()

        def func(dim, tag, x, y, z, lc):
            return q(x, y, z, lc, *args)
        self._fast_cb = func              # keep a reference while gmsh uses it
        # gmsh worker threads would all call back into Python and serialise on the GIL
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.mesh.setSizeCallback(func)

    def set_ratio(self, ratio: float) -> float:
        newids = self._amr_new == 1
        self._amr_ratios[newids] = ratio
        return self._amr_ratios[newids][0]


def amr(m) -> AMRPoints:
    """The refinement points of model ``m`` (EMerge 3 has none of its own: Hertzon's are attached)."""
    obj = getattr(m.mesher, "_amrobj", None)
    if obj is None:
        obj = m.mesher._amrobj = AMRPoints()
    return obj
