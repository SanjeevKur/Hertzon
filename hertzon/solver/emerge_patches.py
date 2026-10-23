"""Run-time fixes for EMerge 3 (applied once in the solver worker): thin conductor DoF numbering, port mode
solve retry, mesh retry for broken CAD faces, curved elements, and the grid accelerated size field of
Hertzon's re-mesh refinement. docs/EMERGE_INTEGRATION.md lists each one with its status upstream."""
from __future__ import annotations

import os

import numpy as np

_DONE = False


def native(name: str) -> bool:
    """True when the Hertzon replacement ``name`` is switched off for a comparison with plain EMerge:
    HERTZON_EMERGE_NATIVE=mt_pardiso,thin_conductor,sheet_error,eigs_retry,mesh_fallback"""
    return name in os.environ.get("HERTZON_EMERGE_NATIVE", "").replace(" ", "").split(",")


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
        if not native("mt_pardiso"):
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
    apply_curved()
    if not native("mesh_fallback"):
        apply_mesh_fallback()
    if not native("eigs_retry"):
        apply_eigs_retry()
    if not native("thin_conductor"):
        apply_thin_conductor()


# ------------------------------------------------------------------------------------------------
# Adaptive refinement size field
# ------------------------------------------------------------------------------------------------
# size(x) = min(lc, min_i max(s_i, A_i + c*|x - p_i|))  over *all* refinement points (they accumulate pass
# after pass) for every gmsh size query (emerge_compat.AMRPoints, the size field of EMerge 2.8.9). Evaluated
# with a uniform grid over the points and a pruned shell search; the kernel is compiled once (cached on disk).

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
