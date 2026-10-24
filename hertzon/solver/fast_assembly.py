"""Faster solve for EMerge 3 (same results, applied once in the solver worker): PARDISO factorizes the complex
symmetric system matrix as LDL^T (mtype 6) instead of an unsymmetric LU, and the PEC rows and columns are
removed in parallel instead of with scipy fancy indexing (single core). EMerge 3 builds the element
matrices and the CSC pattern in its compiled library, which replaced Hertzon's assembly for EMerge 2.8.9.
Set HERTZON_SLOW_ASSEMBLY=1 for EMerge's original code, HERTZON_PARDISO_STRUCT=1 for its LU only.
"""
from __future__ import annotations

import inspect
import os
import re
import textwrap
import threading

import numpy as np
from numba import njit, prange

_DONE = False
STATS: dict = {}


def _main_thread() -> bool:
    """numba's default (workqueue) threading layer aborts the process when parallel kernels run in two
    threads at once: EMerge's parallel sweeps solve in worker threads, which use the serial *_s copies."""
    return threading.current_thread() is threading.main_thread()


# -- PEC rows/columns removed in parallel --------------------------------------------------------------

@njit(cache=True, nogil=True, parallel=True)
def _sub_counts(indptr, indices, keep, newid):
    n = keep.shape[0]
    cnt = np.zeros(n + 1, dtype=np.int64)
    for j in prange(n):
        c = keep[j]
        m = 0
        for k in range(indptr[c], indptr[c + 1]):
            if newid[indices[k]] >= 0:
                m += 1
        cnt[j + 1] = m
    return cnt


@njit(cache=True, nogil=True)
def _sub_counts_s(indptr, indices, keep, newid):
    n = keep.shape[0]
    cnt = np.zeros(n + 1, dtype=np.int64)
    for j in range(n):
        c = keep[j]
        m = 0
        for k in range(indptr[c], indptr[c + 1]):
            if newid[indices[k]] >= 0:
                m += 1
        cnt[j + 1] = m
    return cnt


@njit(cache=True, nogil=True, parallel=True)
def _sub_fill(indptr, indices, data, keep, newid, outptr, outind, outdata):
    for j in prange(keep.shape[0]):
        c = keep[j]
        p = outptr[j]
        for k in range(indptr[c], indptr[c + 1]):
            r = newid[indices[k]]
            if r >= 0:
                outind[p] = r
                outdata[p] = data[k]
                p += 1


@njit(cache=True, nogil=True)
def _sub_fill_s(indptr, indices, data, keep, newid, outptr, outind, outdata):
    for j in range(keep.shape[0]):
        c = keep[j]
        p = outptr[j]
        for k in range(indptr[c], indptr[c + 1]):
            r = newid[indices[k]]
            if r >= 0:
                outind[p] = r
                outdata[p] = data[k]
                p += 1


def submatrix(A, ids):
    """A[:, ids][ids, :] for a CSC matrix and increasing ids."""
    from scipy.sparse import csc_matrix
    ids = np.asarray(ids)
    if (getattr(A, "format", "") != "csc" or A.shape[0] != A.shape[1] or ids.ndim != 1 or ids.size == 0
            or ids.dtype.kind not in "iu" or np.any(np.diff(ids) <= 0)):
        return A[:, ids][ids, :]
    keep = ids.astype(np.int64)
    newid = np.full(A.shape[0], -1, dtype=np.int64)
    newid[keep] = np.arange(keep.size, dtype=np.int64)
    par = _main_thread()
    cnt = (_sub_counts if par else _sub_counts_s)(A.indptr, A.indices, keep, newid)
    outptr = np.cumsum(cnt)
    idx_t = np.int32 if outptr[-1] < 2**31 - 1 else np.int64
    outptr = outptr.astype(idx_t)
    outind = np.empty(outptr[-1], dtype=idx_t)
    outdata = np.empty(outptr[-1], dtype=A.dtype)
    (_sub_fill if par else _sub_fill_s)(A.indptr, A.indices, A.data, keep, newid, outptr, outind, outdata)
    return csc_matrix((outdata, outind, outptr), shape=(keep.size, keep.size))


# -- complex symmetric PARDISO -----------------------------------------------------------------------------
# EMerge factorizes the (complex symmetric) system matrix as "structurally symmetric" (mtype 3, LU).
# mtype 6 (complex symmetric, LDL^T on the upper triangle) needs about half the work and memory; on a
# 350k DoF patch model the numeric factorization took 7.3 s instead of 11.8 s with the same residual.
# Used only when EMerge classifies the matrix as symmetric (not with periodic boundaries) and a random
# bilinear test confirms it; any PARDISO error repeats the solve the original way.

@njit(cache=True, nogil=True, parallel=True)
def _upper_counts(indptr, indices):
    n = indptr.shape[0] - 1
    cnt = np.zeros(n + 1, dtype=np.int64)
    for j in prange(n):
        m = 0
        diag = False
        for k in range(indptr[j], indptr[j + 1]):
            if indices[k] >= j:
                m += 1
                if indices[k] == j:
                    diag = True
        cnt[j + 1] = m if diag else -1
    return cnt


@njit(cache=True, nogil=True)
def _upper_counts_s(indptr, indices):
    n = indptr.shape[0] - 1
    cnt = np.zeros(n + 1, dtype=np.int64)
    for j in range(n):
        m = 0
        diag = False
        for k in range(indptr[j], indptr[j + 1]):
            if indices[k] >= j:
                m += 1
                if indices[k] == j:
                    diag = True
        cnt[j + 1] = m if diag else -1
    return cnt


@njit(cache=True, nogil=True, parallel=True)
def _upper_fill(indptr, indices, data, outptr, outind, outdata):
    for j in prange(indptr.shape[0] - 1):
        p = outptr[j]
        for k in range(indptr[j], indptr[j + 1]):
            if indices[k] >= j:
                outind[p] = indices[k]
                outdata[p] = data[k]
                p += 1


@njit(cache=True, nogil=True)
def _upper_fill_s(indptr, indices, data, outptr, outind, outdata):
    for j in range(indptr.shape[0] - 1):
        p = outptr[j]
        for k in range(indptr[j], indptr[j + 1]):
            if indices[k] >= j:
                outind[p] = indices[k]
                outdata[p] = data[k]
                p += 1


def upper_symmetric(A):
    """Upper triangle (CSR) of a complex symmetric CSC/CSR matrix, or None when that is not safe."""
    from scipy.sparse import csr_matrix
    if (getattr(A, "format", "") not in ("csc", "csr") or A.shape[0] != A.shape[1] or A.dtype != np.complex128
            or not A.has_sorted_indices or A.shape[0] < 2):
        return None
    rng = np.random.default_rng(12345)
    n = A.shape[0]
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    s1, s2 = y @ (A @ x), x @ (A @ y)
    if not abs(s1 - s2) <= 1e-9 * (abs(s1) + abs(s2)):
        return None
    # the arrays of a symmetric CSC matrix are its CSR arrays: keep column >= row in every row
    par = _main_thread()
    cnt = (_upper_counts if par else _upper_counts_s)(A.indptr, A.indices)
    if np.any(cnt[1:] < 0):           # PARDISO needs every diagonal entry in the pattern
        return None
    outptr = np.cumsum(cnt).astype(np.int32)
    outind = np.empty(outptr[-1], dtype=np.int32)
    outdata = np.empty(outptr[-1], dtype=np.complex128)
    (_upper_fill if par else _upper_fill_s)(A.indptr, A.indices, A.data, outptr, outind, outdata)
    U = csr_matrix((outdata, outind, outptr), shape=A.shape)
    return U


def _pardiso_configure(orig):
    def _configure(self, A):
        orig(self, A)
        mt = getattr(self, "_hzn_mtype", None)
        if mt is not None and self.complex:
            self.MATRIX_TYPE = mt
    _configure._hertzon_fast = True
    return _configure


def _pardiso_solve(orig):
    def solve(self, A, b, precon, id: int = -1):  # noqa: A002
        from emerge._emerge.solve_interfaces.pardiso_interface import PARDISOMType
        U = None
        if getattr(self, "_hzn_sym", False) and os.environ.get("HERTZON_PARDISO_STRUCT") != "1":
            try:
                U = upper_symmetric(A)
            except Exception:  # noqa: BLE001
                U = None
        mode = "sym" if U is not None else "struct"
        if getattr(self, "_hzn_mode", mode) != mode:     # the analysis belongs to the other matrix type
            self.fact_symb = False
        self._hzn_mode = mode
        if U is not None:
            self.solver._hzn_mtype = PARDISOMType.COMP_SYM
            try:
                out = orig(self, U, b, precon, id)
                _record_factor(self.solver, A.shape[0], "complex symmetric (LDLT)")
                return out
            except Exception as ex:  # noqa: BLE001
                from loguru import logger
                logger.warning(f"Symmetric PARDISO failed ({ex}); repeating with the unsymmetric factorization")
                try:
                    self.solver.clear_memory()
                except Exception:  # noqa: BLE001
                    pass
                self.fact_symb = False
                self._hzn_mode = "struct"
            finally:
                self.solver._hzn_mtype = None
        out = orig(self, A, b, precon, id)
        _record_factor(self.solver, A.shape[0], "unsymmetric (LU)")
        return out
    solve._hertzon_fast = True
    return solve


# memory of the last PARDISO factorization (iparm 15-17, in kB), for the memory check of parallel sweeps
FACTOR: dict = {}


def _record_factor(pardiso, n: int, kind: str) -> None:
    try:
        ip = pardiso.IPARM
        kb = max(int(ip[14]), int(ip[15]) + int(ip[16]))
        if kb > 0:
            FACTOR.update(bytes=kb * 1024.0, unknowns=int(n), kind=kind)
    except Exception:  # noqa: BLE001
        pass


def _set_symmetry(self, complex_symmetric: bool) -> None:
    self._hzn_sym = bool(complex_symmetric)


# -- source patches --------------------------------------------------------------------------------------


def _patched(fn, subs):
    src = textwrap.dedent(inspect.getsource(fn))
    n = 0
    for rx, rep in subs:
        src, k = rx.subn(rep, src)
        n += k
    if not n:
        return None, 0
    mod = inspect.getmodule(fn)
    ns = dict(vars(mod))
    ns["_hzn_fa"] = __import__(__name__, fromlist=["x"])
    code = compile(src, inspect.getsourcefile(fn) or "<hertzon>", "exec")
    exec(code, ns)  # noqa: S102
    new = ns[fn.__name__]
    new.__qualname__ = fn.__qualname__
    new._hertzon_fast = True
    return new, n


def apply() -> None:
    """Install the faster code (before emerge_patches.apply_curved wraps the assembly)."""
    global _DONE
    if _DONE or os.environ.get("HERTZON_SLOW_ASSEMBLY") == "1":
        return
    _DONE = True
    from emerge._emerge import solver
    fn = solver.SolveRoutine.solve
    try:
        new, n = _patched(fn, [(re.compile(r"Asel = A\[:, solve_ids\]\[solve_ids, :\]"),
                                "Asel = _hzn_fa.submatrix(A, solve_ids)")])
    except Exception:  # noqa: BLE001
        new, n = None, 0
    if new is not None:
        solver.SolveRoutine.solve = new
    STATS["solve"] = n
    if os.environ.get("HERTZON_PARDISO_STRUCT") != "1":
        try:
            from emerge._emerge.solve_interfaces import pardiso_interface as PI
            if not getattr(PI.PardisoInterface._configure, "_hertzon_fast", False):
                PI.PardisoInterface._configure = _pardiso_configure(PI.PardisoInterface._configure)
            if not getattr(solver.SolverPardiso.solve, "_hertzon_fast", False):
                solver.SolverPardiso.solve = _pardiso_solve(solver.SolverPardiso.solve)
                solver.SolverPardiso.set_symmetry = _set_symmetry
            STATS["pardiso"] = "complex symmetric"
        except Exception:  # noqa: BLE001
            STATS["pardiso"] = "unchanged"


