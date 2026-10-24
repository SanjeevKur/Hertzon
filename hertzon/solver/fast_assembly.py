"""Faster solve for EMerge 3 (same results, applied once in the solver worker): PARDISO factorizes the complex
symmetric system matrix as LDL^T (mtype 6) instead of an unsymmetric LU. EMerge removes the PEC rows and
columns in parallel itself since FennisRobert/EMerge#78, and builds the element matrices and the CSC pattern
in its compiled library. Set HERTZON_SLOW_ASSEMBLY=1 or HERTZON_PARDISO_STRUCT=1 for EMerge's LU.
"""
from __future__ import annotations

import os
import threading

import numpy as np
from numba import njit, prange

_DONE = False
STATS: dict = {}


def _main_thread() -> bool:
    """numba's default (workqueue) threading layer aborts the process when parallel kernels run in two
    threads at once: EMerge's parallel sweeps solve in worker threads, which use the serial *_s copies."""
    return threading.current_thread() is threading.main_thread()


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


def apply() -> None:
    """Install the faster PARDISO factorization."""
    global _DONE
    if _DONE or os.environ.get("HERTZON_SLOW_ASSEMBLY") == "1":
        return
    _DONE = True
    from emerge._emerge import solver
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


