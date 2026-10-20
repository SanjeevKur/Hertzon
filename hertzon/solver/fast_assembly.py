"""Faster FEM matrix assembly for EMerge 2.8.9 (same matrices, applied once in the solver worker).

Between "Using ... basis functions" and "Starting solve" EMerge spends about 13 s per pass on a 250k
element model, mostly outside its parallel element kernel:

- Nedelec2() fills the boundary index arrays in a Python loop over all triangles (empty_tri_rowcol);
- the CSC pattern of the system matrix is built from 400 COO entries per tetrahedron with a serial
  counting pass over all of them (~100M entries);
- E - k0^2 B is formed as two full size temporary arrays before it is scattered into the CSC matrix;
- the boundary (Robin) matrix is added with a sparse matrix addition that copies the whole matrix;
- conductor tetrahedra and the free degrees of freedom are found with Python loops;
- the PEC rows and columns are removed with scipy fancy indexing (single core);
- PARDISO factorizes the complex symmetric matrix with an unsymmetric LU (see below).

The replacements below compute the same matrices: the element matrices still come from EMerge's
kernel, the CSC pattern is built from the tetrahedron to DoF map in the order EMerge's own builder
uses (identical permutation), and sums are only reordered within rounding (~1e-16).
Set HERTZON_SLOW_ASSEMBLY=1 to use EMerge's original code.
"""
from __future__ import annotations

import inspect
import os
import re
import textwrap
import threading
import time

import numpy as np
from numba import njit, prange

_DONE = False
STATS: dict = {}


# -- element matrices without the COO row/column arrays ------------------------------------------------

def _kernels():
    from emerge._emerge.physics.microwave.assembly.curlcurl import (
        local_tet_to_edgeid, local_tet_to_triid, ned2_tet_stiff_mass, parse_dofcode)
    from emerge._emerge.mth.optimized import matinv
    return local_tet_to_edgeid, local_tet_to_triid, ned2_tet_stiff_mass, parse_dofcode, matinv


_lte, _ltt, _ned2, _parse, _matinv = _kernels()


def _main_thread() -> bool:
    """numba's default (workqueue) threading layer aborts the process when parallel kernels run in two
    threads at once: EMerge's parallel sweeps solve in worker threads, which use the serial *_s copies."""
    return threading.current_thread() is threading.main_thread()


@njit(cache=True, nogil=True, parallel=True)
def _element_data(nodes, tets, tris, edges, tet_to_edge, tet_to_tri, ur, er, tetids, dofcodes):
    """dataE, dataB of EMerge's _matrix_builder (same layout), without its rows/cols arrays."""
    ndof = dofcodes.shape[0]
    nn = ndof * ndof
    n = tetids.shape[0]
    dataE = np.empty(n * nn, dtype=np.complex128)
    dataB = np.empty(n * nn, dtype=np.complex128)
    typearry, indexarry = _parse(dofcodes)
    for iassy in prange(n):
        itet = tetids[iassy]
        p = iassy * nn
        local_tri_map = _ltt(tet_to_tri, tets, tris, itet)
        local_edge_map = _lte(tet_to_edge, tets, edges, itet)
        Esub, Bsub = _ned2(nodes[:, tets[:, itet]], local_edge_map, local_tri_map, _matinv(ur[:, :, itet]),
                           er[:, :, itet], dofcodes, typearry, indexarry)
        for a in range(ndof):
            for b in range(ndof):
                dataE[p + a * ndof + b] = Esub[a, b]
                dataB[p + a * ndof + b] = Bsub[a, b]
    return dataE, dataB


@njit(cache=True, nogil=True, parallel=True)
def _csc_pattern(tet_to_field, tetids, N):
    """precompute_csc_pattern(rows, cols, N) of EMerge for the COO layout of _matrix_builder.

    Entry k = iassy*ndof^2 + a*ndof + b has row t2f[a, tet] and column t2f[b, tet]. Same CSC pattern;
    entries of one row are kept in COO order, so only the order of summing duplicates differs."""
    ndof = tet_to_field.shape[0]
    nn = ndof * ndof
    n = tetids.shape[0]
    # DoF -> (element, local index), elements in increasing order
    cnt = np.zeros(N + 1, dtype=np.int64)
    for i in range(n):
        t = tetids[i]
        for b in range(ndof):
            cnt[tet_to_field[b, t] + 1] += 1
    inv_off = np.zeros(N + 1, dtype=np.int64)
    for c in range(N):
        inv_off[c + 1] = inv_off[c] + cnt[c + 1]
    cursor = inv_off[:-1].copy()
    inv_el = np.empty(inv_off[N], dtype=np.int64)
    inv_b = np.empty(inv_off[N], dtype=np.int64)
    for i in range(n):
        t = tetids[i]
        for b in range(ndof):
            c = tet_to_field[b, t]
            inv_el[cursor[c]] = i
            inv_b[cursor[c]] = b
            cursor[c] += 1
    col_offsets = np.zeros(N + 1, dtype=np.int64)
    for c in range(N):
        col_offsets[c + 1] = col_offsets[c] + (inv_off[c + 1] - inv_off[c]) * ndof

    nnz_coo = n * nn
    perm = np.empty(nnz_coo, dtype=np.int64)
    unique_counts = np.zeros(N, dtype=np.int64)
    maxcount = 0
    for c in range(N):
        maxcount = max(maxcount, col_offsets[c + 1] - col_offsets[c])
    # one buffer per chunk of columns: allocating per column does not scale with many threads.
    # key = row * nnz_coo + k sorts by row, then by COO index (duplicates are summed in COO order)
    nchunk = min(N, 1024)
    for ch in prange(nchunk):
        keys = np.empty(maxcount, dtype=np.int64)
        for c in range(ch * N // nchunk, (ch + 1) * N // nchunk):
            start = col_offsets[c]
            count = col_offsets[c + 1] - start
            if count == 0:
                continue
            j = 0
            for q in range(inv_off[c], inv_off[c + 1]):
                i = inv_el[q]
                t = tetids[i]
                p = i * nn + inv_b[q]
                for a in range(ndof):
                    keys[j] = tet_to_field[a, t] * nnz_coo + p + a * ndof
                    j += 1
            kk = keys[:count]
            kk.sort()
            uniques = 1
            prev = kk[0] // nnz_coo
            perm[start] = kk[0] - prev * nnz_coo
            for j in range(1, count):
                r = kk[j] // nnz_coo
                if r != prev:
                    uniques += 1
                    prev = r
                perm[start + j] = kk[j] - r * nnz_coo
            unique_counts[c] = uniques

    indptr = np.zeros(N + 1, dtype=np.int64)
    for c in range(N):
        indptr[c + 1] = indptr[c] + unique_counts[c]
    indices = np.empty(indptr[N], dtype=np.int64)
    target = np.empty(nnz_coo, dtype=np.int64)
    for c in prange(N):
        start = col_offsets[c]
        count = col_offsets[c + 1] - start
        if count == 0:
            continue
        pos = indptr[c]
        k = perm[start]
        prev = tet_to_field[(k % nn) // ndof, tetids[k // nn]]
        indices[pos] = prev
        target[start] = pos
        for j in range(1, count):
            k = perm[start + j]
            r = tet_to_field[(k % nn) // ndof, tetids[k // nn]]
            if r != prev:
                pos += 1
                indices[pos] = r
                prev = r
            target[start + j] = pos
    return indptr, indices, target, col_offsets, perm


def tet_mass_stiffness_matrices(field, er, ur, conductor_tets, cscmap=None):
    """curlcurl.tet_mass_stiffness_matrices with the same result."""
    from emerge._emerge.mth.csc_cast import CSCMapping
    mesh = field.mesh
    tet_to_field = np.ascontiguousarray(field.get_tet_to_field(), dtype=np.int64)
    tet_assy = np.arange(mesh.tets.shape[1], dtype=np.int64)
    if conductor_tets.shape[0] > 0:
        tet_assy = np.delete(tet_assy, conductor_tets)
    i64 = lambda a: np.asarray(a, dtype=np.int64)  # noqa: E731
    t0 = time.perf_counter()
    dataE, dataB = _element_data(np.asarray(mesh.nodes, np.float64), i64(mesh.tets), i64(mesh.tris),
                                 i64(mesh.edges), i64(mesh.tet_to_edge), i64(mesh.tet_to_tri), ur, er, tet_assy,
                                 i64(field.dofcodes3d))
    t1 = time.perf_counter()
    if cscmap is None:
        cscmap = CSCMapping(*_csc_pattern(tet_to_field, tet_assy, int(field.n_field)), int(field.n_field))
    t2 = time.perf_counter()
    try:      # once per mesh (a discrete sweep assembles every frequency)
        if STATS.get("logged_mesh") == (id(mesh), tet_assy.shape[0]):
            raise LookupError
        STATS["logged_mesh"] = (id(mesh), tet_assy.shape[0])
        from loguru import logger
        logger.info(f"Element matrices: {tet_assy.shape[0]} tetrahedra in {t1 - t0:.2f} s"
                    + (f", sparsity pattern {t2 - t1:.2f} s" if t2 - t1 > 0.005 else ""))
    except Exception:  # noqa: BLE001
        pass
    return dataE, dataB, cscmap


# -- E - k0^2 B scattered into CSC without temporaries -------------------------------------------------

@njit(cache=True, nogil=True, parallel=True)
def _scatter_combined(E, B, s, perm, target, col_offsets, out):
    N = col_offsets.shape[0] - 1
    for i in prange(N):
        for k in range(col_offsets[i], col_offsets[i + 1]):
            q = perm[k]
            out[target[k]] += E[q] - B[q] * s
    return out



@njit(cache=True, nogil=True)
def _scatter_combined_s(E, B, s, perm, target, col_offsets, out):
    N = col_offsets.shape[0] - 1
    for i in range(N):
        for k in range(col_offsets[i], col_offsets[i + 1]):
            q = perm[k]
            out[target[k]] += E[q] - B[q] * s
    return out


def combine(cscmap, E, B, s):
    """cscmap.to_csc(E - B * s)"""
    from scipy.sparse import csc_matrix
    if not (E.dtype == B.dtype == np.complex128 and E.shape == B.shape == cscmap.perm.shape):
        return cscmap.to_csc(E - B * s)
    out = np.zeros(cscmap.nnz, dtype=np.complex128)
    scatter = _scatter_combined if _main_thread() else _scatter_combined_s
    scatter(E, B, complex(s), cscmap.perm, cscmap.target_csc, cscmap.col_offsets, out)
    if not hasattr(cscmap, "_hzn_idx"):   # scipy converts the int64 index arrays to int32 for every matrix
        idx_t = np.int32 if cscmap.nnz < 2**31 - 1 else np.int64
        cscmap._hzn_idx = (cscmap.indices.astype(idx_t), cscmap.indptr.astype(idx_t))
    ind, ptr = cscmap._hzn_idx      # copies: the matrix must not share its index arrays with the cache
    return csc_matrix((out, ind.copy(), ptr.copy()), shape=(cscmap.N, cscmap.N))


# -- boundary matrix added into the existing pattern ---------------------------------------------------

@njit(cache=True, nogil=True)
def _positions(indptr, indices, r, c):
    pos = np.empty(r.shape[0], dtype=np.int64)
    for k in range(r.shape[0]):
        lo = indptr[c[k]]
        hi = indptr[c[k] + 1]
        j = lo + np.searchsorted(indices[lo:hi], r[k])
        if j >= hi or indices[j] != r[k]:
            return pos, k
        pos[k] = j
    return pos, -1


@njit(cache=True, nogil=True)
def _add_at(data, pos, v):
    for k in range(pos.shape[0]):
        data[pos[k]] += v[k]


def add_boundary(K, field, data, rowcol=None):
    """K + field.generate_csc(data, rowcol); in place when the entries are in K's pattern (always for
    triangle DoFs, which belong to tetrahedra)."""
    rows, cols = (field._rows, field._cols) if rowcol is None else rowcol
    ids = np.flatnonzero(data)
    if (getattr(K, "format", "") != "csc" or K.dtype != np.complex128 or not np.iscomplexobj(data)
            or not K.has_sorted_indices):
        return K + field.generate_csc(data, rowcol)
    r = np.asarray(rows[ids], np.int64)
    c = np.asarray(cols[ids], np.int64)
    pos, miss = _positions(K.indptr, K.indices, r, c)
    if miss >= 0:
        return K + field.generate_csc(data, rowcol)
    _add_at(K.data, pos, np.asarray(data[ids], np.complex128))
    return K


# -- small Python loops ----------------------------------------------------------------------------------

def conductor_tets(cond, peclim, surfimplim):
    c = np.asarray(cond[0, 0, :])
    return np.flatnonzero((c > peclim) | (c > surfimplim)).astype(np.int64)


def free_ids(NF, pec_ids):
    keep = np.ones(int(NF), dtype=bool)
    if pec_ids:
        keep[np.fromiter((int(i) for i in pec_ids), dtype=np.int64, count=len(pec_ids))] = False
    return np.flatnonzero(keep).astype(np.int64)


def empty_tri_rowcol(self, other_side: bool = False):
    """FEMBasis.empty_tri_rowcol: block (a, b) of triangle t is (t2f[a, t], t2f[b, t])."""
    N = self.n_tri_dofs
    t2f = self.tri_to_field_os if other_side else self.tri_to_field
    T = np.ascontiguousarray(np.asarray(t2f)[:N, :self.n_tris].T, dtype=np.int64)
    return np.repeat(T, N, axis=1).ravel(), np.tile(T, (1, N)).ravel()


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

_ASSEMBLER_SUBS = [
    (re.compile(r"conductor_tets = \[\]\n\s*for itet in range\(field\.n_tets\):\n\s*if \(\n"
                r"\s*cond\[0, 0, itet\] > self\.settings\.mw_3d_peclim\n"
                r"\s*or cond\[0, 0, itet\] > self\.settings\.mw_3d_surfimplim\n"
                r"\s*\):\n\s*conductor_tets\.append\(itet\)\n"),
     "conductor_tets = _hzn_fa.conductor_tets(cond, self.settings.mw_3d_peclim, "
     "self.settings.mw_3d_surfimplim)\n"),
    (re.compile(r"self\.cached_cscmap\.to_csc\(\s*(\w+) - (\w+) \* \(K0\*\*2\)\s*\)"),
     r"_hzn_fa.combine(self.cached_cscmap, \1, \2, K0**2)"),
    (re.compile(r"(\w+) \+= field\.generate_csc\((B_matrix_robin(?:_2)?)((?:, \(rows, cols\))?)\)"),
     r"\1 = _hzn_fa.add_boundary(\1, field, \2\3)"),
    (re.compile(r"np\.array\(\[i for i in range\(NF\) if i not in pec_ids\]\)"),
     "_hzn_fa.free_ids(NF, pec_ids)"),
]


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
    from emerge._emerge.elements import femdata
    from emerge._emerge.physics.microwave.assembly import assembler, curlcurl
    from emerge._emerge import solver
    femdata.FEMBasis.empty_tri_rowcol = empty_tri_rowcol
    curlcurl.tet_mass_stiffness_matrices = tet_mass_stiffness_matrices
    for name in ("assemble_freq_matrix", "assemble_scattering_matrix", "assemble_eig_matrix"):
        fn = getattr(assembler.Assembler, name, None)
        if fn is None or getattr(fn, "_hertzon_fast", False):
            continue
        try:
            new, n = _patched(fn, _ASSEMBLER_SUBS)
        except Exception:  # noqa: BLE001
            new, n = None, 0
        if new is not None:
            setattr(assembler.Assembler, name, new)
        STATS[name] = n
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
