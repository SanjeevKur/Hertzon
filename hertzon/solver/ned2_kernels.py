"""Nedelec-2 basis function kernels of EMerge 2.8.9 (numba), for Hertzon's curved element matrices.

EMerge 3 evaluates its basis functions in its compiled library, which numba code cannot call. The curved
tetrahedra (curvilinear.py) need them point-wise inside a numba kernel, so the numba versions of EMerge
2.8.9 are kept here unchanged: compiled/ccbf.py (basis functions), and matinv, local_mapping,
local_tet_to_triid, local_tet_to_edgeid and tet_coefficients. The DoF codes and the local edge/face
order are the same in EMerge 3 (tests/test_core.py compares the straight element matrices). The explicit
numba signatures were removed, so nothing is compiled on import, only what a curved model calls.

Copied from EMerge 2.8.9, Copyright (C) 2025 Robert Fennis, GPL-2.0-or-later.
"""
from numba import njit
import numpy as np

# This will be an overview of the complete order basis functions
# A binary number will be used to represent each basis function. There wil be 4 codes(bits) for the type
# 000xxxxx = Nodal basis function
# 001xxxxx = Edge bassis function
# 010xxxxx = Face basis function
# 100xxxxx = Volume basis function

# This leaves 5 bits for the basis function number (16 total should be enough.)
NODE_TYPE = 0b00000000
EDGE_TYPE = 0b01000000
FACE_TYPE = 0b10000000
VOLU_TYPE = 0b11000000

MASK_TYPE = 0b11000000
MASK_INDEX = 0b00111111

@njit(cache=True)
def parse_dofcode(dofcodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns three arrays that contain information regarding the type of basis function and edge indices.

    Args:
        dofcodes (np.ndarray): The list of all DoF codes sorted

    Returns:
        tuple[int, int, np.ndarray, np.ndarray]: The type array (0=edge, 1=face), index array(0-6 etc.), 
    """
    typearray = np.empty_like(dofcodes, dtype=np.int64)
    indexarray = np.empty_like(dofcodes, dtype=np.int64)
    i = 0
    ne = np.zeros((2**6,), dtype=np.int64)
    nf = np.zeros((2**6,), dtype=np.int64)
    for code in dofcodes:
        idofcode = code & 0b00111111
        if code & 0b11000000==64:
            typearray[i] = 0
            indexarray[i] = ne[idofcode]
            ne[idofcode] += 1
        else:
            typearray[i] = 1
            indexarray[i] = nf[idofcode]
            nf[idofcode] += 1
        i += 1
    return typearray, indexarray
        

@njit(cache=True)
def get_type_index(number: int):
    index = number & MASK_INDEX
    bftype = number & MASK_TYPE
    return bftype, index



############################################################
#                      FUNCTIONS HERE                     #
############################################################
@njit(cache=True, nogil=True)
def _ne0_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ai + bi*xs + ci*ys
    bx = -_t0*bi + _t1*bj
    by = -_t0*ci + _t1*cj
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _ne1_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ai + bi*xs + ci*ys
    bx = _t0*bi + _t1*bj
    by = _t0*ci + _t1*cj
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _ne2_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = bj*xs
    _t1 = cj*ys
    _t2 = ai + bi*xs + ci*ys
    _t3 = -_t0 - _t1 + _t2 - aj
    _t4 = _t0 + _t1 + aj
    bx = -_t3*(-_t2*bj + _t4*bi)
    by = -_t3*(-_t2*cj + _t4*ci)
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _ne3_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = bj*xs
    _t1 = cj*ys
    _t2 = _t0 + _t1 + aj
    _t3 = ai + bi*xs + ci*ys
    _t4 = _t2*_t3
    _t5 = -_t0 - _t1 + _t3 - aj
    _t6 = _t2*_t5
    _t7 = _t3*_t5
    bx = _t4*(bi - bj) + _t6*bi + _t7*bj
    by = _t4*(ci - cj) + _t6*ci + _t7*cj
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _ne4_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ai + bi*xs + ci*ys
    _t1 = aj + bj*xs + cj*ys
    bx = -_t0*(-_t0*bj + _t1*bi)
    by = -_t0*(-_t0*cj + _t1*ci)
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _ne5_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ai + bi*xs + ci*ys
    bx = -_t0*(_t0*bi - _t1*bj)
    by = -_t0*(_t0*ci - _t1*cj)
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _curl_ne0_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    out[:] = (2*bi*cj - 2*bj*ci*np.ones_like(coords[0,:]))

@njit(cache=True, nogil=True)
def _curl_ne1_2d(coeff, coords, i, j, k, out):
    out[:] = 0.0

@njit(cache=True, nogil=True)
def _curl_ne2_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = bj*xs
    _t1 = cj*ys
    _t2 = _t0 + _t1 + aj
    _t3 = ai + bi*xs + ci*ys
    out[:] = (-(bi - bj)*(_t2*ci - _t3*cj) + (ci - cj)*(_t2*bi - _t3*bj) + 2*(bi*cj - bj*ci)*(-_t0 - _t1 + _t3 - aj))

@njit(cache=True, nogil=True)
def _curl_ne3_2d(coeff, coords, i, j, k, out):
    out[:] = 0.0

@njit(cache=True, nogil=True)
def _curl_ne4_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ai + bi*xs + ci*ys
    _t1 = aj + bj*xs + cj*ys
    out[:] = (2*_t0*(bi*cj - bj*ci) - bi*(-_t0*cj + _t1*ci) + ci*(-_t0*bj + _t1*bi))

@njit(cache=True, nogil=True)
def _curl_ne5_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ai + bi*xs + ci*ys
    out[:] = (2*_t0*(bi*cj - bj*ci) - bj*(_t0*ci - _t1*cj) + cj*(_t0*bi - _t1*bj))

@njit(cache=True, nogil=True)
def _div_ne0_2d(coeff, coords, i, j, k, out):
    out[:] = 0.0

@njit(cache=True, nogil=True)
def _div_ne1_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    out[:] = (2*bi*bj + 2*ci*cj*np.ones_like(coords[0,:]))

@njit(cache=True, nogil=True)
def _div_ne2_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ai + bi*xs + ci*ys
    out[:] = (-(bi - bj)*(_t0*bi - _t1*bj) - (ci - cj)*(_t0*ci - _t1*cj))

@njit(cache=True, nogil=True)
def _div_ne3_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = bi - bj
    _t1 = bj*xs
    _t2 = cj*ys
    _t3 = _t1 + _t2 + aj
    _t4 = ai + bi*xs + ci*ys
    _t5 = ci - cj
    _t6 = -_t1 - _t2 + _t4 - aj
    out[:] = (2*_t0*_t3*bi + 2*_t0*_t4*bj + 2*_t3*_t5*ci + 2*_t4*_t5*cj + 2*_t6*bi*bj + 2*_t6*ci*cj)

@njit(cache=True, nogil=True)
def _div_ne4_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ai + bi*xs + ci*ys
    out[:] = (-bi*(_t0*bi - _t1*bj) - ci*(_t0*ci - _t1*cj))

@njit(cache=True, nogil=True)
def _div_ne5_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ai + bi*xs + ci*ys
    out[:] = (-bj*(_t0*bi - _t1*bj) - cj*(_t0*ci - _t1*cj))

@njit(cache=True, nogil=True)
def _nf0_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ai + bi*xs + ci*ys
    _t1 = ak + bk*xs + ck*ys
    _t2 = aj + bj*xs + cj*ys
    bx = -_t0*(_t1*bj - _t2*bk)
    by = -_t0*(_t1*cj - _t2*ck)
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _nf1_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ak + bk*xs + ck*ys
    _t2 = ai + bi*xs + ci*ys
    bx = -_t0*(_t1*bi - _t2*bk)
    by = -_t0*(_t1*ci - _t2*ck)
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _nf2_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ak + bk*xs + ck*ys
    _t1 = aj + bj*xs + cj*ys
    _t2 = ai + bi*xs + ci*ys
    bx = _t0*(_t1*bi - _t2*bj)
    by = _t0*(_t1*ci - _t2*cj)
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _nf3_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ak + bk*xs + ck*ys
    _t2 = _t0*_t1
    _t3 = ai + bi*xs + ci*ys
    _t4 = _t1*_t3
    _t5 = 2*_t0*_t3
    bx = _t2*bi + _t4*bj - _t5*bk
    by = _t2*ci + _t4*cj - _t5*ck
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _nf4_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ai + bi*xs + ci*ys
    _t1 = ak + bk*xs + ck*ys
    _t2 = _t0*_t1
    _t3 = aj + bj*xs + cj*ys
    _t4 = _t0*_t3
    _t5 = 2*_t1*_t3
    bx = _t2*bj + _t4*bk - _t5*bi
    by = _t2*cj + _t4*ck - _t5*ci
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _nf5_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ak + bk*xs + ck*ys
    _t2 = _t0*_t1
    _t3 = ai + bi*xs + ci*ys
    _t4 = _t0*_t3
    _t5 = 2*_t1*_t3
    bx = _t2*bi + _t4*bk - _t5*bj
    by = _t2*ci + _t4*ck - _t5*cj
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _nf6_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ak + bk*xs + ck*ys
    _t2 = _t0*_t1
    _t3 = ai + bi*xs + ci*ys
    _t4 = _t1*_t3
    _t5 = _t0*_t3
    bx = _t2*bi + _t4*bj + _t5*bk
    by = _t2*ci + _t4*cj + _t5*ck
    out[0,:] = bx
    out[1,:] = by

@njit(cache=True, nogil=True)
def _curl_nf0_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ak + bk*xs + ck*ys
    _t1 = aj + bj*xs + cj*ys
    out[:] = (-bi*(_t0*cj - _t1*ck) + ci*(_t0*bj - _t1*bk) + 2*(bj*ck - bk*cj)*(ai + bi*xs + ci*ys))

@njit(cache=True, nogil=True)
def _curl_nf1_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ak + bk*xs + ck*ys
    _t1 = ai + bi*xs + ci*ys
    out[:] = (-bj*(_t0*ci - _t1*ck) + cj*(_t0*bi - _t1*bk) + 2*(bi*ck - bk*ci)*(aj + bj*xs + cj*ys))

@njit(cache=True, nogil=True)
def _curl_nf2_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ai + bi*xs + ci*ys
    out[:] = (bk*(_t0*ci - _t1*cj) - ck*(_t0*bi - _t1*bj) - 2*(bi*cj - bj*ci)*(ak + bk*xs + ck*ys))

@njit(cache=True, nogil=True)
def _curl_nf3_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ai + bi*xs + ci*ys
    out[:] = (-3*_t0*bi*ck + 3*_t0*bk*ci - 3*_t1*bj*ck + 3*_t1*bk*cj)

@njit(cache=True, nogil=True)
def _curl_nf4_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ak + bk*xs + ck*ys
    _t1 = aj + bj*xs + cj*ys
    out[:] = (3*_t0*bi*cj - 3*_t0*bj*ci + 3*_t1*bi*ck - 3*_t1*bk*ci)

@njit(cache=True, nogil=True)
def _curl_nf5_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ak + bk*xs + ck*ys
    _t1 = ai + bi*xs + ci*ys
    out[:] = (-3*_t0*bi*cj + 3*_t0*bj*ci + 3*_t1*bj*ck - 3*_t1*bk*cj)

@njit(cache=True, nogil=True)
def _curl_nf6_2d(coeff, coords, i, j, k, out):
    out[:] = 0.0

@njit(cache=True, nogil=True)
def _div_nf0_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ak + bk*xs + ck*ys
    _t1 = aj + bj*xs + cj*ys
    out[:] = (-bi*(_t0*bj - _t1*bk) - ci*(_t0*cj - _t1*ck))

@njit(cache=True, nogil=True)
def _div_nf1_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ak + bk*xs + ck*ys
    _t1 = ai + bi*xs + ci*ys
    out[:] = (-bj*(_t0*bi - _t1*bk) - cj*(_t0*ci - _t1*ck))

@njit(cache=True, nogil=True)
def _div_nf2_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ai + bi*xs + ci*ys
    out[:] = (bk*(_t0*bi - _t1*bj) + ck*(_t0*ci - _t1*cj))

@njit(cache=True, nogil=True)
def _div_nf3_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = aj + bj*xs + cj*ys
    _t1 = ai + bi*xs + ci*ys
    _t2 = ak + bk*xs + ck*ys
    out[:] = (-_t0*bi*bk - _t0*ci*ck - _t1*bj*bk - _t1*cj*ck + 2*_t2*bi*bj + 2*_t2*ci*cj)

@njit(cache=True, nogil=True)
def _div_nf4_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ak + bk*xs + ck*ys
    _t1 = aj + bj*xs + cj*ys
    _t2 = ai + bi*xs + ci*ys
    out[:] = (-_t0*bi*bj - _t0*ci*cj - _t1*bi*bk - _t1*ci*ck + 2*_t2*bj*bk + 2*_t2*cj*ck)

@njit(cache=True, nogil=True)
def _div_nf5_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ak + bk*xs + ck*ys
    _t1 = ai + bi*xs + ci*ys
    _t2 = aj + bj*xs + cj*ys
    out[:] = (-_t0*bi*bj - _t0*ci*cj - _t1*bj*bk - _t1*cj*ck + 2*_t2*bi*bk + 2*_t2*ci*ck)

@njit(cache=True, nogil=True)
def _div_nf6_2d(coeff, coords, i, j, k, out):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    _t0 = ak + bk*xs + ck*ys
    _t1 = aj + bj*xs + cj*ys
    _t2 = ai + bi*xs + ci*ys
    out[:] = (2*_t0*bi*bj + 2*_t0*ci*cj + 2*_t1*bi*bk + 2*_t1*ci*ck + 2*_t2*bj*bk + 2*_t2*cj*ck)

@njit(cache=True, nogil=True)
def _ne0_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ai + bi*xs + ci*ys + di*zs
    bx = -_t0*bi + _t1*bj
    by = -_t0*ci + _t1*cj
    bz = -_t0*di + _t1*dj
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _ne1_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ai + bi*xs + ci*ys + di*zs
    bx = _t0*bi + _t1*bj
    by = _t0*ci + _t1*cj
    bz = _t0*di + _t1*dj
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _ne2_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = bj*xs
    _t1 = cj*ys
    _t2 = dj*zs
    _t3 = ai + bi*xs + ci*ys + di*zs
    _t4 = -_t0 - _t1 - _t2 + _t3 - aj
    _t5 = _t0 + _t1 + _t2 + aj
    bx = -_t4*(-_t3*bj + _t5*bi)
    by = -_t4*(-_t3*cj + _t5*ci)
    bz = -_t4*(-_t3*dj + _t5*di)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _ne3_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = bj*xs
    _t1 = cj*ys
    _t2 = dj*zs
    _t3 = _t0 + _t1 + _t2 + aj
    _t4 = ai + bi*xs + ci*ys + di*zs
    _t5 = _t3*_t4
    _t6 = -_t0 - _t1 - _t2 + _t4 - aj
    _t7 = _t3*_t6
    _t8 = _t4*_t6
    bx = _t5*(bi - bj) + _t7*bi + _t8*bj
    by = _t5*(ci - cj) + _t7*ci + _t8*cj
    bz = _t5*(di - dj) + _t7*di + _t8*dj
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _ne4_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ai + bi*xs + ci*ys + di*zs
    _t1 = aj + bj*xs + cj*ys + dj*zs
    bx = -_t0*(-_t0*bj + _t1*bi)
    by = -_t0*(-_t0*cj + _t1*ci)
    bz = -_t0*(-_t0*dj + _t1*di)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _ne5_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ai + bi*xs + ci*ys + di*zs
    bx = -_t0*(_t0*bi - _t1*bj)
    by = -_t0*(_t0*ci - _t1*cj)
    bz = -_t0*(_t0*di - _t1*dj)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curl_ne0_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    bx = 2*ci*dj - 2*cj*di
    by = -2*bi*dj + 2*bj*di
    bz = 2*bi*cj - 2*bj*ci
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curl_ne1_3d(coeff, coords, i, j, k, out):
    out[:,:] = 0.0

@njit(cache=True, nogil=True)
def _curl_ne2_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = di - dj
    _t1 = bj*xs
    _t2 = cj*ys
    _t3 = dj*zs
    _t4 = _t1 + _t2 + _t3 + aj
    _t5 = ai + bi*xs + ci*ys + di*zs
    _t6 = _t4*ci - _t5*cj
    _t7 = ci - cj
    _t8 = _t4*di - _t5*dj
    _t9 = -2*_t1 - 2*_t2 - 2*_t3 + 2*_t5 - 2*aj
    _t10 = _t4*bi - _t5*bj
    _t11 = bi - bj
    bx = _t0*_t6 - _t7*_t8 + _t9*(ci*dj - cj*di)
    by = -_t0*_t10 + _t11*_t8 - _t9*(bi*dj - bj*di)
    bz = _t10*_t7 - _t11*_t6 + _t9*(bi*cj - bj*ci)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curl_ne3_3d(coeff, coords, i, j, k, out):
    out[:,:] = 0.0

@njit(cache=True, nogil=True)
def _curl_ne4_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ai + bi*xs + ci*ys + di*zs
    _t1 = 2*_t0
    _t2 = aj + bj*xs + cj*ys + dj*zs
    _t3 = -_t0*cj + _t2*ci
    _t4 = -_t0*dj + _t2*di
    _t5 = -_t0*bj + _t2*bi
    bx = _t1*(ci*dj - cj*di) + _t3*di - _t4*ci
    by = -_t1*(bi*dj - bj*di) + _t4*bi - _t5*di
    bz = _t1*(bi*cj - bj*ci) - _t3*bi + _t5*ci
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curl_ne5_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = 2*_t0
    _t2 = ai + bi*xs + ci*ys + di*zs
    _t3 = _t0*ci - _t2*cj
    _t4 = _t0*di - _t2*dj
    _t5 = _t0*bi - _t2*bj
    bx = _t1*(ci*dj - cj*di) + _t3*dj - _t4*cj
    by = -_t1*(bi*dj - bj*di) + _t4*bj - _t5*dj
    bz = _t1*(bi*cj - bj*ci) - _t3*bj + _t5*cj
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _div_ne0_3d(coeff, coords, i, j, k, out):
    out[:] = 0.0

@njit(cache=True, nogil=True)
def _div_ne1_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    out[:] = (2*bi*bj + 2*ci*cj + 2*di*dj*np.ones_like(coords[0,:]))

@njit(cache=True, nogil=True)
def _div_ne2_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ai + bi*xs + ci*ys + di*zs
    out[:] = (-(bi - bj)*(_t0*bi - _t1*bj) - (ci - cj)*(_t0*ci - _t1*cj) - (di - dj)*(_t0*di - _t1*dj))

@njit(cache=True, nogil=True)
def _div_ne3_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = bi - bj
    _t1 = bj*xs
    _t2 = cj*ys
    _t3 = dj*zs
    _t4 = _t1 + _t2 + _t3 + aj
    _t5 = ai + bi*xs + ci*ys + di*zs
    _t6 = ci - cj
    _t7 = di - dj
    _t8 = -_t1 - _t2 - _t3 + _t5 - aj
    out[:] = (2*_t0*_t4*bi + 2*_t0*_t5*bj + 2*_t4*_t6*ci + 2*_t4*_t7*di + 2*_t5*_t6*cj + 2*_t5*_t7*dj + 2*_t8*bi*bj + 2*_t8*ci*cj + 2*_t8*di*dj)

@njit(cache=True, nogil=True)
def _div_ne4_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ai + bi*xs + ci*ys + di*zs
    out[:] = (-bi*(_t0*bi - _t1*bj) - ci*(_t0*ci - _t1*cj) - di*(_t0*di - _t1*dj))

@njit(cache=True, nogil=True)
def _div_ne5_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ai + bi*xs + ci*ys + di*zs
    out[:] = (-bj*(_t0*bi - _t1*bj) - cj*(_t0*ci - _t1*cj) - dj*(_t0*di - _t1*dj))

@njit(cache=True, nogil=True)
def _nf0_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ai + bi*xs + ci*ys + di*zs
    _t1 = ak + bk*xs + ck*ys + dk*zs
    _t2 = aj + bj*xs + cj*ys + dj*zs
    bx = -_t0*(_t1*bj - _t2*bk)
    by = -_t0*(_t1*cj - _t2*ck)
    bz = -_t0*(_t1*dj - _t2*dk)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _nf1_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ak + bk*xs + ck*ys + dk*zs
    _t2 = ai + bi*xs + ci*ys + di*zs
    bx = -_t0*(_t1*bi - _t2*bk)
    by = -_t0*(_t1*ci - _t2*ck)
    bz = -_t0*(_t1*di - _t2*dk)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _nf2_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ak + bk*xs + ck*ys + dk*zs
    _t1 = aj + bj*xs + cj*ys + dj*zs
    _t2 = ai + bi*xs + ci*ys + di*zs
    bx = _t0*(_t1*bi - _t2*bj)
    by = _t0*(_t1*ci - _t2*cj)
    bz = _t0*(_t1*di - _t2*dj)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _nf3_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ak + bk*xs + ck*ys + dk*zs
    _t2 = _t0*_t1
    _t3 = ai + bi*xs + ci*ys + di*zs
    _t4 = _t1*_t3
    _t5 = 2*_t0*_t3
    bx = _t2*bi + _t4*bj - _t5*bk
    by = _t2*ci + _t4*cj - _t5*ck
    bz = _t2*di + _t4*dj - _t5*dk
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _nf4_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ai + bi*xs + ci*ys + di*zs
    _t1 = ak + bk*xs + ck*ys + dk*zs
    _t2 = _t0*_t1
    _t3 = aj + bj*xs + cj*ys + dj*zs
    _t4 = _t0*_t3
    _t5 = 2*_t1*_t3
    bx = _t2*bj + _t4*bk - _t5*bi
    by = _t2*cj + _t4*ck - _t5*ci
    bz = _t2*dj + _t4*dk - _t5*di
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _nf5_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ak + bk*xs + ck*ys + dk*zs
    _t2 = _t0*_t1
    _t3 = ai + bi*xs + ci*ys + di*zs
    _t4 = _t0*_t3
    _t5 = 2*_t1*_t3
    bx = _t2*bi + _t4*bk - _t5*bj
    by = _t2*ci + _t4*ck - _t5*cj
    bz = _t2*di + _t4*dk - _t5*dj
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _nf6_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ak + bk*xs + ck*ys + dk*zs
    _t2 = _t0*_t1
    _t3 = ai + bi*xs + ci*ys + di*zs
    _t4 = _t1*_t3
    _t5 = _t0*_t3
    bx = _t2*bi + _t4*bj + _t5*bk
    by = _t2*ci + _t4*cj + _t5*ck
    bz = _t2*di + _t4*dj + _t5*dk
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curl_nf0_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = 2*ai + 2*bi*xs + 2*ci*ys + 2*di*zs
    _t1 = ak + bk*xs + ck*ys + dk*zs
    _t2 = aj + bj*xs + cj*ys + dj*zs
    _t3 = _t1*cj - _t2*ck
    _t4 = _t1*dj - _t2*dk
    _t5 = _t1*bj - _t2*bk
    bx = _t0*(cj*dk - ck*dj) + _t3*di - _t4*ci
    by = -_t0*(bj*dk - bk*dj) + _t4*bi - _t5*di
    bz = _t0*(bj*ck - bk*cj) - _t3*bi + _t5*ci
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curl_nf1_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = 2*aj + 2*bj*xs + 2*cj*ys + 2*dj*zs
    _t1 = ak + bk*xs + ck*ys + dk*zs
    _t2 = ai + bi*xs + ci*ys + di*zs
    _t3 = _t1*ci - _t2*ck
    _t4 = _t1*di - _t2*dk
    _t5 = _t1*bi - _t2*bk
    bx = _t0*(ci*dk - ck*di) + _t3*dj - _t4*cj
    by = -_t0*(bi*dk - bk*di) + _t4*bj - _t5*dj
    bz = _t0*(bi*ck - bk*ci) - _t3*bj + _t5*cj
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curl_nf2_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = 2*ak + 2*bk*xs + 2*ck*ys + 2*dk*zs
    _t1 = aj + bj*xs + cj*ys + dj*zs
    _t2 = ai + bi*xs + ci*ys + di*zs
    _t3 = _t1*ci - _t2*cj
    _t4 = _t1*di - _t2*dj
    _t5 = _t1*bi - _t2*bj
    bx = -_t0*(ci*dj - cj*di) - _t3*dk + _t4*ck
    by = _t0*(bi*dj - bj*di) - _t4*bk + _t5*dk
    bz = -_t0*(bi*cj - bj*ci) + _t3*bk - _t5*ck
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curl_nf3_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = _t0*dk
    _t2 = ai + bi*xs + ci*ys + di*zs
    _t3 = _t2*dk
    bx = 3*_t0*ck*di - 3*_t1*ci + 3*_t2*ck*dj - 3*_t3*cj
    by = -3*_t0*bk*di + 3*_t1*bi - 3*_t2*bk*dj + 3*_t3*bj
    bz = -3*_t0*bi*ck + 3*_t0*bk*ci - 3*_t2*bj*ck + 3*_t2*bk*cj
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curl_nf4_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ak + bk*xs + ck*ys + dk*zs
    _t1 = _t0*dj
    _t2 = aj + bj*xs + cj*ys + dj*zs
    _t3 = _t2*dk
    bx = -3*_t0*cj*di + 3*_t1*ci - 3*_t2*ck*di + 3*_t3*ci
    by = 3*_t0*bj*di - 3*_t1*bi + 3*_t2*bk*di - 3*_t3*bi
    bz = 3*_t0*bi*cj - 3*_t0*bj*ci + 3*_t2*bi*ck - 3*_t2*bk*ci
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curl_nf5_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ak + bk*xs + ck*ys + dk*zs
    _t1 = _t0*dj
    _t2 = ai + bi*xs + ci*ys + di*zs
    _t3 = _t2*dj
    bx = 3*_t0*cj*di - 3*_t1*ci + 3*_t2*cj*dk - 3*_t3*ck
    by = -3*_t0*bj*di + 3*_t1*bi - 3*_t2*bj*dk + 3*_t3*bk
    bz = -3*_t0*bi*cj + 3*_t0*bj*ci + 3*_t2*bj*ck - 3*_t2*bk*cj
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curl_nf6_3d(coeff, coords, i, j, k, out):
    out[:,:] = 0.0

@njit(cache=True, nogil=True)
def _div_nf0_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ak + bk*xs + ck*ys + dk*zs
    _t1 = aj + bj*xs + cj*ys + dj*zs
    out[:] = (-bi*(_t0*bj - _t1*bk) - ci*(_t0*cj - _t1*ck) - di*(_t0*dj - _t1*dk))

@njit(cache=True, nogil=True)
def _div_nf1_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ak + bk*xs + ck*ys + dk*zs
    _t1 = ai + bi*xs + ci*ys + di*zs
    out[:] = (-bj*(_t0*bi - _t1*bk) - cj*(_t0*ci - _t1*ck) - dj*(_t0*di - _t1*dk))

@njit(cache=True, nogil=True)
def _div_nf2_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ai + bi*xs + ci*ys + di*zs
    out[:] = (bk*(_t0*bi - _t1*bj) + ck*(_t0*ci - _t1*cj) + dk*(_t0*di - _t1*dj))

@njit(cache=True, nogil=True)
def _div_nf3_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = aj + bj*xs + cj*ys + dj*zs
    _t1 = ai + bi*xs + ci*ys + di*zs
    _t2 = ak + bk*xs + ck*ys + dk*zs
    out[:] = (-_t0*bi*bk - _t0*ci*ck - _t0*di*dk - _t1*bj*bk - _t1*cj*ck - _t1*dj*dk + 2*_t2*bi*bj + 2*_t2*ci*cj + 2*_t2*di*dj)

@njit(cache=True, nogil=True)
def _div_nf4_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ak + bk*xs + ck*ys + dk*zs
    _t1 = aj + bj*xs + cj*ys + dj*zs
    _t2 = ai + bi*xs + ci*ys + di*zs
    out[:] = (-_t0*bi*bj - _t0*ci*cj - _t0*di*dj - _t1*bi*bk - _t1*ci*ck - _t1*di*dk + 2*_t2*bj*bk + 2*_t2*cj*ck + 2*_t2*dj*dk)

@njit(cache=True, nogil=True)
def _div_nf5_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ak + bk*xs + ck*ys + dk*zs
    _t1 = ai + bi*xs + ci*ys + di*zs
    _t2 = aj + bj*xs + cj*ys + dj*zs
    out[:] = (-_t0*bi*bj - _t0*ci*cj - _t0*di*dj - _t1*bj*bk - _t1*cj*ck - _t1*dj*dk + 2*_t2*bi*bk + 2*_t2*ci*ck + 2*_t2*di*dk)

@njit(cache=True, nogil=True)
def _div_nf6_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    _t0 = ak + bk*xs + ck*ys + dk*zs
    _t1 = aj + bj*xs + cj*ys + dj*zs
    _t2 = ai + bi*xs + ci*ys + di*zs
    out[:] = (2*_t0*bi*bj + 2*_t0*ci*cj + 2*_t0*di*dj + 2*_t1*bi*bk + 2*_t1*ci*ck + 2*_t1*di*dk + 2*_t2*bj*bk + 2*_t2*cj*ck + 2*_t2*dj*dk)

@njit(cache=True, nogil=True)
def _curlcurl_ne0_3d(coeff, coords, i, j, k, out):
    out[:,:] = 0.0

@njit(cache=True, nogil=True)
def _curlcurl_ne1_3d(coeff, coords, i, j, k, out):
    out[:,:] = 0.0

@njit(cache=True, nogil=True)
def _curlcurl_ne2_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    _t0 = ci - cj
    _t1 = bi*cj - bj*ci
    _t2 = di - dj
    _t3 = bi*dj - bj*di
    _t4 = bi - bj
    _t5 = ci*dj - cj*di
    bx = 3*_t0*_t1 + 3*_t2*_t3
    by = -3*_t1*_t4 + 3*_t2*_t5
    bz = -3*_t0*_t5 - 3*_t3*_t4
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curlcurl_ne3_3d(coeff, coords, i, j, k, out):
    out[:,:] = 0.0

@njit(cache=True, nogil=True)
def _curlcurl_ne4_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    _t0 = bi*cj - bj*ci
    _t1 = bi*dj - bj*di
    _t2 = ci*dj - cj*di
    bx = 3*_t0*ci + 3*_t1*di
    by = -3*_t0*bi + 3*_t2*di
    bz = -3*_t1*bi - 3*_t2*ci
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curlcurl_ne5_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    _t0 = bi*cj - bj*ci
    _t1 = bi*dj - bj*di
    _t2 = ci*dj - cj*di
    bx = 3*_t0*cj + 3*_t1*dj
    by = -3*_t0*bj + 3*_t2*dj
    bz = -3*_t1*bj - 3*_t2*cj
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curlcurl_nf0_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    _t0 = bj*ck - bk*cj
    _t1 = bj*dk - bk*dj
    _t2 = cj*dk - ck*dj
    bx = 3*_t0*ci + 3*_t1*di
    by = -3*_t0*bi + 3*_t2*di
    bz = -3*_t1*bi - 3*_t2*ci
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curlcurl_nf1_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    _t0 = bi*ck - bk*ci
    _t1 = bi*dk - bk*di
    _t2 = ci*dk - ck*di
    bx = 3*_t0*cj + 3*_t1*dj
    by = -3*_t0*bj + 3*_t2*dj
    bz = -3*_t1*bj - 3*_t2*cj
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curlcurl_nf2_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    _t0 = bi*cj - bj*ci
    _t1 = bi*dj - bj*di
    _t2 = ci*dj - cj*di
    bx = -3*_t0*ck - 3*_t1*dk
    by = 3*_t0*bk - 3*_t2*dk
    bz = 3*_t1*bk + 3*_t2*ck
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curlcurl_nf3_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    bx = -3*bi*cj*ck - 3*bi*dj*dk - 3*bj*ci*ck - 3*bj*di*dk + 6*bk*ci*cj + 6*bk*di*dj
    by = 6*bi*bj*ck - 3*bi*bk*cj - 3*bj*bk*ci - 3*ci*dj*dk - 3*cj*di*dk + 6*ck*di*dj
    bz = 6*bi*bj*dk - 3*bi*bk*dj - 3*bj*bk*di + 6*ci*cj*dk - 3*ci*ck*dj - 3*cj*ck*di
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curlcurl_nf4_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    bx = 6*bi*cj*ck + 6*bi*dj*dk - 3*bj*ci*ck - 3*bj*di*dk - 3*bk*ci*cj - 3*bk*di*dj
    by = -3*bi*bj*ck - 3*bi*bk*cj + 6*bj*bk*ci + 6*ci*dj*dk - 3*cj*di*dk - 3*ck*di*dj
    bz = -3*bi*bj*dk - 3*bi*bk*dj + 6*bj*bk*di - 3*ci*cj*dk - 3*ci*ck*dj + 6*cj*ck*di
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curlcurl_nf5_3d(coeff, coords, i, j, k, out):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    bx = -3*bi*cj*ck - 3*bi*dj*dk + 6*bj*ci*ck + 6*bj*di*dk - 3*bk*ci*cj - 3*bk*di*dj
    by = -3*bi*bj*ck + 6*bi*bk*cj - 3*bj*bk*ci - 3*ci*dj*dk + 6*cj*di*dk - 3*ck*di*dj
    bz = -3*bi*bj*dk + 6*bi*bk*dj - 3*bj*bk*di - 3*ci*cj*dk + 6*ci*ck*dj - 3*cj*ck*di
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz

@njit(cache=True, nogil=True)
def _curlcurl_nf6_3d(coeff, coords, i, j, k, out):
    out[:,:] = 0.0

@njit(cache=True, nogil=True)
def _eval_f_2d(coeff, coords, i, j, k, code, out):
    bftype, index = get_type_index(code)

    if bftype == 64:
        if index == 0:
            _ne0_2d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _ne1_2d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _ne2_2d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _ne3_2d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _ne4_2d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _ne5_2d(coeff, coords, i,j,k, out)
            return
    if bftype == 128:
        if index == 0:
            _nf0_2d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _nf1_2d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _nf2_2d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _nf3_2d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _nf4_2d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _nf5_2d(coeff, coords, i,j,k, out)
            return
        elif index == 6:
            _nf6_2d(coeff, coords, i,j,k, out)
            return
    raise ValueError('Unrecognized basis function type.')

@njit(cache=True, nogil=True)
def _eval_f_3d(coeff, coords, i, j, k, code, out):
    bftype, index = get_type_index(code)

    if bftype == 64:
        if index == 0:
            _ne0_3d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _ne1_3d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _ne2_3d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _ne3_3d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _ne4_3d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _ne5_3d(coeff, coords, i,j,k, out)
            return
    if bftype == 128:
        if index == 0:
            _nf0_3d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _nf1_3d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _nf2_3d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _nf3_3d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _nf4_3d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _nf5_3d(coeff, coords, i,j,k, out)
            return
        elif index == 6:
            _nf6_3d(coeff, coords, i,j,k, out)
            return
    raise ValueError('Unrecognized basis function type.')

@njit(cache=True, nogil=True)
def _eval_curl_f_2d(coeff, coords, i, j, k, code, out):
    bftype, index = get_type_index(code)

    if bftype == 64:
        if index == 0:
            _curl_ne0_2d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _curl_ne1_2d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _curl_ne2_2d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _curl_ne3_2d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _curl_ne4_2d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _curl_ne5_2d(coeff, coords, i,j,k, out)
            return
    if bftype == 128:
        if index == 0:
            _curl_nf0_2d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _curl_nf1_2d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _curl_nf2_2d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _curl_nf3_2d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _curl_nf4_2d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _curl_nf5_2d(coeff, coords, i,j,k, out)
            return
        elif index == 6:
            _curl_nf6_2d(coeff, coords, i,j,k, out)
            return
    raise ValueError('Unrecognized basis function type.')

@njit(cache=True, nogil=True)
def _eval_curl_f_3d(coeff, coords, i, j, k, code, out):
    bftype, index = get_type_index(code)

    if bftype == 64:
        if index == 0:
            _curl_ne0_3d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _curl_ne1_3d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _curl_ne2_3d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _curl_ne3_3d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _curl_ne4_3d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _curl_ne5_3d(coeff, coords, i,j,k, out)
            return
    if bftype == 128:
        if index == 0:
            _curl_nf0_3d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _curl_nf1_3d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _curl_nf2_3d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _curl_nf3_3d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _curl_nf4_3d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _curl_nf5_3d(coeff, coords, i,j,k, out)
            return
        elif index == 6:
            _curl_nf6_3d(coeff, coords, i,j,k, out)
            return
    raise ValueError('Unrecognized basis function type.')

@njit(cache=True, nogil=True)
def _eval_div_f_2d(coeff, coords, i, j, k, code, out):
    bftype, index = get_type_index(code)

    if bftype == 64:
        if index == 0:
            _div_ne0_2d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _div_ne1_2d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _div_ne2_2d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _div_ne3_2d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _div_ne4_2d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _div_ne5_2d(coeff, coords, i,j,k, out)
            return
    if bftype == 128:
        if index == 0:
            _div_nf0_2d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _div_nf1_2d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _div_nf2_2d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _div_nf3_2d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _div_nf4_2d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _div_nf5_2d(coeff, coords, i,j,k, out)
            return
        elif index == 6:
            _div_nf6_2d(coeff, coords, i,j,k, out)
            return
    raise ValueError('Unrecognized basis function type.')

@njit(cache=True, nogil=True)
def _eval_div_f_3d(coeff, coords, i, j, k, code, out):
    bftype, index = get_type_index(code)

    if bftype == 64:
        if index == 0:
            _div_ne0_3d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _div_ne1_3d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _div_ne2_3d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _div_ne3_3d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _div_ne4_3d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _div_ne5_3d(coeff, coords, i,j,k, out)
            return
    if bftype == 128:
        if index == 0:
            _div_nf0_3d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _div_nf1_3d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _div_nf2_3d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _div_nf3_3d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _div_nf4_3d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _div_nf5_3d(coeff, coords, i,j,k, out)
            return
        elif index == 6:
            _div_nf6_3d(coeff, coords, i,j,k, out)
            return
    raise ValueError('Unrecognized basis function type.')

@njit(cache=True, nogil=True)
def _eval_curlcurl_f_3d(coeff, coords, i, j, k, code, out):
    bftype, index = get_type_index(code)

    if bftype == 64:
        if index == 0:
            _curlcurl_ne0_3d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _curlcurl_ne1_3d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _curlcurl_ne2_3d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _curlcurl_ne3_3d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _curlcurl_ne4_3d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _curlcurl_ne5_3d(coeff, coords, i,j,k, out)
            return
    if bftype == 128:
        if index == 0:
            _curlcurl_nf0_3d(coeff, coords, i,j,k, out)
            return
        elif index == 1:
            _curlcurl_nf1_3d(coeff, coords, i,j,k, out)
            return
        elif index == 2:
            _curlcurl_nf2_3d(coeff, coords, i,j,k, out)
            return
        elif index == 3:
            _curlcurl_nf3_3d(coeff, coords, i,j,k, out)
            return
        elif index == 4:
            _curlcurl_nf4_3d(coeff, coords, i,j,k, out)
            return
        elif index == 5:
            _curlcurl_nf5_3d(coeff, coords, i,j,k, out)
            return
        elif index == 6:
            _curlcurl_nf6_3d(coeff, coords, i,j,k, out)
            return
    raise ValueError('Unrecognized basis function type.')


# ---- from emerge/_emerge/mth/optimized.py (2.8.9) ----

@njit(cache=True, nogil=True)
def matinv(M: np.ndarray) -> np.ndarray:
    """Optimized matrix inverse of 3x3 matrix

    Args:
        M (np.ndarray): Input matrix M of shape (3,3)

    Returns:
        np.ndarray: The matrix inverse inv(M)
    """
    out = np.zeros((3, 3), dtype=np.complex128)

    if (
        M[0, 1] == 0
        and M[0, 2] == 0
        and M[1, 0] == 0
        and M[1, 2] == 0
        and M[2, 0] == 0
        and M[2, 1] == 0
    ):
        out[0, 0] = 1 / M[0, 0]
        out[1, 1] = 1 / M[1, 1]
        out[2, 2] = 1 / M[2, 2]
    else:
        det = (
            M[0, 0] * M[1, 1] * M[2, 2]
            - M[0, 0] * M[1, 2] * M[2, 1]
            - M[0, 1] * M[1, 0] * M[2, 2]
            + M[0, 1] * M[1, 2] * M[2, 0]
            + M[0, 2] * M[1, 0] * M[2, 1]
            - M[0, 2] * M[1, 1] * M[2, 0]
        )
        out[0, 0] = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
        out[0, 1] = -M[0, 1] * M[2, 2] + M[0, 2] * M[2, 1]
        out[0, 2] = M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]
        out[1, 0] = -M[1, 0] * M[2, 2] + M[1, 2] * M[2, 0]
        out[1, 1] = M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]
        out[1, 2] = -M[0, 0] * M[1, 2] + M[0, 2] * M[1, 0]
        out[2, 0] = M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]
        out[2, 1] = -M[0, 0] * M[2, 1] + M[0, 1] * M[2, 0]
        out[2, 2] = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        out = out / det
    return out


@njit(cache=True, nogil=True)
def local_mapping(vertex_ids, triangle_ids):
    """
    Parameters
    ----------
    vertex_ids   : 1-D int64 array (length 4)
        Global vertex indices of one tetrahedron, in *its* order
        (v0, v1, v2, v3).

    triangle_ids : 2-D int64 array (nTri × 3)
        Each row is a global-ID triple of one face that belongs to this tet.

    Returns
    -------
    local_tris   : 2-D int64 array (nTri × 3)
        Same triangles, but every entry replaced by the local index
        0,1,2,3 that the vertex has inside this tetrahedron.
        (Guaranteed to be ∈{0,1,2,3}; no -1 ever appears if the input
        really belongs to the tet.)
    """
    ndim = triangle_ids.shape[0]
    ntri = triangle_ids.shape[1]
    out = np.zeros(triangle_ids.shape, dtype=np.int64)

    for t in range(ntri):  # each triangle
        for j in range(ndim):  # each vertex in that triangle
            gid = triangle_ids[j, t]  # global ID to look up

            # linear search over the four tet vertices
            for k in range(4):
                if vertex_ids[k] == gid:
                    out[j, t] = k  # store local index 0-3
                    break  # stop the k-loop

    return out


# ---- from emerge/_emerge/physics/microwave/assembly/curlcurl.py (2.8.9) ----

@njit(cache=True, nogil=True)
def local_tet_to_triid(tet_to_tri, tets, tris, itet) -> np.ndarray:
    """Returns the triangle node indices in the right order given a tet-index"""
    tri_ids = tet_to_tri[:, itet]
    global_tri_map = tris[:, tri_ids]
    out = local_mapping(tets[:, itet], global_tri_map)
    return out


@njit(cache=True, nogil=True)
def local_tet_to_edgeid(tet_to_edge, tets, edges, itet) -> np.ndarray:
    """Returns the edge node indices in the right order given a tet-index"""
    global_edge_map = edges[:, tet_to_edge[:, itet]]
    return local_mapping(tets[:, itet], global_edge_map)


@njit(
    cache=True,
    nogil=True,
)
def tet_coefficients(xs, ys, zs):
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    aas = np.empty((4,), dtype=np.float64)
    bbs = np.empty((4,), dtype=np.float64)
    ccs = np.empty((4,), dtype=np.float64)
    dds = np.empty((4,), dtype=np.float64)

    V = np.abs(
        -x1 * y2 * z3 
        + x1 * y2 * z4
        + x1 * y3 * z2
        - x1 * y3 * z4
        - x1 * y4 * z2
        + x1 * y4 * z3
        + x2 * y1 * z3
        - x2 * y1 * z4
        - x2 * y3 * z1
        + x2 * y3 * z4
        + x2 * y4 * z1
        - x2 * y4 * z3
        - x3 * y1 * z2
        + x3 * y1 * z4
        + x3 * y2 * z1
        - x3 * y2 * z4
        - x3 * y4 * z1
        + x3 * y4 * z2
        + x4 * y1 * z2
        - x4 * y1 * z3
        - x4 * y2 * z1
        + x4 * y2 * z3
        + x4 * y3 * z1
        - x4 * y3 * z2
    ) / 6

    aas[0] = (
        x2 * y3 * z4
        - x2 * y4 * z3
        - x3 * y2 * z4
        + x3 * y4 * z2
        + x4 * y2 * z3
        - x4 * y3 * z2
    )
    aas[1] = (
        -x1 * y3 * z4
        + x1 * y4 * z3
        + x3 * y1 * z4
        - x3 * y4 * z1
        - x4 * y1 * z3
        + x4 * y3 * z1
    )
    aas[2] = (
        x1 * y2 * z4
        - x1 * y4 * z2
        - x2 * y1 * z4
        + x2 * y4 * z1
        + x4 * y1 * z2
        - x4 * y2 * z1
    )
    aas[3] = (
        -x1 * y2 * z3
        + x1 * y3 * z2
        + x2 * y1 * z3
        - x2 * y3 * z1
        - x3 * y1 * z2
        + x3 * y2 * z1
    )
    bbs[0] = -y2 * z3 + y2 * z4 + y3 * z2 - y3 * z4 - y4 * z2 + y4 * z3
    bbs[1] = y1 * z3 - y1 * z4 - y3 * z1 + y3 * z4 + y4 * z1 - y4 * z3
    bbs[2] = -y1 * z2 + y1 * z4 + y2 * z1 - y2 * z4 - y4 * z1 + y4 * z2
    bbs[3] = y1 * z2 - y1 * z3 - y2 * z1 + y2 * z3 + y3 * z1 - y3 * z2
    ccs[0] = x2 * z3 - x2 * z4 - x3 * z2 + x3 * z4 + x4 * z2 - x4 * z3
    ccs[1] = -x1 * z3 + x1 * z4 + x3 * z1 - x3 * z4 - x4 * z1 + x4 * z3
    ccs[2] = x1 * z2 - x1 * z4 - x2 * z1 + x2 * z4 + x4 * z1 - x4 * z2
    ccs[3] = -x1 * z2 + x1 * z3 + x2 * z1 - x2 * z3 - x3 * z1 + x3 * z2
    dds[0] = -x2 * y3 + x2 * y4 + x3 * y2 - x3 * y4 - x4 * y2 + x4 * y3
    dds[1] = x1 * y3 - x1 * y4 - x3 * y1 + x3 * y4 + x4 * y1 - x4 * y3
    dds[2] = -x1 * y2 + x1 * y4 + x2 * y1 - x2 * y4 - x4 * y1 + x4 * y2
    dds[3] = x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2

    return aas, bbs, ccs, dds, V
