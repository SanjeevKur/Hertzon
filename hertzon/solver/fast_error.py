"""Multi-threaded a-posteriori error estimate for the adaptive passes.

EMerge's ``compute_error_single`` loops serially over all tetrahedra (about 0.15 ms each, so a 400k
element mesh needs a minute per excitation and frequency). The loop body writes only to slots of its
own tetrahedron and to the neighbour's slot of the shared face, which no other tetrahedron writes, so
the iterations are independent. The kernel is generated from the installed EMerge source with the tet
loop as ``prange`` and compiled once (numba disk cache next to the generated module). Any difference
in the source, or a failed compile, keeps EMerge's serial kernel.
"""
from __future__ import annotations

import hashlib
import inspect
from contextlib import contextmanager
import os
import time
from pathlib import Path

import numpy as np

_STATE: dict = {}

_LOOP = "    for itet in range(N_TETS):\n"
# the NaN diagnostics print and raise inside the loop; in a parallel loop a flag is raised afterwards
_NAN_START = "        if np.isnan(\n            np.abs(-Gt / (erc * EPS0) * Qt * Qt - Gt * urc * MU0 * np.sum(Jt * Jt))\n        ):\n"
_NAN_END = '            raise ValueError(f"NaN detected at tet {itet}")\n'
# list handling raises IndexError and so leaves the loop early: numba then runs it serially
_REPL = (
    ("""            adj_tets = [int(tri_to_tet[j, triids[iface]]) for j in range(2)]
            adj_tets = [num for num in adj_tets if num not in (itet, -1234)]

            if len(adj_tets) == 0:
                continue
""", """            itet_adj = -1
            found = False
            for j in range(2):
                num = int(tri_to_tet[j, triids[iface]])
                if not found and num != itet and num != -1234:
                    itet_adj = num
                    found = True
            if not found:
                continue
"""),
    ("""                itet_adj = adj_tets[0]
                iface_adj = np.argwhere(tet_to_tri[:, itet_adj] == triids[iface])[0][0]

""", "\n"),
    ("""            itet_adj = adj_tets[0]
            iface_adj = np.argwhere(tet_to_tri[:, itet_adj] == triids[iface])[0][0]
""", """            iface_adj = 0
            for j in range(4):
                if tet_to_tri[j, itet_adj] == triids[iface]:
                    iface_adj = j
"""),
)


# The face residuals are only used integrated over each face, and the integral is linear: integrating
# inside the loop keeps 4 numbers per face instead of the values at every quadrature point (about 6 kB
# per tetrahedron, 1.6 GB for 250k elements, allocated, zeroed and swept several times per excitation).
_INT_ALLOC = """    Qf_face1 = np.zeros((4, N2D, N_TETS), dtype=np.complex128)
    Qf_face2 = np.zeros((4, N2D, N_TETS), dtype=np.complex128)
    Jf_face1 = np.zeros((4, 3, N2D, N_TETS), dtype=np.complex128)
    Jf_face2 = np.zeros((4, 3, N2D, N_TETS), dtype=np.complex128)

    areas_face_residual = np.zeros((4, N2D, N_TETS), dtype=np.float64)
"""
_INT_ALLOC_NEW = """    Qf_i1 = np.zeros((4, N_TETS), dtype=np.complex128)
    Qf_i2 = np.zeros((4, N_TETS), dtype=np.complex128)
    Jf_i1 = np.zeros((4, 3, N_TETS), dtype=np.complex128)
    Jf_i2 = np.zeros((4, 3, N_TETS), dtype=np.complex128)
    WQ = DPTS_2D[0, :].copy()
"""
_INT_REPL = (
    (_INT_ALLOC, _INT_ALLOC_NEW),
    ("            areas_face_residual[iface, :, itet] = area\n", ""),
    ("                Jf_face1[iface, :, :, itet] = Jtan\n", """                for d in range(3):
                    acc = 0j
                    for q in range(N2D):
                        acc += Jtan[d, q] * WQ[q]
                    Jf_i1[iface, d, itet] = acc * area
"""),
    ("""            Jf_face1[iface, :, :, itet] = cross_c_arry(normal, Jf_all[:, slc1:slc2])
            Jf_face2[iface_adj, :, :, itet_adj] = -cross_c_arry(
                normal, Jf_all[:, slc1:slc2]
            )

            Qf_face1[iface, :, itet] = dot_c_arry(normal, Qf_all[:, slc1:slc2])
            Qf_face2[iface_adj, :, itet_adj] = -dot_c_arry(normal, Qf_all[:, slc1:slc2])
""", """            jf = cross_c_arry(normal, Jf_all[:, slc1:slc2])
            qf = dot_c_arry(normal, Qf_all[:, slc1:slc2])
            for d in range(3):
                acc = 0j
                for q in range(N2D):
                    acc += jf[d, q] * WQ[q]
                Jf_i1[iface, d, itet] = acc * area
                Jf_i2[iface_adj, d, itet_adj] = -acc * area
            acc = 0j
            for q in range(N2D):
                acc += qf[q] * WQ[q]
            Qf_i1[iface, itet] = acc * area
            Qf_i2[iface_adj, itet_adj] = -acc * area
"""),
)
_INT_EPI_START = "    # Compute 2D Gauss quadrature weight matrix\n"
_INT_EPI_END = "    Jf_int_z = np.sum(Jf_delta[:, 2, :, :] * areas_face_residual * fWs, axis=1)\n"
_INT_EPI_NEW = """    Qf_int = Qf_i1 - Qf_i2
    Jf_int_x = Jf_i1[:, 0, :] - Jf_i2[:, 0, :]
    Jf_int_y = Jf_i1[:, 1, :] - Jf_i2[:, 1, :]
    Jf_int_z = Jf_i1[:, 2, :] - Jf_i2[:, 2, :]
"""


# All excitations of a field in one pass (the ports of a model, 11 or 18 for a Rotman lens): EMerge evaluates
# every basis function at the quadrature points once per tetrahedron and excitation; they do not depend on
# the excitation, so here they are evaluated once and reused. The arithmetic is EMerge's (face integrals as
# above); work arrays are allocated per block of tetrahedra, so the threads do not wait for the allocator.
_MULTI = '''"""Generated by Hertzon: EMerge's adaptive_mesh.compute_error_single for all excitations of a field at once."""
import numpy as np
from numba import njit, prange
from emerge._emerge.physics.microwave import adaptive_mesh as _am
globals().update({k: v for k, v in vars(_am).items() if not k.startswith('__')})
from numba import njit, prange  # noqa: E402,F811


@njit(cache=True, parallel=True, error_model="numpy")
def compute_error_multi(nodes, tets, tris, edges, centers, Efields, edge_lengths, areas, tet_to_edge, tet_to_tri,
                        tri_to_tet, tet_to_field, er, ur, pec_tris, k0, dofcodes3d):
    NEX = Efields.shape[1]
    tet_is_pec = np.zeros((tris.shape[1],), dtype=np.bool_)
    tet_is_pec[pec_tris] = True
    N_TETS = tets.shape[1]
    N2D = DPTS_2D.shape[1]
    NV = DPTS_3D.shape[1]
    NF = 4 * N2D
    WV = DPTS_3D[0, :].copy()
    WQ = DPTS_2D[0, :].copy()
    W0 = k0 * C0
    Y0 = np.sqrt(1 / MU0)
    typearray, indexarray = parse_dofcode(dofcodes3d)
    ND = dofcodes3d.shape[0]

    alpha_t = np.zeros((NEX, N_TETS), dtype=np.complex128)
    max_elem_size = np.zeros((N_TETS,), dtype=np.float64)
    Qf_i1 = np.zeros((NEX, 4, N_TETS), dtype=np.complex128)
    Qf_i2 = np.zeros((NEX, 4, N_TETS), dtype=np.complex128)
    Jf_i1 = np.zeros((NEX, 4, 3, N_TETS), dtype=np.complex128)
    Jf_i2 = np.zeros((NEX, 4, 3, N_TETS), dtype=np.complex128)
    Rf_face = np.zeros((4, N_TETS), dtype=np.float64)
    adj_tets_mat = -np.ones((4, N_TETS), dtype=np.int64)

    BS = 64
    NB = (N_TETS + BS - 1) // BS
    for ib in prange(NB):
        # work arrays once per block of tetrahedra, not per tetrahedron
        intpts = np.empty((3, NV), dtype=np.float64)
        fcoords = np.empty((3, NF), dtype=np.float64)
        dummy = np.zeros((3, 1), dtype=np.float64)
        coeff = np.empty((4, 4), dtype=np.float64)
        coeffd = np.empty((4, 4), dtype=np.float64)
        Fv = np.zeros((ND, 3, NV), dtype=np.complex128)
        Ff = np.zeros((ND, 3, NF), dtype=np.complex128)
        Cf = np.zeros((ND, 3, NF), dtype=np.complex128)
        Dv = np.zeros((ND, NV), dtype=np.complex128)
        CC = np.zeros((ND, 3), dtype=np.complex128)
        t3v = np.zeros((3, NV), dtype=np.complex128)
        t3f = np.zeros((3, NF), dtype=np.complex128)
        tdv = np.zeros((NV,), dtype=np.complex128)
        tcc = np.zeros((3, 1), dtype=np.complex128)
        Ef = np.empty((ND,), dtype=np.complex128)
        Ev = np.empty((3, NV), dtype=np.complex128)
        Eface = np.empty((3, NF), dtype=np.complex128)
        Cface = np.empty((3, NF), dtype=np.complex128)
        fnorm = np.empty((4, 3), dtype=np.complex128)
        farea = np.empty((4,), dtype=np.float64)
        fadj = np.empty((4,), dtype=np.int64)
        fadjface = np.empty((4,), dtype=np.int64)
        fpec = np.empty((4,), dtype=np.bool_)
        for itet in range(ib * BS, min(N_TETS, (ib + 1) * BS)):
            erc = er[itet]
            urc = ur[itet]
            vertices = nodes[:, tets[:, itet]]
            for q in range(NV):
                for d in range(3):
                    intpts[d, q] = (DPTS_3D[1, q] * vertices[d, 0] + DPTS_3D[2, q] * vertices[d, 1]
                                    + DPTS_3D[3, q] * vertices[d, 2] + DPTS_3D[4, q] * vertices[d, 3])
            g_node_ids = tets[:, itet]
            l_edge_ids = local_mapping(g_node_ids, edges[:, tet_to_edge[:, itet]])
            l_tri_ids = local_mapping(g_node_ids, tris[:, tet_to_tri[:, itet]])
            triids = tet_to_tri[:, itet]
            size_max = np.max(edge_lengths[tet_to_edge[:, itet]])
            TET_VOLUME = compute_volume(vertices[0, :], vertices[1, :], vertices[2, :])
            Rt = ((3 * TET_VOLUME) / (4 * np.pi)) ** (1 / 3)
            a_s, b_s, c_s, d_s, V = tet_coefficients(vertices[0, :], vertices[1, :], vertices[2, :])
            for j in range(4):
                coeff[0, j] = a_s[j]
                coeff[1, j] = b_s[j]
                coeff[2, j] = c_s[j]
                coeff[3, j] = d_s[j]
                coeffd[0, j] = a_s[j] / (6 * V)
                coeffd[1, j] = b_s[j] / (6 * V)
                coeffd[2, j] = c_s[j] / (6 * V)
                coeffd[3, j] = d_s[j] / (6 * V)
            for itri in range(4):
                t = triids[itri]
                for d in range(3):
                    x1 = nodes[d, tris[0, t]]
                    x2 = nodes[d, tris[1, t]]
                    x3 = nodes[d, tris[2, t]]
                    for q in range(N2D):
                        fcoords[d, itri * N2D + q] = DPTS_2D[1, q] * x1 + DPTS_2D[2, q] * x2 + DPTS_2D[3, q] * x3
            # basis functions at the quadrature points: shared by all excitations
            for idof in range(ND):
                i_index = indexarray[idof]
                if typearray[idof] == 0:
                    i1 = l_edge_ids[0, i_index]
                    j1 = l_edge_ids[1, i_index]
                    k1 = 0
                else:
                    i1 = l_tri_ids[0, i_index]
                    j1 = l_tri_ids[1, i_index]
                    k1 = l_tri_ids[2, i_index]
                code = dofcodes3d[idof]
                _eval_f_3d(coeff, intpts, i1, j1, k1, code, t3v)
                Fv[idof] = t3v
                _eval_f_3d(coeff, fcoords, i1, j1, k1, code, t3f)
                Ff[idof] = t3f
                _eval_curl_f_3d(coeff, fcoords, i1, j1, k1, code, t3f)
                Cf[idof] = t3f
                _eval_div_f_3d(coeffd, intpts, i1, j1, k1, code, tdv)
                Dv[idof] = tdv
                _eval_curlcurl_f_3d(coeff, dummy, i1, j1, k1, code, tcc)
                for d in range(3):
                    CC[idof, d] = tcc[d, 0]
            tetc = centers[:, itet].copy()
            max_elem_size[itet] = size_max
            for iface in range(4):
                tri_index = triids[iface]
                fpec[iface] = tet_is_pec[tri_index]
                i1 = tris[0, tri_index]
                i2 = tris[1, tri_index]
                i3 = tris[2, tri_index]
                nrm = calc_inward_normal(nodes[:, i1], nodes[:, i2], nodes[:, i3], tetc)
                for d in range(3):
                    fnorm[iface, d] = nrm[d]
                area = areas[tri_index]
                farea[iface] = area
                Rf_face[iface, itet] = (area / np.pi) ** 0.5
                adj = -1
                for j in range(2):
                    num = int(tri_to_tet[j, tri_index])
                    if adj < 0 and num != itet and num != -1234:
                        adj = num
                fadj[iface] = adj
                fadjface[iface] = 0
                if adj >= 0:
                    for j in range(4):
                        if tet_to_tri[j, adj] == tri_index:
                            fadjface[iface] = j
                    adj_tets_mat[iface, itet] = adj
            Gt = 1j * W0 * np.exp(-1j * k0 * Rt) / (4 * np.pi * Rt + DEPS)
            ci = -1 / (1j * MU0 * W0)
            for ex in range(NEX):
                for idof in range(ND):
                    Ef[idof] = Efields[tet_to_field[idof, itet], ex]
                # Qt term
                div = 0j
                for q in range(NV):
                    acc = 0j
                    for idof in range(ND):
                        acc += Ef[idof] * Dv[idof, q]
                    div += WV[q] * (erc * acc)
                Qt = TET_VOLUME * EPS0 * div
                # Jt term
                jj = 0j
                for d in range(3):
                    rv1 = 0j
                    for idof in range(ND):
                        rv1 += Ef[idof] * CC[idof, d]
                    rv1 = rv1 * (1 / urc)
                    s = 0j
                    for q in range(NV):
                        acc = 0j
                        for idof in range(ND):
                            acc += Ef[idof] * Fv[idof, d, q]
                        s += (-(k0 ** 2) * (erc * acc) + rv1) * WV[q]
                    jt = -TET_VOLUME * (1 / (1j * W0 * MU0)) * s
                    jj += jt * jt
                alpha_t[ex, itet] = -Gt / (erc * EPS0 + DEPS) * Qt * Qt - Gt * urc * MU0 * jj
                # face residuals
                for d in range(3):
                    for q in range(NF):
                        a1 = 0j
                        a2 = 0j
                        for idof in range(ND):
                            a1 += Ef[idof] * Ff[idof, d, q]
                            a2 += Ef[idof] * Cf[idof, d, q]
                        Eface[d, q] = a1
                        Cface[d, q] = a2
                for iface in range(4):
                    adj = fadj[iface]
                    if adj < 0:
                        continue
                    n0 = fnorm[iface, 0]
                    n1 = fnorm[iface, 1]
                    n2 = fnorm[iface, 2]
                    area = farea[iface]
                    s0 = iface * N2D
                    if fpec[iface]:
                        sc = Y0 * np.sqrt(1 / urc)
                        j0 = 0j
                        j1_ = 0j
                        j2 = 0j
                        for q in range(N2D):
                            e0 = Eface[0, s0 + q]
                            e1 = Eface[1, s0 + q]
                            e2 = Eface[2, s0 + q]
                            c0 = -(n1 * e2 - n2 * e1)
                            c1 = -(n2 * e0 - n0 * e2)
                            c2 = -(n0 * e1 - n1 * e0)
                            j0 += sc * (n1 * c2 - n2 * c1) * WQ[q]
                            j1_ += sc * (n2 * c0 - n0 * c2) * WQ[q]
                            j2 += sc * (n0 * c1 - n1 * c0) * WQ[q]
                        Jf_i1[ex, iface, 0, itet] = j0 * area
                        Jf_i1[ex, iface, 1, itet] = j1_ * area
                        Jf_i1[ex, iface, 2, itet] = j2 * area
                        continue
                    ia = fadjface[iface]
                    j0 = 0j
                    j1_ = 0j
                    j2 = 0j
                    qq = 0j
                    for q in range(N2D):
                        h0 = ci * ((1 / urc) * Cface[0, s0 + q])
                        h1 = ci * ((1 / urc) * Cface[1, s0 + q])
                        h2 = ci * ((1 / urc) * Cface[2, s0 + q])
                        j0 += (n1 * h2 - n2 * h1) * WQ[q]
                        j1_ += (n2 * h0 - n0 * h2) * WQ[q]
                        j2 += (n0 * h1 - n1 * h0) * WQ[q]
                        qq += (n0 * (erc * EPS0 * Eface[0, s0 + q]) + n1 * (erc * EPS0 * Eface[1, s0 + q])
                               + n2 * (erc * EPS0 * Eface[2, s0 + q])) * WQ[q]
                    Jf_i1[ex, iface, 0, itet] = j0 * area
                    Jf_i1[ex, iface, 1, itet] = j1_ * area
                    Jf_i1[ex, iface, 2, itet] = j2 * area
                    Jf_i2[ex, ia, 0, adj] = -j0 * area
                    Jf_i2[ex, ia, 1, adj] = -j1_ * area
                    Jf_i2[ex, ia, 2, adj] = -j2 * area
                    Qf_i1[ex, iface, itet] = qq * area
                    Qf_i2[ex, ia, adj] = -qq * area

    error = np.zeros((N_TETS,), dtype=np.float64)
    for it in prange(N_TETS):
        best = 0.0
        bad = False
        for ex in range(NEX):
            at = alpha_t[ex, it]
            af = 0j
            for f in range(4):
                Rf = Rf_face[f, it]
                Gf = 1j * W0 * np.exp(-1j * k0 * Rf) / (4 * np.pi * Rf + DEPS)
                Qi = Qf_i1[ex, f, it] - Qf_i2[ex, f, it]
                Jx = Jf_i1[ex, f, 0, it] - Jf_i2[ex, f, 0, it]
                Jy = Jf_i1[ex, f, 1, it] - Jf_i2[ex, f, 1, it]
                Jz = Jf_i1[ex, f, 2, it] - Jf_i2[ex, f, 2, it]
                aD = -Gf / (er[it] * EPS0) * (Qi * Qi) - Gf * (ur[it] * MU0) * (Jx * Jx + Jy * Jy + Jz * Jz)
                it2 = adj_tets_mat[f, it]
                aN = alpha_t[ex, it2] if it2 != -1 else 0j
                af += (at / (at + aN + DEPS)) * aD
            e = np.abs(at + af) ** 0.5
            if e != e:
                bad = True
            elif e > best:
                best = e
        error[it] = np.nan if bad else best
    return error, max_elem_size
'''


def _integrated(body: str) -> str:
    """Face integrals inside the loop (see above); the body unchanged when the source differs."""
    out = body
    for old, new in _INT_REPL:
        if out.count(old) != 1:
            return body
        out = out.replace(old, new)
    a, b = out.find(_INT_EPI_START), out.find(_INT_EPI_END)
    if a < 0 or b < a:
        return body
    out = out[:a] + _INT_EPI_NEW + out[b + len(_INT_EPI_END):]
    if "areas_face_residual" in out or "Qf_face" in out or "Jf_face" in out:
        return body
    return out


def _source() -> str | None:
    from emerge._emerge.physics.microwave import adaptive_mesh as am
    src = inspect.getsource(am)
    i = src.find("def compute_error_single(")
    if i < 0:
        return None
    start = src.rfind("@njit(", 0, i)
    end = src.find("    return error, max_elem_size\n", i)
    if start < 0 or end < 0:
        return None
    fn = src[start:end + len("    return error, max_elem_size\n")]
    head, body = fn[:fn.index("def compute_error_single(")], fn[fn.index("def compute_error_single("):]
    if head.count("cache=True") != 1 or body.count(_LOOP) != 1:
        return None
    # numpy error model: a division by zero gives inf/nan instead of raising (which forbids threads)
    head = head.replace("cache=True", 'cache=True,\n    parallel=True,\n    error_model="numpy"')
    a, b = body.find(_NAN_START), body.find(_NAN_END)
    if a < 0 or b < a:
        return None
    body = body[:a] + body[b + len(_NAN_END):]
    for old, new in _REPL:
        if body.count(old) != 1:
            return None
        body = body.replace(old, new)
    body = body.replace(_LOOP, "    for itet in prange(N_TETS):\n")
    body = _integrated(body)
    return (
        '"""Generated by Hertzon from EMerge\'s adaptive_mesh.compute_error_single (parallel tet loop)."""\n'
        "from numba import prange\n"
        "from emerge._emerge.physics.microwave import adaptive_mesh as _am\n"
        "globals().update({k: v for k, v in vars(_am).items() if not k.startswith('__')})\n"
        "from numba import prange  # noqa: E402,F811\n\n\n"
        + head + body.replace("def compute_error_single(", "def compute_error_parallel(")
    )


def cached() -> bool:
    """True when the compiled kernels are on disk (no compile at the next start)."""
    src = _source() if "src" not in _STATE else _STATE["src"]
    _STATE["src"] = src
    if src is None:
        return True
    tag = hashlib.sha1(src.encode()).hexdigest()[:12]
    mtag = hashlib.sha1((src + _MULTI).encode()).hexdigest()[:12]
    pc = _cache_dir() / "__pycache__"
    return (any(pc.glob(f"error_estimate_{tag}.compute_error_parallel-*.nbi"))
            and any(pc.glob(f"error_multi_{mtag}.compute_error_multi-*.nbi")))


def _cache_dir() -> Path:
    return Path(os.environ.get("HERTZON_HOME") or (Path.home() / ".hertzon")) / "kernels"


def _load(prefix: str, src: str, tag: str, fname: str):
    """Write the generated module next to its numba disk cache and import it (compiles, or loads the cache)."""
    import importlib.util
    import sys
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{prefix}_{tag}.py"
    if not path.exists() or path.read_text(encoding="utf-8") != src:
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(src, encoding="utf-8")
        os.replace(tmp, path)
        for f in d.glob(f"{prefix}_*.py"):     # kernels of earlier EMerge or Hertzon versions
            if f != path:
                try:
                    f.unlink()
                except OSError:
                    pass
    name = f"_hertzon_{prefix}_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, fname)


def kernel():
    """The parallel kernel, or None (EMerge's serial one is used then)."""
    if "fn" in _STATE:
        return _STATE["fn"]
    _STATE["fn"] = None
    if os.environ.get("HERTZON_SERIAL_ERROR") == "1":
        return None
    try:
        src = _source()
        if src is None:
            return None
        tag = hashlib.sha1(src.encode()).hexdigest()[:12]
        _STATE["fn"] = _load("error_estimate", src, tag, "compute_error_parallel")   # signature given: compiles here
    except Exception:  # noqa: BLE001
        _STATE["fn"] = None
    return _STATE["fn"]


def multi_kernel():
    """The kernel for all excitations at once (written for the EMerge source _source() accepts), or None."""
    if "multi" in _STATE:
        return _STATE["multi"]
    _STATE["multi"] = None
    if os.environ.get("HERTZON_SERIAL_ERROR") == "1" or os.environ.get("HERTZON_ERROR_PER_PORT") == "1":
        return None
    try:
        src = _source()
        if src is None:
            return None
        tag = hashlib.sha1((src + _MULTI).encode()).hexdigest()[:12]
        _STATE["multi"] = _load("error_multi", _MULTI, tag, "compute_error_multi")
    except Exception:  # noqa: BLE001
        _STATE["multi"] = None
    return _STATE["multi"]


def _group_size(n_tets: int, n_ex: int) -> int:
    """Excitations per kernel call: the face integrals take about 0.5 kB per tetrahedron and excitation."""
    try:
        import psutil
        budget = 0.25 * psutil.virtual_memory().available
    except Exception:  # noqa: BLE001
        budget = 2 * 2**30
    return max(1, min(n_ex, int(budget // (528 * max(n_tets, 1)))))


def apply() -> bool:
    """Route EMerge's error estimate through the parallel kernel. True when it is active."""
    from emerge._emerge.physics.microwave import adaptive_mesh as am
    fn = kernel()
    if fn is None:
        return False
    serial = _STATE.setdefault("serial", am.compute_error_single)

    check = os.environ.get("HERTZON_CHECK_ERROR") == "1"

    tune = _Tuner()

    def compute_error_single(*args):
        t0 = time.time()
        with tune.threads(int(args[1].shape[1])):
            error, sizes = fn(*args)
        if not (error == error).all():     # NaN: let the serial kernel report the element
            return serial(*args)
        if check:                          # development: compare with EMerge's serial kernel
            t1 = time.time()
            e0, s0 = serial(*args)
            t2 = time.time()
            rel = float(np.max(np.abs(error - e0)) / max(float(np.max(np.abs(e0))), 1e-300))
            print(f"[error estimate] parallel {t1 - t0:.3f} s, serial {t2 - t1:.3f} s, "
                  f"max rel. difference {rel:.2e}, sizes equal {bool(np.array_equal(sizes, s0))}", flush=True)
        return error, sizes

    am.compute_error_single = compute_error_single

    multi = multi_kernel()
    per_port = _STATE.setdefault("estimate", am.compute_error_estimate)
    if multi is None:
        return True

    def compute_error_estimate(field, pec_tris):
        """All excitations in one pass over the mesh: the basis functions at the quadrature points are
        evaluated once per tetrahedron instead of once per excitation."""
        keys = list(field._fields.keys())
        if not keys:
            return per_port(field, pec_tris)
        mesh = field.mesh
        nt = int(mesh.tets.shape[1])
        pec = np.sort(np.unique(np.array(pec_tris))).astype(np.int64)
        grp = _group_size(nt, len(keys))
        t0 = time.time()
        error = None
        for i in range(0, len(keys), grp):
            E = np.ascontiguousarray(np.stack([np.asarray(field._fields[k], dtype=np.complex128)
                                               for k in keys[i:i + grp]], axis=1))
            with tune.threads(nt):
                err, sizes = multi(mesh.nodes, mesh.tets, mesh.tris, mesh.edges, mesh.centers, E,
                                   mesh.edge_lengths, mesh.areas, mesh.tet_to_edge, mesh.tet_to_tri,
                                   mesh.tri_to_tet, field.basis.tet_to_field, field._der, field._dur, pec,
                                   float(field.k0), field.basis.dofcodes3d)
            error = err if error is None else np.maximum(error, err)
        if not (error == error).all():     # NaN: the per-excitation path reports the element
            return per_port(field, pec_tris)
        if check:
            t1 = time.time()
            e0, s0 = per_port(field, pec_tris)
            t2 = time.time()
            rel = float(np.max(np.abs(error - e0)) / max(float(np.max(np.abs(e0))), 1e-300))
            print(f"[error estimate] {len(keys)} excitations together {t1 - t0:.3f} s, one by one {t2 - t1:.3f} s, "
                  f"max rel. difference {rel:.2e}, sizes equal {bool(np.array_equal(sizes, s0))}", flush=True)
        return error, sizes

    am.compute_error_estimate = compute_error_estimate
    return True


class _Tuner:
    """Thread count of the parallel kernel, measured on the machine. Every tetrahedron allocates small
    work arrays, so with many cores the threads queue at the memory allocator (64 cores were only about 6
    times faster than one). The first calls of a pass (one per excitation and frequency) try all cores,
    a half, a quarter, ...; the fastest per tetrahedron is kept, and measured again when the mesh has
    grown four times."""

    def __init__(self):
        import numba
        self.numba = numba
        self.max = int(numba.config.NUMBA_NUM_THREADS)
        self.best = None
        self.size = 0
        self.trials: dict[int, float] = {}
        self.todo: list[int] = []

    def _start(self, nt: int) -> None:
        cur = int(self.numba.get_num_threads())
        cands, n = [], cur
        while n >= 4 and len(cands) < 5:
            cands.append(n)
            n //= 2
        self.todo, self.trials, self.size = cands if len(cands) > 1 else [], {}, nt
        if not self.todo:
            self.best = cur

    @contextmanager
    def threads(self, nt: int):
        if os.environ.get("HERTZON_ERROR_THREADS"):
            self.best = int(os.environ["HERTZON_ERROR_THREADS"])
        elif nt >= 20000 and (self.size == 0 or nt >= 4 * self.size) and not self.todo:
            self._start(nt)
        cur = int(self.numba.get_num_threads())
        n = self.todo[0] if self.todo else (self.best or cur)
        n = max(1, min(n, cur))
        self.numba.set_num_threads(n)
        t0 = time.time()
        try:
            yield
        finally:
            self.numba.set_num_threads(cur)
            if self.todo and self.todo[0] == n:
                self.trials[n] = (time.time() - t0) / max(nt, 1)
                self.todo.pop(0)
                if not self.todo:
                    self.best = min(self.trials, key=self.trials.get)
                    _STATE["threads"] = (self.best, dict(self.trials))
