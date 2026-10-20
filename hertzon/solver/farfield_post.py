"""Far fields as post-processing (an Infinite Sphere can be added or edited after the solve).

The solver stores the tangential E/H on the open boundary (radiation/PML faces) once per solution in
``huygens_<solution>.npz``. Every infinite sphere is then evaluated from that file with the
Stratton-Chu integral, without the field solution or a new analysis.
"""
from __future__ import annotations

import numpy as np

Z0 = 376.730313412


_CEXP = None


def _cexp(x: np.ndarray) -> np.ndarray:
    """exp(1j * x) on all cores (numpy's exp is single-threaded and dominated the far field time)."""
    global _CEXP
    if _CEXP is None:
        try:
            from numba import njit, prange
            import math

            @njit(parallel=True, cache=True, fastmath=False)
            def k(a):
                out = np.empty(a.shape, np.complex128)
                for i in prange(a.shape[0]):
                    for j in range(a.shape[1]):
                        out[i, j] = complex(math.cos(a[i, j]), math.sin(a[i, j]))
                return out
            _CEXP = k
        except Exception:  # noqa: BLE001
            _CEXP = lambda a: np.exp(1j * a)  # noqa: E731
    return _CEXP(np.ascontiguousarray(x, np.float64))


def stratton_chu(E: np.ndarray, H: np.ndarray, vis: np.ndarray, wns: np.ndarray,
                 th: np.ndarray, ph: np.ndarray, k0: float, progress=None) -> np.ndarray:
    """Far field E (3, N) (r*exp(jkr) normalised) radiated by the surface fields (same formula as EMerge)."""
    return stratton_chu_multi([E], [H], vis, wns, th, ph, k0, progress)[0]


def stratton_chu_multi(Es: list, Hs: list, vis: np.ndarray, wns: np.ndarray, th: np.ndarray, ph: np.ndarray,
                       k0: float, progress=None) -> list:
    """stratton_chu for several excitations at one frequency: the phase matrix exp(jk r.x) is the same for
    all of them, so it is computed once and multiplied with every excitation in one BLAS call."""
    Es = [np.asarray(E, np.complex128) for E in Es]
    Hs = [np.asarray(H, np.complex128) for H in Hs]
    keep = np.zeros(vis.shape[1], bool)
    for E in Es:                        # sources that matter for any excitation
        mag = np.sqrt((np.abs(E) ** 2).sum(0))
        if mag.size:
            keep |= mag > mag.max() * 1e-3
    vis, wns = vis[:, keep], wns[:, keep]
    r = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]).T   # (N, 3)
    ne = len(Es)
    # columns: n x E then n x H of every excitation (M, 6 * ne)
    src = np.concatenate([np.cross(wns.T, E[:, keep].T) for E in Es] + [np.cross(wns.T, H[:, keep].T) for H in Hs],
                         axis=1)
    out = np.zeros((ne, r.shape[0], 3), np.complex128)
    # sum over the sources first: A = sum G n x E, B = sum G n x H (r is constant per direction)
    step = max(1, 4_000_000 // max(1, src.shape[0]))
    for a in range(0, r.shape[0], step):
        rr = r[a:a + step]
        G = _cexp(k0 * (rr @ vis))                               # (n, M)
        AB = (G @ src).reshape(len(rr), 2, ne, 3)
        for i in range(ne):
            A, B = AB[:, 0, i], AB[:, 1, i]
            out[i, a:a + step] = -1j * k0 / (4 * np.pi) * np.cross(rr, A - Z0 * np.cross(rr, B))
        if progress is not None:
            progress(min(1.0, (a + step) / r.shape[0]))
    return [o.T for o in out]


def compute_sphere(huy: dict, props: dict, ev, design=None, plane_wave=None, progress=None) -> dict:
    """The far field file content (farfield_<sphere>_<solution>.npz) of one infinite sphere."""
    th = np.arange(ev.angle(props["ThetaStart"]), ev.angle(props["ThetaStop"]) + 1e-9,
                   max(ev.angle(props["ThetaStep"]), 1e-3))
    ph = np.arange(ev.angle(props["PhiStart"]), ev.angle(props["PhiStop"]) + 1e-9,
                   max(ev.angle(props["PhiStep"]), 1e-3))
    res = {"theta": th, "phi": ph}
    R = None
    csn = props.get("CoordinateSystem", "Global") or "Global"
    if csn != "Global" and design is not None:
        from ..model import coordsys
        try:
            O, R = coordsys.frame(design, csn, ev)
            res["cs_origin"], res["cs_R"] = O, R
        except (KeyError, ValueError):
            R = None
    T, P = np.meshgrid(th, ph, indexing="ij")
    Tg, Pg = T.ravel(), P.ravel()
    if R is not None:
        rl = np.stack([np.sin(Tg) * np.cos(Pg), np.sin(Tg) * np.sin(Pg), np.cos(Tg)])
        rg = R @ rl
        Tg, Pg = np.arccos(np.clip(rg[2], -1, 1)), np.arctan2(rg[1], rg[0])
    vis, wns = huy["vis"], huy["wns"]
    if "E0" in huy:
        res["E0"] = huy["E0"]
    keys = [k for k in huy if k.startswith("E_")]
    by_freq: dict[str, list] = {}
    for key in keys:
        by_freq.setdefault(key.split("_")[1], []).append(key)
    done = 0
    for k, group in by_freq.items():
        f = float(huy[f"freq_{k}"][0])
        k0 = 2 * np.pi * f / 299792458.0
        ports = [key.split("_")[2] for key in group]
        Es = stratton_chu_multi([huy[key] for key in group], [huy[f"H_{k}_{p}"] for p in ports], vis, wns, Tg, Pg,
                                k0, None if progress is None else
                                (lambda x, d=done, n=len(group): progress((d + x * n) / len(keys))))
        done += len(group)
        for key, port, E in zip(group, ports, Es):
            if R is not None:
                E = R.T @ E
            res[key] = E.reshape((3,) + T.shape)
            res[f"Prad_{k}_{port}"] = huy[f"Prad_{k}_{port}"]
        res[f"freq_{k}"] = np.array([f])
        if "mono_dir" in huy:
            tb, pb = [float(x) for x in huy["mono_dir"]]
            key = group[-1]
            res[f"Emono_{k}"] = stratton_chu(huy[key], huy[f"H_{k}_{ports[-1]}"], vis, wns,
                                             np.array([tb]), np.array([pb]), k0).reshape(3)
            res["mono_dir"] = huy["mono_dir"]
    return res


# the GUI replaces this with a runner that computes in a worker thread and shows a progress bar
RUN = lambda text, fn: fn(None)  # noqa: E731


def farfield(st, design, sphere: str, label: str = "LastAdaptive", ev=None):
    """Far field data of a sphere: computed from the stored boundary fields when the sphere is new
    or was edited after the solve, cached in farfield_<sphere>_<label>.npz."""
    import json
    sph = next((s for s in design.spheres if s.name == sphere), None)
    huy = st.npz(f"huygens_{label}.npz")
    name = f"farfield_{sphere}_{label}.npz"
    if sph is None or huy is None:
        return st.npz(name)
    sig = json.dumps({k: str(v) for k, v in sorted(sph.props.items())}) + str(int(huy["stamp"][0]))
    old = st.npz(name)
    if old is not None and "sig" in old and str(old["sig"][0]) == sig:
        return old
    if ev is None:
        ev = design.evaluator(None)
    res = RUN(f"Computing far field '{sphere}'",
              lambda prog: compute_sphere(huy, sph.props, ev, design, progress=prog))
    res["sig"] = np.array([sig])
    try:
        st.save_npz(name, **res)
    except OSError:
        pass
    return res
