"""Incident plane wave vectors (Cartesian or Spherical vector setup, zero phase position)."""
from __future__ import annotations

import numpy as np

from .expr import split_list


def _vec(ev, text: str, default) -> np.ndarray:
    try:
        parts = split_list(text)
        if len(parts) == 3:
            return np.array([float(np.real(ev.value(p))) for p in parts])
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(default, float)


def vectors(props: dict, ev) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit propagation vector k, unit E0 vector (made orthogonal to k) and the zero phase position (m)."""
    if props.get("VectorInput", "Spherical") == "Cartesian":
        k = _vec(ev, props.get("KVector", "0,0,1"), (0, 0, 1))
        e = _vec(ev, props.get("E0Vector", "1,0,0"), (1, 0, 0))
        k = k / (np.linalg.norm(k) or 1.0)
        e = e - (e @ k) * k
        if np.linalg.norm(e) < 1e-9:                      # E along k: take any perpendicular direction
            e = np.cross(k, [0, 0, 1] if abs(k[2]) < 0.9 else [1, 0, 0])
        e = e / np.linalg.norm(e)
    else:
        th, ph = ev.angle(props.get("Theta", "0deg")), ev.angle(props.get("Phi", "0deg"))
        psi = ev.angle(props.get("Polarization", "0deg"))
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        k = -np.array([st * cp, st * sp, ct])
        e = np.cos(psi) * np.array([ct * cp, ct * sp, -st]) + np.sin(psi) * np.array([-sp, cp, 0.0])
    try:
        origin = np.asarray(ev.point(props.get("ZeroPhasePosition", "0,0,0") or "0,0,0"), float)
    except Exception:  # noqa: BLE001
        origin = np.zeros(3)
    return k, e, origin


def incidence_angles(k: np.ndarray, e: np.ndarray) -> tuple[float, float, float]:
    """(theta, phi, psi) in degrees with k = -r(theta, phi) and E = cos(psi) theta^ + sin(psi) phi^."""
    r = -k
    th = float(np.arccos(np.clip(r[2], -1, 1)))
    ph = float(np.arctan2(r[1], r[0])) if np.hypot(r[0], r[1]) > 1e-12 else 0.0
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    t_hat = np.array([ct * cp, ct * sp, -st])
    p_hat = np.array([-sp, cp, 0.0])
    psi = float(np.arctan2(e @ p_hat, e @ t_hat))
    return np.degrees(th), np.degrees(ph), np.degrees(psi)
