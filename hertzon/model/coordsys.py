"""Relative and face coordinate systems (Modeler > Coordinate System).

A coordinate system has an origin and axes given in its reference system (Global or another CS).
`frame()` resolves the chain to global: origin O (metres) and rotation R whose columns are the
local x, y, z axes in global coordinates. Local -> global: p_g = O + R @ p_l.
"""
from __future__ import annotations

import math

import numpy as np

GLOBAL = "Global"
MODES = ("Axis/Position", "Euler Angle ZXZ", "Euler Angle ZYZ")


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-15:
        raise ValueError("axis of zero length")
    return v / n


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def local_frame(cs, ev) -> tuple[np.ndarray, np.ndarray]:
    """Origin and axes of cs in its reference system."""
    p = cs.props
    O = np.asarray(ev.point(p.get("Origin", "0,0,0")), float)
    mode = p.get("Mode", MODES[0])
    if mode == "Axis/Position":
        # X axis vector and a vector in the XY plane, both relative to the origin
        x = _unit(ev.point(p.get("XAxis", "1,0,0")))
        yp = np.asarray(ev.point(p.get("YPoint", "0,1,0")), float)
        y = yp - (yp @ x) * x
        y = _unit(y)
        z = np.cross(x, y)
        R = np.c_[x, y, z]
    else:
        a, b, c = (ev.angle(p.get(k, "0deg")) for k in ("Phi", "Theta", "Psi"))
        mid = _rx if mode == "Euler Angle ZXZ" else _ry
        R = _rz(a) @ mid(b) @ _rz(c)
    return O, R


def frame(design, name: str, ev, _seen=None) -> tuple[np.ndarray, np.ndarray]:
    """Global origin (m) and rotation of coordinate system `name`."""
    if not name or name == GLOBAL:
        return np.zeros(3), np.eye(3)
    _seen = set() if _seen is None else _seen
    if name in _seen:
        raise ValueError(f"circular coordinate system reference through '{name}'")
    _seen.add(name)
    cs = design.cs(name)
    O0, R0 = frame(design, cs.reference, ev, _seen)
    o, r = local_frame(cs, ev)
    return O0 + R0 @ o, R0 @ r


def matrix(O, R) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = O
    return M
