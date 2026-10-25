"""Automatic integration line of a lumped port or lumped RLC sheet: from the conductor on one side of the
sheet to the conductor on the other side, centre of the contact to centre of the contact."""
from __future__ import annotations

import numpy as np


def boundary_segments(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Outline of a triangulated sheet as (K, 2, 3) segments (edges used by one triangle only)."""
    if len(tris) == 0:
        return np.zeros((0, 2, 3))
    e = np.sort(np.r_[tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1)
    u, n = np.unique(e, axis=0, return_counts=True)
    u = u[n == 1]
    return np.asarray(points, float)[u]


def _samples(seg: np.ndarray, n: int = 16) -> np.ndarray:
    """n points per segment, ends included, so a contact shorter than a sheet edge is still found."""
    t = np.linspace(0.0, 1.0, n)[None, :, None]
    return (seg[:, :1] * (1 - t) + seg[:, 1:] * t).reshape(-1, 3)


def point_triangle_distance(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Distance of each point (N,3) to the nearest triangle (M,3,3)."""
    out = np.full(len(P), np.inf)
    if len(P) == 0 or len(T) == 0:
        return out
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, ac = b - a, c - a
    for s in range(0, len(P), 256):
        p = P[s:s + 256, None, :]
        ap = p - a
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        bp = p - b
        d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        cp = p - c
        d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        den = va + vb + vc
        den = np.where(np.abs(den) > 1e-300, den, 1e-300)
        v, w = vb / den, vc / den
        q = a + v[..., None] * ab + w[..., None] * ac           # inside the triangle
        # outside: nearest point on the three edges
        def on_edge(p0, p1):
            d = p1 - p0
            t = np.clip(((p - p0) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-300), 0, 1)
            return p0 + t[..., None] * d
        inside = (va >= 0) & (vb >= 0) & (vc >= 0)
        cand = [np.where(inside[..., None], q, np.inf)] + [on_edge(a, b), on_edge(b, c), on_edge(c, a)]
        dist = np.min([np.linalg.norm(p - x, axis=-1) for x in cand], axis=0)
        out[s:s + 256] = dist.min(1)
    return out


def between_conductors(sheet_points: np.ndarray, sheet_tris: np.ndarray,
                       conductors: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Start and end point of the line between the two conductors that touch the sheet outline.

    ``conductors`` maps a name to its surface triangles (M,3,3). The two conductors with the longest contact
    are used; the line runs from the centre of one contact to the centre of the other, pointing along the
    positive direction of its main axis (ground to trace for a vertical port). None when fewer than two
    conductors touch the sheet."""
    seg = boundary_segments(sheet_points, sheet_tris)
    if len(seg) == 0:
        return None
    P = _samples(seg)
    lo, hi = P.min(0), P.max(0)
    size = float(np.linalg.norm(hi - lo))
    if size <= 0:
        return None
    tol = 1e-3 * size
    contacts = []
    for name, T in conductors.items():
        T = np.asarray(T, float).reshape(-1, 3, 3)
        if len(T) == 0:
            continue
        # only triangles near the sheet
        keep = ((T.min(1) <= hi + tol) & (T.max(1) >= lo - tol)).all(1)
        T = T[keep]
        if len(T) == 0:
            continue
        hit = P[point_triangle_distance(P, T) <= tol]
        if len(hit) == 0:
            continue
        clo, chi = hit.min(0), hit.max(0)
        if np.linalg.norm(chi - clo) >= 0.98 * size:
            continue                     # touches the whole outline: the sheet lies on it, not between
        contacts.append((float(np.linalg.norm(chi - clo)), name, (clo + chi) / 2))
    if len(contacts) < 2:
        return None
    contacts.sort(key=lambda c: -c[0])
    p0, p1 = contacts[0][2], contacts[1][2]
    v = p1 - p0
    if np.linalg.norm(v) <= tol:
        return None
    if v[int(np.argmax(np.abs(v)))] < 0:
        p0, p1 = p1, p0
    return p0, p1
