"""Center point arcs (AngularArc polyline segments), shared by the geometry kernel and the draw preview."""
from __future__ import annotations

import numpy as np

# plane of the arc -> its normal; positive angles turn counter-clockwise about it (XZ is the same plane as ZX)
ARC_PLANES = {"XY": (0.0, 0.0, 1.0), "YZ": (1.0, 0.0, 0.0), "ZX": (0.0, 1.0, 0.0), "XZ": (0.0, 1.0, 0.0)}


def plane_normal(plane: str) -> np.ndarray:
    return np.asarray(ARC_PLANES.get(str(plane).strip().upper(), ARC_PLANES["XY"]), float)


def arc_center(start, center, plane: str = "XY") -> np.ndarray:
    """The center moved into the arc plane through the start point."""
    s, c = np.asarray(start, float), np.asarray(center, float)
    ax = plane_normal(plane)
    return c + ((s - c) @ ax) * ax


def angular_arc(start, center, angle: float, plane: str = "XY", n: int = 0) -> np.ndarray:
    """Points of the arc from start about center by angle (radians, counter-clockwise about the plane normal).

    The center is moved into the plane of the start point. n = 0 gives one point per started 90 degrees
    (the pieces handed to OCC), otherwise n points; the first point is start, the last the end point."""
    s = np.asarray(start, float)
    c = arc_center(start, center, plane)
    ax = plane_normal(plane)
    v = s - c
    if n <= 0:
        n = max(2, int(np.ceil(abs(angle) / (np.pi / 2) - 1e-9)) + 1)
    t = np.linspace(0.0, angle, n)
    w = np.cross(ax, v)
    return c + np.cos(t)[:, None] * v + np.sin(t)[:, None] * w
