"""Gerber X2 and Excellon export of a design (File > Export > Gerber).

How the board geometry is obtained
----------------------------------
* The design is built with the normal OpenCASCADE kernel (:mod:`hertzon.kernel.occ`) into a gmsh
  model of its own, so the export sees the real CAD geometry (exact lines and circular arcs, SI metres).
* Everything is moved into the *board frame*: the chosen plane (XY, YZ or ZX) of a coordinate system
  becomes ``z = 0`` with the plane's first axis as board X and its second axis as board Y. All copper
  layers, the outline and the drill files share this frame and one export origin. Nothing is mirrored:
  every layer is written as seen from the board top (+normal).
* Objects are classified from the model data:

  - copper: sheets (or faces of solids) with a *Perfect E* or *Finite Conductivity* boundary, and solids
    made of a conducting material (PEC or conductivity >= 1e5 S/m);
  - vias: conducting solids whose curved faces are coaxial cylinders along the board normal and that
    reach two or more copper layers;
  - board: solids of a dielectric material (outline and non-plated holes);
  - never exported: the air region, non-model objects, port sheets (excitations), sheets carrying
    radiation/PML/symmetry/master-slave/RLC/impedance boundaries and vacuum/air bodies.

* Copper layers are found from the elevations (board Z) of the copper: planar sheets parallel to the
  board give one elevation, conducting solids not thicker than ``max_copper_thickness`` give an interval;
  touching or overlapping intervals form one layer. Layers are numbered from the top (L1 = Top).
* Layer geometry: sheets parallel to the board are used directly with their own edges (2D contours).
  Conducting solids are cut by a plane parallel to the board at half their height (OCC section), or at
  the layer elevation when assigned to a layer by hand; there is no projection of whole bodies.
* All faces of a layer are fragmented together (OCC boolean); edges shared by two fragments are internal,
  the remaining edges are chained into closed contours. Contours are classified by orientation and
  nesting: outer contours are written as dark regions (%LPD), holes/slots as clear regions (%LPC), and
  islands inside holes are written after the clear polarity of their surrounding copper.
* Edges that are straight lines or circular arcs (also full circles) are written exactly (G01, G02/G03 in
  G75 mode with I/J centre offsets). Other curves (ellipses, splines) are replaced by a polyline whose
  chord deviation is at most ``tolerance``.
* Drill data: plated holes (PTH) come only from via solids, non-plated holes (NPTH) only from circular
  holes through the board body that hold no via. Cut-outs in copper are never turned into drills.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from ..kernel import occ as K
from ..model import coordsys
from ..model.design import Design, Project
from ..model.materials import is_conductor

UNITS = {"mm": 1e3, "inch": 1.0 / 0.0254}          # metres -> file unit
PLANES = {"XY": (0, 1, 2), "YZ": (1, 2, 0), "ZX": (2, 0, 1)}   # (u, v, normal) columns of the CS axes
COPPER_BOUNDARIES = {"Perfect E", "Finite Conductivity"}
HELPER_BOUNDARIES = {"Radiation", "PML", "Symmetry", "Primary", "Secondary", "Perfect H", "Impedance",
                     "Lumped RLC", "Layered Impedance"}
MAX_DRILL = 6.5e-3          # larger circular board holes stay in the outline as milled cut-outs
PROFILE_WIDTH = 0.1e-3      # line width of the board outline (Profile) in metres

Pt = tuple[float, float]


# ================================================================================================
# data
# ================================================================================================
@dataclass
class Options:
    units: str = "mm"                    # "mm" or "inch"
    cs: str = "Global"                   # coordinate system of the board plane
    plane: str = "XY"                    # plane of that coordinate system: XY, YZ or ZX
    origin: str = "cs"                   # "cs" (CS origin), "outline" (lower left of outline), "point"
    origin_point: Pt = (0.0, 0.0)        # metres, board X/Y, used for origin == "point"
    tolerance: float = 1e-6              # metres, chord tolerance for curves that are no lines/arcs
    max_copper_thickness: float = 0.2e-3  # metres, thicker conductors are no layer copper
    outline: bool = True
    drills: bool = True
    roles: dict[str, str] = field(default_factory=dict)          # item key -> "skip"/"outline"/"via"/"L<n>"
    layer_names: dict[str, str] = field(default_factory=dict)    # layer id -> user name
    layer_enabled: dict[str, bool] = field(default_factory=dict)  # layer id -> write the file


@dataclass
class Msg:
    level: str                           # "error", "warning" or "info"
    text: str
    objects: list[str] = field(default_factory=list)


@dataclass
class Item:
    """One exportable thing: a whole object or one face of a solid with a conductor boundary."""
    key: str
    obj: str
    face: int = -1
    kind: str = "solid"                  # "sheet", "solid" or "line"
    category: str = "other"              # copper / conductor / via / board / excluded / ambiguous / other / error
    note: str = ""
    z0: float = math.nan
    z1: float = math.nan
    via: tuple[float, float, float] | None = None    # centre x, y and drill diameter (board frame)
    suggested: str = "skip"
    role: str = "skip"
    tags: list = field(default_factory=list, repr=False)

    @property
    def label(self) -> str:
        return self.obj if self.face < 0 else f"{self.obj} (face {self.face})"


@dataclass
class Loop:
    """Closed contour in the board frame (metres). Segments: ("L", p0, p1) or ("A", p0, p1, c, ccw)."""
    segs: list[tuple]
    _poly: np.ndarray | None = field(default=None, repr=False)

    def polygon(self) -> np.ndarray:
        if self._poly is None:
            pts = []
            for s in self.segs:
                pts += _sample_seg(s)[:-1]
            self._poly = np.asarray(pts, float).reshape(-1, 2)
        return self._poly

    @property
    def area(self) -> float:
        p = self.polygon()
        x, y = p[:, 0], p[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def contains(self, q: Pt) -> bool:
        p = self.polygon()
        x, y = p[:, 0], p[:, 1]
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            hit = ((y > q[1]) != (y2 > q[1])) & (q[0] < (x2 - x) * (q[1] - y) / (y2 - y) + x)
        return bool(np.count_nonzero(hit) % 2)

    def bbox(self) -> tuple[float, float, float, float]:
        p = self.polygon()
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())

    def circle(self) -> tuple[float, float, float] | None:
        """(cx, cy, diameter) when the contour is one full circle."""
        if not self.segs or any(s[0] != "A" for s in self.segs):
            return None
        c = self.segs[0][3]
        r = math.dist(c, self.segs[0][1])
        if any(math.dist(s[3], c) > 1e-9 * r + 1e-12 or abs(math.dist(s[3], s[1]) - r) > 1e-9 * r + 1e-12
               for s in self.segs) or len({s[4] for s in self.segs}) != 1:
            return None
        if abs(sum(_arc_sweep(s) for s in self.segs) - 2 * math.pi) > 1e-6:
            return None
        return c[0], c[1], 2 * r

    def reversed(self) -> "Loop":
        return Loop([_rev(s) for s in reversed(self.segs)])

    def copper_point(self) -> Pt:
        """A point just left of the first segment: inside a counter-clockwise outer contour, outside
        (in the surrounding copper) for a clockwise hole."""
        pts = _sample_seg(self.segs[0])
        a, b = np.asarray(pts[0]), np.asarray(pts[1])
        m = (a + b) / 2
        d = b - a
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1.0)
        bb = self.bbox()
        eps = max(min(np.linalg.norm(d) * 1e-3, max(bb[2] - bb[0], bb[3] - bb[1]) * 1e-6), 1e-12)
        return float(m[0] + n[0] * eps), float(m[1] + n[1] * eps)


@dataclass
class Component:
    """One connected copper (or board) area: counter-clockwise outer contour, clockwise holes."""
    outer: Loop
    holes: list[Loop]
    depth: int = 0

    def contains(self, q: Pt) -> bool:
        return self.outer.contains(q) and not any(h.contains(q) for h in self.holes)


@dataclass
class Layer:
    id: str
    number: int
    z: float
    z0: float
    z1: float
    name: str = ""
    side: str = "Inr"                    # Top / Inr / Bot
    enabled: bool = True
    items: list[str] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)


@dataclass
class Drill:
    x: float
    y: float
    d: float
    plated: bool
    span: tuple[int, int]
    source: str


@dataclass
class Result:
    options: Options
    design: str
    items: list[Item]
    layers: list[Layer]
    outline: list[Component]
    drills: list[Drill]
    messages: list[Msg]
    origin: Pt = (0.0, 0.0)

    @property
    def errors(self) -> list[Msg]:
        return [m for m in self.messages if m.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def item(self, key: str) -> Item:
        for it in self.items:
            if it.key == key:
                return it
        raise KeyError(key)


class ExportError(RuntimeError):
    pass


# ================================================================================================
# small geometry helpers (board frame, metres)
# ================================================================================================
def _same(a: Pt, b: Pt, eps: float = 1e-12) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def _rev(s: tuple) -> tuple:
    if s[0] == "L":
        return ("L", s[2], s[1])
    return ("A", s[2], s[1], s[3], not s[4])


def _arc_sweep(s: tuple) -> float:
    """Swept angle (> 0) of an arc segment; 2 pi for a full circle."""
    _, a, b, c, ccw = s
    t0 = math.atan2(a[1] - c[1], a[0] - c[0])
    t1 = math.atan2(b[1] - c[1], b[0] - c[0])
    d = (t1 - t0) if ccw else (t0 - t1)
    d %= 2 * math.pi
    return 2 * math.pi if d < 1e-12 else d


def _sample_seg(s: tuple, per_circle: int = 96) -> list[Pt]:
    if s[0] == "L":
        return [s[1], s[2]]
    _, a, b, c, ccw = s
    r = math.dist(a, c)
    sw = _arc_sweep(s)
    n = max(2, int(math.ceil(per_circle * sw / (2 * math.pi))))
    t0 = math.atan2(a[1] - c[1], a[0] - c[0])
    sg = 1.0 if ccw else -1.0
    pts = [(c[0] + r * math.cos(t0 + sg * sw * i / n), c[1] + r * math.sin(t0 + sg * sw * i / n)) for i in range(n)]
    return [a] + pts[1:] + [b]


def _circumcircle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-30:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return ux, uy


def _seg_dist(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    d = b - a
    L = float(d @ d)
    if L == 0:
        return float(np.linalg.norm(p - a))
    t = min(1.0, max(0.0, float((p - a) @ d) / L))
    return float(np.linalg.norm(p - (a + t * d)))


# ================================================================================================
# gmsh / OCC access
# ================================================================================================
def _vertex_xy(v: int) -> Pt:
    x = gmsh.model.getValue(0, v, [])
    return float(x[0]), float(x[1])


def _curve_points(c: int, ts) -> np.ndarray:
    return np.asarray(gmsh.model.getValue(1, c, list(map(float, ts))), float).reshape(-1, 3)


def curve_kind(c: int) -> str:
    """'line', 'arc' or 'free' (approximated) for curve tag c."""
    return _curve_segments(c, 1e-6, kind_only=True)


def _curve_segments(c: int, tol: float, kind_only: bool = False):
    """Segments of curve c from its first to its last vertex (gmsh parametrisation direction)."""
    verts = list(gmsh.model.getAdjacencies(1, c)[1])
    t0, t1 = (float(v) for v in np.ravel(gmsh.model.getParametrizationBounds(1, c)))
    if len(verts) >= 2:
        pa, pb = _vertex_xy(verts[0]), _vertex_xy(verts[1])
    else:   # closed curve: one vertex (or none)
        p = _curve_points(c, [t0])[0]
        pa = pb = _vertex_xy(verts[0]) if verts else (float(p[0]), float(p[1]))
    typ = gmsh.model.getType(1, c)
    if typ == "Line":
        return "line" if kind_only else [("L", pa, pb)]
    ts = np.linspace(t0, t1, 65)
    P = _curve_points(c, ts)[:, :2]
    closed = _same(pa, pb, 1e-12) or len(verts) < 2
    size = float(np.ptp(P, axis=0).max()) or 1.0
    i1, i2 = (21, 43) if closed else (32, 64)
    cc = _circumcircle(P[0], P[i1], P[i2])
    if cc is not None:
        r = math.dist(cc, P[0])
        dev = np.abs(np.hypot(P[:, 0] - cc[0], P[:, 1] - cc[1]) - r)
        if r < 1e3 * size and dev.max() <= 1e-9 * max(r, 1e-6):
            if kind_only:
                return "arc"
            ang = np.unwrap(np.arctan2(P[:, 1] - cc[1], P[:, 0] - cc[0]))
            ccw = bool(ang[-1] > ang[0])
            return [("A", pa, pb, (float(cc[0]), float(cc[1])), ccw)]
    # straight curve of another type (e.g. a spline that is a line)
    if not closed and max(_seg_dist(p, pa, pb) for p in P) <= 1e-9 * size:
        return "line" if kind_only else [("L", pa, pb)]
    if kind_only:
        return "free"
    # adaptive chord flattening: every chord deviates at most tol from the curve
    out: list[Pt] = [pa]

    def rec(ta, tb, a, b, depth):
        tm = 0.5 * (ta + tb)
        q = _curve_points(c, [0.25 * (3 * ta + tb), tm, 0.25 * (ta + 3 * tb)])[:, :2]
        if depth >= 2 and (depth >= 20 or max(_seg_dist(p, a, b) for p in q) <= tol):
            out.append(b)
            return
        m = (float(q[1][0]), float(q[1][1]))
        rec(ta, tm, a, m, depth + 1)
        rec(tm, tb, m, b, depth + 1)

    rec(t0, t1, pa, pb, 0)
    out[-1] = pb
    return [("L", out[i], out[i + 1]) for i in range(len(out) - 1) if not _same(out[i], out[i + 1], 0.0)]


def _faces_of(dt) -> list[int]:
    out = []
    for d, t in dt:
        if d == 2:
            out.append(t)
        elif d == 3:
            out += [ft for _, ft in gmsh.model.getBoundary([(3, t)], combined=False, oriented=False)]
    return list(dict.fromkeys(abs(f) for f in out))


def _zrange(dt) -> tuple[float, float]:
    """Board Z range. The OCC bounding box carries a small gap, so the vertices are used where they
    reach the box (prisms, cylinders: the usual board geometry)."""
    lo, hi = math.inf, -math.inf
    for d, t in dt:
        b = gmsh.model.occ.getBoundingBox(d, t)
        lo, hi = min(lo, b[2]), max(hi, b[5])
    vz = [gmsh.model.getValue(0, v, [])[2] for _, v in gmsh.model.getBoundary(dt, combined=False, recursive=True)]
    if vz:
        if min(vz) - lo <= 2e-7:
            lo = min(vz)
        if hi - max(vz) <= 2e-7:
            hi = max(vz)
    return lo, hi


def _face_flat_z(f: int) -> float | None:
    """Elevation of a face parallel to the board (z = const), else None."""
    b = gmsh.model.occ.getBoundingBox(2, f)
    size = max(b[3] - b[0], b[4] - b[1], 1e-9)
    # the OCC bounding box is enlarged by a small gap (~1e-7): coarse test only, samples decide
    if b[5] - b[2] > 1e-6 * size + 1e-6:
        return None
    u0, u1 = (float(v) for v in np.ravel(gmsh.model.getParametrizationBounds(2, f))[[0, 2]])
    v0, v1 = (float(v) for v in np.ravel(gmsh.model.getParametrizationBounds(2, f))[[1, 3]])
    uv = [x for u in np.linspace(u0, u1, 5) for v in np.linspace(v0, v1, 5) for x in (u, v)]
    z = np.asarray(gmsh.model.getValue(2, f, uv), float).reshape(-1, 3)[:, 2]
    if np.ptp(z) > 1e-9 * size + 1e-12:
        return None
    return float(np.mean(z))


def _cylinder_axis(f: int):
    """(cx, cy, r) when face f is a cylinder with axis along board Z, else None."""
    b = np.ravel(gmsh.model.getParametrizationBounds(2, f))
    uv = [x for u in np.linspace(b[0], b[2], 7) for v in np.linspace(b[1], b[3], 3) for x in (u, v)]
    P = np.asarray(gmsh.model.getValue(2, f, uv), float).reshape(-1, 3)
    N = np.asarray(gmsh.model.getNormal(f, uv), float).reshape(-1, 3)
    if np.abs(N[:, 2]).max() > 1e-7 or np.ptp(N[:, :2], axis=0).max() < 1e-6:
        return None
    # centre: points minus radius * normal coincide
    xy = P[:, :2]
    nxy = N[:, :2]
    # least squares centre c and radius r with p = c + s r n  (s = +-1 by orientation)
    A = np.zeros((2 * len(xy), 3))
    A[0::2, 0] = 1
    A[1::2, 1] = 1
    A[0::2, 2] = nxy[:, 0]
    A[1::2, 2] = nxy[:, 1]
    sol, *_ = np.linalg.lstsq(A, xy.ravel(), rcond=None)
    cx, cy, rr = sol
    r = abs(rr)
    if r <= 0 or np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r).max() > 1e-7 * r + 1e-12:
        return None
    return float(cx), float(cy), float(r)


def _union(faces: list[tuple[int, int]], tol: float) -> list[Component]:
    """Fragment all faces (z = const), keep edges used once, chain them into closed contours."""
    occ = gmsh.model.occ
    if not faces:
        return []
    if len(faces) > 1:
        out, _ = occ.fragment(faces, [])
    else:
        out = faces
    occ.synchronize()
    frags = [t for d, t in out if d == 2]
    use: dict[int, int] = {}
    oriented: list[tuple[int, int]] = []
    for f in frags:
        b = np.ravel(gmsh.model.getParametrizationBounds(2, f))
        n = gmsh.model.getNormal(f, [0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])])
        flip = n[2] < 0
        for _, c in gmsh.model.getBoundary([(2, f)], combined=False, oriented=True, recursive=False):
            c = -c if flip else c
            use[abs(c)] = use.get(abs(c), 0) + 1
            oriented.append((c, f))
    edges = [(c, f) for c, f in oriented if use[abs(c)] == 1]
    ends: dict[int, tuple[int, int]] = {}
    for c, _ in edges:
        v = list(gmsh.model.getAdjacencies(1, abs(c))[1])
        a, b = (v[0], v[1]) if len(v) >= 2 else ((v[0], v[0]) if v else (-abs(c), -abs(c)))
        ends[c] = (a, b) if c > 0 else (b, a)
    start: dict[int, list[tuple[int, int]]] = {}
    for e in edges:
        start.setdefault(ends[e[0]][0], []).append(e)
    seg_cache: dict[int, list] = {}

    def segs(c: int) -> list:
        if abs(c) not in seg_cache:
            seg_cache[abs(c)] = _curve_segments(abs(c), tol)
        s = seg_cache[abs(c)]
        return s if c > 0 else [_rev(x) for x in reversed(s)]

    loops: list[Loop] = []
    left = set(edges)
    while left:
        e = min(left)
        first_v = ends[e[0]][0]
        chain = []
        while True:
            left.discard(e)
            start[ends[e[0]][0]].remove(e)
            chain.append(e)
            v = ends[e[0]][1]
            if v == first_v:
                break
            nxt = start.get(v, [])
            if not nxt:
                raise ExportError("open contour after the union (non-manifold copper geometry)")
            same = [x for x in nxt if x[1] == e[1]]
            e = (same or nxt)[0]
        sl = [s for c, _ in chain for s in segs(c)]
        if sl:
            loops.append(Loop(sl))
    return _nest(loops)


def _nest(loops: list[Loop]) -> list[Component]:
    """Group contours into components (outer CCW + holes CW) with nesting depth."""
    loops = [lp for lp in loops if abs(lp.area) > 0]
    outers = [lp for lp in loops if lp.area > 0]
    holes = [lp for lp in loops if lp.area < 0]
    comps = [Component(o, []) for o in outers]
    for h in holes:
        # the hole belongs to the smallest outer contour around the copper just outside of it
        q = h.copper_point()
        best = None
        for c in comps:
            if c.outer.contains(q) and (best is None or c.outer.area < best.outer.area):
                best = c
        if best is None:
            raise ExportError("hole contour outside of any copper area")
        best.holes.append(h)
    for c in comps:
        p = c.outer.copper_point()
        c.depth = sum(1 for o in comps if o is not c and o.outer.contains(p)) + \
            sum(1 for o in comps for h in o.holes if h.contains(p))
    comps.sort(key=lambda c: (c.depth, -c.outer.area))
    return comps


# ================================================================================================
# classification
# ================================================================================================
def _material(project: Project, name: str):
    try:
        return project.material(name)
    except KeyError:
        return None


def _is_air(mat) -> bool:
    if mat is None:
        return True
    try:
        return (not is_conductor(mat)) and abs(float(mat.permittivity) - 1) < 0.01 and \
            float(mat.conductivity or 0) == 0
    except (TypeError, ValueError):
        return False


def _classify(project: Project, des: Design) -> list[Item]:
    ports: dict[str, str] = {}
    for e in des.excitations:
        for o in e.objects:
            ports.setdefault(o, e.name)
        for f in e.faces:
            if f.face < 0:
                ports.setdefault(f.obj, e.name)
    obj_bnd: dict[str, list[str]] = {}
    face_bnd: dict[tuple[str, int], str] = {}
    for b in des.boundaries:
        if not b.enabled:
            continue
        for o in b.objects:
            obj_bnd.setdefault(o, []).append(b.kind)
        for f in b.faces:
            if f.face < 0:
                obj_bnd.setdefault(f.obj, []).append(b.kind)
            elif b.kind in COPPER_BOUNDARIES:
                face_bnd[(f.obj, f.face)] = b.name
    items: list[Item] = []
    for o in des.live_objects():
        dim = o.dim
        it = Item(o.name, o.name, kind={3: "solid", 2: "sheet"}.get(dim, "line"))
        kinds = obj_bnd.get(o.name, [])
        mat = _material(project, o.material)
        if o.history[0].kind == "CreateRegion":
            it.category, it.note = "excluded", "air region (simulation helper)"
        elif not o.model:
            it.category, it.note = "excluded", "non-model object"
        elif o.name in ports:
            it.category, it.note = "excluded", f"port sheet (excitation '{ports[o.name]}')"
        elif dim < 2:
            it.category, it.note = "excluded", "line object"
        elif any(k in COPPER_BOUNDARIES for k in kinds):
            it.category = "copper" if dim == 2 else "conductor"
            it.note = ", ".join(k for k in kinds if k in COPPER_BOUNDARIES) + " boundary"
        elif dim == 3 and mat is not None and is_conductor(mat):
            it.category, it.note = "conductor", f"conducting material '{o.material}'"
        elif kinds:
            it.category, it.note = "excluded", f"{', '.join(sorted(set(kinds)))} boundary (simulation helper)"
        elif dim == 3 and _is_air(mat):
            it.category, it.note = "excluded", f"{o.material} body"
        elif dim == 3:
            it.category, it.note = "board", f"dielectric '{o.material}'"
        else:
            it.category, it.note = "other", "sheet without conductor boundary"
        items.append(it)
        for (on, fi), bname in sorted(face_bnd.items()):
            if on == o.name and it.category in ("board", "other", "excluded") and o.model \
                    and o.history[0].kind != "CreateRegion":
                items.append(Item(f"{o.name}:{fi}", o.name, fi, "sheet", "copper",
                                  f"face with boundary '{bname}'"))
    return items


def _transform(dts: list, O: np.ndarray, A: np.ndarray) -> None:
    """Map global coordinates into the board frame: local = A^T (p - O).

    Done as a translation and one rotation: a general affine transform would turn circles and cylinders
    into B-splines in OpenCASCADE."""
    if not dts:
        return
    occ = gmsh.model.occ
    if np.linalg.norm(O) > 0:
        occ.translate(dts, *(-np.asarray(O, float)))
    R = np.asarray(A, float).T
    c = max(-1.0, min(1.0, 0.5 * (np.trace(R) - 1)))
    ang = math.acos(c)
    if ang > 1e-12:
        if math.pi - ang > 1e-6:
            axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        else:                       # 180 degrees: axis from (R + I) / 2
            B = 0.5 * (R + np.eye(3))
            axis = B[:, int(np.argmax(np.diag(B)))]
        axis = axis / np.linalg.norm(axis)
        occ.rotate(dts, 0, 0, 0, *axis, ang)
    occ.synchronize()


def board_frame(des: Design, ev, cs: str, plane: str) -> tuple[np.ndarray, np.ndarray]:
    O, R = coordsys.frame(des, cs, ev)
    perm = PLANES[plane]
    return np.asarray(O, float), np.asarray(R, float)[:, list(perm)]


# ================================================================================================
# main entry
# ================================================================================================
def run(project: Project, design: str | Design, options: Options | None = None) -> Result:
    """Analyse the design, suggest/apply the layer assignment and compute the board geometry.

    Builds the design into its own gmsh model and removes it again; the result is plain data."""
    opts = options or Options()
    des = design if isinstance(design, Design) else project.design(design)
    ev = des.evaluator(project)
    msgs: list[Msg] = []
    prev = None
    if gmsh.isInitialized():
        try:
            prev = gmsh.model.getCurrent()
        except Exception:  # noqa: BLE001
            prev = None
    K.ensure_gmsh("hertzon_gerber")
    gmsh.model.remove()
    gmsh.model.add("hertzon_gerber")
    try:
        return _run(project, des, ev, opts, msgs)
    finally:
        try:
            gmsh.model.remove()
        except Exception:  # noqa: BLE001
            pass
        if prev and prev != "hertzon_gerber":
            try:
                gmsh.model.setCurrent(prev)
            except Exception:  # noqa: BLE001
                pass


def _run(project: Project, des: Design, ev, opts: Options, msgs: list[Msg]) -> Result:
    if opts.units not in UNITS:
        raise ValueError(f"unknown units {opts.units!r}")
    if opts.tolerance <= 0:
        raise ValueError("the curve tolerance must be positive")
    items = _classify(project, des)
    ctx = K.build_design(des, ev)
    try:
        O, A = board_frame(des, ev, opts.cs, opts.plane)
    except (KeyError, ValueError) as e:
        raise ExportError(f"coordinate system: {e}") from e
    tops = []
    for dt in ctx.built.values():
        tops += [x for x in dt if x not in tops]
    _transform(tops, O, A)
    occ = gmsh.model.occ

    # ---- geometry per item ----------------------------------------------------------------------
    eps_z = 1e-9
    for it in items:
        if it.category in ("excluded",):
            continue
        if it.obj in ctx.errors:
            it.category, it.note = "error", f"cannot be built: {ctx.errors[it.obj]}"
            continue
        dt = ctx.built.get(it.obj, [])
        if not dt:
            it.category, it.note = "error", "no geometry"
            continue
        if it.face >= 0:
            fl = K.object_faces(dt)
            if it.face >= len(fl):
                it.category, it.note = "error", f"face {it.face} does not exist"
                continue
            it.tags = [(2, fl[it.face])]
        else:
            it.tags = list(dt)
        if it.kind == "sheet" or it.face >= 0:
            zs = [_face_flat_z(f) for f in _faces_of(it.tags)]
            if any(z is None for z in zs) or (zs and max(zs) - min(zs) > eps_z):
                if it.category == "copper":
                    it.category = "ambiguous"
                    it.note += "; not planar or not parallel to the board plane"
                it.z0 = it.z1 = math.nan
            else:
                it.z0 = it.z1 = zs[0]
        else:
            it.z0, it.z1 = _zrange(it.tags)
            if it.category == "conductor":
                cyl = [_cylinder_axis(f) for f in _faces_of(it.tags) if gmsh.model.getType(2, f) != "Plane"]
                if cyl and all(c is not None for c in cyl):
                    c0 = cyl[0]
                    if all(math.hypot(c[0] - c0[0], c[1] - c0[1]) <= 1e-7 * c0[2] + 1e-12 for c in cyl):
                        it.via = (c0[0], c0[1], 2 * max(c[2] for c in cyl))
                if it.z1 - it.z0 > opts.max_copper_thickness:
                    it.category = "via?" if it.via else "ambiguous"
                    if not it.via:
                        it.note += "; thicker than the copper limit and no via cylinder"

    # ---- layers from copper elevations -----------------------------------------------------------
    ivs = sorted((it.z0, it.z1, it.key) for it in items
                 if it.category in ("copper", "conductor") and not math.isnan(it.z0))
    groups: list[list] = []
    for z0, z1, key in ivs:
        if groups and z0 <= groups[-1][1] + eps_z:
            groups[-1][1] = max(groups[-1][1], z1)
            groups[-1][2].append(key)
        else:
            groups.append([z0, z1, [key]])
    groups.sort(key=lambda g: -g[1])
    layers: list[Layer] = []
    n = len(groups)
    for i, (z0, z1, keys) in enumerate(groups, 1):
        side = "Top" if i == 1 else ("Bot" if i == n else "Inr")
        default = "Top" if i == 1 else ("Bottom" if i == n else f"Inner{i - 1}")
        lid = f"L{i}"
        layers.append(Layer(lid, i, 0.5 * (z0 + z1), z0, z1, opts.layer_names.get(lid) or default, side,
                            opts.layer_enabled.get(lid, True), list(keys)))
    by_id = {L.id: L for L in layers}

    def layers_in(z0, z1):
        """Layers whose copper touches or overlaps the board Z interval [z0, z1]."""
        return [L for L in layers if L.z0 <= z1 + eps_z and L.z1 >= z0 - eps_z]

    # vias: thick coaxial conductors reaching two or more layers
    for it in items:
        if it.category == "via?":
            if len(layers_in(it.z0, it.z1)) >= 2:
                it.category = "via"
                it.note += f"; via, drill {it.via[2] * 1e3:.4g} mm"
            else:
                it.category = "ambiguous"
                it.note += "; cylinder that does not connect two copper layers"

    # ---- suggestion and roles -------------------------------------------------------------------
    for it in items:
        if it.category in ("copper", "conductor"):
            it.suggested = next((L.id for L in layers if it.key in L.items), "skip")
        elif it.category == "via":
            it.suggested = "via" if opts.drills else "skip"
        elif it.category == "board":
            it.suggested = "outline" if opts.outline else "skip"
        else:
            it.suggested = "skip"
        role = opts.roles.get(it.key, it.suggested)
        if role.startswith("L") and role not in by_id:
            role = it.suggested
        it.role = role
    for it in items:
        if it.role != "skip" and it.category in ("error", "excluded"):
            msgs.append(Msg("error", f"{it.label}: {it.note}; cannot be exported", [it.obj]))
            it.role = "skip"
    errs = [it for it in items if it.category == "error"]
    if errs:
        msgs.append(Msg("warning", "Objects that could not be built are not exported: " +
                        ", ".join(it.label for it in errs), [it.obj for it in errs]))
    amb = [it for it in items if it.category == "ambiguous" and it.role == "skip"]
    if amb:
        msgs.append(Msg("warning", "Ambiguous conductor geometry, not exported (assign a layer to override): " +
                        "; ".join(f"{it.label} ({it.note})" for it in amb), [it.obj for it in amb]))

    # ---- copper per layer ---------------------------------------------------------------------
    approx: set[str] = set()
    for L in layers:
        L.items = [it.key for it in items if it.role == L.id]
        faces: list[tuple[int, int]] = []
        for it in items:
            if it.role != L.id:
                continue
            new = _layer_faces(it, L, layers, opts, msgs)
            if new is None:
                continue
            if not new:
                msgs.append(Msg("error", f"{it.label}: no copper in the plane of layer {L.name}", [it.obj]))
                continue
            for f in _faces_of(new):
                for _, c in gmsh.model.getBoundary([(2, f)], combined=False, oriented=False):
                    if curve_kind(abs(c)) == "free":
                        approx.add(it.label)
            faces += new
        try:
            L.components = _union(faces, opts.tolerance)
        except ExportError as e:
            msgs.append(Msg("error", f"Layer {L.name}: {e}", [it.obj for it in items if it.role == L.id]))
    if approx:
        msgs.append(Msg("info", f"Curves that are neither lines nor circular arcs are written as polylines "
                                f"(chord tolerance {opts.tolerance * 1e6:.3g} um): " + ", ".join(sorted(approx)),
                        sorted(approx)))

    # ---- board outline --------------------------------------------------------------------------
    outline: list[Component] = []
    ofaces = []
    for it in items:
        if it.role != "outline":
            continue
        if it.kind == "sheet" or it.face >= 0:
            if math.isnan(it.z0):
                msgs.append(Msg("error", f"{it.label}: outline sheet is not parallel to the board plane", [it.obj]))
                continue
            new = occ.copy([(2, f) for f in _faces_of(it.tags)])
            occ.translate(new, 0, 0, -it.z0)
        else:
            new = _section(it.tags, 0.5 * (it.z0 + it.z1))
            occ.translate(new, 0, 0, -0.5 * (it.z0 + it.z1))
        occ.synchronize()
        ofaces += new
    try:
        outline = _union(ofaces, opts.tolerance) if ofaces else []
    except ExportError as e:
        msgs.append(Msg("error", f"Board outline: {e}"))

    # ---- drills ---------------------------------------------------------------------------------
    drills: list[Drill] = []
    for it in items:
        if it.role != "via":
            continue
        if not it.via:
            msgs.append(Msg("error", f"{it.label}: not a cylinder along the board normal, cannot be drilled",
                            [it.obj]))
            continue
        span = layers_in(it.z0, it.z1)
        if len(span) < 2:
            msgs.append(Msg("warning", f"{it.label}: plated hole reaches only "
                                       f"{span[0].name if span else 'no copper layer'}", [it.obj]))
        nums = [L.number for L in span] or [1]
        drills.append(Drill(it.via[0], it.via[1], it.via[2], True, (min(nums), max(nums)), it.label))
        for L in span:
            if L.components and not _copper_around(L, it.via):
                msgs.append(Msg("warning", f"Via {it.label} has no copper pad on layer {L.name}", [it.obj]))
    board_names = ", ".join(it.label for it in items if it.role == "outline")
    for comp in outline:
        keep = []
        for h in comp.holes:
            circ = h.circle()
            if circ is None:
                keep.append(h)
                continue
            plated = [d for d in drills if math.hypot(d.x - circ[0], d.y - circ[1]) <= max(1e-7, 1e-3 * circ[2])]
            if plated:
                continue            # hole of a via barrel: drilled by the PTH file
            if circ[2] > MAX_DRILL or not opts.drills:
                keep.append(h)      # large opening: milled along the profile
                continue
            nl = max(len(layers), 1)
            drills.append(Drill(circ[0], circ[1], circ[2], False, (1, nl), board_names))
        comp.holes = keep

    for d in drills:
        if d.plated:
            continue
        pts = [(d.x, d.y)] + [(d.x + 0.45 * d.d * math.cos(k * math.pi / 4), d.y + 0.45 * d.d * math.sin(k * math.pi / 4))
                              for k in range(8)]
        hit = [L.name for L in layers if any(_covered(L.components, q) for q in pts)]
        if hit:
            msgs.append(Msg("warning", f"Non-plated hole d = {d.d * 1e3:.4g} mm at ({d.x * 1e3:.4g}, {d.y * 1e3:.4g}) mm "
                                       f"of {d.source} goes through copper on layer(s) {', '.join(hit)}",
                            [d.source]))
    if not any(L.components for L in layers):
        msgs.append(Msg("error", "No copper geometry to export (no layer has copper)."))
    if opts.outline and not outline:
        msgs.append(Msg("warning", "No board outline: no dielectric body is assigned to the outline."))

    origin = (0.0, 0.0)
    if opts.origin == "point":
        origin = tuple(opts.origin_point)  # type: ignore[assignment]
    elif opts.origin == "outline":
        src = [c.outer for c in outline] or [c.outer for L in layers for c in L.components]
        if src:
            bb = np.array([lp.bbox() for lp in src])
            origin = (float(bb[:, 0].min()), float(bb[:, 1].min()))
            if not outline:
                msgs.append(Msg("info", "Origin: lower left corner of the copper (there is no board outline)."))
    for it in items:
        it.tags = []
    return Result(opts, des.name, items, layers, outline, drills, msgs, origin)


def _section(dt, z: float) -> list[tuple[int, int]]:
    """Cross section of solid(s) with the plane board-Z = z (exact OCC boolean)."""
    occ = gmsh.model.occ
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for d, t in dt:
        b = occ.getBoundingBox(d, t)
        lo = np.minimum(lo, b[:2])
        hi = np.maximum(hi, b[3:5])
    m = max(float((hi - lo).max()), 1e-6)
    rect = occ.addRectangle(lo[0] - m, lo[1] - m, z, (hi - lo)[0] + 2 * m, (hi - lo)[1] + 2 * m)
    vols = [x for x in dt if x[0] == 3]
    out, _ = occ.intersect(vols, [(2, rect)], removeObject=False, removeTool=True)
    occ.synchronize()
    return [x for x in out if x[0] == 2]


def _layer_faces(it: Item, L: Layer, layers: list[Layer], opts: Options, msgs: list[Msg]):
    occ = gmsh.model.occ
    if it.kind == "sheet" or it.face >= 0:
        if math.isnan(it.z0):
            msgs.append(Msg("error", f"{it.label}: sheet is not planar or not parallel to the board plane",
                            [it.obj]))
            return None
        if not (L.z0 - 1e-9 <= it.z0 <= L.z1 + 1e-9):
            msgs.append(Msg("warning", f"{it.label}: sheet at board Z = {it.z0 * 1e3:.4g} mm is placed on layer "
                                       f"{L.name} (Z = {L.z * 1e3:.4g} mm) as assigned", [it.obj]))
        new = occ.copy([(2, f) for f in _faces_of(it.tags)])
        occ.translate(new, 0, 0, L.z - it.z0)
        occ.synchronize()
        return new
    # solid: half height when it belongs to the layer, else the layer plane
    if L.z0 - 1e-9 <= it.z0 and it.z1 <= L.z1 + 1e-9:
        zc = 0.5 * (it.z0 + it.z1)
    elif it.z0 < L.z < it.z1:
        zc = L.z
        msgs.append(Msg("info", f"{it.label}: cut in the plane of layer {L.name} (Z = {L.z * 1e3:.4g} mm)",
                        [it.obj]))
    else:
        msgs.append(Msg("error", f"{it.label}: body (Z {it.z0 * 1e3:.4g}..{it.z1 * 1e3:.4g} mm) does not reach "
                                 f"layer {L.name}", [it.obj]))
        return None
    new = _section(it.tags, zc)
    occ.translate(new, 0, 0, L.z - zc)
    occ.synchronize()
    return new


def _copper_around(L: Layer, via) -> bool:
    cx, cy, d = via
    r = 0.5 * d * 1.02 + 1e-6
    for k in range(8):
        a = k * math.pi / 4
        q = (cx + r * math.cos(a), cy + r * math.sin(a))
        if _covered(L.components, q):
            return True
    return False


def _covered(comps: list[Component], q: Pt) -> bool:
    """Copper at q, following the write order (later components paint over earlier holes)."""
    hit = False
    for c in comps:
        if c.outer.contains(q):
            hit = True
            if any(h.contains(q) for h in c.holes):
                hit = False
    return hit


# ================================================================================================
# writers
# ================================================================================================
class _Fmt:
    def __init__(self, units: str, origin: Pt):
        self.s = UNITS[units]
        self.units = units
        self.ox, self.oy = origin

    def q(self, v: float) -> int:
        return int(round(v * self.s * 1e6))

    def xy(self, p: Pt) -> tuple[int, int]:
        return self.q(p[0] - self.ox), self.q(p[1] - self.oy)

    def dec(self, v: float) -> str:
        return f"{v * self.s:.4f}" if self.units == "mm" else f"{v * self.s:.5f}"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("hertzon")
    except Exception:  # noqa: BLE001
        return "dev"


def _header(res: Result, function: str, polarity: str = "Positive") -> list[str]:
    u = res.options.units
    fs = "%FSLAX46Y46*%" if u == "mm" else "%FSLAX26Y26*%"
    return [
        f"G04 Hertzon Gerber X2 export of design {_safe(res.design)}*",
        f"%TF.GenerationSoftware,Hertzon,Hertzon,{_version()}*%",
        f"%TF.CreationDate,{_now()}*%",
        f"%TF.SameCoordinates,{_safe(res.design)}*%",
        f"%TF.FileFunction,{function}*%",
        f"%TF.FilePolarity,{polarity}*%",
        fs,
        f"G04 board frame: plane {res.options.plane} of coordinate system {_safe(res.options.cs)}, "
        f"viewed from the top, not mirrored*",
        "%MOMM*%" if u == "mm" else "%MOIN*%",
        "%LPD*%",
        "G01*",
        "G75*",
    ]


def _safe(s: str) -> str:
    """Characters allowed in Gerber strings/attributes (no '*', '%', ',')."""
    return "".join(ch if (ch.isalnum() or ch in " _-.:+()/") else "_" for ch in str(s))


def _path(f: _Fmt, lp: Loop) -> list[str]:
    """Contour statements; the first point is a D02 move."""
    out = []
    x, y = f.xy(lp.segs[0][1])
    out.append(f"X{x}Y{y}D02*")
    cur = (x, y)
    for s in lp.segs:
        bx, by = f.xy(s[2])
        if s[0] == "L" or (_arc_sweep(s) < 2 * math.pi and (bx, by) == cur):
            if (bx, by) != cur:
                out.append(f"X{bx}Y{by}D01*")
            cur = (bx, by)
            continue
        cx, cy = f.xy(s[3])
        out.append(("G03*" if s[4] else "G02*"))
        out.append(f"X{bx}Y{by}I{cx - cur[0]}J{cy - cur[1]}D01*")
        out.append("G01*")
        cur = (bx, by)
    return out


def _region(f: _Fmt, lp: Loop) -> list[str]:
    return ["G36*"] + _path(f, lp) + ["G37*"]


def gerber_copper(res: Result, L: Layer) -> str:
    f = _Fmt(res.options.units, res.origin)
    lines = _header(res, f"Copper,L{L.number},{L.side}")
    lines.append(f"G04 layer {_safe(L.name)} at board Z = {L.z * 1e3:.6g} mm, "
                 f"objects: {_safe(', '.join(L.items))}*")
    lines.append("%TA.AperFunction,Conductor*%")
    for comp in L.components:
        lines.append("%LPD*%")
        lines += _region(f, comp.outer)
        if comp.holes:
            lines.append("%LPC*%")
            for h in comp.holes:
                lines += _region(f, h.reversed())
    lines.append("%LPD*%")
    lines.append("%TD*%")
    lines.append("M02*")
    return "\n".join(lines) + "\n"


def gerber_profile(res: Result) -> str:
    f = _Fmt(res.options.units, res.origin)
    lines = _header(res, "Profile,NP")
    w = PROFILE_WIDTH * f.s
    lines.append("%TA.AperFunction,Profile*%")
    lines.append(f"%ADD10C,{w:.6f}*%")
    lines.append("%TD*%")
    lines.append("D10*")
    for comp in res.outline:
        for lp in [comp.outer] + comp.holes:
            lines += _path(f, lp)
    lines.append("M02*")
    return "\n".join(lines) + "\n"


def excellon(res: Result, drills: list[Drill], plated: bool, span: tuple[int, int]) -> str:
    f = _Fmt(res.options.units, res.origin)
    kind = "PTH" if plated else "NPTH"
    func = f"{'Plated' if plated else 'NonPlated'},{span[0]},{span[1]},{kind}"
    lines = ["M48",
             f"; DRILL file of Hertzon design {res.design}",
             "; FORMAT={-:-/ absolute / " + ("metric" if res.options.units == "mm" else "inch") + " / decimal}",
             f"; #@! TF.GenerationSoftware,Hertzon,Hertzon,{_version()}",
             f"; #@! TF.CreationDate,{_now()}",
             f"; #@! TF.SameCoordinates,{_safe(res.design)}",
             f"; #@! TF.FileFunction,{func}",
             "; #@! TF.FilePolarity,Positive",
             "FMAT,2",
             "METRIC" if res.options.units == "mm" else "INCH",
             "G90"]
    sizes = sorted({round(d.d * f.s, 4 if f.units == "mm" else 5) for d in drills})
    lines.append(";TYPE=PLATED" if plated else ";TYPE=NON_PLATED")
    for i, s in enumerate(sizes, 1):
        lines.append(f"; #@! TA.AperFunction,{'Plated,PTH,ViaDrill' if plated else 'NonPlated,NPTH,ComponentDrill'}")
        lines.append(f"T{i}C{s:.4f}" if f.units == "mm" else f"T{i}C{s:.5f}")
    lines += ["%", "G05"]
    for i, s in enumerate(sizes, 1):
        lines.append(f"T{i}")
        for d in drills:
            if round(d.d * f.s, 4 if f.units == "mm" else 5) == s:
                lines.append(f"X{f.dec(d.x - f.ox)}Y{f.dec(d.y - f.oy)}")
    lines.append("M30")
    return "\n".join(lines) + "\n"


def file_plan(res: Result, base: str) -> dict[str, str]:
    """File name -> content of everything the export writes. Raises ExportError on errors."""
    if not res.ok:
        raise ExportError("; ".join(m.text for m in res.errors))
    files: dict[str, str] = {}
    for L in res.layers:
        if L.enabled and L.components:
            files[f"{base}-L{L.number}_{_fname(L.name)}.gbr"] = gerber_copper(res, L)
    if res.options.outline and res.outline:
        files[f"{base}-Profile.gbr"] = gerber_profile(res)
    if res.options.drills:
        groups: dict[tuple, list[Drill]] = {}
        for d in res.drills:
            groups.setdefault((d.plated, d.span), []).append(d)
        for (plated, span), ds in sorted(groups.items(), key=lambda kv: (not kv[0][0], kv[0][1])):
            name = f"{base}-PTH_L{span[0]}-L{span[1]}.drl" if plated else f"{base}-NPTH.drl"
            files[name] = excellon(res, ds, plated, span)
    if not any(n.endswith(".gbr") and "Profile" not in n for n in files):
        raise ExportError("No copper layer is selected for export.")
    files[f"{base}-export-report.txt"] = report(res, list(files))
    return files


def _fname(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s) or "layer"


def report(res: Result, names: list[str]) -> str:
    o = res.options
    u = o.units
    s = UNITS[u]
    lines = [f"Hertzon Gerber export, design {res.design}", f"created {_now()}", "",
             f"units: {u}   board plane: {o.plane} of coordinate system {o.cs}   "
             f"origin (board frame): {res.origin[0] * s:.6g}, {res.origin[1] * s:.6g} {u}",
             "All files share one coordinate system and are viewed from the top; no layer is mirrored.",
             f"curve tolerance (only curves that are neither lines nor arcs): {o.tolerance * 1e6:.3g} um", "",
             "Layers (top to bottom):"]
    for L in res.layers:
        lines.append(f"  L{L.number} {L.name:<12} {L.side}  board Z = {L.z * s:.6g} {u}  "
                     f"{'exported' if L.enabled else 'not exported'}  objects: {', '.join(L.items) or '-'}")
    lines += ["", "Drills:"]
    for d in res.drills:
        lines.append(f"  {'PTH ' if d.plated else 'NPTH'} L{d.span[0]}-L{d.span[1]}  "
                     f"d = {d.d * s:.6g} {u}  at {(d.x - res.origin[0]) * s:.6g}, {(d.y - res.origin[1]) * s:.6g}"
                     f"  ({d.source})")
    lines += ["", "Objects:"]
    for it in res.items:
        lines.append(f"  {it.label:<24} {it.role:<8} {it.category:<10} {it.note}")
    if res.messages:
        lines += ["", "Messages:"] + [f"  {m.level}: {m.text}" for m in res.messages]
    lines += ["", "Files:"] + [f"  {n}" for n in names]
    return "\n".join(lines) + "\n"


def write(res: Result, folder: str | Path, base: str) -> list[Path]:
    """Write all files. Everything is generated before the first file is written."""
    files = file_plan(res, base)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, text in files.items():
        p = folder / name
        p.write_text(text, encoding="ascii", errors="replace", newline="\n")
        out.append(p)
    return out
