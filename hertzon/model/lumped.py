"""SMD components (R, L, C) between two traces: package dimensions, land pattern, the geometry that is
generated (pads, optional terminations, the element sheet with its integration line) and the component
library of Touchstone files.

All lengths in this module are millimetres."""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Package:
    name: str           # shown in the lists, e.g. "0402" or "Coilcraft 0402HP"
    size: str           # EIA size code
    L: float            # body length (terminal to terminal)
    W: float            # body width
    T: float            # body height
    e: float            # terminal length at each end (wraparound)
    vendor: str = ""
    kind: str = ""      # L / C / R when the series is one kind of component
    prefix: str = ""    # part numbers of the series start with this
    note: str = ""


# EIA chip sizes (nominal body dimensions of chip resistors and capacitors)
GENERIC = [
    Package("0201", "0201", 0.60, 0.30, 0.30, 0.15),
    Package("0402", "0402", 1.00, 0.50, 0.50, 0.25),
    Package("0603", "0603", 1.60, 0.80, 0.80, 0.35),
    Package("0805", "0805", 2.00, 1.25, 0.85, 0.50),
    Package("1206", "1206", 3.20, 1.60, 0.85, 0.50),
]

# RF series of the library (body dimensions from the data sheets: maximum length and height, nominal
# width, terminal wraparound)
SERIES = [
    Package("Coilcraft 0402HP", "0402", 1.09, 0.61, 0.61, 0.18, "Coilcraft", "L", "0402HP",
            "A max 1.09, B 0.51-0.71, C max 0.61, wraparound 0.18"),
    Package("Coilcraft 0402HPH", "0402", 1.12, 0.66, 0.66, 0.18, "Coilcraft", "L", "0402HPH",
            "A max 1.12, B 0.66, C max 0.66, wraparound 0.18"),
    Package("Coilcraft 0402CS", "0402", 1.19, 0.64, 0.66, 0.18, "Coilcraft", "L", "0402CS",
            "A max 1.19, B max 0.64, C max 0.66, wraparound 0.18"),
    Package("Coilcraft 0603HP", "0603", 1.75, 0.98, 0.94, 0.24, "Coilcraft", "L", "0603HP",
            "A max 1.75, B 0.86-1.09, C max 0.94, wraparound 0.24"),
    Package("Coilcraft 0603CS", "0603", 1.80, 1.12, 1.02, 0.18, "Coilcraft", "L", "0603CS",
            "A max 1.80, B max 1.12, C max 1.02, wraparound 0.18"),
    Package("Murata LQW15AN", "0402", 1.00, 0.60, 0.50, 0.20, "Murata", "L", "LQW15AN",
            "L 1.0, W 0.6, T 0.5; terminal length typical"),
    Package("Murata LQP03TN", "0201", 0.60, 0.30, 0.30, 0.15, "Murata", "L", "LQP03TN",
            "0201 (0603 metric) nominal; terminal length typical"),
    Package("Murata GJM03", "0201", 0.60, 0.30, 0.30, 0.15, "Murata", "C", "GJM03",
            "0201 (0603 metric) nominal; terminal length typical"),
    Package("Murata GJM15", "0402", 1.00, 0.50, 0.50, 0.25, "Murata", "C", "GJM15",
            "0402 (1005 metric) nominal; terminal length typical"),
]

PACKAGES = {p.name: p for p in GENERIC + SERIES}

# typical parasitics of a chip component mounted on its pads (series ESL in nH, ESR in ohm)
PARASITICS = {"0201": (0.15, 0.05), "0402": (0.30, 0.05), "0603": (0.45, 0.05), "0805": (0.60, 0.05),
              "1206": (0.90, 0.05)}


def series_for(part: str) -> Package | None:
    """Library series of a part number (longest matching prefix)."""
    p = part.upper()
    hits = [s for s in SERIES if p.startswith(s.prefix.upper())]
    return max(hits, key=lambda s: len(s.prefix)) if hits else None


@dataclass(frozen=True)
class Land:
    G: float            # gap between the inner pad edges
    Z: float            # outer extent of the land pattern
    X: float            # pad width


def land_pattern(p: Package) -> Land:
    """Land pattern from the body dimensions as in IPC-7351 (nominal density): toe fillet 0.10 mm below
    size 0603 and 0.35 mm from it on, no heel and side fillet."""
    toe = 0.10 if p.L < 1.5 else 0.35
    G = max(p.L - 2 * p.e, 0.1 * p.L)
    return Land(G=round(G, 4), Z=round(p.L + 2 * toe, 4), X=round(p.W, 4))


# --------------------------------------------------------------------------------------------
# Placement between two traces
# --------------------------------------------------------------------------------------------
@dataclass
class TraceEnds:
    """The two trace ends that face each other across the gap (global coordinates in mm)."""
    axis: int                   # gap direction (0, 1, 2)
    sign: int                   # +1: trace B lies in the positive axis direction of trace A
    normal: int                 # axis normal to the traces (the board normal)
    top: float                  # top surface of the traces along the normal
    bottom: float               # bottom of the traces (= top for sheets)
    a_end: float                # end of trace A along the gap axis
    b_end: float                # end of trace B
    centre: float               # centre of the facing ends along the third axis

    @property
    def gap(self) -> float:
        return abs(self.b_end - self.a_end)

    @property
    def lateral(self) -> int:
        return 3 - self.axis - self.normal


def trace_ends(a: np.ndarray, b: np.ndarray) -> TraceEnds:
    """Facing ends of two traces from their surface points (N,3). The traces have to be separated along
    one coordinate axis; the board normal is the axis in which both are thinnest."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    alo, ahi, blo, bhi = a.min(0), a.max(0), b.min(0), b.max(0)
    sep = np.maximum(blo - ahi, alo - bhi)          # > 0: separated along that axis
    ext = np.maximum(ahi - alo, bhi - blo)
    normal = int(np.argmin(ext))
    cand = [i for i in range(3) if i != normal and sep[i] > 0]
    if not cand:
        raise ValueError("the two traces have to be separated by a gap along the X, Y or Z axis")
    axis = max(cand, key=lambda i: sep[i])
    sign = 1 if blo[axis] > ahi[axis] else -1
    a_end = ahi[axis] if sign > 0 else alo[axis]
    b_end = blo[axis] if sign > 0 else bhi[axis]
    lat = 3 - axis - normal
    tol = 1e-3 * max(float(ext.max()), 1e-9)
    ra = a[np.abs(a[:, axis] - a_end) <= tol][:, lat]
    rb = b[np.abs(b[:, axis] - b_end) <= tol][:, lat]
    lo, hi = max(ra.min(), rb.min()), min(ra.max(), rb.max())
    centre = (lo + hi) / 2 if lo <= hi else (ra.mean() + rb.mean()) / 2
    top = max(ahi[normal], bhi[normal])
    bottom = min(alo[normal], blo[normal])
    if abs(ahi[normal] - bhi[normal]) > tol:
        raise ValueError("the two traces have to lie in the same plane")
    return TraceEnds(axis, sign, normal, float(top), float(bottom), float(a_end), float(b_end), float(centre))


@dataclass
class Box:
    """Axis aligned box (or a sheet when one size is 0) in mm: lower corner and sizes."""
    name: str
    lo: tuple[float, float, float]
    size: tuple[float, float, float]
    role: str                   # pad / termination / element

    @property
    def sheet_axis(self) -> int | None:
        z = [i for i in range(3) if abs(self.size[i]) < 1e-12]
        return z[0] if z else None


@dataclass
class ComponentGeometry:
    boxes: list[Box] = field(default_factory=list)
    line: tuple[tuple, tuple] = ((0, 0, 0), (0, 0, 0))     # integration line of the element sheet
    warnings: list[str] = field(default_factory=list)


def component_geometry(ends: TraceEnds, pkg: Package, land: Land, name: str, terminations: bool = False,
                       solid_pads: bool = False) -> ComponentGeometry:
    """Pads from the trace ends to the land pattern gap G (none when the gap already is G), the element
    sheet in the gap between the pads, and with ``terminations`` the metal end caps of the body with the
    element sheet on top of the body between them. The component is centred in the gap."""
    out = ComponentGeometry()
    ax, nrm, lat, s = ends.axis, ends.normal, ends.lateral, ends.sign
    gap = ends.gap
    if gap < land.G - 1e-9:
        raise ValueError(f"the gap between the traces ({gap:.4g} mm) is smaller than the gap of the "
                         f"{pkg.name} land pattern ({land.G:.4g} mm)")
    mid = (ends.a_end + ends.b_end) / 2
    pad_len = (gap - land.G) / 2
    if pad_len > land.Z - land.G + 1e-9:          # more than twice the pad length of the land pattern
        out.warnings.append(f"the pads reach {pad_len:.3g} mm to the traces: a gap of about "
                            f"{land.Z:.3g} mm fits the {pkg.name} land pattern")

    def box(nm, a0, a1, w, n0, n1, role, c=None):
        lo, size = [0.0] * 3, [0.0] * 3
        lo[ax], size[ax] = min(a0, a1), abs(a1 - a0)
        cc = ends.centre if c is None else c
        lo[lat], size[lat] = cc - w / 2, w
        lo[nrm], size[nrm] = n0, n1 - n0
        out.boxes.append(Box(nm, tuple(lo), tuple(size), role))

    pad_bottom = ends.bottom if solid_pads else ends.top
    if pad_len > 1e-9:
        box(f"{name}_PadA", ends.a_end, mid - s * land.G / 2, land.X, pad_bottom, ends.top, "pad")
        box(f"{name}_PadB", mid + s * land.G / 2, ends.b_end, land.X, pad_bottom, ends.top, "pad")
    if terminations:
        top = ends.top + pkg.T
        inner = pkg.L / 2 - pkg.e
        box(f"{name}_TermA", mid - s * pkg.L / 2, mid - s * inner, pkg.W, ends.top, top, "termination")
        box(f"{name}_TermB", mid + s * inner, mid + s * pkg.L / 2, pkg.W, ends.top, top, "termination")
        box(f"{name}", mid - s * inner, mid + s * inner, pkg.W, top, top, "element")
        z = top
        half = inner
    else:
        box(f"{name}", mid - s * land.G / 2, mid + s * land.G / 2, pkg.W, ends.top, ends.top, "element")
        z = ends.top
        half = land.G / 2
    p0, p1 = [0.0] * 3, [0.0] * 3
    p0[lat] = p1[lat] = ends.centre
    p0[nrm] = p1[nrm] = z
    p0[ax], p1[ax] = mid - s * half, mid + s * half
    if p1[ax] < p0[ax]:
        p0, p1 = p1, p0
    out.line = (tuple(p0), tuple(p1))
    return out


# --------------------------------------------------------------------------------------------
# Values from part numbers and the library of Touchstone files
# --------------------------------------------------------------------------------------------
def value_from_part(part: str) -> tuple[str, str]:
    """(kind, value text) read from a part number: inductors '2N0' / '12N' (nH), Murata capacitors
    'GJM1555C1H1R0...' (three digit pF code). ('', '') when unknown."""
    p = part.upper()
    m = re.match(r"^G[JR]M\d{4}\w\w\w(\d[R\d]\d)", p)
    if m:
        code = m.group(1)
        if "R" in code:
            return "C", f"{float(code.replace('R', '.')):g}pF"
        return "C", f"{int(code[:2]) * 10 ** int(code[2]):g}pF"
    m = re.search(r"(\d+)N(\d*)", p.split("-", 1)[-1]) or re.search(r"(\d+)N(\d*)", p)
    if m:
        return "L", f"{float(m.group(1) + '.' + (m.group(2) or '0')):g}nH"
    m = re.search(r"R(\d+)N?", p.split("-", 1)[-1])
    if m and series_for(p) is not None and series_for(p).kind == "L":
        return "L", f"{float('0.' + m.group(1)):g}nH"
    return "", ""


@dataclass
class LibraryPart:
    part: str
    vendor: str
    series: str          # package name in PACKAGES ("" = unknown series)
    kind: str            # L / C / R
    value: str
    file: str            # Touchstone file in the library folder
    ports: int


class Library:
    """Touchstone files of components imported by the user (the vendor files are not shipped)."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.parts: list[LibraryPart] = []
        idx = self.folder / "index.json"
        if idx.exists():
            try:
                self.parts = [LibraryPart(**d) for d in json.loads(idx.read_text())]
            except Exception:  # noqa: BLE001
                self.parts = []

    def save(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "index.json").write_text(json.dumps([asdict(p) for p in self.parts], indent=1))

    def import_files(self, paths: list[Path]) -> tuple[int, list[str]]:
        """Copies .s1p/.s2p files into the library. Returns the number of parts and the skipped files."""
        from . import touchstone
        skipped, n = [], 0
        known = {p.part: p for p in self.parts}
        for src in paths:
            src = Path(src)
            if src.suffix.lower() not in (".s1p", ".s2p"):
                continue
            try:
                f, S, _ = touchstone.read(src)
            except Exception as e:  # noqa: BLE001
                skipped.append(f"{src.name}: {e}")
                continue
            part = src.stem
            ser = series_for(part)
            kind, value = value_from_part(part)
            dst = self.folder / "files" / (ser.vendor if ser else "Other") / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dst.resolve():
                shutil.copyfile(src, dst)
            known[part] = LibraryPart(part, ser.vendor if ser else "", ser.name if ser else "",
                                      kind or (ser.kind if ser else ""), value, str(dst), S.shape[1])
            n += 1
        self.parts = sorted(known.values(), key=lambda p: (p.vendor, p.series, p.part))
        self.save()
        return n, skipped

    def import_folder(self, folder: str | Path) -> tuple[int, list[str]]:
        """All Touchstone files of a folder (with subfolders) or of a ZIP archive."""
        import tempfile
        import zipfile
        folder = Path(folder)
        if folder.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(folder) as z:
                    for m in z.namelist():
                        if m.lower().endswith((".s1p", ".s2p")) and ".." not in Path(m).parts:
                            z.extract(m, tmp)
                return self.import_files(sorted(Path(tmp).rglob("*.s[12][pP]")))
        return self.import_files(sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".s1p", ".s2p")))

    def search(self, text: str = "", kind: str = "") -> list[LibraryPart]:
        t = text.strip().upper()
        return [p for p in self.parts if (not kind or p.kind == kind)
                and (not t or t in p.part.upper() or t in p.value.upper() or t in p.series.upper())]


# --------------------------------------------------------------------------------------------
# Inserting a component into a design
# --------------------------------------------------------------------------------------------
KINDS = {"Resistor": "R", "Inductor": "L", "Capacitor": "C", "Touchstone": "S"}
NAME_PREFIX = {"R": "R", "L": "L", "C": "C", "S": "U"}


def rlc_props(kind: str, value: str, pkg: Package, parasitics: bool, touchstone: str = "",
              mount: str = "Series") -> dict[str, str]:
    """Properties of the Lumped RLC boundary of a component: the ideal element, or in series with the
    typical ESL and ESR of its size (``parasitics``), or the Touchstone file."""
    from . import schema
    p = schema.defaults(schema.BOUNDARIES["Lumped RLC"])
    p.update(UseResistance="False", UseInductance="False", UseCapacitance="False")
    if kind == "S":
        p.update(Model="Touchstone", TouchstoneFile=touchstone, TouchstoneMount=mount)
        return p
    p["Model"] = "Series RLC"
    esl, esr = PARASITICS.get(pkg.size, (0.0, 0.0))
    if kind == "R":
        p.update(UseResistance="True", Resistance=value)
    elif kind == "L":
        p.update(UseInductance="True", Inductance=value)
    else:
        p.update(UseCapacitance="True", Capacitance=value)
    if parasitics:
        if kind != "R":
            p.update(UseResistance="True", Resistance=f"{esr:g}ohm")
        if kind != "L":
            p.update(UseInductance="True", Inductance=f"{esl:g}nH")
    if kind == "R" and not parasitics:
        p["Model"] = "Parallel RLC"              # a plain resistor reads the same in both
    return p


def insert_component(project, design, trace_a: str, trace_b: str, ends: TraceEnds, pkg: Package, land: Land,
                     kind: str, value: str = "", name: str = "", parasitics: bool = True,
                     terminations: bool = False, touchstone: str = "", mount: str = "Series"):
    """Adds the pads, terminations and element sheet of a component between two traces and its Lumped RLC
    boundary. The pads are made like trace A: of its material, or added to the Perfect E / Finite
    Conductivity boundary that makes it a conductor (copper when neither). Returns (boundary, objects,
    warnings)."""
    from .design import Assignment
    from .expr import LENGTH_UNITS
    from .materials import is_conductor
    from . import schema
    to_model = 1e-3 / LENGTH_UNITS[design.units]
    fmt = lambda v: f"{v * to_model:.6g}"  # noqa: E731
    name = design.unique_name(name or NAME_PREFIX[kind] + "1")
    obj_a = design.obj(trace_a)
    solid = obj_a.dim == 3
    geo = component_geometry(ends, pkg, land, name, terminations=terminations, solid_pads=solid)
    try:
        metal = is_conductor(project.material(obj_a.material))
    except KeyError:
        metal = False
    bnd = None if metal else next((b for b in design.boundaries if b.kind in ("Perfect E", "Finite Conductivity")
                                   and trace_a in b.objects), None)
    pad_mat = obj_a.material if metal else ("vacuum" if bnd is not None else "copper")
    axis = "XYZ"[ends.normal]
    made = []
    for bx in geo.boxes:
        lo = ",".join(fmt(v) for v in bx.lo)
        if bx.sheet_axis is None:
            props = dict(Position=lo, XSize=fmt(bx.size[0]), YSize=fmt(bx.size[1]), ZSize=fmt(bx.size[2]))
            o = design.add_object("CreateBox", name=bx.name, material="vacuum", **props)
        else:
            sa, sb = schema.RECT_SIZES[axis]
            props = {"Position": lo, "Axis": axis, sa: fmt(bx.size["XYZ".index(sa[0])]),
                     sb: fmt(bx.size["XYZ".index(sb[0])])}
            o = design.add_object("CreateRectangle", name=bx.name, material="vacuum", **props)
        mat = {"pad": pad_mat, "termination": "pec", "element": "vacuum"}[bx.role]
        o.material = mat
        o.group = name
        if bx.role != "element":
            try:
                o.solve_inside = not is_conductor(project.material(mat))
            except KeyError:
                pass
            if bx.role == "pad" and bnd is not None:
                bnd.objects.append(o.name)
            o.color = "#d4a017" if bx.role == "pad" else "#9a9a9a"
        else:
            o.color = "#16a085"
        made.append(o)
    element = next(o for o in made if o.name == name)
    props = rlc_props(kind, value, pkg, parasitics and not terminations, touchstone, mount)
    props["Component"] = pkg.name + ("" if kind != "S" else f" ({Path(touchstone).stem})")
    props["IntegrationLine"] = ";".join(",".join(fmt(v) for v in p) for p in geo.line)
    b = Assignment(design.unique_name(name + "_RLC"), "Lumped RLC", props, objects=[element.name])
    design.boundaries.append(b)
    return b, made, geo.warnings
