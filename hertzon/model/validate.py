"""Design validation check (the "Validation Check" dialog)."""
from __future__ import annotations

from dataclasses import dataclass

from .design import Project, Design
from .expr import ExprError
from . import schema


@dataclass
class Check:
    category: str
    level: str          # ok | warning | error
    message: str


CATEGORIES = ["Design Settings", "3D Model", "Boundaries and Excitations", "Mesh Operations", "Analysis Setup",
              "Radiation"]


def validate(project: Project, design: Design, tess: dict | None = None, errors: dict | None = None) -> list[Check]:
    out: list[Check] = []
    ev = design.evaluator(project)
    add = lambda c, l, m: out.append(Check(c, l, m))  # noqa: E731

    # Design settings / variables
    bad = False
    for name, expr in {**project.variables, **design.variables}.items():
        err = ev.check(expr)
        if err:
            add("Design Settings", "error", f"Variable '{name}': {err}")
            bad = True
    if not bad:
        add("Design Settings", "ok", f"Solution type {design.settings.get('SolutionType', 'Modal')}, "
                                     f"units {design.units}")

    # 3D model
    live = [o for o in design.live_objects() if o.model]
    if not live:
        add("3D Model", "error", "The model contains no objects")
    for o in live:
        try:
            project.material(o.material)
        except KeyError:
            add("3D Model", "error", f"'{o.name}': unknown material '{o.material}'")
        for c in o.history:
            for k, v in c.props.items():
                p = next((p for p in schema.GEOMETRY.get(c.kind, []) if p.name == k), None)
                if p and p.type in ("length", "expr", "int", "angle", "float") and str(v).strip():
                    err = ev.check(v)
                    if err:
                        add("3D Model", "error", f"'{o.name}' {c.kind}.{k}: {err}")
    for n, e in (errors or {}).items():
        add("3D Model", "error", f"'{n}': {e}")
    if tess is not None:
        solids = [(o.name, tess[o.name]) for o in live if o.name in tess and tess[o.name].dim == 3
                  and o.history[0].kind != "CreateRegion"]
        for i in range(len(solids)):
            for j in range(i + 1, len(solids)):
                a, ta = solids[i]
                b, tb = solids[j]
                if _boxes_overlap(ta.bbox, tb.bbox) and not _contains(ta.bbox, tb.bbox) and not _contains(tb.bbox, ta.bbox):
                    add("3D Model", "warning", f"Objects '{a}' and '{b}' may intersect; the smaller object takes "
                                               "precedence in the overlap")
    if not any(c.level == "error" for c in out if c.category == "3D Model"):
        add("3D Model", "ok", f"{len(live)} objects")

    # boundaries and excitations
    names = {o.name for o in design.live_objects()}
    for a in design.boundaries + design.excitations:
        refs = {f.obj for f in a.faces} | set(a.objects)
        missing = refs - names
        if missing:
            add("Boundaries and Excitations", "error", f"'{a.name}' references missing object(s): {', '.join(missing)}")
        if not refs and a.kind != "Incident Plane Wave":
            add("Boundaries and Excitations", "error", f"'{a.name}' has no assignment")
        if a.kind in schema.UNSUPPORTED_BOUNDARIES:
            add("Boundaries and Excitations", "warning", f"'{a.name}': {schema.UNSUPPORTED_BOUNDARIES[a.kind]}")
        for k, v in a.props.items():
            p = next((p for p in schema.BOUNDARIES.get(a.kind, []) + schema.EXCITATIONS.get(a.kind, [])
                      if p.name == k), None)
            if p and p.type in ("length", "expr", "angle", "float") and str(v).strip() and ev.check(v):
                add("Boundaries and Excitations", "error", f"'{a.name}'.{k}: {ev.check(v)}")
        if a.kind in ("Radiation", "PML"):
            for o in sorted(refs & names):
                try:
                    mat = project.material(design.obj(o).material)
                    cond = float(mat.conductivity or 0)
                except Exception:  # noqa: BLE001
                    continue
                if mat.pec or cond > 1e4:
                    add("Boundaries and Excitations", "error",
                        f"'{a.name}' is on '{o}' whose material is {mat.name}: the radiation region must be a "
                        "dielectric (air or vacuum), a conductor leaves nothing to solve")
    if tess and not design.ports() and any(e.kind == "Incident Plane Wave" and e.enabled
                                           for e in design.excitations):
        for b in design.boundaries:
            if b.kind != "Radiation" or not b.enabled:
                continue
            for o in {f.obj for f in b.faces}:
                t = tess.get(o)
                if t is None:
                    continue
                n = len(t.face_normal)
                have = {f.face for f in b.faces if f.obj == o}
                if len(have) < n:
                    add("Boundaries and Excitations", "warning",
                        f"'{b.name}' covers {len(have)} of {n} faces of '{o}'. The incident wave enters only "
                        "through radiation faces; the other faces are Perfect E and reflect it. Assign the "
                        "radiation boundary to all outer faces (or use Auto Open).")
    pws = [e for e in design.excitations if e.kind == "Incident Plane Wave" and e.enabled]
    if pws and not design.ports():
        rads = [b for b in design.boundaries if b.kind == "Radiation" and b.enabled]
        if design.settings.get("AutoOpen", "Off") == "Off" and not rads:
            add("Boundaries and Excitations", "error", f"'{pws[0].name}' needs a radiation boundary (or Auto Open): "
                                                       "the incident wave enters through it")
        elif tess and rads:
            from . import planewave
            region = [o for b in rads for o in {f.obj for f in b.faces} | set(b.objects) if o in tess]
            for e in pws:
                try:
                    _, _, origin = planewave.vectors(e.props, ev)
                except Exception:  # noqa: BLE001
                    continue
                inside = any(all(tess[o].bbox[i] - 1e-9 <= origin[i] <= tess[o].bbox[i + 3] + 1e-9 for i in range(3))
                             for o in region)
                if region and not inside:
                    add("Boundaries and Excitations", "warning",
                        f"'{e.name}': the zero phase position lies outside the radiation boundary region "
                        f"({', '.join(sorted(set(region)))}). Place it inside the region.")
    ports = design.ports()
    nums = []
    for e in ports:
        try:
            nums.append(ev.integer(e.props["PortNumber"]))
        except ExprError:
            pass
    if design.settings.get("SolutionType") != "Eigenmode":
        if not ports and not any(e.kind == "Incident Plane Wave" for e in design.excitations):
            add("Boundaries and Excitations", "error", "No excitations defined")
        if sorted(nums) != list(range(1, len(nums) + 1)):
            add("Boundaries and Excitations", "error", f"Port numbers must be 1..N without gaps, found {sorted(nums)}")
    open_bc = design.settings.get("AutoOpen", "Off") != "Off" or any(
        b.kind in ("Radiation", "PML") for b in design.boundaries)
    if not open_bc:
        add("Boundaries and Excitations", "ok", "Outer boundary is Perfect E (closed problem)")
    if not any(c.level == "error" for c in out if c.category == "Boundaries and Excitations"):
        add("Boundaries and Excitations", "ok", f"{len(design.boundaries)} boundaries, {len(design.excitations)} excitations")

    # mesh operations
    for m in design.mesh_ops:
        refs = {f.obj for f in m.faces} | set(m.objects)
        if refs - names:
            add("Mesh Operations", "error", f"'{m.name}' references missing objects")
        if not refs:
            add("Mesh Operations", "error", f"'{m.name}' has no assignment")
    if not any(c.category == "Mesh Operations" for c in out):
        add("Mesh Operations", "ok", f"{len(design.mesh_ops)} mesh operations")

    # analysis
    if not design.setups:
        add("Analysis Setup", "error", "No solution setup defined")
    for s in design.setups:
        props = schema.EIGEN_SETUP if s.eigen else schema.SETUP
        for p in props:
            v = s.props.get(p.name, p.default)
            if p.type in ("freq",) and schema.is_visible(p, s.props):
                err = ev.check(v)
                if err:
                    add("Analysis Setup", "error", f"{s.name}.{p.name}: {err}")
        for sw in s.sweeps:
            for p in schema.SWEEP:
                if p.type == "freq" and schema.is_visible(p, sw.props):
                    err = ev.check(sw.props.get(p.name, p.default))
                    if err:
                        add("Analysis Setup", "error", f"{s.name}:{sw.name}.{p.name}: {err}")
    if design.setups and not any(c.level == "error" for c in out if c.category == "Analysis Setup"):
        add("Analysis Setup", "ok", f"{len(design.setups)} setup(s)")

    # radiation
    if design.spheres and not open_bc:
        add("Radiation", "warning", "Far field setup without radiation/PML boundary or Auto Open")
    elif design.spheres:
        add("Radiation", "ok", f"{len(design.spheres)} far field setup(s)")
    return out


def _boxes_overlap(a, b) -> bool:
    return all(a[i] < b[i + 3] - 1e-12 and b[i] < a[i + 3] - 1e-12 for i in range(3))


def _contains(a, b) -> bool:
    return all(a[i] <= b[i] + 1e-12 and b[i + 3] <= a[i + 3] + 1e-12 for i in range(3))


def worst(checks: list[Check], category: str) -> str:
    lv = [c.level for c in checks if c.category == category]
    if "error" in lv:
        return "error"
    if "warning" in lv:
        return "warning"
    return "ok"
