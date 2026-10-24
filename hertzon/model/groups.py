"""Object groups: several objects (also of different kinds) that are selected and transformed as one unit.

A group is the ``group`` name its member objects carry, so it is saved with the objects and needs no list of
its own. Groups are flat: grouping objects that are already in a group moves them into the new group."""
from __future__ import annotations

import math

import numpy as np

from .expr import ExprError, split_list

TRANSFORMS = ("Move", "Rotate", "Mirror", "Scale")


def groups(design) -> dict[str, list[str]]:
    """Group name -> live member objects, in object order (groups without live members are left out)."""
    out: dict[str, list[str]] = {}
    for o in design.live_objects():
        if o.group:
            out.setdefault(o.group, []).append(o.name)
    return out


def members(design, group: str) -> list[str]:
    return [o.name for o in design.live_objects() if o.group == group]


def expand(design, names: list[str]) -> list[str]:
    """The objects plus all other members of their groups, without duplicates."""
    out: list[str] = []
    for n in names:
        g = design.obj(n).group if design.has_obj(n) else ""
        for m in (members(design, g) if g else [n]):
            if m not in out:
                out.append(m)
    return out


def make_group(design, names: list[str], name: str = "") -> str:
    """Puts the objects (and the other members of groups they are in) into one new group; returns its name."""
    objs = expand(design, names)
    if not objs:
        raise ValueError("no objects to group")
    name = design.unique_name(name.strip() or "Group1")
    for n in objs:
        design.obj(n).group = name
    return name


def ungroup(design, names: list[str]) -> list[str]:
    """Dissolves the groups of the given objects or group names; returns the dissolved group names."""
    gone = []
    for n in names:
        g = design.obj(n).group if design.has_obj(n) else n
        if g and g not in gone:
            gone.append(g)
    for o in design.objects:
        if o.group in gone:
            o.group = ""
    return gone


def rename_group(design, old: str, new: str) -> str:
    new = new.strip()
    if not new:
        raise ValueError("the name is empty")
    if new != old and new in design.all_names():
        raise ValueError(f"'{new}' is already used")
    for o in design.objects:
        if o.group == old:
            o.group = new
    return new


def copy_groups(design, copies: dict[str, str]) -> None:
    """Copies of grouped objects (source name -> copy name) form new groups of their own instead of joining
    the group of their source."""
    new: dict[str, str] = {}
    for cp in copies.values():
        o = design.obj(cp)
        if not o.group:
            continue
        if o.group not in new:
            new[o.group] = design.unique_name(o.group, design.all_names() | set(new.values()))
        o.group = new[o.group]


# ---- integration lines follow their objects ------------------------------------------------------------------
def _rotate(p, c, axis, ang):
    k = np.zeros(3)
    k["XYZ".index(axis)] = 1.0
    v = p - c
    return c + v * math.cos(ang) + np.cross(k, v) * math.sin(ang) + k * (k @ v) * (1 - math.cos(ang))


def follow_lines(design, ev, objs: list[str], kind: str, props: dict) -> list[str]:
    """Applies a Move / Rotate / Mirror / Scale of the objects also to the integration lines of the lumped
    ports and lumped elements that lie on these objects only, so a moved component keeps its line. A move
    keeps the expressions (variables) of the line; the other transforms write numbers. Returns the names of
    the assignments whose line was changed."""
    if kind not in TRANSFORMS:
        return []
    moved = set(objs)
    if any(design.obj(n).orientation not in ("", "Global") for n in objs if design.has_obj(n)):
        return []                                    # commands of rotated objects are in their own system
    scale = ev.unit_scale
    num = lambda v: f"{v / scale:.9g}"  # noqa: E731
    done = []
    for a in (*design.boundaries, *design.excitations):
        line = a.props.get("IntegrationLine", "").strip()
        on = set(a.objects) | {f.obj for f in a.faces}
        if not line or not on or not on <= moved:
            continue
        try:
            pts = [split_list(p) for p in line.split(";") if p.strip()]
            P = [np.asarray(ev.point(",".join(p))) for p in pts]
            if kind == "Move":
                d = ev.vector(props["Vector"])
                out = []
                for p, x in zip(pts, P):
                    comp = []
                    for i, s in enumerate(p):
                        if abs(d[i]) < 1e-15:
                            comp.append(s)
                        elif ev._eval(s)[1]:                 # expression with units: keep it, add the shift
                            comp.append(f"{s}{'+' if d[i] >= 0 else '-'}{num(abs(d[i]))}{design.units}")
                        else:
                            comp.append(num(x[i] + d[i]))
                    out.append(",".join(comp))
            else:
                if kind == "Rotate":
                    c = np.asarray(ev.point(props.get("Center", "0,0,0") or "0,0,0"))
                    Q = [_rotate(x, c, props.get("Axis", "Z"), ev.angle(props["Angle"])) for x in P]
                elif kind == "Mirror":
                    b = np.asarray(ev.point(props["Base"]))
                    n = np.asarray(ev.vector(props["Normal"]), float)
                    n = n / (np.linalg.norm(n) + 1e-300)
                    Q = [x - 2 * ((x - b) @ n) * n for x in P]
                else:
                    c = np.asarray(ev.point(props.get("Center", "0,0,0") or "0,0,0"))
                    f = ev.value(props["Factor"])
                    Q = [c + f * (x - c) for x in P]
                out = [",".join(num(v) for v in q) for q in Q]
        except (ExprError, KeyError, ValueError):
            continue
        a.props["IntegrationLine"] = ";".join(out)
        done.append(a.name)
    return done
