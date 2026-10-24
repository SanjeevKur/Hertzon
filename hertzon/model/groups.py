"""Object groups: several objects (also of different kinds) that are selected and transformed as one unit.

A group is the ``group`` name its member objects carry, so it is saved with the objects and needs no list of
its own. Groups are flat: grouping objects that are already in a group moves them into the new group."""
from __future__ import annotations


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
