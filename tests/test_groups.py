"""Object groups: grouping, ungrouping, saving, selection as one unit and lumped components as groups."""
import numpy as np
import pytest

from hertzon.model import groups as G
from hertzon.model import lumped as LU
from hertzon.model.design import Assignment, Project


def _design():
    p = Project("Groups")
    d = p.designs[0]
    d.add_object("CreateBox", name="Box1")
    d.add_object("CreateRectangle", name="Rect1")
    d.add_object("CreateCylinder", name="Cyl1")
    d.add_object("CreatePolyline", name="Line1")
    return p, d


def test_group_ungroup_rename():
    p, d = _design()
    g = G.make_group(d, ["Box1", "Rect1", "Line1"])           # solids, sheets and lines in one group
    assert g == "Group1" and G.groups(d) == {"Group1": ["Box1", "Rect1", "Line1"]}
    assert G.expand(d, ["Rect1", "Cyl1"]) == ["Box1", "Rect1", "Line1", "Cyl1"]
    assert "Group1" in d.all_names()                          # objects do not take a group's name
    assert d.add_object("CreateBox", name="Group1").name == "Group2"
    # grouping a member with another object moves the whole group into the new one (groups are flat)
    g2 = G.make_group(d, ["Box1", "Cyl1"])
    assert g2 == "Group3" and G.groups(d) == {"Group3": ["Box1", "Rect1", "Cyl1", "Line1"]}
    assert G.rename_group(d, "Group3", "Filter") == "Filter" and G.members(d, "Filter")[0] == "Box1"
    with pytest.raises(ValueError):
        G.rename_group(d, "Filter", "Cyl1")
    assert G.ungroup(d, ["Rect1"]) == ["Filter"] and G.groups(d) == {}
    G.make_group(d, ["Box1", "Rect1"])
    assert G.ungroup(d, ["Group1"]) == ["Group1"] and not any(o.group for o in d.objects)


def test_groups_are_saved(tmp_path):
    p, d = _design()
    G.make_group(d, ["Box1", "Cyl1"], "Housing")
    p.save(tmp_path / "g.hertzon")
    q = Project.load(tmp_path / "g.hertzon")
    assert G.groups(q.designs[0]) == {"Housing": ["Box1", "Cyl1"]}
    assert G.groups(p.clone().designs[0]) == {"Housing": ["Box1", "Cyl1"]}       # undo snapshots too


def test_copies_form_own_groups():
    p, d = _design()
    G.make_group(d, ["Box1", "Rect1"])
    for src in ("Box1", "Rect1"):
        o = d.add_object("CreateBox", name=src + "_1")
        o.group = "Group1"
    G.copy_groups(d, {"Box1": "Box1_1", "Rect1": "Rect1_1"})
    assert G.groups(d) == {"Group1": ["Box1", "Rect1"], "Group2": ["Box1_1", "Rect1_1"]}


def test_consumed_members_leave_the_group():
    p, d = _design()
    G.make_group(d, ["Box1", "Cyl1", "Rect1"])
    d.obj("Cyl1").consumed_by = "Box1"
    assert G.groups(d) == {"Group1": ["Box1", "Rect1"]}


def test_lumped_component_is_a_group():
    """An SMD component placed from the lumped element menu is one group: pads and element sheet."""
    p = Project("Free")
    d = p.designs[0]
    pkg = LU.PACKAGES["0603"]
    b, made, _ = LU.insert_component(p, d, None, None, None, pkg, LU.land_pattern(pkg), "C", "1pF",
                                     place=LU.Placement((5.0, 7.0, 0.8), axis=1, normal=2))
    assert G.groups(d) == {"C1": ["C1_PadA", "C1_PadB", "C1"]}
    b2, made2, _ = LU.insert_component(p, d, None, None, None, pkg, LU.land_pattern(pkg), "L", "1nH",
                                       place=LU.Placement((0, 0, 0.8), axis=0, normal=2))
    assert G.groups(d)["L1"] == ["L1_PadA", "L1_PadB", "L1"]


def test_integration_line_follows_its_objects():
    p = Project("Free")
    d = p.designs[0]
    pkg = LU.PACKAGES["0603"]
    b, made, _ = LU.insert_component(p, d, None, None, None, pkg, LU.land_pattern(pkg), "C", "1pF",
                                     place=LU.Placement((5.0, 7.0, 0.8), axis=1, normal=2))
    names = G.members(d, "C1")
    ev = d.evaluator(p)
    before = np.asarray(ev.points(b.props["IntegrationLine"])) * 1e3
    # a move keeps the variables of the placed part
    assert G.follow_lines(d, ev, names, "Move", {"Vector": "1,-2,0"}) == ["C1_RLC"]
    assert b.props["IntegrationLine"] == "C1_x+1mm,C1_y-0.45mm-2mm,C1_z;C1_x+1mm,C1_y+0.45mm-2mm,C1_z"
    assert np.allclose(np.asarray(ev.points(b.props["IntegrationLine"])) * 1e3, before + [1, -2, 0])
    # rotation by 90 degrees about Z through the part's centre turns the line into X
    G.follow_lines(d, ev, names, "Rotate", {"Axis": "Z", "Angle": "90deg", "Center": "6,5,0"})
    after = np.asarray(ev.points(b.props["IntegrationLine"])) * 1e3
    assert np.allclose(after, [[6.45, 5, 0.8], [5.55, 5, 0.8]])
    G.follow_lines(d, ev, names, "Mirror", {"Base": "0,0,0", "Normal": "1,0,0"})
    assert np.allclose(np.asarray(ev.points(b.props["IntegrationLine"])) * 1e3, after * [-1, 1, 1])
    # a line on objects that are not all moved stays where it is
    line = b.props["IntegrationLine"]
    assert G.follow_lines(d, ev, names[:2], "Move", {"Vector": "1,0,0"}) == []
    assert b.props["IntegrationLine"] == line
    # plain numbers stay plain numbers in model units
    a = Assignment("P1", "Lumped Port", {"IntegrationLine": "0,0,0;0,0,1"}, objects=["C1"])
    d.excitations.append(a)
    G.follow_lines(d, ev, ["C1"], "Scale", {"Factor": "2", "Center": "0,0,0"})
    assert a.props["IntegrationLine"] == "0,0,0;0,0,2"
