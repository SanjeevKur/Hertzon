"""Integration lines belong to their object: they follow its Move, Rotate, Mirror and Scale."""
import itertools

import numpy as np
import pytest

from hertzon.model import lumped as LU
from hertzon.model import objframe as OF
from hertzon.model.design import Assignment, Command, FaceRef, Project, project_from_dict


def _placed():
    p = Project("Free")
    d = p.designs[0]
    pkg = LU.PACKAGES["0603"]
    b, made, _ = LU.insert_component(p, d, None, None, None, pkg, LU.land_pattern(pkg), "C", "1pF",
                                     place=LU.Placement((5.0, 7.0, 0.8), axis=1, normal=2))
    return p, d, b, made


def _line_mm(p, d, b):
    return np.asarray(OF.global_points(d, b, "IntegrationLine", d.evaluator(p))) * 1e3


def test_line_follows_moved_component():
    p, d, b, made = _placed()
    text = b.props["IntegrationLine"]
    before = _line_mm(p, d, b)
    assert np.allclose(before, [[5, 6.55, 0.8], [5, 7.45, 0.8]])
    for o in made:                                   # the component moved as a whole
        o.history.append(Command("Move", {"Vector": "10,-3,0"}))
    assert np.allclose(_line_mm(p, d, b), before + [10, -3, 0])
    assert b.props["IntegrationLine"] == text        # stored text untouched, variables still work
    d.variables["C1_y"] = "8mm"
    assert np.allclose(_line_mm(p, d, b)[:, 1], [7.55 - 3, 8.45 - 3])


def test_line_follows_rotate_mirror_scale_and_suppress():
    p, d, b, made = _placed()
    el = d.obj("C1")
    el.history.append(Command("Rotate", {"Axis": "Z", "Angle": "90deg", "Center": "5,7,0"}))
    assert np.allclose(_line_mm(p, d, b), [[5.45, 7, 0.8], [4.55, 7, 0.8]])
    el.history.append(Command("Mirror", {"Base": "0,0,0", "Normal": "1,0,0"}))
    assert np.allclose(_line_mm(p, d, b), [[-5.45, 7, 0.8], [-4.55, 7, 0.8]])
    el.history[-1].suppressed = True
    el.history.append(Command("Scale", {"Factor": "2", "Center": "5,7,0.8"}))
    assert np.allclose(_line_mm(p, d, b), [[5.9, 7, 0.8], [4.1, 7, 0.8]])


def test_matrix_matches_the_geometry_kernel():
    from hertzon.kernel import occ as K
    p = Project()
    d = p.designs[0]
    o = d.add_object("CreateBox", Position="1,2,3", XSize="2", YSize="1", ZSize="0.5")
    o.history += [Command("Move", {"Vector": "1,-2,0.5"}),
                  Command("Rotate", {"Axis": "X", "Angle": "30deg", "Center": "0,1,0"}),
                  Command("Mirror", {"Base": "1,1,1", "Normal": "1,1,0"}),
                  Command("Scale", {"Factor": "1.5", "Center": "0,0,1"}),
                  Command("Rotate", {"Axis": "Y", "Angle": "-50deg", "Center": "0,0,0"})]
    ev = d.evaluator(p)
    K.ensure_gmsh("objframe_test")
    t = K.tessellate(K.build_design(d, ev))[o.name]
    M = OF.object_matrix(d, o, ev)
    corners = np.array([[1 + 2 * i, 2 + j, 3 + 0.5 * k] for i, j, k in itertools.product((0, 1), repeat=3)]) * 1e-3
    mapped = corners @ M[:3, :3].T + M[:3, 3]
    P = np.asarray(t.points, float)
    for c in mapped:
        assert np.min(np.linalg.norm(P - c, axis=1)) < 1e-9


def test_picked_line_stored_in_object_coordinates():
    p = Project()
    d = p.designs[0]
    d.add_object("CreateRectangle", name="S", Position="0,0,0", Axis="Z", XSize="1", YSize="2")
    a = Assignment("P1", "Lumped Port", {"IntegrationLine": ""}, faces=[FaceRef("S")])
    ev = d.evaluator(p)
    assert OF.to_object_coords(d, a, "w/2,0,0;w/2,2,0", ev) == "w/2,0,0;w/2,2,0"     # not moved: kept as is
    d.obj("S").history.append(Command("Move", {"Vector": "10,0,0"}))
    local = OF.to_object_coords(d, a, "10.5,0,0;10.5,2,0", ev)
    assert np.allclose(np.asarray(ev.points(local)) * 1e3, [[0.5, 0, 0], [0.5, 2, 0]])
    a.props["IntegrationLine"] = local
    assert np.allclose(np.asarray(OF.global_points(d, a, "IntegrationLine", ev)) * 1e3, [[10.5, 0, 0], [10.5, 2, 0]])


def test_older_files_are_converted():
    """Files before version 2 stored the line in global coordinates of the moved object."""
    p = Project()
    d = p.designs[0]
    o = d.add_object("CreateRectangle", name="S", Position="0,0,0", Axis="Z", XSize="1", YSize="2")
    o.history.append(Command("Move", {"Vector": "10,0,0"}))
    d.add_object("CreateRectangle", name="T", Position="0,0,0", Axis="Z", XSize="1", YSize="2")
    d.excitations.append(Assignment("P1", "Lumped Port", {"IntegrationLine": "10.5,0,0;10.5,2,0"}, objects=["S"]))
    d.boundaries.append(Assignment("R1", "Lumped RLC", {"IntegrationLine": "x0,0,0;x0,2,0"}, objects=["T"]))
    d.variables["x0"] = "0.5"
    dd = p.to_dict()
    assert dd["version"] == 2
    assert project_from_dict(dd).designs[0].excitations[0].props["IntegrationLine"] == "10.5,0,0;10.5,2,0"
    dd["version"] = 1
    q = project_from_dict(dd).designs[0]
    ev = q.evaluator()
    a = q.excitations[0]
    assert np.allclose(np.asarray(ev.points(a.props["IntegrationLine"])) * 1e3, [[0.5, 0, 0], [0.5, 2, 0]])
    assert np.allclose(np.asarray(OF.global_points(q, a, "IntegrationLine", ev)) * 1e3,
                       [[10.5, 0, 0], [10.5, 2, 0]])
    assert q.boundaries[0].props["IntegrationLine"] == "x0,0,0;x0,2,0"       # unmoved object: expressions kept


def test_orientation_is_applied():
    from hertzon.model.design import CoordSys
    p = Project()
    d = p.designs[0]
    c = CoordSys("CS1", "Relative", "Global")
    c.props.update({"Origin": "1,2,3"})
    d.coord_systems.append(c)
    o = d.add_object("CreateRectangle", name="S", Position="0,0,0", Axis="Z", XSize="1", YSize="2")
    o.orientation = "CS1"
    a = Assignment("P1", "Lumped Port", {"IntegrationLine": "0.5,0,0;0.5,2,0"}, objects=["S"])
    assert np.allclose(np.asarray(OF.global_points(d, a, "IntegrationLine", d.evaluator(p))) * 1e3,
                       [[1.5, 2, 3], [1.5, 4, 3]])


@pytest.mark.parametrize("mode,props", [
    ("Axis/Position", {"XAxis": "1,1,0", "YPoint": "-1,1,1"}),
    ("Euler Angle ZXZ", {"Phi": "30deg", "Theta": "40deg", "Psi": "10deg"}),
    ("Euler Angle ZYZ", {"Phi": "0deg", "Theta": "180deg", "Psi": "0deg"}),      # a half turn
])
def test_fillet_and_chamfer_on_an_object_in_a_relative_cs(mode, props):
    """An object defined in a relative coordinate system keeps exact planar faces, so its edges can be filleted
    and chamfered, and it lands where the coordinate system puts it."""
    from hertzon.kernel import occ as K
    from hertzon.model import coordsys
    from hertzon.model.design import CoordSys
    K.ensure_gmsh("test_objframe")
    p = Project()
    d = p.designs[0]
    cs = CoordSys("CS")
    cs.props.update(Mode=mode, Origin="3,-2,5", **props)
    d.coord_systems.append(cs)
    for name, op in (("F", Command.new("Fillet", Radius="0.5", Edges="0,2,4,6")),
                     ("C", Command.new("Chamfer", Distance="0.5", Edges="0")), ("B", None)):
        o = d.add_object("CreateBox", name=name, Position="1,2,0", XSize="10", YSize="20", ZSize="2")
        o.orientation = "CS"
        if op is not None:
            o.history.append(op)
    ctx = K.build_design(d, d.evaluator(p))
    assert not ctx.errors
    O, R = coordsys.frame(d, "CS", d.evaluator(p))
    verts = np.asarray(K.tessellate(ctx)["B"].vertices)
    for c in np.array([[1, 2, 0], [11, 22, 2], [11, 2, 0]]) * 1e-3:
        assert np.min(np.linalg.norm(verts - (O + R @ c), axis=1)) < 1e-9
