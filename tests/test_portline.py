"""Automatic integration line of lumped ports and lumped RLC sheets (between the touching conductors)."""
import os
import types
from pathlib import Path

import numpy as np
import pytest

from hertzon.model.portline import between_conductors, boundary_segments

EX = Path(__file__).resolve().parents[1] / "examples"


def rect(x0, x1, y0, y1, z=0.0):
    """Sheet in the plane z as points and two triangles."""
    P = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)
    return P, np.array([[0, 1, 2], [0, 2, 3]])


def vrect(x0, x1, z0, z1, y=0.0):
    """Sheet in the plane y."""
    P = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], float)
    return P, np.array([[0, 1, 2], [0, 2, 3]])


def tris(sheet):
    P, T = sheet
    return P[T]


def test_boundary_of_a_rectangle():
    seg = boundary_segments(*rect(0, 2, 0, 1))
    assert len(seg) == 4
    assert sorted(np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)) == pytest.approx([1, 1, 2, 2])


def test_rlc_sheet_in_the_gap_between_two_traces():
    """Gap of 1 mm between two 3 mm traces along y: the line spans the gap at the centre of the trace width."""
    sheet = rect(-1.5, 1.5, 10, 11, 1.6)
    cond = {"A": tris(rect(-1.5, 1.5, 0, 10, 1.6)), "B": tris(rect(-1.5, 1.5, 11, 20, 1.6)),
            "Ground": tris(rect(-10, 10, 0, 20, 0))}
    p0, p1 = between_conductors(*sheet, cond)
    assert p0 == pytest.approx([0, 10, 1.6]) and p1 == pytest.approx([0, 11, 1.6])


def test_sheet_wider_than_the_traces_uses_the_contact():
    """Only the part of the sheet edge that touches the trace counts: the line is centred on the traces."""
    sheet = rect(-3, 3, 10, 11, 1.6)
    cond = {"A": tris(rect(0, 2, 0, 10, 1.6)), "B": tris(rect(0, 2, 11, 20, 1.6))}
    p0, p1 = between_conductors(*sheet, cond)
    assert p0 == pytest.approx([1, 10, 1.6], abs=0.05) and p1 == pytest.approx([1, 11, 1.6], abs=0.05)


def test_vertical_port_points_from_ground_to_trace():
    sheet = vrect(-1.5, 1.5, 0, 1.6)
    cond = {"Trace": tris(rect(-1.5, 1.5, 0, 20, 1.6)), "Ground": tris(rect(-10, 10, -5, 20, 0))}
    p0, p1 = between_conductors(*sheet, cond)
    assert p0 == pytest.approx([0, 0, 0]) and p1 == pytest.approx([0, 0, 1.6])


def test_solid_trace_ends_and_a_sheet_lying_on_a_conductor():
    """Box traces count by their end faces; a conductor under the whole sheet is not one of its two sides."""
    def box(x0, x1, y0, y1, z0, z1):
        faces = [rect(x0, x1, y0, y1, z0), rect(x0, x1, y0, y1, z1), vrect(x0, x1, z0, z1, y0),
                 vrect(x0, x1, z0, z1, y1)]
        return np.concatenate([tris(f) for f in faces])
    sheet = rect(-1, 1, 5, 6, 1.0)
    cond = {"A": box(-1, 1, 0, 5, 1.0, 1.035), "B": box(-1, 1, 6, 12, 1.0, 1.035),
            "Under": tris(rect(-5, 5, 0, 12, 1.0))}
    p0, p1 = between_conductors(*sheet, cond)
    assert p0 == pytest.approx([0, 5, 1]) and p1 == pytest.approx([0, 6, 1])


def test_no_line_without_two_conductors():
    sheet = rect(-1, 1, 10, 11)
    assert between_conductors(*sheet, {}) is None
    assert between_conductors(*sheet, {"A": tris(rect(-1, 1, 0, 10))}) is None
    assert between_conductors(*sheet, {"Far": tris(rect(-1, 1, 30, 40)), "A": tris(rect(-1, 1, 0, 10))}) is None


def _window(doc):
    """The integration line helpers of the main window on a document, without the window."""
    from hertzon.gui import main_window as MW
    win = types.SimpleNamespace(doc=doc, des=doc.design)
    for n in ("_face_triangles", "_conductor_surfaces", "_auto_integration_line", "_fmt", "_pt"):
        setattr(win, n, types.MethodType(getattr(MW.MainWindow, n), win))
    return win


def test_auto_line_on_the_microstrip_example():
    """The port sheets of the example get the line from the ground to the trace that is stored in it."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from hertzon.gui.doc import Document
    from hertzon.model.design import Project
    app = QApplication.instance() or QApplication([])  # noqa: F841
    doc = Document(Project.load(EX / "Microstrip.hzn"))
    doc.rebuild()
    win = _window(doc)
    assert win._auto_integration_line([], ["Port1"]) == "0,0,0;0,0,1.6"
    assert win._auto_integration_line([], ["Port2"]) == "0,40,0;0,40,1.6"


def test_auto_line_of_an_rlc_sheet_between_two_traces():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from hertzon.gui.doc import Document
    from hertzon.model.design import Project
    app = QApplication.instance() or QApplication([])  # noqa: F841
    p = Project("RLC")
    d = p.designs[0]
    d.add_object("CreateBox", name="Substrate", material="FR4_epoxy", Position="-10,0,0", XSize="20", YSize="21",
                 ZSize="1.6")
    d.add_object("CreateRectangle", name="T1", material="pec", Position="-1.5,0,1.6", Axis="Z", XSize="3",
                 YSize="10")
    d.add_object("CreateRectangle", name="T2", material="pec", Position="-1.5,14,1.6", Axis="Z", XSize="3",
                 YSize="7")
    # the gap is longer than the traces are wide: across the shorter side would be wrong
    d.add_object("CreateRectangle", name="R1", Position="-1.5,10,1.6", Axis="Z", XSize="3", YSize="4")
    doc = Document(p)
    doc.rebuild()
    win = _window(doc)
    assert win._auto_integration_line([], ["R1"]) == "0,10,1.6;0,14,1.6"
    # conductors given by a boundary instead of a material
    for o in d.objects:
        if o.name in ("T1", "T2"):
            o.material = "vacuum"
    from hertzon.model.design import Assignment
    d.boundaries.append(Assignment("PerfE1", "Perfect E", {}, objects=["T1", "T2"]))
    doc.commit("model")
    assert win._auto_integration_line([], ["R1"]) == "0,10,1.6;0,14,1.6"
