"""Gerber X2 / Excellon export (File > Export > Gerber).

The files are read back with two independent parsers: gerbonara (Gerber + Excellon, attributes and
geometry) and pygerber (raster rendering, which checks polarity, holes and islands by copper area)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hertzon.model.design import Project, Command, Assignment, CoordSys  # noqa: E402
from hertzon.export import gerber as G  # noqa: E402

PI = math.pi


def _sheet(d, kind, name, **props):
    o = d.add_object(kind, name=name, **props)
    return o


def board_project() -> Project:
    """Two-layer test board (mm): rectangle, circle, plane with a cut-out and an island in it,
    overlapping rectangle + circle, ellipse, thin solid trace, via with pads, NPTH hole and a slot."""
    p = Project()
    d = p.designs[0]
    d.units = "mm"
    sub = d.add_object("CreateBox", name="Substrate", material="FR4_epoxy", Position="0,0,0",
                       XSize="40", YSize="30", ZSize="1.6")
    hole = d.add_object("CreateCylinder", name="MountHole", CenterPosition="5,5,0", Axis="Z", Radius="1.6",
                        Height="1.6")
    slot = d.add_object("CreateBox", name="Slot", Position="34,4,0", XSize="2", YSize="6", ZSize="1.6")
    hole.consumed_by = slot.consumed_by = sub.name
    sub.history.append(Command.new("Subtract", Tools="MountHole,Slot", KeepOriginals="False"))
    top = []
    top.append(_sheet(d, "CreateRectangle", "Rect", Position="2,20,1.6", Axis="Z", XSize="10", YSize="5"))
    top.append(_sheet(d, "CreateCircle", "Disk", CenterPosition="20,22,1.6", Axis="Z", Radius="3"))
    plane = _sheet(d, "CreateRectangle", "Plane", Position="14,4,1.6", Axis="Z", XSize="12", YSize="10")
    cut = _sheet(d, "CreateCircle", "Cutout", CenterPosition="20,9,1.6", Axis="Z", Radius="2")
    cut.consumed_by = plane.name
    plane.history.append(Command.new("Subtract", Tools="Cutout", KeepOriginals="False"))
    top.append(plane)
    top.append(_sheet(d, "CreateCircle", "Island", CenterPosition="20,9,1.6", Axis="Z", Radius="1"))
    top.append(_sheet(d, "CreateRectangle", "OverlapA", Position="30,18,1.6", Axis="Z", XSize="6", YSize="6"))
    top.append(_sheet(d, "CreateCircle", "OverlapB", CenterPosition="36,21,1.6", Axis="Z", Radius="3"))
    top.append(_sheet(d, "CreateEllipse", "Oval", CenterPosition="7,11,1.6", Axis="Z", MajorRadius="2",
                      Ratio="0.5"))
    top.append(_sheet(d, "CreateCircle", "PadTop", CenterPosition="28,6,1.6", Axis="Z", Radius="0.8"))
    d.add_object("CreateBox", name="Trace", material="copper", Position="2,15,1.6", XSize="15", YSize="1",
                 ZSize="0.035")
    gnd = _sheet(d, "CreateRectangle", "Ground", Position="0,0,0", Axis="Z", XSize="40", YSize="30")
    # copper clearance around the mounting hole and the slot (a copper cut-out, never a drill)
    c1 = _sheet(d, "CreateCircle", "GndClear", CenterPosition="5,5,0", Axis="Z", Radius="2.5")
    c2 = _sheet(d, "CreateRectangle", "GndSlot", Position="33.5,3.5,0", Axis="Z", XSize="3", YSize="7")
    c1.consumed_by = c2.consumed_by = gnd.name
    gnd.history.append(Command.new("Subtract", Tools="GndClear,GndSlot", KeepOriginals="False"))
    d.add_object("CreateCylinder", name="Via", material="copper", CenterPosition="28,6,0", Axis="Z",
                 Radius="0.4", Height="1.6")
    port = _sheet(d, "CreateRectangle", "PortSheet", Position="2,0,0", Axis="Y", XSize="1", ZSize="1.6")
    rad = _sheet(d, "CreateRectangle", "RadSheet", Position="0,0,8", Axis="Z", XSize="40", YSize="30")
    d.add_object("CreateRegion")
    d.boundaries.append(Assignment("CopperTop", "Finite Conductivity", {}, objects=[o.name for o in top]))
    d.boundaries.append(Assignment("PerfE_Ground", "Perfect E", {}, objects=[gnd.name]))
    d.boundaries.append(Assignment("Rad1", "Radiation", {}, objects=[rad.name]))
    d.excitations.append(Assignment("1", "Lumped Port", {"PortNumber": "1"}, objects=[port.name]))
    return p


TOP_AREA = (10 * 5 + PI * 9 + (120 - PI * 4) + PI * 1 + (36 + PI * 9 / 2) + PI * 2 * 1 + PI * 0.64
            + 15 * 1)          # mm^2
BOTTOM_AREA = 40 * 30 - PI * 6.25 - 3 * 7


@pytest.fixture(scope="module")
def board():
    p = board_project()
    return p, G.run(p, p.designs[0], G.Options())


@pytest.fixture(scope="module")
def files(board, tmp_path_factory):
    out = tmp_path_factory.mktemp("gerber")
    paths = G.write(board[1], out, "board")
    return {p.name: p for p in paths}


def _bb(obj, *a):
    (x0, y0), (x1, y1) = obj.bounding_box(*a)
    return (x0, y0, x1, y1)


def _roles(res):
    return {it.label: (it.category, it.role) for it in res.items}


def test_classification_and_layers(board):
    _, res = board
    r = _roles(res)
    assert res.ok, res.messages
    assert [(L.name, L.side) for L in res.layers] == [("Top", "Top"), ("Bottom", "Bot")]
    assert res.layers[0].z0 == pytest.approx(1.6e-3) and res.layers[0].z1 == pytest.approx(1.635e-3)
    assert res.layers[1].z == pytest.approx(0)
    for n in ("Rect", "Disk", "Plane", "Island", "OverlapA", "OverlapB", "Oval", "PadTop", "Trace"):
        assert r[n][1] == "L1", n
    assert r["Ground"][1] == "L2"
    assert r["Via"] == ("via", "via")
    assert r["Substrate"] == ("board", "outline")
    for n in ("PortSheet", "RadSheet", "Region1"):        # simulation helpers are never exported
        assert r[n] == ("excluded", "skip"), n
    top = res.layers[0]
    # union: OverlapA + OverlapB is one component; the island sits inside the hole of the plane
    assert len(top.components) == 8
    plane = next(c for c in top.components if c.outer.area == pytest.approx(120e-6))
    assert len(plane.holes) == 1 and plane.holes[0].circle() == pytest.approx((20e-3, 9e-3, 4e-3))
    island = next(c for c in top.components if c.outer.circle() and c.outer.circle()[2] == pytest.approx(2e-3))
    assert island.depth == 2 and top.components.index(island) > top.components.index(plane)


def test_file_set_and_attributes(files):
    gerbonara = pytest.importorskip("gerbonara")
    assert sorted(files) == ["board-L1_Top.gbr", "board-L2_Bottom.gbr", "board-NPTH.drl", "board-PTH_L1-L2.drl",
                             "board-Profile.gbr", "board-export-report.txt"]
    want = {"board-L1_Top.gbr": ("Copper", "L1", "Top"), "board-L2_Bottom.gbr": ("Copper", "L2", "Bot"),
            "board-Profile.gbr": ("Profile", "NP")}
    for name, ff in want.items():
        g = gerbonara.GerberFile.open(files[name])
        assert g.file_attrs[".FileFunction"] == ff
        assert g.file_attrs[".FilePolarity"] == ("Positive",)
        assert g.file_attrs[".SameCoordinates"] == ("Design1",)
        assert g.unit == gerbonara.utils.MM
    text = files["board-L1_Top.gbr"].read_text()
    assert "%FSLAX46Y46*%" in text and "%MOMM*%" in text and "G75*" in text and text.rstrip().endswith("M02*")
    assert "%LPC*%" in text and "%TA.AperFunction,Conductor*%" in text


def test_geometry_read_back_with_gerbonara(files):
    gerbonara = pytest.importorskip("gerbonara")
    top = gerbonara.GerberFile.open(files["board-L1_Top.gbr"])
    (x0, y0), (x1, y1) = top.bounding_box(gerbonara.utils.MM)
    assert (x0, y0, x1, y1) == pytest.approx((2, 4, 39, 25), abs=1e-6)
    bot = gerbonara.GerberFile.open(files["board-L2_Bottom.gbr"])
    assert _bb(bot, gerbonara.utils.MM) == pytest.approx((0, 0, 40, 30), abs=1e-6)
    regions = [o for o in top.objects if isinstance(o, gerbonara.graphic_objects.Region)]
    dark = [r for r in regions if r.polarity_dark]
    clear = [r for r in regions if not r.polarity_dark]
    assert len(dark) == 8 and len(clear) == 1
    assert len([o for o in bot.objects if not o.polarity_dark]) == 2      # clearance circle and slot
    # arcs are kept exact: centre and radius of the overlapping circle and the disk
    centres = {(round(c[0], 6), round(c[1], 6)) for r in regions for ac in r.arc_centers if ac for c in [ac[1]]}
    assert {(36.0, 21.0), (20.0, 22.0), (20.0, 9.0), (28.0, 6.0)} <= centres
    disk = next(r for r in dark if any(ac and math.dist(ac[1], (20, 22)) < 1e-9 for ac in r.arc_centers))
    assert len(disk.outline) <= 2 and math.dist(disk.outline[0], (20, 22)) == pytest.approx(3, abs=1e-6)
    # the clear region is the round cut-out of the plane, not the island
    assert _bb(clear[0]) == pytest.approx((18, 7, 22, 11), abs=1e-6)


def test_copper_area_by_independent_rendering(files):
    """pygerber renders the files; copper area and holes/islands are compared with the model."""
    pytest.importorskip("pygerber")
    from pygerber.gerberx3.api.v2 import GerberFile, ColorScheme
    from pygerber.common.rgba import RGBA
    from PIL import Image
    bw = ColorScheme(background_color=RGBA.from_hex("#000000"), clear_color=RGBA.from_hex("#000000"),
                     solid_color=RGBA.from_hex("#FFFFFF"), clear_region_color=RGBA.from_hex("#000000"),
                     solid_region_color=RGBA.from_hex("#FFFFFF"))
    dpmm = 40
    for name, area in (("board-L1_Top.gbr", TOP_AREA), ("board-L2_Bottom.gbr", BOTTOM_AREA)):
        parsed = GerberFile.from_file(files[name]).parse()
        png = files[name].with_suffix(".png")
        parsed.render_raster(png, dpmm=dpmm, color_scheme=bw)
        img = np.asarray(Image.open(png).convert("L")) > 127
        assert img.sum() / dpmm ** 2 == pytest.approx(area, rel=0.01), name
        if "Top" in name:
            info = parsed.get_info()
            mx, my = float(info.min_x_mm), float(info.max_y_mm)

            def px(x, y):
                return img[int((my - y) * dpmm), int((x - mx) * dpmm)]
            assert px(20, 9)            # island in the cut-out
            assert not px(21.5, 9)      # ring of the cut-out is clear
            assert px(24, 12)           # plane
            assert px(37.5, 21)         # half disk of the overlap
            assert not px(30, 10)       # no copper between objects


def test_drill_files(files):
    gerbonara = pytest.importorskip("gerbonara")
    pth = gerbonara.ExcellonFile.open(files["board-PTH_L1-L2.drl"])
    npth = gerbonara.ExcellonFile.open(files["board-NPTH.drl"])
    assert pth.is_plated and npth.is_nonplated
    assert [(o.x, o.y, o.aperture.diameter) for o in pth.objects] == [pytest.approx((28, 6, 0.8))]
    # only the board hole: neither the copper cut-out (20, 9) nor the milled slot become drills
    assert [(o.x, o.y, o.aperture.diameter) for o in npth.objects] == [pytest.approx((5, 5, 3.2))]
    assert "TF.FileFunction,Plated,1,2,PTH" in files["board-PTH_L1-L2.drl"].read_text()
    assert "TF.FileFunction,NonPlated,1,2,NPTH" in files["board-NPTH.drl"].read_text()


def test_profile_keeps_slot_and_drops_drilled_hole(files):
    gerbonara = pytest.importorskip("gerbonara")
    prof = gerbonara.GerberFile.open(files["board-Profile.gbr"])
    lines = [o for o in prof.objects if isinstance(o, gerbonara.graphic_objects.Line)]
    arcs = [o for o in prof.objects if isinstance(o, gerbonara.graphic_objects.Arc)]
    assert not arcs                                            # the round hole is in the NPTH file
    segs = {tuple(sorted(((round(o.x1, 6), round(o.y1, 6)), (round(o.x2, 6), round(o.y2, 6))))) for o in lines}
    assert ((34.0, 4.0), (34.0, 10.0)) in segs and ((36.0, 4.0), (36.0, 10.0)) in segs
    assert ((0.0, 0.0), (40.0, 0.0)) in segs
    assert lines[0].aperture.diameter == pytest.approx(0.1)


def test_common_coordinate_system_and_origin(tmp_path):
    gerbonara = pytest.importorskip("gerbonara")
    p = board_project()
    d = p.designs[0]
    d.coord_systems.append(CoordSys("BoardCS", props={"Mode": "Euler Angle ZXZ", "Origin": "10,5,0",
                                                      "Phi": "90deg", "Theta": "0deg", "Psi": "0deg"}))
    res = G.run(p, d, G.Options(cs="BoardCS", origin="point", origin_point=(1e-3, -2e-3)))
    assert res.ok, res.messages
    files = {x.name: x for x in G.write(res, tmp_path, "b")}
    # board x = global y - 5, board y = -(global x - 10); then minus the origin (1, -2)
    bot = gerbonara.GerberFile.open(files["b-L2_Bottom.gbr"])
    assert _bb(bot, gerbonara.utils.MM) == pytest.approx((-6, -28, 24, 12), abs=1e-6)
    pth = gerbonara.ExcellonFile.open(files["b-PTH_L1-L2.drl"])
    assert (pth.objects[0].x, pth.objects[0].y) == pytest.approx((6 - 5 - 1, -(28 - 10) + 2))
    top = gerbonara.GerberFile.open(files["b-L1_Top.gbr"])
    centres = {(round(c[0], 6), round(c[1], 6)) for r in top.objects for ac in r.arc_centers if ac for c in [ac[1]]}
    assert (21 - 5 - 1, -(36 - 10) + 2) in centres           # centre of OverlapB in the same frame


def test_origin_at_outline_and_inch(tmp_path):
    gerbonara = pytest.importorskip("gerbonara")
    p = board_project()
    p.designs[0].objects[0].history[0].props["Position"] = "-3,-7,0"     # board corner no longer at 0,0
    p.designs[0].objects[0].history[0].props["XSize"] = "43"
    p.designs[0].objects[0].history[0].props["YSize"] = "37"
    res = G.run(p, p.designs[0], G.Options(origin="outline", units="inch"))
    assert res.ok, res.messages
    files = {x.name: x for x in G.write(res, tmp_path, "b")}
    text = files["b-L2_Bottom.gbr"].read_text()
    assert "%MOIN*%" in text and "%FSLAX26Y26*%" in text
    bot = gerbonara.GerberFile.open(files["b-L2_Bottom.gbr"])
    (x0, y0), (x1, y1) = bot.bounding_box(gerbonara.utils.MM)
    assert (x0, y0, x1, y1) == pytest.approx((3, 7, 43, 37), abs=1e-4)
    npth = gerbonara.ExcellonFile.open(files["b-NPTH.drl"])
    o = npth.objects[0]
    assert npth.objects[0].unit == gerbonara.utils.Inch
    assert (o.x * 25.4, o.y * 25.4, o.aperture.diameter * 25.4) == pytest.approx((8, 12, 3.2), abs=1e-3)


def test_npth_through_copper_is_reported():
    p = board_project()
    gnd = p.designs[0].obj("Ground")
    gnd.history.pop()                           # ground plane without clearance around the mounting hole
    res = G.run(p, p.designs[0], G.Options())
    assert res.ok
    assert any(m.level == "warning" and "Non-plated hole" in m.text and "Bottom" in m.text for m in res.messages)
    assert not any("Non-plated" in m.text for m in G.run(board_project(), "Design1").messages)


def test_curve_tolerance(tmp_path):
    """The ellipse is the only curve that is approximated; its chords stay within the tolerance."""
    gerbonara = pytest.importorskip("gerbonara")
    counts = {}
    for tol in (10e-6, 1e-6):
        p = board_project()
        res = G.run(p, p.designs[0], G.Options(tolerance=tol))
        files = {x.name: x for x in G.write(res, tmp_path / str(tol), "b")}
        top = gerbonara.GerberFile.open(files["b-L1_Top.gbr"])
        oval = next(r for r in top.objects if _bb(r)[:2] == pytest.approx((5, 10), abs=1e-3))
        pts = np.array(oval.outline)
        counts[tol] = len(pts)
        a, b = 2.0, 1.0
        mid = 0.5 * (pts + np.roll(pts, -1, axis=0)) - (7, 11)
        F = (mid[:, 0] / a) ** 2 + (mid[:, 1] / b) ** 2 - 1
        grad = np.hypot(2 * mid[:, 0] / a ** 2, 2 * mid[:, 1] / b ** 2)
        assert np.max(np.abs(F) / grad) <= tol * 1e3 * 1.05 + 2e-6
        assert any(m.level == "info" and "Oval" in m.objects for m in res.messages)
    assert counts[1e-6] > counts[10e-6]


def test_errors_prevent_export(tmp_path):
    p = Project()
    d = p.designs[0]
    d.add_object("CreateBox", name="Sub", material="FR4_epoxy", XSize="10", YSize="10", ZSize="1")
    tilt = d.add_object("CreateRectangle", name="Tilted", Position="0,0,0", Axis="X", YSize="5", ZSize="1")
    d.boundaries.append(Assignment("PE", "Perfect E", {}, objects=[tilt.name]))
    res = G.run(p, d, G.Options())
    assert not res.ok
    assert any("No copper" in m.text for m in res.errors)
    assert any(m.level == "warning" and "Tilted" in m.objects for m in res.messages)
    with pytest.raises(G.ExportError):
        G.write(res, tmp_path, "x")
    assert not list(tmp_path.iterdir())                     # nothing written
    # forcing a simulation helper onto a layer is an error too
    p2 = board_project()
    res2 = G.run(p2, p2.designs[0], G.Options(roles={"Region1": "L1"}))
    assert not res2.ok and any("Region1" in m.objects for m in res2.errors)


@pytest.mark.parametrize("name,ncomp", [("PatchAntenna", (1, 1)), ("HairpinFilter", (3, 1))])
def test_examples_export(name, ncomp, tmp_path):
    p = Project.load(ROOT / "examples" / f"{name}.hzn")
    res = G.run(p, p.designs[0], G.Options(origin="outline"))
    assert res.ok, res.messages
    assert tuple(len(L.components) for L in res.layers) == ncomp
    names = [x.name for x in G.write(res, tmp_path, name)]
    assert f"{name}-Profile.gbr" in names and not any(n.endswith(".drl") for n in names)


def test_export_dialog(tmp_path, monkeypatch):
    """The dialog suggests the assignment, follows a manual change and writes the files."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from hertzon.gui.gerber_dialog import GerberExportDialog
    app = QApplication.instance() or QApplication([])

    class Doc:
        def __init__(self, p):
            self.project, self.design = p, p.designs[0]

        def evaluator(self):
            return self.design.evaluator(self.project)

    p = board_project()
    dlg = GerberExportDialog(None, Doc(p))
    assert dlg.res.ok and dlg.export_btn.isEnabled()
    assert dlg.layers.rowCount() == 2 and dlg.objects.rowCount() == len(dlg.res.items)
    row = [dlg.objects.item(r, 0).text() for r in range(dlg.objects.rowCount())].index("Disk")
    combo = dlg.objects.cellWidget(row, 2)
    assert combo.currentData() == "L1"
    combo.setCurrentIndex(combo.findData("L2"))                 # move the disk to the bottom layer
    dlg.recompute()
    assert dlg.res.item("Disk").role == "L2"
    dlg.layers.item(1, 1).setText("GND")                        # rename the bottom layer
    dlg.folder.setText(str(tmp_path))
    dlg.base.setText("dlg")
    written = dlg.export(interactive=False)
    names = sorted(x.name for x in written)
    assert "dlg-L2_GND.gbr" in names and "dlg-PTH_L1-L2.drl" in names
    assert "Export finished" in dlg.status.text()
    # the Export button must call export() interactively (clicked(bool) must not reach 'interactive')
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMessageBox
    shown = []

    def close_boxes():
        for w in app.topLevelWidgets():
            if isinstance(w, QMessageBox) and w.isVisible():
                shown.append(w.text())
                w.done(0)
    for k in range(1, 30):
        QTimer.singleShot(100 * k, close_boxes)
    dlg.export_btn.click()
    assert shown and any("Overwrite" in t or "Export finished" in t for t in shown)
    dlg.grab().save(str(tmp_path / "dialog.png"))
    app.processEvents()
