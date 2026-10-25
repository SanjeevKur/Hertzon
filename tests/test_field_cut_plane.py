"""Cut planes through per-element fields: the field jumps at material boundaries and metal sheets, and the cut
must keep both sides apart instead of giving every boundary point the value of one side."""
from __future__ import annotations

import types

import numpy as np
import pyvista as pv
import pytest

from hertzon.gui import fields as F
from hertzon.solver.results import ResultStore

LOW, HIGH = 1.0, 100.0          # |E| below and above the material boundary z = 0.5 mm


@pytest.fixture
def plot(tmp_path, monkeypatch):
    cube = pv.ImageData(dimensions=(5, 5, 5), spacing=(0.25e-3,) * 3).triangulate()
    nodes = np.asarray(cube.points, float)
    tets = np.asarray(cube.cells_dict[pv.CellType.TETRA], np.int64)
    upper = nodes[tets].mean(axis=1)[:, 2] > 0.5e-3
    mag = np.where(upper, HIGH, LOW)
    E = np.repeat(np.column_stack([np.zeros(len(tets)), np.zeros(len(tets)), mag]), 4, axis=0).astype(complex)
    st = ResultStore(tmp_path)
    st.save_npz("mesh.npz", nodes=nodes, tets=tets, tet_obj=upper.astype(np.int64),
                obj_names=np.array(["Lower", "Upper"]))
    st.save_npz("fields_LastAdaptive.npz", tet4=np.array([1]), freq_0=np.array([1e9]), E_0_1=E, H_0_1=E / 377)
    monkeypatch.setattr(F, "store_for", lambda doc, solution: (ResultStore(tmp_path), "LastAdaptive"))
    monkeypatch.setattr(F, "source_vector", lambda doc, st, fields, grid, which, fi, port:
                        fields[f"{which}_{fi}_{port}"].astype(np.complex128))
    F._cache.clear()
    doc = types.SimpleNamespace(design=types.SimpleNamespace(units="mm"))
    return doc, types.SimpleNamespace(actors={})


@pytest.mark.parametrize("plot_type", ["Flat Shaded", "Smooth Shaded"])
def test_cut_keeps_the_jump_at_a_material_boundary(plot, plot_type):
    doc, view = plot
    meshes, _ = F.make_overlay(doc, view, "Mag_E", "Setup1 : LastAdaptive", plane="XZ@",
                               attrs={"PlotType": plot_type})
    s = meshes[0][0]
    side = np.round(s.cell_data_to_point_data().point_data["_obj"])
    vals = np.asarray(s.point_data["Mag_E"])
    on = np.isclose(np.asarray(s.points)[:, 2], 0.5e-3)
    assert on.any()                                         # the cut has points on the boundary
    assert np.allclose(vals[side == 1], HIGH) and np.allclose(vals[side == 0], LOW)
    assert set(np.round(vals[on])) == {LOW, HIGH}           # both sides are drawn there


def test_drawn_plane_is_cut_from_the_solver_mesh(plot):
    """A drawn plane (a sheet outside the solver mesh) shows the jump too, cropped to the sheet."""
    grid, fields = F.load_grid(*F.store_for(None, ""))
    work = grid.copy(deep=False)
    work.point_data["cmag"] = np.linalg.norm(np.abs(fields["E_0_1"]), axis=1)
    sheet = pv.Rectangle([[0.1e-3, 0.4e-3, 0.2e-3], [0.9e-3, 0.4e-3, 0.2e-3], [0.9e-3, 0.4e-3, 0.8e-3]])
    s = F._sheet_cut(work, sheet.triangulate())
    assert np.allclose(s.bounds, (0.1e-3, 0.9e-3, 0.4e-3, 0.4e-3, 0.2e-3, 0.8e-3), atol=1e-9)
    side = np.round(s.cell_data_to_point_data().point_data["_obj"])
    vals = np.asarray(s.point_data["cmag"])
    assert np.allclose(vals[side == 1], HIGH) and np.allclose(vals[side == 0], LOW)
    on = np.isclose(np.asarray(s.points)[:, 2], 0.5e-3)
    assert on.any() and set(np.round(vals[on])) == {LOW, HIGH}
    assert F._sheet_cut(work, pv.Sphere(radius=0.2e-3, center=(0.5e-3,) * 3)) is None   # not flat


def test_drawn_plane_is_cropped_exactly_to_its_outline(plot):
    """Round outlines and holes: the plot covers exactly the sheet, without whole cut triangles sticking out
    over the edge or missing next to it."""
    grid, fields = F.load_grid(*F.store_for(None, ""))
    work = grid.copy(deep=False)
    work.point_data["cmag"] = np.linalg.norm(np.abs(fields["E_0_1"]), axis=1)
    ring = pv.Disc(center=(0.5e-3, 0.4e-3, 0.5e-3), normal=(0, 1, 0), inner=0.1e-3, outer=0.35e-3, c_res=48)
    s = F._sheet_cut(work, ring)
    assert s.area == pytest.approx(ring.area, rel=1e-6)
    r = np.linalg.norm(np.asarray(s.points)[:, [0, 2]] - 0.5e-3, axis=1)
    assert r.min() >= 0.1e-3 * np.cos(np.pi / 48) - 1e-9 and r.max() <= 0.35e-3 + 1e-9
    tilted = pv.Rectangle([[0.1e-3, 0.2e-3, 0.1e-3], [0.9e-3, 0.7e-3, 0.1e-3], [0.9e-3, 0.7e-3, 0.9e-3]])
    assert F._sheet_cut(work, tilted).area == pytest.approx(tilted.area, rel=1e-6)


def test_plot_on_a_helper_sheet_follows_the_sheet():
    """Resizing or moving a non-model sheet draws its field plot again on the new shape; a model object,
    an unchanged sheet or a plot that is not on screen are left alone."""
    from hertzon.gui import main_window as MW

    objs = {"Cut": types.SimpleNamespace(model=False), "Patch": types.SimpleNamespace(model=True)}
    plots = [types.SimpleNamespace(name="Mag_E1", objects=["Cut"], faces=[]),
             types.SimpleNamespace(name="Mag_E2", objects=["Patch"], faces=[]),
             types.SimpleNamespace(name="Mag_E3", objects=["Cut"], faces=[])]
    des = types.SimpleNamespace(field_plots=plots, has_obj=lambda n: n in objs, obj=lambda n: objs[n])
    drawn = []
    win = types.SimpleNamespace(des=des, doc=types.SimpleNamespace(tess={"Cut": object(), "Patch": object()}),
                                view=types.SimpleNamespace(overlay_actors={"Mag_E1": 1, "Mag_E2": 1}),
                                anim_bar=types.SimpleNamespace(names=[], close_player=lambda: None),
                                _draw_field_plot=lambda n, loud=True: drawn.append(n))
    for n in ("_helper_geometry", "_follow_helper_plots"):
        setattr(win, n, types.MethodType(getattr(MW.MainWindow, n), win))
    win._plot_geom = {p.name: win._helper_geometry(p) for p in plots}
    win._follow_helper_plots()
    assert drawn == []                                      # nothing changed
    win.doc.tess = {"Cut": object(), "Patch": object()}     # both rebuilt (e.g. the sheet was resized)
    win._follow_helper_plots()
    assert drawn == ["Mag_E1"]                              # not the model object, not the hidden plot
