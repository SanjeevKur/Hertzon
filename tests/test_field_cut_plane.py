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
