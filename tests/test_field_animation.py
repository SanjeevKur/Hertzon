"""Phase animation of field plots: the plot is prepared once and every frame only turns the field, which
must give the same picture as building the plot at that phase."""
from __future__ import annotations

import types

import numpy as np
import pyvista as pv
import pytest

from hertzon.gui import fields as F
from hertzon.solver.results import ResultStore

A = np.array([1.0 + 2.0j, -0.5 + 0.3j, 0.2 - 1.0j])
B = np.array([[3.0 - 1.0j, 0.5j, 0.0], [0.0, -2.0 + 1.0j, 1.0], [0.4j, 0.0, 1.5 - 0.5j]])


def _field(x):
    """A complex field linear in the position: interpolation from the tet corners is exact."""
    return A[None, :] + x @ B.T


@pytest.fixture
def plot(tmp_path, monkeypatch):
    cube = pv.ImageData(dimensions=(5, 5, 5), spacing=(0.25e-3,) * 3).triangulate()
    nodes = np.asarray(cube.points, float)
    tets = np.asarray(cube.cells_dict[pv.CellType.TETRA], np.int64)
    st = ResultStore(tmp_path)
    st.save_npz("mesh.npz", nodes=nodes, tets=tets, tet_obj=np.zeros(len(tets), np.int64), obj_names=np.array(["Box"]))
    st.save_npz("fields_LastAdaptive.npz", tet4=np.array([1]), freq_0=np.array([1e9]),
                E_0_1=_field(nodes[tets].reshape(-1, 3)), H_0_1=_field(nodes[tets].reshape(-1, 3)) / 377)
    monkeypatch.setattr(F, "store_for", lambda doc, solution: (ResultStore(tmp_path), "LastAdaptive"))
    monkeypatch.setattr(F, "source_vector", lambda doc, st, fields, grid, which, fi, port:
                        fields[f"{which}_{fi}_{port}"].astype(np.complex128))
    F._cache.clear()
    doc = types.SimpleNamespace(design=types.SimpleNamespace(units="mm"))
    view = types.SimpleNamespace(actors={})
    return doc, view


@pytest.mark.parametrize("phase", [0.0, 37.0, 90.0, 200.0])
def test_prepared_plot_turns_the_field(plot, phase):
    doc, view = plot
    attrs = {"PlotType": "Flat Shaded"}
    prep = F.prepare_overlay(doc, view, "Mag_E", "Setup1 : LastAdaptive", plane="XY@", attrs=attrs)
    meshes, info = F.overlay_at(prep, phase)
    s = meshes[0][0]
    want = np.linalg.norm(np.real(_field(np.asarray(s.points)) * np.exp(1j * np.radians(phase))), axis=1)
    assert np.allclose(s.point_data["Mag_E"], want, rtol=1e-9, atol=1e-12 * want.max())
    assert f"phase {phase:g} deg" in info
    again, _ = F.make_overlay(doc, view, "Mag_E", "Setup1 : LastAdaptive", 0, 1, phase, plane="XY@", attrs=attrs)
    assert np.array_equal(again[0][0].point_data["Mag_E"], s.point_data["Mag_E"])


def test_prepared_arrows_follow_the_phase(plot):
    doc, view = plot
    prep = F.prepare_overlay(doc, view, "Vector_E", "Setup1 : LastAdaptive", plane="XY@")
    for phase in (0.0, 120.0):
        meshes, _ = F.overlay_at(prep, phase)
        sub, _ = F._arrow_points(prep, 0, prep["sampled"][0])
        vec = np.real(_field(np.asarray(sub.points)) * np.exp(1j * np.radians(phase)))
        assert np.allclose(F._at_phase(sub, np.cos(np.radians(phase)), np.sin(np.radians(phase))).point_data["re"],
                           vec, rtol=1e-9, atol=1e-12 * np.abs(vec).max())
        assert meshes and meshes[0][0].n_points
    assert len(prep["arrows"]) == 1                 # the arrow points are found once, not per frame


def test_grid_cache_rereads_changed_files(plot, tmp_path):
    st = ResultStore(tmp_path)
    g1, f1 = F.load_grid(st, "LastAdaptive")
    g2, f2 = F.load_grid(st, "LastAdaptive")
    assert g1 is g2 and f1 is f2                    # nothing read again for the next frame
    nodes = st.npz("mesh.npz")["nodes"]
    tets = st.npz("mesh.npz")["tets"]
    st.save_npz("fields_LastAdaptive.npz", tet4=np.array([1]), freq_0=np.array([2e9]),
                E_0_1=2 * _field(nodes[tets].reshape(-1, 3)), H_0_1=_field(nodes[tets].reshape(-1, 3)))
    _, f3 = F.load_grid(st, "LastAdaptive")
    assert float(f3["freq_0"][0]) == 2e9


def test_phase_animation_defaults_to_35_steps():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from hertzon.gui.animation import AnimationSetupDialog
    app = QApplication.instance() or QApplication([])  # noqa: F841
    fp = types.SimpleNamespace(name="P", props={})
    dlg = AnimationSetupDialog(None, fp, [1e9], [1])
    assert dlg.nph.value() == 35
    frames = dlg.frames(fp)
    assert len(frames) == 34 and frames[0][1]["phase"] == 0.0     # 0..360: the last step equals the first
