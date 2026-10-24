"""Fast unit tests (no FEM solve). Run: python -m pytest tests"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hertzon.model.design import Project, Command  # noqa: E402
from hertzon.model.expr import Evaluator, ExprError  # noqa: E402
from hertzon.model.validate import validate  # noqa: E402
from hertzon.kernel import occ as K  # noqa: E402
from hertzon.solver import bridge as B  # noqa: E402
from hertzon.solver.script import design_to_script  # noqa: E402

EX = ROOT / "examples"


def test_expressions_and_units():
    ev = Evaluator({"w": "2mm", "h": "w/2"}, "mm")
    assert ev.length("w+1mm") == pytest.approx(3e-3)
    assert ev.length("3") == pytest.approx(3e-3)       # bare numbers are model units
    assert ev.length("h") == pytest.approx(1e-3)
    assert ev.freq("2.45") == pytest.approx(2.45e9)
    assert ev.freq("100MHz") == pytest.approx(1e8)
    assert ev.angle("90") == pytest.approx(np.pi / 2)
    assert ev.point("1,2,w") == pytest.approx((1e-3, 2e-3, 2e-3))
    with pytest.raises(ExprError):
        ev.length("undefined_var")


def test_project_roundtrip(tmp_path):
    p = Project.load(EX / "Microstrip.hzn")
    out = tmp_path / "copy.hzn"
    p.save(out)
    q = Project.load(out)
    assert json.dumps(p.to_dict(), sort_keys=True) == json.dumps(q.to_dict(), sort_keys=True)


@pytest.mark.parametrize("name", ["WaveguidePost", "Microstrip", "PatchAntenna"])
def test_examples_build_and_validate(name):
    p = Project.load(EX / f"{name}.hzn")
    d = p.designs[0]
    K.ensure_gmsh("test_" + name)
    ctx = K.build_design(d, d.evaluator(p))
    assert not ctx.errors
    tess = K.tessellate(ctx)
    assert all(len(t.tris) for t in tess.values())
    checks = validate(p, d, tess, ctx.errors)
    assert not [c for c in checks if c.level == "error"], checks


def test_booleans_keep_top_dimension():
    p = Project()
    d = p.designs[0]
    d.add_object("CreateSphere", CenterPosition="0,0,0", Radius="2")
    d.add_object("CreateCone", CenterPosition="10,0,0", Axis="Z", LowerRadius="2", UpperRadius="1", Height="4")
    d.obj("Cone1").consumed_by = "Sphere1"
    d.obj("Sphere1").history.append(Command.new("Unite", Tools="Cone1", KeepOriginals="False"))
    K.ensure_gmsh("test_bool")
    ctx = K.build_design(d, d.evaluator(p))
    assert {dim for dim, _ in ctx.built["Sphere1"]} == {3}
    assert len(ctx.built["Sphere1"]) == 2


def test_sweep_frequencies():
    ev = Evaluator({}, "mm")
    f = B.sweep_frequencies({"Distribution": "Linear Step", "Start": "1GHz", "Stop": "2GHz", "Step": "0.25GHz"}, ev)
    assert np.allclose(f, [1e9, 1.25e9, 1.5e9, 1.75e9, 2e9])
    f = B.sweep_frequencies({"Distribution": "Linear Count", "Start": "1GHz", "Stop": "2GHz", "Count": "11"}, ev)
    assert len(f) == 11
    f = B.sweep_frequencies({"Distribution": "Log Scale", "Start": "1GHz", "Stop": "100GHz",
                             "SamplesPerDecade": "10"}, ev)
    assert f[0] == pytest.approx(1e9) and f[-1] == pytest.approx(1e11) and len(f) == 21


def test_aligned_delta_ignores_port_sign_flip():
    S = np.array([[0.1 + 0.2j, 0.9j], [0.9j, 0.1 - 0.2j]])
    flip = np.diag([1, -1])
    d, *_ = B.aligned_delta(S, flip @ S @ flip)
    assert d == pytest.approx(0, abs=1e-12)


def test_renormalization_matches_z_matrix():
    S = np.array([[[0.2 + 0.1j, 0.7], [0.7, 0.1 - 0.3j]]])
    ports = [{"number": 1, "renorm_ohm": 75.0}, {"number": 2, "renorm_ohm": 75.0}]
    S2, z = B.postprocess_ports([1e9], S, [50, 50], ports)
    back, _ = B.postprocess_ports([1e9], S2, z, [{"number": 1, "renorm_ohm": 50.0}, {"number": 2, "renorm_ohm": 50.0}])
    assert np.allclose(back, S)
    # one-port check: open-circuit-like load keeps |S|=1
    S1 = np.array([[[1j]]])
    S1r, _ = B.postprocess_ports([1e9], S1, [50], [{"number": 1, "renorm_ohm": 20.0}])
    assert abs(S1r[0, 0, 0]) == pytest.approx(1)


def test_script_export_compiles():
    p = Project.load(EX / "WaveguidePost.hzn")
    src = design_to_script(p, p.designs[0].name)
    compile(src, "exported.py", "exec")
    assert "bridge.prepare(m)" in src


def test_rectangle_sizes_named_by_axis():
    """CreateRectangle takes XSize/YSize/ZSize (the two in the sheet plane); old Width/Height still load."""
    from hertzon.model import schema
    for axis, (a, b) in schema.RECT_SIZES.items():
        old = Command("CreateRectangle", {"Position": "0,0,0", "Axis": axis, "Width": "2", "Height": "5"})
        assert "Width" not in old.props and old.props[a] == "2" and old.props[b] == "5"
        vis = [p.name for p in schema.GEOMETRY["CreateRectangle"] if schema.is_visible(p, old.props)]
        assert [n for n in vis if n.endswith("Size")] == sorted([a, b])
    pr = Project()
    d = pr.designs[0]
    d.add_object("CreateRectangle", name="R", Position="0,0,0", Axis="Y", Width="2", Height="5")
    assert d.obj("R").history[0].props["ZSize"] == "2" and d.obj("R").history[0].props["XSize"] == "5"
    tess = K.tessellate(K.build_design(d, d.evaluator(pr)))
    P = np.asarray(tess["R"].points)
    assert np.ptp(P, axis=0) == pytest.approx([5e-3, 0, 2e-3], abs=1e-9)


def test_antenna_report_expressions():
    """Antenna parameters: dB() of a gain or power is 10 log10, of the axial ratio 20 log10."""
    from hertzon.gui.reports import eval_expr
    data = {"freq": np.array([1.0, 2.0]), "xaxis": "Pass",
            "antenna": {"PeakGain": np.array([2.0, 4.0]), "AxialRatioAtMax": np.array([2.0, 1.0])}}
    assert eval_expr(data, "dB(PeakGain)") == pytest.approx(10 * np.log10([2.0, 4.0]))
    assert eval_expr(data, "dB(AxialRatioAtMax)") == pytest.approx(20 * np.log10([2.0, 1.0]))
    assert eval_expr(data, "PeakGain*2") == pytest.approx([4.0, 8.0])


def test_refinement_method_renamed():
    """Old setups keep their refinement method under the new names, and their solutions stay valid."""
    from hertzon.model.design import Setup
    from hertzon.model import schema
    from hertzon.solver.keys import setup_key
    assert schema.upgrade_values({"RefinementMethod": "Remesh"})["RefinementMethod"] == "Remesh (slow)"
    a, b = Setup("S"), Setup("S")
    a.props["RefinementMethod"] = "Local (Delaunay, experimental)"
    b.props["RefinementMethod"] = "Local (Delaunay)"
    assert setup_key(a) == setup_key(b)
    assert schema.defaults(schema.SETUP)["RefinementMethod"] == "Local (Delaunay)"


def test_far_field_follows_edit_sources(tmp_path):
    """Gain patterns and antenna parameters use the excitation of Edit Sources, not always port 1."""
    import json as _json
    import types
    from hertzon.solver.results import ResultStore
    from hertzon.solver.ffquant import excited_quantities, farfield_quantities
    st = ResultStore(tmp_path)
    ports = [{"name": "P1", "number": 1, "kind": "Lumped Port"}, {"name": "P2", "number": 2, "kind": "Lumped Port"}]
    st.write_json("ports.json", ports)
    S = np.array([[[0.3 + 0.1j, 0.05], [0.05, 0.2j]]])
    st.save_sparams("LastAdaptive", np.array([1e9]), S, np.array([50.0, 50.0]), ports)
    th, ph = np.linspace(0, np.pi, 19), np.linspace(0, 2 * np.pi, 37)
    T, P = np.meshgrid(th, ph, indexing="ij")
    E1 = np.array([np.cos(T), 0 * T, -np.sin(T)]).astype(complex)          # broadside to +x/-x
    E2 = np.array([0 * T, np.cos(T) + 0.5, 0 * T]).astype(complex)
    ff = {"theta": th, "phi": ph, "E_0_1": E1, "E_0_2": E2, "freq_0": np.array([1e9]),
          "Prad_0_1": np.array([0.5]), "Prad_0_2": np.array([0.4])}
    des = types.SimpleNamespace(settings={}, excitations=[],
                                ports=lambda: [types.SimpleNamespace(name="P1"), types.SimpleNamespace(name="P2")])
    q0 = excited_quantities(st, des, ff, "LastAdaptive", 0)
    assert q0["GainTotal"] == pytest.approx(farfield_quantities(ff, 1, 0, pacc=1 - 0.1 - 0.05 ** 2)["GainTotal"])
    des.settings["Sources"] = _json.dumps({"P1": {"mag": 0, "phase": 0}, "P2": {"mag": 1, "phase": 0}})
    q2 = excited_quantities(st, des, ff, "LastAdaptive", 0)
    assert q2["GainTotal"] == pytest.approx(farfield_quantities(ff, 2, 0, pacc=1 - 0.04 - 0.05 ** 2)["GainTotal"])
    des.settings["Sources"] = _json.dumps({"P1": {"mag": 2, "phase": 0}, "P2": {"mag": 2, "phase": 90}})
    q12 = excited_quantities(st, des, ff, "LastAdaptive", 0)
    a = np.sqrt(2) * np.array([1, 1j])
    assert q12["_Pinc"][0] == pytest.approx(4.0)
    assert q12["_Pacc"][0] == pytest.approx(4.0 - np.linalg.norm(S[0] @ a) ** 2)
    assert q12["rETotal"] == pytest.approx(np.sqrt(np.abs(farfield_quantities(
        {**ff, "E_0_9": a[0] * E1 + a[1] * E2, "Prad_0_9": np.array([1.0])}, 9, 0)["rETotal"]) ** 2))


def test_parallel_error_kernel_source():
    """The parallel error estimate is generated from EMerge's kernel: the tet loop must become a prange
    without exits (else numba runs it serially), everything else unchanged."""
    from hertzon.solver import fast_error
    from hertzon.solver.emerge_compat import EMERGE3
    if EMERGE3:
        pytest.skip("EMerge 3 has its own compiled error estimate")
    src = fast_error._source()
    assert src is not None
    assert "for itet in prange(N_TETS):" in src and "parallel=True" in src
    assert "adj_tets = [" not in src and "raise ValueError" not in src
    compile(src, "error_estimate.py", "exec")


def test_edit_sources_apply_keeps_dialog_open():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QDialogButtonBox
    from hertzon.gui.dialogs import EditSourcesDialog
    from hertzon.model import sources as SRC
    app = QApplication.instance() or QApplication([])  # noqa: F841
    proj = Project.load(EX / "PatchAntenna.hzn")
    des = proj.designs[0]
    got = []
    dlg = EditSourcesDialog(None, des, on_apply=lambda s, f: got.append((s, f)))
    name = dlg.names[0]
    dlg.mag[name].setValue(2.5)
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Apply).click()
    assert got and got[0][0][name]["mag"] == 2.5 and not dlg.result()
    assert SRC.defaults(des)


def test_fast_assembly_patches_and_sparse_helpers():
    """The faster assembly replaces EMerge's code by source patterns; they must still match, and the
    sparse helpers must give scipy's results."""
    import scipy.sparse as sp
    from hertzon.solver import fast_assembly as FA
    from hertzon.solver.emerge_compat import EMERGE3
    from emerge._emerge.physics.microwave.assembly import assembler
    from emerge._emerge import solver
    if not EMERGE3:             # EMerge 3 assembles in its compiled library: only the solve is patched
        n = sum(FA._patched(assembler.Assembler.assemble_freq_matrix, FA._ASSEMBLER_SUBS)[1:])
        assert n == 5
    assert FA._patched(solver.SolveRoutine.solve, [(FA.re.compile(r"Asel = A\[:, solve_ids\]\[solve_ids, :\]"),
                                                    "x")])[1] == 1
    rng = np.random.default_rng(0)
    M = sp.random(300, 300, density=0.05, random_state=1) + 1j * sp.random(300, 300, density=0.05, random_state=2)
    A = (M + M.T + sp.identity(300)).tocsc()
    A.sort_indices()
    ids = np.sort(rng.choice(300, 250, replace=False))
    S0, S1 = A[:, ids][ids, :], FA.submatrix(A, ids)
    S0.sort_indices()
    assert np.array_equal(S0.indptr, S1.indptr) and np.array_equal(S0.indices, S1.indices)
    assert np.array_equal(S0.data, S1.data)
    U = FA.upper_symmetric(A)
    assert U is not None and abs(U - sp.triu(A)).max() == 0
    assert FA.upper_symmetric((A + sp.random(300, 300, density=0.01, random_state=3)).tocsc()) is None
    assert np.array_equal(FA.free_ids(10, {3, 7, 0}), [1, 2, 4, 5, 6, 8, 9])


def test_curved_kernels_match_emerge_element_matrices():
    """The curved elements use EMerge 2.8.9's numba basis functions (ned2_kernels); with zero edge
    displacement they must give the element matrices of the installed EMerge (also EMerge 3's compiled
    assembly: same DoF codes and local edge/face order)."""
    from hertzon.solver import curvilinear as CV
    from hertzon.solver import ned2_kernels as NK
    from emerge._emerge.elements.dofsets import ElementSpace
    try:
        from emr import curlcurl as CC         # EMerge 3
    except ImportError:
        from emerge._emerge.physics.microwave.assembly import curlcurl as CC
    rng = np.random.default_rng(1)
    nodes = np.array([[0, 1, 0, 0, 1], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1]], float) * 1e-3 + rng.normal(0, 1e-4, (3, 5))
    tets = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], np.int64)           # two tets sharing face 1-2-3
    st = np.sort(tets, axis=0)
    edges = np.unique(np.concatenate([st[[a, b]] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))],
                                     axis=1), axis=1)
    tris = np.unique(np.concatenate([st[list(t)] for t in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))], axis=1),
                     axis=1)
    ekey = {tuple(e): i for i, e in enumerate(edges.T.tolist())}
    tkey = {tuple(t): i for i, t in enumerate(tris.T.tolist())}
    t2e = np.array([[ekey[(st[a, k], st[b, k])] for k in range(2)]
                    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))], np.int64)
    t2t = np.array([[tkey[tuple(st[list(t), k])] for k in range(2)]
                    for t in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))], np.int64)
    er = np.repeat(np.array([[2.2, 0.1, 0], [0.1, 2.0, 0], [0, 0, 1.7]], complex)[:, :, None], 2, axis=2)
    ur = np.repeat((np.eye(3, dtype=complex) * (1.3 - 0.1j))[:, :, None], 2, axis=2)
    _, build = CV.kernels()
    for sp in ElementSpace:
        codes = np.array(sp.get_set().set3d.codes, dtype=np.int64)
        nd = len(codes)
        t2f = np.arange(2 * nd, dtype=np.int64).reshape(nd, 2)
        E0, B0 = CC._matrix_builder(nodes, tets, tris, edges, t2f, t2e, t2t, ur, er, np.arange(2, dtype=np.int64),
                                    codes)[:2]
        E1, B1 = np.zeros(2 * nd * nd, complex), np.zeros(2 * nd * nd, complex)
        build(nodes, tets, tris, edges, t2e, t2t, ur, er, np.arange(2, dtype=np.int64), np.arange(2, dtype=np.int64),
              np.zeros((2, 6, 3)), codes, CV._QP, E1, B1)
        assert np.abs(E1 - np.asarray(E0)).max() < 1e-9 * np.abs(E0).max(), sp
        assert np.abs(B1 - np.asarray(B0)).max() < 1e-9 * np.abs(B0).max(), sp
    assert NK.matinv(ur[:, :, 0])[0, 0] == pytest.approx(1 / (1.3 - 0.1j))


def test_emerge_material_library():
    """EMerge's material library (emsutil) appears next to Hertzon's own one; frequency dependent
    materials keep EMerge's data for the solve."""
    from hertzon.model.materials import EMERGE_DISPERSION, EMERGE_LIBRARY, emerge_library, emerge_object
    lib = emerge_library()
    if not lib:
        pytest.skip("emsutil not installed")
    assert len(lib) > 100
    ro = lib["Rogers RO4350B"]
    assert ro.library == EMERGE_LIBRARY and float(ro.permittivity) == pytest.approx(3.48)
    assert float(ro.loss_tangent) == pytest.approx(0.0037) and ro.emsutil_key == "DIEL_RO4350B"
    assert not {"pec", "copper", "air"} & {n.lower() for n in lib}       # Hertzon's own ones win
    p = Project()
    assert p.material("Rogers RO4350B") is ro and ro in p.all_materials()
    disp = [m for m in lib.values() if m.dispersion == EMERGE_DISPERSION]
    for m in disp:
        assert emerge_object(m.emsutil_key) is not None


def test_thin_conductor_keeps_the_assembled_dof_numbering():
    """EMerge assembles with get_tet_to_field() (the numbering before the thin conductor split); plots, far
    fields and the error estimate read tet_to_field, which the patch keeps equal to it."""
    import inspect
    from hertzon.solver import emerge_patches as EP
    from emerge._emerge.elements.nedelec2 import Nedelec2
    assert "_tet_to_field_orig" in inspect.getsource(Nedelec2.get_tet_to_field)
    EP.apply_thin_conductor()
    b = object.__new__(Nedelec2)
    b._partitioned = True                     # EMerge's own split returns at once
    b._tet_to_field_orig = np.arange(12).reshape(3, 4)
    b.tet_to_field = b._tet_to_field_orig + 100
    Nedelec2.partition_dof(b, [[1]])
    assert np.array_equal(b.tet_to_field, b._tet_to_field_orig) and b.get_tet_to_field() is b._tet_to_field_orig


def test_finite_conductivity_thickness_without_two_sided():
    """Older files: a thickness meant the two-sided thin conductor; loading keeps that and the result key."""
    from hertzon.model.design import project_from_dict
    from hertzon.solver.keys import model_key
    p = Project.load(EX / "Microstrip.hzn")
    d = p.to_dict()
    for b in d["designs"][0]["boundaries"]:
        b["props"].pop("TwoSided", None)
    old = project_from_dict(json.loads(json.dumps(d)))
    des = old.designs[0]
    cu = next(b for b in des.boundaries if b.kind == "Finite Conductivity")
    assert cu.props["UseThickness"] == "True" and cu.props["TwoSided"] == "True"
    key = model_key(old, des, des.setups[0])
    cu.props.pop("TwoSided")
    assert model_key(old, des, des.setups[0]) == key
    # one-sided with thickness is a different problem
    cu.props.update(TwoSided="False")
    assert model_key(old, des, des.setups[0]) != key


def test_object_colours_in_other_notations_are_read():
    from hertzon.model.materials import parse_color
    from hertzon.model.design import project_from_dict
    assert parse_color("0,128,96") == parse_color("(0 128 96)") == parse_color("#008060") == "#008060"
    assert parse_color("#abc") == "#aabbcc" and parse_color("nonsense") is None
    p = Project.load(EX / next(f.name for f in EX.glob("*.hzn")))
    d = p.to_dict()
    d["designs"][0]["objects"][0]["color"] = "0,128,96"
    d["materials"] = [{"name": "M", "color": "(10 20 30)"}]
    q = project_from_dict(d)
    assert q.designs[0].objects[0].color == "#008060"
    assert q.materials[0].color == "#0a141e"


@pytest.mark.parametrize("boxes", [0, 10])
def test_circles_in_flat_faces_are_drawn_round(boxes):
    # thin ring (rat race): without curvature sizing its circles had a dozen segments and a zigzag outline
    p = Project()
    d = p.designs[0]
    d.add_object("CreateCircle", name="Ring", CenterPosition="0,0,0", Axis="Z", Radius="18")
    d.add_object("CreateCircle", name="Hole", CenterPosition="0,0,0", Axis="Z", Radius="17")
    d.obj("Hole").consumed_by = "Ring"
    d.obj("Ring").history.append(Command.new("Subtract", Tools="Hole", KeepOriginals="False"))
    for i in range(boxes):      # many flat faces and a cylinder: the two pass display mesh
        d.add_object("CreateBox", Position=f"{30 + 3 * i},0,0", XSize="1", YSize="1", ZSize="1")
    if boxes:
        d.add_object("CreateCylinder", CenterPosition="-40,0,0", Axis="Z", Radius="3", Height="2")
    K.ensure_gmsh("test_ring")
    ctx = K.build_design(d, d.evaluator(p))
    tess = K.tessellate(ctx)
    circles = [e for e in tess["Ring"].edges if len(e) > 2]
    assert len(circles) == 2 and all(len(e) >= 60 for e in circles)
    if boxes:
        assert len(tess["Cylinder1"].tris) < 400


def test_results_dir_stays_inside_results_folder():
    from hertzon.solver.results import results_dir
    proj = Path("/work/demo/filter.hzn")
    base = proj.with_suffix(".hznresults").resolve()
    for design, setup in [("..", ".."), (".", ".."), ("", ""), ("../..", "x"), ("a/../..", "...")]:
        d = results_dir(proj, design, setup).resolve()
        assert d.parent.parent == base, (design, setup, d)


def test_worker_routines_are_independent():
    """EMerge 3's AutomaticRoutine is a singleton: the parallel sweep threads must still get routines and
    PARDISO handles of their own (shared ones crashed in MKL now and then)."""
    from hertzon.solver import emerge_patches
    emerge_patches.apply()
    from emerge._emerge import solver as S
    r = S.AutomaticRoutine()
    a, b = r.duplicate(), r.duplicate()
    assert a is not r and b is not a
    for key, s in r.solvers.items():
        assert a.solvers[key] is not s and a.solvers[key] is not b.solvers[key]


def test_subtract_dialog_moves_single_objects():
    """Subtract: the arrow buttons move the selected objects between Blank Parts and Tool Parts."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QAbstractItemView, QDialogButtonBox
    from hertzon.gui.dialogs import SubtractDialog
    app = QApplication.instance() or QApplication([])  # noqa: F841
    dlg = SubtractDialog(["Cylinder1"], ["Cylinder2", "Cylinder3"])
    ok = dlg.buttons.button(QDialogButtonBox.Ok)
    assert dlg.blank_list.selectionMode() == QAbstractItemView.ExtendedSelection
    assert dlg.tool_list.selectionMode() == QAbstractItemView.ExtendedSelection
    assert not dlg.to_tool.isEnabled() and not dlg.to_blank.isEnabled() and ok.isEnabled()
    dlg.tool_list.item(1).setSelected(True)                     # Cylinder3 <- back to the blank parts
    dlg.to_blank.click()
    assert dlg.blanks() == ["Cylinder1", "Cylinder3"] and dlg.tools() == ["Cylinder2"]
    dlg.blank_list.clearSelection()
    for i in range(2):                                          # both blanks -> tools: no blank part left
        dlg.blank_list.item(i).setSelected(True)
    dlg.to_tool.click()
    assert dlg.blanks() == [] and dlg.tools() == ["Cylinder2", "Cylinder1", "Cylinder3"] and not ok.isEnabled()
    dlg.swap()
    assert dlg.blanks() == ["Cylinder2", "Cylinder1", "Cylinder3"] and dlg.tools() == [] and not ok.isEnabled()


def test_subtract_with_several_blank_parts():
    """Every blank part is cut by all tool parts; the tools are cloned for all but the last blank."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import types
    from PySide6.QtWidgets import QApplication, QDialog
    from hertzon.gui import main_window as MW
    app = QApplication.instance() or QApplication([])  # noqa: F841
    proj = Project()
    des = proj.designs[0]
    for name, x in (("A", 0), ("B", 20), ("T", 5)):
        des.add_object("CreateBox", name, Position=f"{x},0,0", XSize="10", YSize="10", ZSize="10")

    class Dlg:
        def __init__(self, blanks, tools, parent=None):
            self.clone = types.SimpleNamespace(isChecked=lambda: False)

        def exec(self):
            return QDialog.Accepted

        def blanks(self):
            return ["A", "B"]

        def tools(self):
            return ["T"]
    win = types.SimpleNamespace(des=des, doc=types.SimpleNamespace(checkpoint=lambda: None, commit=lambda w: None,
                                                                   select=lambda p: None),
                                selected_objects=lambda n: ["A", "T"], _drop_refs=lambda objs: None)
    old = MW.SubtractDialog
    MW.SubtractDialog = Dlg
    try:
        MW.MainWindow.boolean(win, "Subtract")
    finally:
        MW.SubtractDialog = old
    cuts = {o.name: [c for c in o.history if c.kind == "Subtract"] for o in des.objects}
    assert cuts["A"][0].props["Tools"] == "T_tool" and des.obj("T_tool").consumed_by == "A"
    assert cuts["B"][0].props["Tools"] == "T" and des.obj("T").consumed_by == "B"


def test_projection_keeps_the_model_size():
    """View > Orthographic: switching projection keeps the visible height (the parallel scale matches)."""
    import math
    import types
    from hertzon.gui.viewer import ModelerView
    from vtkmodules.vtkRenderingCore import vtkCamera
    cam = vtkCamera()
    cam.SetPosition(0, 0, 100)
    cam.SetFocalPoint(0, 0, 0)
    cam.SetViewAngle(30)
    got = []
    v = types.SimpleNamespace(plotter=types.SimpleNamespace(camera=cam, render=lambda: None), on_projection=got.append)
    ModelerView.set_orthographic(v, True)
    assert cam.GetParallelProjection() and got == [True]
    assert cam.GetParallelScale() == pytest.approx(100 * math.tan(math.radians(15)))
    ModelerView.set_orthographic(v, False)
    assert not cam.GetParallelProjection() and got == [True, False]


def test_radiation_pattern_of_a_sweep_frequency_and_rectangular(tmp_path):
    """Radiation Pattern of a sweep with fields: the frequency is chosen in the report, and the cut can be
    drawn over the angle (rectangular) with markers."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from hertzon.gui.doc import Document, result_dir
    from hertzon.gui.reports import ReportWindow, ff_solution_list
    from hertzon.model.design import Report
    from hertzon.solver.results import ResultStore
    app = QApplication.instance() or QApplication([])  # noqa: F841
    proj = Project.load(EX / "PatchAntenna.hzn")
    proj.save(tmp_path / "p.hzn")
    doc = Document(proj)
    st = ResultStore(result_dir(doc, "Setup1"))
    ports = [{"name": "1", "number": 1, "kind": "Lumped Port"}]
    st.write_json("ports.json", ports)
    st.save_sparams("Sweep", np.array([1e9, 2e9]), np.zeros((2, 1, 1), complex), np.array([50.0]), ports)
    rng = np.random.default_rng(1)
    n = 40
    vis = rng.normal(size=(3, n)) * 0.01
    wns = rng.normal(size=(3, n)) * 1e-4
    E = (rng.normal(size=(3, n)) + 1j * rng.normal(size=(3, n)))
    H = (rng.normal(size=(3, n)) + 1j * rng.normal(size=(3, n))) / 377
    st.save_npz("huygens_Sweep.npz", vis=vis, wns=wns, stamp=np.array([1]),
                freq_0=np.array([1e9]), E_0_1=E, H_0_1=H, Prad_0_1=np.array([0.5]),
                freq_1=np.array([2e9]), E_1_1=2 * E, H_1_1=2 * H, Prad_1_1=np.array([2.0]))
    assert ff_solution_list(doc) == ["Setup1 : Sweep"]
    rep = Report("Radiation Pattern 1", "Radiation Pattern", "Setup1 : Sweep", ["dB(GainTotal)"])
    w = ReportWindow(doc, rep)
    assert "color:#c0392b" not in w.info.text(), w.info.text()
    assert [w.ff_freq.itemText(i) for i in range(w.ff_freq.count())] == ["1 GHz", "2 GHz"]
    assert "1 GHz" in w.info.text()
    w.ff_freq.setCurrentIndex(1)
    assert rep.props["freq"] == "2000000000" and "2 GHz" in w.info.text()
    assert w._ff()["_Prad"][0] == pytest.approx(2.0)
    # polar plot: markers are angles in deg on the polar axes, the beamwidth rows go to the marker table
    assert w._polar() and w.fig.axes[0].name == "polar"
    x, y = w._trace_xy("dB(GainTotal)")
    w.marker_at("max")
    assert w.markers == [pytest.approx(x[np.argmax(y)])]
    assert w._from_ax(np.radians(350)) == pytest.approx(-10)
    # the Phi cut is labelled over theta -180..180 (as defined on the sphere), the Theta cut over phi 0..360
    fmt = w.fig.axes[0].xaxis.get_major_formatter()
    assert [fmt(np.radians(a)) for a in (0, 90, 180, 270)] == ["0\u00b0", "90\u00b0", "\u00b1180\u00b0", "-90\u00b0"]
    rep.props["cut"] = "Theta"
    w.refresh()
    assert w.fig.axes[0].xaxis.get_major_formatter()(np.radians(270)) == "270\u00b0"
    rep.props["cut"] = "Phi"
    w.refresh()
    rep.props.update(bandwidth="dB(GainTotal)", bw_level="-3")
    w.refresh()
    rows = [w.mtable.item(r, 0).text() for r in range(w.mtable.rowCount())]
    assert rows[0] == "m1" and "BW @ -3 dB" in rows
    w._clear_markers()
    w.ff_plot.setCurrentText("Rectangular")
    assert rep.props["plot"] == "Rectangular" and w._xy_plot()
    ax = w.fig.axes[0]
    assert ax.name == "rectilinear" and ax.get_xlabel() == "Theta [deg]"
    x, y = w._trace_xy("dB(GainTotal)")
    assert x.min() == pytest.approx(-180) and x.max() == pytest.approx(180)
    w.marker_at("max")
    assert len(w.markers) == 1 and w.markers[0] == pytest.approx(x[np.argmax(y)])
    assert w.mtable.isVisibleTo(w) and w.mtable.rowCount() == 1


def test_beamwidth_and_side_lobes():
    from hertzon.gui.reports import beamwidth
    x = np.linspace(-180, 180, 361)                       # Phi cut over the full circle
    y = 10 * np.log10(np.maximum(np.cos(np.radians(x) / 2) ** 4, 1e-9))
    b = beamwidth(x, y, -3)
    half = 2 * np.degrees(np.arccos(0.5 ** 0.25))         # cos^4(t/2) = 1/2
    assert b["width"] == pytest.approx(2 * half, abs=0.5) and b["lo"] == pytest.approx(-half, abs=0.3)
    assert not b["open"] and "sll" not in b
    x = np.linspace(0, 360, 361)                           # Theta cut 0..360 with the beam across 0/360
    b = beamwidth(x, 10 * np.log10(np.maximum(np.cos(np.radians(x) / 2) ** 4, 1e-9)), -3)
    assert b["width"] == pytest.approx(2 * half, abs=0.5) and b["max_at"] == pytest.approx(0)
    assert b["lo"] == pytest.approx(360 - half, abs=0.3) and b["hi"] == pytest.approx(half, abs=0.3)
    x = np.linspace(-90, 90, 181)                          # uniform array: first side lobe -13.26 dB
    u = 4 * np.sin(np.radians(x))
    b = beamwidth(x, 20 * np.log10(np.abs(np.sinc(u)) + 1e-12), -3)
    assert b["sll"] == pytest.approx(-13.26, abs=0.1) and abs(b["sll_at"]) == pytest.approx(21, abs=1)


def test_keyboard_shortcuts_dialog_opens_and_saves(monkeypatch):
    """Tools > Keyboard Shortcuts: the dialog opens, a new key is set on the command and kept in the settings,
    and it is applied again at the next start."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QKeySequence
    from PySide6.QtWidgets import QApplication, QDialog, QKeySequenceEdit, QMainWindow
    from hertzon.gui import main_window as MW
    app = QApplication.instance() or QApplication([])  # noqa: F841
    win = QMainWindow()
    win.settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "hertzon-test", "shortcuts")
    win.settings.clear()
    m = win.menuBar().addMenu("&Tools")
    act = m.addAction("Edit Materials...")

    def accept(dlg):
        for e in dlg.findChildren(QKeySequenceEdit):
            e.setKeySequence(QKeySequence("Ctrl+Shift+M"))
        return QDialog.Accepted
    monkeypatch.setattr(QDialog, "exec", accept)
    MW.MainWindow.edit_shortcuts(win)
    assert act.shortcut().toString() == "Ctrl+Shift+M"
    act.setShortcut(QKeySequence())
    MW.MainWindow._apply_shortcuts(win)
    assert act.shortcut().toString() == "Ctrl+Shift+M"
    win.settings.clear()


def test_center_point_arc_polyline():
    """AngularArc segments: start, center and angle; the angle decides the end point (the stored one is a copy)."""
    import gmsh
    p = Project()
    d = p.designs[0]
    d.add_object("CreatePolyline", Points="1,0,0;0,0,0;0,1,0;-2,1,0", SegmentTypes="AngularArc:90:XY,Line")
    d.add_object("CreatePolyline", Points="1,0,0;0,0,0;1,0,0", SegmentTypes="AngularArc:360deg",
                 Closed="True", Covered="True")
    # half circle in the ZX plane with a wrong stored end point, closed by a line
    d.add_object("CreatePolyline", Points="0,0,2;0,0,0;5,5,5", SegmentTypes="AngularArc:-180:ZX",
                 Closed="True", Covered="True")
    K.ensure_gmsh("test_angular_arc")
    ctx = K.build_design(d, d.evaluator(p))
    assert not ctx.errors
    size = lambda name: sum(gmsh.model.occ.getMass(*x) for x in ctx.built[name])  # noqa: E731
    assert size("Polyline1") == pytest.approx((np.pi / 2 + 2) * 1e-3)
    assert size("Polyline2") == pytest.approx(np.pi * 1e-6)
    assert size("Polyline3") == pytest.approx(2 * np.pi * 1e-6)
    lo = gmsh.model.occ.getBoundingBox(*ctx.built["Polyline3"][0])
    assert lo[0] == pytest.approx(-2e-3, abs=1e-6) and lo[3] == pytest.approx(0, abs=1e-6)   # -180 about +Y: -X


def test_center_point_arc_drawing():
    """Draw > Arc > Center Point: start, center, then the end angle follows the cursor around the center."""
    import types
    from hertzon.gui import viewer as Vw
    names = ("add_draw_point", "finish_polyline", "_track_sweep", "_arc_plane", "_phase_hint")
    View = type("View", (), {k: Vw.ModelerView.__dict__[k] for k in names})
    got = []
    v = View()
    v.grid_plane, v.axis_lock = "XY", None
    v.drawFinished = types.SimpleNamespace(emit=lambda kind, pts: got.append((kind, pts)))
    v.drawPhaseChanged = types.SimpleNamespace(emit=lambda s: None)
    v._update_preview = lambda: None
    v.cancel_draw = lambda: setattr(v, "draw", None)
    v.draw = Vw.DrawState("CreatePolyline", 3, seg_mode="CenterArc")
    v.add_draw_point(np.array([1.0, 0, 0]))
    v.add_draw_point(np.array([0.0, 0, 0]))
    assert "end angle" in v._phase_hint()
    for a in np.radians([60, 120, 179, 200, 270]):     # past 180 degrees counter-clockwise
        v._track_sweep(np.array([np.cos(a), np.sin(a), 0]))
    v.add_draw_point(np.array([0.0, -3.0, 0]))           # 270 deg; only the direction counts
    kind, pts = got[0]
    assert kind == "CreatePolyline" and v.last_segment_types == "AngularArc:270:XY"
    assert np.allclose(pts[-1], [0, -1, 0])
    # the spec and the preview of a clockwise arc
    modes = ["CenterArc", "CenterArc:-90", "Line"]
    assert Vw.segment_spec(modes, "XY") == "AngularArc:-90:XY,Line"
    P = Vw.densify([np.array([1.0, 0, 0]), np.zeros(3), np.array([0, -1.0, 0]), np.array([0, -2.0, 0])], modes)
    assert np.allclose(np.linalg.norm(P[:-1], axis=1), 1) and np.allclose(P[-1], [0, -2, 0])


def test_first_start_note(monkeypatch):
    """Solver kernels not compiled yet: the main window says so in the Message Manager and the status bar."""
    import types
    from hertzon.gui.main_window import MainWindow
    from hertzon.solver import bridge
    msgs, status = [], []
    win = types.SimpleNamespace(msgs=types.SimpleNamespace(add=lambda kind, text: msgs.append((kind, text))),
                                statusBar=lambda: types.SimpleNamespace(showMessage=lambda t, ms=0: status.append(t)))
    monkeypatch.setattr(bridge, "_first_run", lambda: False)
    MainWindow._first_start_note(win)
    assert not msgs and not status
    monkeypatch.setattr(bridge, "_first_run", lambda: True)
    MainWindow._first_start_note(win)
    assert msgs[0][0] == "warning" and "First start detected" in msgs[0][1]
    assert "compiles the solver kernels" in status[0]


def test_solid_overlaps_and_resolve():
    """Exact overlap check as the reference tool: with material override a metal partially in a dielectric is
    fine, two different dielectrics (or metals) partially overlapping are an error, contained objects are only
    listed when a conductor swallows them; without material override every partial overlap of different
    materials is an error and the inner object always wins. Resolve subtracts the winner from the loser."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import types
    from PySide6.QtWidgets import QApplication
    from hertzon.gui import main_window as MW
    from hertzon.model.validate import classify_overlaps
    app = QApplication.instance() or QApplication([])  # noqa: F841
    p = Project()
    d = p.designs[0]
    d.add_object("CreateBox", "Sub", Position="0,0,0", XSize="20", YSize="20", ZSize="2").material = "FR4_epoxy"
    d.add_object("CreateBox", "Glue", Position="15,0,0", XSize="10", YSize="10", ZSize="2").material = "Teflon (tm)"
    d.add_object("CreateCylinder", "Via", CenterPosition="5,5,-1", Axis="Z", Radius="0.5", Height="4").material = "copper"
    d.add_object("CreateBox", "Shield", Position="40,0,0", XSize="10", YSize="10", ZSize="10").material = "pec"
    d.add_object("CreateBox", "Inner", Position="42,2,2", XSize="2", YSize="2", ZSize="2").material = "FR4_epoxy"
    d.add_object("CreateBox", "Air", Position="-5,-5,-5", XSize="70", YSize="30", ZSize="30").material = "vacuum"
    ev = d.evaluator(p)
    names = ["Sub", "Glue", "Via", "Shield", "Inner", "Air"]
    ovs = classify_overlaps(p, d, K.solid_overlaps(d, ev, names))
    got = {(o.winner, o.loser): o for o in ovs}
    assert set(got) == {("Glue", "Sub"), ("Via", "Sub"), ("Shield", "Inner")}, [o.message for o in ovs]
    assert got["Glue", "Sub"].level == "error" and got["Via", "Sub"].level == "ok"
    assert "two different dielectrics" in got["Glue", "Sub"].message
    assert got["Shield", "Inner"].contained and "no effect" in got["Shield", "Inner"].message
    checks = validate(p, d, None, {}, ovs)
    assert sum(c.message in {o.message for o in ovs} for c in checks) == 3
    assert [c.message for c in checks if c.level == "error" and c.category == "3D Model"] == [got["Glue", "Sub"].message]
    d.settings["MaterialOverride"] = "False"
    strict = {(o.winner, o.loser): o.level for o in classify_overlaps(p, d, K.solid_overlaps(d, ev, names))}
    assert strict == {("Glue", "Sub"): "error", ("Via", "Sub"): "error"}
    del d.settings["MaterialOverride"]
    win = types.SimpleNamespace(des=d, msgs=types.SimpleNamespace(add=lambda *a: None), validation_check=lambda: None,
                                doc=types.SimpleNamespace(checkpoint=lambda: None, commit=lambda w: None,
                                                          select=lambda picks: None))
    MW.MainWindow.resolve_overlaps(win, [o for o in ovs if not o.contained])
    assert d.obj("Sub").history[-1].kind == "Subtract" and d.obj("Sub").history[-2].kind == "Subtract"
    assert {d.obj(n).consumed_by for n in ("Glue_tool", "Via_tool")} == {"Sub"}
    names = [o.name for o in d.live_objects()]
    left = classify_overlaps(p, d, K.solid_overlaps(d, ev, names))
    assert [(o.winner, o.loser) for o in left] == [("Shield", "Inner")]
    from PySide6.QtWidgets import QPushButton
    from hertzon.gui.dialogs import ValidationDialog
    done = []
    dlg = ValidationDialog(types.SimpleNamespace(project=p, design=d, tess=None, errors={}), None, ovs, done.append)
    every = next(b for b in dlg.findChildren(QPushButton) if b.text().startswith("Resolve All"))
    assert every.text() == "Resolve All Overlaps (2)"
    every.click()
    assert len(done) == 1 and {(o.winner, o.loser) for o in done[0]} == {("Glue", "Sub"), ("Via", "Sub")}


def test_solver_rejects_partial_overlaps():
    """The solver refuses partial overlaps of different materials the same way (script and MCP runs)."""
    import types
    import pytest
    from hertzon.solver.bridge import Bridge
    obj = lambda n, m: types.SimpleNamespace(name=n, material=m)  # noqa: E731
    geo = lambda *t: types.SimpleNamespace(tags=list(t))  # noqa: E731
    vols = [(obj("Sub", "FR4_epoxy"), geo(1, 2, 3), False), (obj("Glue", "Teflon (tm)"), geo(3, 4), False),
            (obj("Via", "copper"), geo(2, 5), True), (obj("Core", "FR4_epoxy"), geo(1), False)]
    fake = types.SimpleNamespace(design=types.SimpleNamespace(settings={}), overlap_vols=vols)
    with pytest.raises(RuntimeError) as e:
        Bridge.check_overlaps(fake)
    assert "'Sub' (FR4_epoxy) and 'Glue' (Teflon (tm))" in str(e.value) and "Via" not in str(e.value)
    Bridge.check_overlaps(types.SimpleNamespace(design=fake.design, overlap_vols=vols[2:]))
    fake.design.settings["MaterialOverride"] = "False"
    with pytest.raises(RuntimeError) as e:
        Bridge.check_overlaps(types.SimpleNamespace(design=fake.design, overlap_vols=[vols[0], vols[2]]))
    assert "Material override" in str(e.value)


def test_maximum_delta_energy_default():
    """Incident wave convergence: Maximum Delta Energy is 0.001 by default."""
    from hertzon.model import schema
    p = next(p for p in schema.SETUP if p.name == "MaximumDeltaEnergy")
    assert p.default == "0.001"


def _split_box(cs=None, **split):
    """Box 0..4 x 0..2 x 50..52 (mm), split; returns the bounding box of the kept part in mm."""
    p = Project()
    d = p.designs[0]
    d.add_object("CreateBox", Position="0,0,50", XSize="4", YSize="2", ZSize="2")
    if cs is not None:
        d.coord_systems.append(cs)
        split["CoordinateSystem"] = cs.name
    d.obj("Box1").history.append(Command.new("Split", **split))
    K.ensure_gmsh("test_split")
    ctx = K.build_design(d, d.evaluator(p))
    assert "Box1" not in ctx.errors, ctx.errors.get("Box1")
    return np.asarray(K._bbox(ctx.built["Box1"])) * 1e3


def test_split_in_object_and_face_coordinate_systems():
    from hertzon.model.design import CoordSys
    # global plane far from the object (the kept half space must still reach it)
    assert _split_box(Plane="XY", Keep="Positive") == pytest.approx([0, 0, 50, 4, 2, 52], abs=1e-3)
    # object CS at the box centre: XY cuts it in half along z
    ocs = CoordSys("ObjectCS1", "Object", "Global")
    ocs.props.update(Origin="2,1,51", Object="Box1")
    assert _split_box(ocs, Plane="XY", Keep="Positive") == pytest.approx([0, 0, 51, 4, 2, 52], abs=1e-3)
    assert _split_box(ocs, Plane="YZ", Keep="Negative") == pytest.approx([0, 0, 50, 2, 2, 52], abs=1e-3)
    # face CS on the side face x=4 (local z = global +x), moved in by 1: XY is the plane x=3
    fcs = CoordSys("FaceCS1", "Face", "Global")
    fcs.props.update(Mode="Axis/Position", Origin="3,1,51", XAxis="0,1,0", YPoint="0,0,1", Face="Box1:1")
    assert _split_box(fcs, Plane="XY", Keep="Positive") == pytest.approx([3, 0, 50, 4, 2, 52], abs=1e-3)
    assert _split_box(fcs, Plane="XY", Keep="Negative") == pytest.approx([0, 0, 50, 3, 2, 52], abs=1e-3)
    # custom plane: Base and Normal are local to the CS (local x = global y)
    assert _split_box(fcs, Plane="Custom", Base="0.5,0,0", Normal="1,0,0", Keep="Positive") == \
        pytest.approx([0, 1.5, 50, 4, 2, 52], abs=1e-3)
