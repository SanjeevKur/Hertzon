"""Far field surface of a scattering problem: a box of Gauss points close around the scatterer (independent of
the mesh, which the adaptive passes refine locally), and the monostatic RCS of a PEC sphere against Mie."""
import importlib.util
import json
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_box_quadrature_closed_and_exact():
    from hertzon.solver.bridge import box_quadrature
    b = np.array([-0.1, -0.2, -0.3, 0.2, 0.25, 0.35])
    vis, wns = box_quadrature(b, 0.07)
    side = b[3:] - b[:3]
    area = 2 * (side[0] * side[1] + side[1] * side[2] + side[0] * side[2])
    assert np.linalg.norm(wns.sum(axis=1)) < 1e-12 * area
    assert abs(np.linalg.norm(wns, axis=0).sum() - area) < 1e-12 * area
    # divergence theorem: the flux of (x^3, 0, 0) is the volume integral of 3 x^2
    flux = np.sum(vis[0] ** 3 * wns[0])
    vol = (b[3] ** 3 - b[0] ** 3) * side[1] * side[2]
    assert abs(flux - vol) < 1e-12
    # a face left out (mirror plane) is missing, the others stay
    _, w2 = box_quadrature(b, 0.07, {(2, 0)})
    assert abs(w2[2].sum() - side[0] * side[1]) < 1e-12


def test_sphere_rcs_against_mie(tmp_path):
    from hertzon.solver import bridge
    from hertzon.solver.farfield_post import stratton_chu_multi
    spec = importlib.util.spec_from_file_location("mie_pec", EXAMPLES / "benchmark" / "sphere_rcs" / "mie_pec.py")
    mie = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mie)
    d = json.loads((EXAMPLES / "SphereRCS.hzn").read_text(encoding="utf-8"))
    design = d["designs"][0]
    setup = design["setups"][0]
    setup["props"].update(RefinementMethod="hzAURA (EMerge)", MaximumPasses="3", MinimumPasses="3",
                          MinimumConvergedPasses="3", MaximumDeltaEnergy="1e-9")
    proj = tmp_path / "sphere.hzn"
    proj.write_text(json.dumps(d), encoding="utf-8")
    assert bridge.run(str(proj), design["name"], setup["name"], str(tmp_path / "out")) == 0
    h = np.load(tmp_path / "out" / "huygens_LastAdaptive.npz")
    vis, wns = h["vis"], h["wns"]
    # the surface is a closed box around the sphere (r = 100 mm), inside the air
    assert np.linalg.norm(wns.sum(axis=1)) < 1e-9
    assert np.all(np.abs(vis).max(axis=0) > 0.1) and np.abs(vis).max() < 0.15
    # lossless scatterer: the scattered power leaves the box
    assert float(h["Prad_0_1"][0]) > 0
    f = float(h["freq_0"][0])
    th, ph = h["mono_dir"]
    F = stratton_chu_multi([h["E_0_1"]], [h["H_0_1"]], vis, wns, np.array([th]), np.array([ph]),
                           2 * np.pi * f / bridge.C0)[0]
    sigma = 4 * np.pi * float(np.sum(np.abs(F) ** 2)) / float(h["E0"][0]) ** 2
    assert abs(10 * np.log10(sigma / mie.rcs_mono(0.1, f)[0])) < 0.5
