"""Far fields with symmetry planes: the stored surface of a half or quarter model, mirrored, gives the far field
and radiated power of the full structure at 1 W (no FEM solve; analytic fields of a short dipole)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hertzon.model.expr import Evaluator  # noqa: E402
from hertzon.solver import farfield_post as FP  # noqa: E402

pytest.importorskip("emr.sc_multi")

F = 1e9
K0 = 2 * np.pi * F / 299792458.0
ETA = 376.730313
PROPS = {"ThetaStart": "0deg", "ThetaStop": "180deg", "ThetaStep": "10deg",
         "PhiStart": "0deg", "PhiStop": "360deg", "PhiStep": "10deg"}


def dipole_fields(X):
    """E and H of a z directed short dipole at the origin (e^{jwt}, all near field terms) at the points X (3, N)."""
    x, y, z = X
    r = np.sqrt(x * x + y * y + z * z)
    th, ph = np.arccos(z / r), np.arctan2(y, x)
    kr = K0 * r
    g = np.exp(-1j * kr)
    Er = ETA / (2 * np.pi * r ** 2) * (1 + 1 / (1j * kr)) * np.cos(th) * g
    Et = 1j * ETA * K0 / (4 * np.pi * r) * (1 + 1 / (1j * kr) - 1 / kr ** 2) * np.sin(th) * g
    Hp = 1j * K0 / (4 * np.pi * r) * (1 + 1 / (1j * kr)) * np.sin(th) * g
    rh = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    th_h = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
    ph_h = np.array([-np.sin(ph), np.cos(ph), 0 * ph])
    return Er * rh + Et * th_h, Hp * ph_h


def octant_points(n=4000, a=0.12):
    """Points (with area weights along the outward normal) of a sphere of radius a in x > 0, y > 0, z > 0."""
    rng = np.random.default_rng(1)
    u = np.abs(rng.normal(size=(3, n)))
    u /= np.linalg.norm(u, axis=0)
    return a * u, u * (4 * np.pi * a * a / (8 * n))


def huygens(X, W, scale=1.0):
    E, H = dipole_fields(X)
    E, H = E * scale, H * scale
    S = 0.5 * np.cross(E.T, np.conj(H).T).T
    return {"vis": X, "wns": W, "freq_0": np.array([F]), "E_0_1": E, "H_0_1": H,
            "Prad_0_1": np.array([float(np.real(np.sum(S * W)))])}


def with_planes(huy, planes):
    huy["sym_kind"] = np.array([p[0] for p in planes])
    huy["sym_axis"] = np.array([p[1] for p in planes], np.int64)
    huy["sym_pos"] = np.array([p[2] for p in planes], np.float64)
    return huy


@pytest.mark.parametrize("planes", [[("H", 0, 0.0)], [("E", 2, 0.0)], [("H", 0, 0.0), ("H", 1, 0.0)],
                                    [("H", 0, 0.0), ("H", 1, 0.0), ("E", 2, 0.0)]])
def test_mirrored_surface_gives_full_far_field(planes):
    """A z dipole is symmetric to Perfect H planes x = 0 and y = 0 and to a Perfect E plane z = 0. The surface of
    the simulated part, excited with 1 W into that part (fields sqrt(2) per plane larger than in the full model
    at 1 W), mirrored at the planes equals the full surface at 1 W."""
    Xo, Wo = octant_points()
    X, W = Xo, Wo        # the full surface: the octant mirrored at all three coordinate planes
    for ax in range(3):
        Xm, Wm = X.copy(), W.copy()
        Xm[ax], Wm[ax] = -Xm[ax], -Wm[ax]
        X, W = np.concatenate([X, Xm], axis=1), np.concatenate([W, Wm], axis=1)
    keep = np.all([X[ax] > 0 for _, ax, _ in planes], axis=0)
    ev = Evaluator({}, "mm")
    full = FP.compute_sphere(huygens(X, W), PROPS, ev)
    part = FP.compute_sphere(with_planes(huygens(X[:, keep], W[:, keep], np.sqrt(2.0 ** len(planes))), planes),
                             PROPS, ev)
    peak = np.abs(full["E_0_1"]).max()
    assert peak > 0
    assert np.abs(part["E_0_1"] - full["E_0_1"]).max() < 1e-9 * peak
    assert float(part["Prad_0_1"][0]) == pytest.approx(float(full["Prad_0_1"][0]), rel=1e-9)


def test_wrong_plane_type_differs():
    """The plane type matters: mirroring the x > 0 half as a Perfect E plane does not give the dipole's field."""
    Xo, Wo = octant_points()
    X, W = Xo, Wo
    for ax in range(3):
        Xm, Wm = X.copy(), W.copy()
        Xm[ax], Wm[ax] = -Xm[ax], -Wm[ax]
        X, W = np.concatenate([X, Xm], axis=1), np.concatenate([W, Wm], axis=1)
    keep = X[0] > 0
    ev = Evaluator({}, "mm")
    full = FP.compute_sphere(huygens(X, W), PROPS, ev)
    part = FP.compute_sphere(with_planes(huygens(X[:, keep], W[:, keep], np.sqrt(2.0)), [("E", 0, 0.0)]), PROPS, ev)
    assert np.abs(part["E_0_1"] - full["E_0_1"]).max() > 0.5 * np.abs(full["E_0_1"]).max()


def test_offset_plane():
    """A symmetry plane at x = x0 mirrors to 2 x0 - x."""
    v, w, Es, Hs = FP.mirrored(np.array([[1.0], [2.0], [3.0]]), np.array([[1.0], [0.0], [0.0]]),
                               [np.array([[1.0], [1.0], [1.0]])], [np.array([[1.0], [1.0], [1.0]])],
                               [("E", 0, 0.5)])
    assert v[:, 1] == pytest.approx([0.0, 2.0, 3.0])
    assert w[:, 1] == pytest.approx([-1.0, 0.0, 0.0])
    assert Es[0][:, 1] == pytest.approx([1.0, -1.0, -1.0])
    assert Hs[0][:, 1] == pytest.approx([-1.0, 1.0, 1.0])
