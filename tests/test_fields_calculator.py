"""Fields Calculator: the language (units, errors), evaluation on element quadrature against an analytic
TE10 waveguide mode, cut planes on element faces, curved elements, and the design round trip."""
from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from hertzon.post.calc import lang as L
from hertzon.post.calc import units as UN
from hertzon.post.calc.evaluate import Evaluator, FieldSource
from hertzon.post.calc.geometry import Mesh, node_lambdas

A, B, LEN, ER, F0 = 22.86e-3, 10.16e-3, 30e-3, 2.2, 9e9


def te10(P, f):
    """TE10 mode in a guide filled with ER, normalised to 1 W."""
    w = 2 * np.pi * f
    k, kc = w * np.sqrt(ER) / L.C0, np.pi / A
    beta = np.sqrt(k * k - kc * kc)
    E0 = np.sqrt(4 * w * L.MU0 / (A * B * beta))
    x, z = P[:, 0], P[:, 2]
    e = np.exp(-1j * beta * z)
    E, H = np.zeros((len(P), 3), complex), np.zeros((len(P), 3), complex)
    E[:, 1] = E0 * np.sin(kc * x) * e
    H[:, 0] = -(beta / (w * L.MU0)) * E0 * np.sin(kc * x) * e
    H[:, 2] = 1j * (kc / (w * L.MU0)) * E0 * np.cos(kc * x) * e
    return E, H, E0, beta


class Waveguide(FieldSource):
    def __init__(self, n=(10, 5, 14), npt=10):
        super().__init__()
        g = pv.ImageData(dimensions=(n[0] + 1, n[1] + 1, n[2] + 1),
                         spacing=(A / n[0], B / n[1], LEN / n[2])).triangulate()
        self.nodes = np.asarray(g.points, float)
        self.tets = np.asarray(g.cells_dict[pv.CellType.TETRA], np.int64)
        self.npt, self.freq = npt, F0
        P = np.einsum("ki,mid->mkd", node_lambdas(npt), self.nodes[self.tets])
        E, H, *_ = te10(P.reshape(-1, 3), F0)
        self._E, self._H = E.reshape(len(self.tets), npt, 3), H.reshape(len(self.tets), npt, 3)

    def nodal(self, which):
        return self._E if which == "E" else self._H

    def material(self, prop):
        return np.full(len(self.tets), {"eps_r": ER, "mu_r": 1.0, "sigma": 0.0, "tand": 1e-3}[prop])

    def object_samples(self, name, dim):
        every = np.arange(len(self.tets))
        if dim == 3:
            return self.mesh.volume_samples(every, name)
        o, loc, n = self.mesh.boundary_faces(every)
        return self.mesh.triangle_samples(self.mesh.face_triangles(o, loc), o, n, label=name)


@pytest.fixture(scope="module")
def wg():
    env = L.Env(geometry={"Guide": L.GeoInfo(frozenset({3, 2}), True)})
    return Evaluator(env, Waveguide())


def value(ev, src):
    v = complex(ev.value(src).value)
    assert abs(v.imag) <= 1e-9 * max(abs(v), 1e-30)       # integrals of real densities stay real
    return v.real


# ---- language ------------------------------------------------------------------------------------
def test_units_and_kinds():
    env = L.Env(geometry={"Guide": L.GeoInfo(frozenset({3, 2}), True)})
    cases = {"E": ("V/m", True, True), "|E|": ("V/m", False, True), "re(S)": ("W/m²", True, True),
             "flux(Guide, re(S))": ("W", False, False), "int_vol(Guide, 0.25*eps0*eps_r*|E|^2)": ("J", False, False),
             "curl(E)": ("V/m²", True, True), "2mm*3GHz": ("Hz·m", False, False)}
    for src, (unit, vec, field) in cases.items():
        i = L.analyze(src, env)
        assert UN.label(i.unit) == unit, src
        assert (i.vec, i.field) == (vec, field), src
    assert UN.fmt(9.634e-3, UN.U(V=1, A=1)) == "9.634 mW"


@pytest.mark.parametrize("src, where", [("E + H", "E + H"), ("flux(Nowhere, re(S))", "Nowhere"),
                                         ("max(Guide |E|)", "|E|"), ("E · ", None)])
def test_errors_point_at_the_text(src, where):
    env = L.Env(geometry={"Guide": L.GeoInfo(frozenset({3, 2}), True)})
    with pytest.raises(L.CalcError) as e:
        L.analyze(src, env)
    if where is not None:
        assert e.value.pos is not None and e.value.pos <= src.index(where) + len(where)


def test_user_functions_rename_and_aliases():
    f = L.parse_function("power(A) = flux(A, re(S))")
    env = L.Env(geometry={"Guide": L.GeoInfo(frozenset({3, 2}), True)}, functions={"power": f},
                named={"Pin": "power(Guide)", "twice": "2*Pin"})
    assert UN.label(L.analyze("twice", env).unit) == "W"
    assert L.rename("Pin + Pin2 + Pin", "Pin", "P_in") == "P_in + Pin2 + P_in"
    assert L.analyze("Re(E)", env).vec                      # Re/Im/abs are accepted spellings
    with pytest.raises(L.CalcError):
        L.analyze("a", L.Env(named={"a": "b", "b": "a"}))     # cycle


# ---- values against the analytic TE10 mode -------------------------------------------------------
def test_power_flux_through_cuts(wg):
    assert value(wg, "flux(cut_xy(15mm), re(S))") == pytest.approx(1.0, rel=0.02)
    # planes on the end faces lie on element faces; flip() reverses the direction
    assert value(wg, "flux(cut_xy(0), re(S))") == pytest.approx(1.0, rel=0.02)
    assert value(wg, "flux(flip(cut_xy(30mm)), re(S))") == pytest.approx(-1.0, rel=0.02)
    r = wg.value("flux(cut_xy(0), re(S)) - flux(cut_xy(30mm), re(S))")
    assert r.is_zero                                         # displayed as approximately 0


def test_energies_loss_and_peak(wg):
    _, _, E0, beta = te10(np.zeros((1, 3)), F0)
    w = 2 * np.pi * F0
    we = 0.25 * L.EPS0 * ER * E0 ** 2 * A * B * LEN / 2
    assert value(wg, "int_vol(Guide, 0.25*eps0*eps_r*|E|^2)") == pytest.approx(we, rel=0.02)
    assert value(wg, "int_vol(Guide, 0.25*mu0*|H|^2)") == pytest.approx(we, rel=0.03)
    assert value(wg, "int_vol(Guide, 0.5*omega*eps0*eps_r*tand*|E|^2)") == pytest.approx(
        2 * w * 1e-3 * we, rel=0.02)
    assert value(wg, "max(Guide, |E|)") == pytest.approx(E0, rel=0.03)
    assert abs(complex(wg.value("at(point(11.43mm, 5.08mm, 0), y(E))").value)) == pytest.approx(E0, rel=0.01)
    assert value(wg, "int_vol(Guide, 1)") == pytest.approx(A * B * LEN, rel=1e-6)
    assert value(wg, "int_surf(Guide, 1)") == pytest.approx(2 * (A * B + A * LEN + B * LEN), rel=1e-6)


def test_derivatives_satisfy_maxwell(wg):
    res = abs(wg.value("max(Guide, |curl(E) + j*omega*mu0*H|) / max(Guide, |omega*mu0*H|)").value)
    assert res < 0.1


def test_phase(wg):
    _, _, E0, beta = te10(np.zeros((1, 3)), F0)
    # the instantaneous field at phase 0 is the real part, a quarter period later the imaginary part
    assert value(wg, "at(point(11.43mm, 5.08mm, 0), inst(y(E)))") == pytest.approx(E0, rel=0.01)
    ev90 = Evaluator(wg.env, wg.src, 90.0)
    assert abs(value(ev90, "at(point(11.43mm, 5.08mm, 0), inst(y(E)))")) < 0.02 * E0


# ---- geometry ------------------------------------------------------------------------------------
def test_curved_elements_integrate_the_true_volume():
    """A tet whose mid-edge nodes are pushed outwards holds more volume than the straight one; zero
    displacement gives the straight volume."""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    tets = np.array([[0, 1, 2, 3]])
    from hertzon.post.calc.geometry import TET10_EDGES
    mid = np.array([(nodes[i] + nodes[j]) / 2 for i, j in TET10_EDGES])
    assert Mesh(nodes, tets).volume_samples(np.arange(1)).W.sum() == pytest.approx(1 / 6)
    flat = Mesh(nodes, tets, curved=(np.array([0]), np.zeros((1, 6, 3))))
    assert flat.volume_samples(np.arange(1)).W.sum() == pytest.approx(1 / 6, rel=1e-6)
    bulged = Mesh(nodes, tets, curved=(np.array([0]), 0.05 * (mid - nodes.mean(axis=0))[None]))
    assert bulged.volume_samples(np.arange(1)).W.sum() > 1 / 6 * 1.01


# ---- design --------------------------------------------------------------------------------------
def test_expressions_are_saved_with_the_design():
    from hertzon.model.design import Design, FieldExpression, design_from_dict
    from dataclasses import asdict
    d = Design("D")
    d.field_exprs.append(FieldExpression("Pin", "flux(Port1, re(S))", "input power"))
    d.field_funcs.append("power(A) = flux(A, re(S))")
    back = design_from_dict(asdict(d))
    assert back.field_exprs[0] == d.field_exprs[0]
    assert back.field_funcs == d.field_funcs


def test_report_traces_follow_a_rename():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from hertzon.gui import calc_bridge as CB
    from hertzon.gui.doc import Document
    from hertzon.model.design import FieldExpression, Project, Report
    doc = Document(Project())
    d = doc.design
    d.add_object("CreateBox", "Box", Position="0,0,0", XSize="1", YSize="1", ZSize="1")
    doc.rebuild()
    d.field_exprs += [FieldExpression("Pin", "flux(Box, re(S))"), FieldExpression("magE", "|E|"),
                      FieldExpression("Evec", "E")]
    d.reports.append(Report("R", "Rectangular", "Setup1 : Sweep", ["Pin @ Setup1 : Sweep", "2*Pin + dB(S(1,1))"]))
    assert CB.value_names(doc) == ["Pin"]                   # only values go into reports
    CB.rename_everywhere(d, "Pin", "P_in")
    assert d.reports[-1].traces == ["P_in @ Setup1 : Sweep", "2*P_in + dB(S(1,1))"]
    assert CB.field_names_in(d, d.reports[-1].traces[1]) == ["P_in"]
