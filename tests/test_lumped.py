"""Lumped Elements: Touchstone components, SMD packages and the generated component geometry."""
import numpy as np
import pytest

from hertzon.model import lumped as LU
from hertzon.model import touchstone as TS
from hertzon.model.design import Assignment, Project
from hertzon.model.schema import BOUNDARIES, defaults


def _s2p(path, z, fmt="RI", unit="GHz", f=(0.1, 1.0, 3.0)):
    """2-port file of a series impedance z between 50 ohm ports."""
    s11 = z / (z + 100)
    s21 = 100 / (z + 100)
    with open(path, "w") as fh:
        fh.write("! test\n# " + unit + " S " + fmt + " R 50\n")
        for x in f:
            row = []
            for s in (s11, s21, s21, s11):
                if fmt == "RI":
                    row += [s.real, s.imag]
                elif fmt == "MA":
                    row += [abs(s), np.degrees(np.angle(s))]
                else:
                    row += [20 * np.log10(abs(s)), np.degrees(np.angle(s))]
            fh.write(f"{x} " + " ".join(f"{v:.12g}" for v in row) + "\n")
    return path


@pytest.mark.parametrize("fmt", ["RI", "MA", "DB"])
def test_touchstone_series_impedance(tmp_path, fmt):
    z = complex(12, -30)
    zf = TS.ComponentImpedance(_s2p(tmp_path / "c.s2p", z, fmt, "MHz", (100, 1000, 3000)))
    assert zf.f == pytest.approx([1e8, 1e9, 3e9])
    assert zf(5e8) == pytest.approx(z, rel=1e-9)
    assert zf(1e10) == pytest.approx(z) and zf.out_of_band == [1e10]


def test_touchstone_shunt_and_one_port(tmp_path):
    z = complex(5, 20)
    s21 = 2 * z / (2 * z + 50)                       # shunt z to ground on a 50 ohm through line
    s11 = s21 - 1
    (tmp_path / "s.s2p").write_text("# Hz S RI R 50\n" + "\n".join(
        f"{f} {s11.real} {s11.imag} {s21.real} {s21.imag} {s21.real} {s21.imag} {s11.real} {s11.imag}"
        for f in (1e9, 2e9)))
    assert TS.ComponentImpedance(tmp_path / "s.s2p", "Shunt")(1.5e9) == pytest.approx(z)
    g = (z - 50) / (z + 50)
    (tmp_path / "o.s1p").write_text(f"# GHz S MA R 50\n1 {abs(g)} {np.degrees(np.angle(g))}\n"
                                    f"2 {abs(g)} {np.degrees(np.angle(g))}\n")
    assert TS.ComponentImpedance(tmp_path / "o.s1p")(1.2e9) == pytest.approx(z)


def test_touchstone_errors(tmp_path):
    (tmp_path / "bad.s2p").write_text("# GHz S RI R 50\n1 0 0 1 0\n")
    with pytest.raises(ValueError):
        TS.read(tmp_path / "bad.s2p")
    (tmp_path / "y.s2p").write_text("# GHz Y RI R 50\n")
    with pytest.raises(ValueError):
        TS.read(tmp_path / "y.s2p")


def test_bridge_impedance_models(tmp_path):
    """Parallel (default, older projects without Model), series and Touchstone Z(f) in the solver bridge."""
    from types import SimpleNamespace
    from hertzon.solver.bridge import Bridge
    proj = Project("p")
    proj.path = str(tmp_path / "p.hzn")
    d = proj.designs[0]
    br = SimpleNamespace(ev=d.evaluator(proj), project=proj, touchstone_elements=[],
                         setup=SimpleNamespace(props={"Frequency": "1GHz"}))
    z = lambda props, f: Bridge.lumped_impedance(br, Assignment("E", "Lumped RLC", props))(f)  # noqa: E731
    p = defaults(BOUNDARIES["Lumped RLC"])
    p.update(UseResistance="True", Resistance="50ohm", UseCapacitance="True", Capacitance="1pF")
    w = 2 * np.pi * 1e9
    old = {k: v for k, v in p.items() if k != "Model"}
    assert z(old, 1e9) == pytest.approx(1 / (1 / 50 + 1j * w * 1e-12))
    assert z({**p, "Model": "Series RLC"}, 1e9) == pytest.approx(50 + 1 / (1j * w * 1e-12))
    _s2p(tmp_path / "part.s2p", complex(3, 40))
    ts = {**p, "Model": "Touchstone", "TouchstoneFile": "part.s2p"}           # relative to the project
    assert z(ts, 1e9) == pytest.approx(complex(3, 40))
    assert br.touchstone_elements and br.touchstone_elements[0][0] == "E"


def test_land_pattern_and_packages():
    p = LU.PACKAGES["0402"]
    land = LU.land_pattern(p)
    assert land.G == pytest.approx(0.5) and land.Z == pytest.approx(1.2) and land.X == pytest.approx(0.5)
    assert LU.land_pattern(LU.PACKAGES["0603"]).Z == pytest.approx(2.3)
    assert LU.series_for("0402HP-2N0XJLW").name == "Coilcraft 0402HP"
    assert LU.series_for("0402HPH-10NXGLU").name == "Coilcraft 0402HPH"
    assert LU.series_for("GJM1555C1H1R0BB01").name == "Murata GJM15"
    assert LU.series_for("XYZ") is None


def _trace(x0, x1, y0, y1, z):
    return np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)


def test_trace_ends_and_planar_geometry():
    ends = LU.trace_ends(_trace(-0.75, 0.75, 0, 14, 0.8), _trace(-0.75, 0.75, 15, 30, 0.8))
    assert (ends.axis, ends.sign, ends.normal, ends.a_end, ends.b_end) == (1, 1, 2, 14, 15)
    pkg = LU.PACKAGES["0402"]
    g = LU.component_geometry(ends, pkg, LU.land_pattern(pkg), "C1")
    by = {b.name: b for b in g.boxes}
    assert set(by) == {"C1_PadA", "C1_PadB", "C1"}
    assert by["C1_PadA"].lo == pytest.approx((-0.25, 14, 0.8)) and by["C1_PadA"].size == pytest.approx((0.5, 0.25, 0))
    assert by["C1"].lo == pytest.approx((-0.25, 14.25, 0.8)) and by["C1"].size == pytest.approx((0.5, 0.5, 0))
    assert g.line == ((0, 14.25, 0.8), (0, 14.75, 0.8))
    # reversed selection order and a gap along -x: the same component
    ends2 = LU.trace_ends(_trace(15, 30, -0.75, 0.75, 0.8), _trace(0, 14, -0.75, 0.75, 0.8))
    assert (ends2.axis, ends2.sign) == (0, -1)
    g2 = LU.component_geometry(ends2, pkg, LU.land_pattern(pkg), "C1")
    assert g2.line == ((14.25, 0, 0.8), (14.75, 0, 0.8))
    assert {b.name: b for b in g2.boxes}["C1"].lo == pytest.approx((14.25, -0.25, 0.8))


def test_geometry_with_terminations_and_too_small_gap():
    ends = LU.trace_ends(_trace(-1, 1, 0, 10, 0), _trace(-1, 1, 10.5, 20, 0))
    pkg = LU.PACKAGES["0402"]
    g = LU.component_geometry(ends, pkg, LU.land_pattern(pkg), "L1", terminations=True)
    by = {b.name: b for b in g.boxes}
    assert "L1_PadA" not in by                         # gap = G: the traces are the pads
    assert by["L1_TermA"].size == pytest.approx((0.5, 0.25, 0.5)) and by["L1_TermA"].lo[1] == pytest.approx(10.25 - 0.5)
    assert by["L1"].lo[2] == pytest.approx(0.5) and g.line == ((0, 10.0, 0.5), (0, 10.5, 0.5))
    with pytest.raises(ValueError):
        LU.component_geometry(LU.trace_ends(_trace(-1, 1, 0, 10, 0), _trace(-1, 1, 10.2, 20, 0)), pkg,
                              LU.land_pattern(pkg), "L1")
    with pytest.raises(ValueError):
        LU.trace_ends(_trace(-1, 1, 0, 10, 0), _trace(-1, 1, 5, 20, 0))      # overlapping, no gap


def _design(pec_material=False):
    p = Project("RLC")
    d = p.designs[0]
    d.add_object("CreateBox", name="Substrate", material="FR4_epoxy", Position="-10,0,0", XSize="20", YSize="30",
                 ZSize="0.8")
    mat = "pec" if pec_material else "vacuum"
    d.add_object("CreateRectangle", name="TraceA", material=mat, Position="-0.75,0,0.8", Axis="Z", XSize="1.5",
                 YSize="14")
    d.add_object("CreateRectangle", name="TraceB", material=mat, Position="-0.75,15,0.8", Axis="Z", XSize="1.5",
                 YSize="15")
    if not pec_material:
        d.boundaries.append(Assignment("PerfE1", "Perfect E", {}, objects=["TraceA", "TraceB"]))
    ends = LU.trace_ends(_trace(-0.75, 0.75, 0, 14, 0.8), _trace(-0.75, 0.75, 15, 30, 0.8))
    return p, d, ends


def test_insert_component_into_a_design():
    p, d, ends = _design()
    pkg = LU.PACKAGES["0603"]
    b, made, warns = LU.insert_component(p, d, "TraceA", "TraceB", ends, pkg, LU.land_pattern(pkg), "C", "2.2pF")
    assert [o.name for o in made] == ["C1_PadA", "C1_PadB", "C1"]
    assert d.boundaries[0].objects == ["TraceA", "TraceB", "C1_PadA", "C1_PadB"]   # pads made like the traces
    assert b.kind == "Lumped RLC" and b.objects == ["C1"] and b.name == "C1_RLC"
    assert b.props["Model"] == "Series RLC" and b.props["Capacitance"] == "2.2pF"
    assert b.props["UseInductance"] == "True" and b.props["Inductance"] == "0.45nH"     # typical ESL
    assert b.props["IntegrationLine"] == "0,14.05,0.8;0,14.95,0.8" and b.props["Component"] == "0603"
    assert made[0].history[0].props["Position"] == "-0.4,14,0.8" and made[0].history[0].props["YSize"] == "0.05"
    # second component gets the next name, metal traces give metal pads
    p, d, ends = _design(pec_material=True)
    pkg = LU.PACKAGES["0402"]
    LU.insert_component(p, d, "TraceA", "TraceB", ends, pkg, LU.land_pattern(pkg), "R", "50ohm", parasitics=False)
    b2, made2, _ = LU.insert_component(p, d, "TraceA", "TraceB", ends, pkg, LU.land_pattern(pkg), "L", "3nH",
                                       terminations=True)
    assert made2[-1].name == "L1" and b2.props["Inductance"] == "3nH" and b2.props["UseResistance"] == "False"
    # sheet pads conduct through a Perfect E boundary (a metal material makes only solids conductors)
    assert d.assignment("R1_Pads").kind == "Perfect E" and d.assignment("R1_Pads").objects == ["R1_PadA", "R1_PadB"]
    assert d.obj("L1_TermA").material == "pec" and d.obj("L1_TermA").history[0].kind == "CreateBox"


def test_insert_touchstone_component(tmp_path):
    p, d, ends = _design()
    pkg = LU.PACKAGES["Murata GJM15"]
    f = _s2p(tmp_path / "GJM1555C1H1R0BB01.s2p", complex(0.1, -150))
    b, _, _ = LU.insert_component(p, d, "TraceA", "TraceB", ends, pkg, LU.land_pattern(pkg), "S", touchstone=str(f))
    assert b.props["Model"] == "Touchstone" and b.props["TouchstoneFile"] == str(f)
    assert b.props["Component"] == "Murata GJM15 (GJM1555C1H1R0BB01)"


def test_place_component_freely():
    """Placed by a click: full land pattern around the point, position in design variables, Perfect E pads."""
    p = Project("Free")
    d = p.designs[0]
    pkg = LU.PACKAGES["0603"]
    land = LU.land_pattern(pkg)                                        # G 0.9, Z 2.3, X 0.8
    place = LU.Placement((5.0, 7.0, 0.8), axis=1, normal=2)
    b, made, warns = LU.insert_component(p, d, None, None, None, pkg, land, "C", "1pF", place=place)
    assert [o.name for o in made] == ["C1_PadA", "C1_PadB", "C1"] and not warns
    assert d.variables == {"C1_x": "5mm", "C1_y": "7mm", "C1_z": "0.8mm"}
    pa = made[0].history[0].props
    assert pa["Position"] == "C1_x-0.4mm,C1_y-1.15mm,C1_z" and pa["YSize"] == "0.7" and pa["XSize"] == "0.8"
    assert d.assignment("C1_Pads").objects == ["C1_PadA", "C1_PadB"]
    ev = d.evaluator(p)
    line = ev.points(b.props["IntegrationLine"])
    assert np.allclose(np.asarray(line) * 1e3, [[5, 6.55, 0.8], [5, 7.45, 0.8]])
    assert np.allclose(np.asarray(ev.point(made[1].history[0].props["Position"])) * 1e3, [4.6, 7.45, 0.8])
    # moving the variable moves pads, element and integration line together
    d.variables["C1_y"] = "10mm"
    ev = d.evaluator(p)
    assert np.allclose(np.asarray(ev.points(b.props["IntegrationLine"]))[:, 1] * 1e3, [9.55, 10.45])
    # a second part gets its own variables; along X in the board plane
    b2, made2, _ = LU.insert_component(p, d, None, None, None, pkg, land, "R", "10ohm",
                                       place=LU.Placement((0, 0, 0), axis=0, normal=2))
    assert made2[-1].name == "R1" and "R1_x" in d.variables
    assert b2.props["IntegrationLine"] == "R1_x-0.45mm,R1_y,R1_z;R1_x+0.45mm,R1_y,R1_z"
    with pytest.raises(ValueError):
        LU.Placement((0, 0, 0), axis=2, normal=2).ends(land)
