"""Units of the Fields Calculator as exponent vectors over V, A, m, s and kg.

V and A are kept as base dimensions (instead of kg m^2 s^-3 A^-1) so that results read as V/m, W, J, Ohm
and S without conversion; kg only appears in densities (SAR in W/kg)."""
from __future__ import annotations

DIMS = ("V", "A", "m", "s", "kg")
Unit = tuple  # (V, A, m, s, kg) exponents


def U(V=0.0, A=0.0, m=0.0, s=0.0, kg=0.0) -> Unit:
    return (float(V), float(A), float(m), float(s), float(kg))


ONE = U()


def mul(a: Unit, b: Unit, k: float = 1.0) -> Unit:
    return tuple(x + k * y for x, y in zip(a, b))


def power(a: Unit, k: float) -> Unit:
    return tuple(x * k for x in a)


def same(a: Unit, b: Unit) -> bool:
    return all(abs(x - y) < 1e-9 for x, y in zip(a, b))


# named units, tried in this order when a unit is printed
NAMED = [("W", U(V=1, A=1)), ("J", U(V=1, A=1, s=1)), ("Ohm", U(V=1, A=-1)), ("S", U(A=1, V=-1)),
         ("F", U(A=1, s=1, V=-1)), ("H", U(V=1, s=1, A=-1)), ("C", U(A=1, s=1)), ("Wb", U(V=1, s=1)),
         ("V", U(V=1)), ("A", U(A=1)), ("Hz", U(s=-1)), ("W/kg", U(V=1, A=1, kg=-1))]

# units written after a number in an expression: (factor to SI, unit)
NUMBER_UNITS = {
    "nm": (1e-9, U(m=1)), "um": (1e-6, U(m=1)), "mm": (1e-3, U(m=1)), "cm": (1e-2, U(m=1)), "m": (1.0, U(m=1)),
    "mil": (25.4e-6, U(m=1)), "in": (25.4e-3, U(m=1)),
    "Hz": (1.0, U(s=-1)), "kHz": (1e3, U(s=-1)), "MHz": (1e6, U(s=-1)), "GHz": (1e9, U(s=-1)), "THz": (1e12, U(s=-1)),
    "deg": (3.141592653589793 / 180, ONE), "rad": (1.0, ONE),
    "W": (1.0, U(V=1, A=1)), "mW": (1e-3, U(V=1, A=1)), "uW": (1e-6, U(V=1, A=1)),
    "V": (1.0, U(V=1)), "mV": (1e-3, U(V=1)), "kV": (1e3, U(V=1)),
    "A": (1.0, U(A=1)), "mA": (1e-3, U(A=1)),
    "ohm": (1.0, U(V=1, A=-1)), "Ohm": (1.0, U(V=1, A=-1)), "S": (1.0, U(A=1, V=-1)),
    "s": (1.0, U(s=1)), "ns": (1e-9, U(s=1)), "ps": (1e-12, U(s=1)),
    "kg": (1.0, U(kg=1)), "g": (1e-3, U(kg=1)),
    "S/m": (1.0, U(A=1, V=-1, m=-1)), "V/m": (1.0, U(V=1, m=-1)), "kV/m": (1e3, U(V=1, m=-1)),
    "A/m": (1.0, U(A=1, m=-1)), "W/kg": (1.0, U(V=1, A=1, kg=-1)),
}
# single letters must touch the number ("2S", "3m"); longer units may follow after a space ("2 mm")
SHORT_UNITS = {u for u in NUMBER_UNITS if len(u) == 1}

_SUP = str.maketrans("-0123456789.", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹·")


def _sup(k: float) -> str:
    return f"{round(k, 3):g}".translate(_SUP)


def label(u: Unit) -> str:
    """'V/m', 'W', 'A/m²', 'W/m³'; '' for dimensionless."""
    if same(u, ONE):
        return ""

    def m_only(r):
        return all(abs(x) < 1e-9 for d, x in zip(DIMS, r) if d != "m")

    def mpow(k):
        if abs(k) < 1e-9:
            return ""
        if k == 1:
            return "·m"
        return f"·m{_sup(k)}" if k > 0 else ("/m" if k == -1 else f"/m{_sup(-k)}")

    for name, nu in NAMED:
        r = mul(u, nu, -1)
        if m_only(r):
            return name + mpow(r[2])
    if m_only(u):
        return "m" if u[2] == 1 else f"m{_sup(u[2])}"
    return "·".join(d + ("" if x == 1 else _sup(x)) for d, x in zip(DIMS, u) if abs(x) > 1e-9)


def describe(u: Unit) -> str:
    return label(u) or "dimensionless"


_PREFIX = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"),
           (1e-12, "p"), (1e-15, "f")]


def fmt(v: float, u: Unit = ONE, digits: int = 4) -> str:
    """A value with an SI prefix: 2.237 kV/m, 9.636 mW, 85.20 pJ."""
    unit = label(u)
    if v != v or v in (float("inf"), float("-inf")):
        return str(v)
    if v == 0:
        return "0" + (f" {unit}" if unit else "")
    if not unit or unit.startswith("m") and unit.rstrip("²³") == "m" or "/kg" in unit:
        return f"{v:.{digits}g}" + (f" {unit}" if unit else "")
    a = abs(v)
    for p, s in _PREFIX:
        if a >= p * 0.9995:
            return f"{v / p:.{digits}g} {s}{unit}"
    return f"{v:.{digits - 1}e} {unit}"
