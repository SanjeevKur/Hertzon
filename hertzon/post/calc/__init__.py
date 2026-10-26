"""Fields Calculator: expressions of E, H and J evaluated, integrated and reduced on stored solutions.

    from hertzon.post import calc
    info = calc.analyze("flux(Port1, re(S))", env)          # type, unit, typeset formula
    res = calc.evaluate("flux(Port1, re(S))", env, source)  # Result with value and unit

No Qt: the GUI, reports and scripts use the same core. The application supplies a FieldSource (the stored
field of one solution, frequency and excitation, with the model's geometry) and an Env (named
expressions, functions, geometry names, coordinate systems, design variables)."""
from __future__ import annotations

from .evaluate import Evaluator, FieldSource, Result, evaluate
from .geometry import Mesh, Samples
from .lang import CalcError, Env, GeoInfo, Info, UserFunction, analyze, parse, parse_function, rename
from . import units

__all__ = ["CalcError", "Env", "Evaluator", "FieldSource", "GeoInfo", "Info", "Mesh", "Result", "Samples",
           "UserFunction", "analyze", "evaluate", "parse", "parse_function", "rename", "units"]
