"""Report expressions on S-parameter data: S(i,j), Y, Z, VSWR, GroupDelay, dB(), output variables.

Used by the report windows and by the optimizer process (no Qt here)."""
from __future__ import annotations

import re

import numpy as np

from .ffquant import ANTENNA_POWER


def network(data: dict, cat: str) -> np.ndarray:
    S = data["S"]
    n = S.shape[1]
    z0 = np.real(np.ravel(data["z0"]))
    if z0.size < n:
        z0 = np.full(n, z0[0] if z0.size else 50.0)
    if cat in ("Y Parameter", "Z Parameter"):
        I = np.eye(n)
        sq = np.diag(np.sqrt(z0))
        out = np.zeros_like(S)
        for k in range(S.shape[0]):
            Z = sq @ np.linalg.solve(I - S[k], I + S[k]) @ sq
            out[k] = Z if cat == "Z Parameter" else np.linalg.inv(Z)
        return out
    return S


def split_trace(trace: str) -> tuple[str, str | None]:
    """'dB(S(1,1)) @ Setup1 : Sweep' -> (expression, solution); without '@' the report's solution is used."""
    expr, sep, sol = trace.partition(" @ ")
    return expr.strip(), (sol.strip() or None) if sep else None


def output_variables(design) -> dict[str, str]:
    """Output Variables of a design: 'name = expression' lines in the design settings."""
    out = {}
    for line in (design.settings.get("OutputVariables", "") if design is not None else "").splitlines():
        name, sep, expr = line.partition("=")
        if sep and name.strip().isidentifier():
            out[name.strip()] = expr.strip()
    return out


_FUNCS = {
    "dB": lambda v: 20 * np.log10(np.maximum(np.abs(v), 1e-300)),
    "dB10": lambda v: 10 * np.log10(np.maximum(np.abs(v), 1e-300)),
    "dB20": lambda v: 20 * np.log10(np.maximum(np.abs(v), 1e-300)),
    "mag": np.abs, "abs": np.abs, "ang_deg": lambda v: np.degrees(np.angle(v)), "ang_rad": np.angle,
    "re": np.real, "im": np.imag, "real": np.real, "imag": np.imag, "conj": np.conj, "sqrt": np.sqrt,
    "log10": np.log10, "ln": np.log, "log": np.log, "exp": np.exp, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "atan": np.arctan, "min": np.minimum, "max": np.maximum, "deg": np.degrees, "rad": np.radians,
}


def _ast_eval(expr: str, names: dict):
    import ast
    import operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
           ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, complex)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name):
            if n.id not in names:
                raise ValueError(f"Unknown name '{n.id}'")
            v = names[n.id]
            return v() if callable(v) and getattr(v, "_lazy", False) else v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
            f = names.get(n.func.id)
            if not callable(f):
                raise ValueError(f"Unknown function '{n.func.id}'")
            return f(*[ev(a) for a in n.args])
        raise ValueError("Unsupported expression")
    return ev(ast.parse(expr.replace("^", "**"), mode="eval"))


def eval_expr(data: dict, expr: str, outvars: dict[str, str] | None = None, _depth: int = 0):
    """Evaluate an report expression to a (possibly complex) array over frequency.

    Knows S(i,j), Y(i,j), Z(i,j), VSWR(i), GroupDelay(x), Freq, pi, j, the usual functions
    (dB, mag, ang_deg, re, im, sqrt, ...), arithmetic and the design's output variables."""
    f = data["freq"]
    cache: dict = {}

    def mat(cat):
        if cat not in cache:
            cache[cat] = network(data, cat)
        return cache[cat]

    def par(cat):
        def g(i, jj):
            M = mat(cat)
            i, jj = int(i) - 1, int(jj) - 1
            if not (0 <= i < M.shape[1] and 0 <= jj < M.shape[2]):
                raise ValueError("Port index out of range")
            return M[:, i, jj]
        return g

    def vswr(i):
        g = np.abs(data["S"][:, int(i) - 1, int(i) - 1])
        return (1 + g) / np.maximum(1 - g, 1e-12)

    def gd(v):
        return -np.gradient(np.unwrap(np.angle(v)), 2 * np.pi * f)

    names = dict(_FUNCS)
    names.update({"S": par("S Parameter"), "Y": par("Y Parameter"), "Z": par("Z Parameter"), "VSWR": vswr,
                  "GroupDelay": gd, "Freq": f, "F": f, "pi": np.pi, "j": 1j, "e": np.e})
    if "antenna" in data:
        names.update(data["antenna"])
        names["Pass"] = f
        # dB() of a gain, directivity or power is 10 log10
        expr = re.sub(r"\bdB\(\s*(" + "|".join(sorted(ANTENNA_POWER)) + r")\s*\)", r"dB10(\1)", expr)
    if _depth < 8:
        for k, v in (outvars or {}).items():
            def lazy(v=v):
                return eval_expr(data, v, outvars, _depth + 1)
            lazy._lazy = True
            names[k] = lazy
    out = _ast_eval(expr, names)
    out = np.asarray(out)
    return np.broadcast_to(out, f.shape).copy() if out.shape != f.shape else out


def eval_trace(data: dict, trace: str, outvars: dict[str, str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate e.g. 'dB(S(1,1))', 'ang_deg(Z(2,1))', 'VSWR(1)', 'mag(S(2,1))^2 + mag(S(1,1))^2'."""
    expr, _ = split_trace(trace)
    v = eval_expr(data, expr, outvars)
    return data["freq"], (np.real(v) if np.iscomplexobj(v) else v.astype(float))
