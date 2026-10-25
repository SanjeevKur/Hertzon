"""Touchstone (.s1p/.s2p) reader and the impedance of a two-terminal component measured with it.

A one-port file is the component itself (Z = Z0 (1+S11)/(1-S11)). A two-port file is the component in a
test fixture, either in series between the two ports (the usual way for chip inductors and capacitors,
Z = 2 Z0 (1-S21)/S21) or as a shunt from the through line to ground (Z = Z0 S21 / (2 (1-S21)))."""
from __future__ import annotations

from pathlib import Path

import numpy as np

FREQ_UNITS = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
MOUNTS = ("Series", "Shunt")


def read(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Frequencies (Hz), S (F, N, N) and the reference impedance of a Touchstone 1.x file with N ports
    (from the extension .sNp)."""
    path = Path(path)
    ext = path.suffix.lower()
    try:
        n = int(ext[2:-1])
    except ValueError:
        raise ValueError(f"{path.name}: not a Touchstone file (.s1p, .s2p ...)") from None
    unit, fmt, z0 = 1e9, "MA", 50.0
    nums: list[float] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            tok = line[1:].upper().split()
            i = 0
            while i < len(tok):
                t = tok[i]
                if t in FREQ_UNITS:
                    unit = FREQ_UNITS[t]
                elif t in ("MA", "DB", "RI"):
                    fmt = t
                elif t == "R" and i + 1 < len(tok):
                    z0 = float(tok[i + 1])
                    i += 1
                elif t != "S":
                    raise ValueError(f"{path.name}: only S-parameter files are supported ('{t}')")
                i += 1
            continue
        if line.startswith("["):
            raise ValueError(f"{path.name}: Touchstone 2.0 keywords are not supported")
        nums.extend(float(x) for x in line.split())
    per = 1 + 2 * n * n
    if not nums or len(nums) % per:
        raise ValueError(f"{path.name}: incomplete data ({len(nums)} numbers, {per} per frequency)")
    d = np.asarray(nums).reshape(-1, per)
    f = d[:, 0] * unit
    a, b = d[:, 1::2], d[:, 2::2]
    if fmt == "RI":
        s = a + 1j * b
    elif fmt == "DB":
        s = 10 ** (a / 20) * np.exp(1j * np.radians(b))
    else:
        s = a * np.exp(1j * np.radians(b))
    S = s.reshape(-1, n, n)
    if n == 2:
        S = S.transpose(0, 2, 1)            # 2 ports are listed S11 S21 S12 S22 (column major)
    return f, S, z0


def impedance(f: np.ndarray, S: np.ndarray, z0: float, mount: str = "Series") -> np.ndarray:
    """Impedance of the two-terminal component over frequency."""
    if S.shape[1] == 1:
        s11 = S[:, 0, 0]
        return z0 * (1 + s11) / (1 - s11)
    s21 = S[:, 1, 0]
    if mount == "Shunt":
        return z0 * s21 / (2 * (1 - s21))
    return 2 * z0 * (1 - s21) / s21


class ComponentImpedance:
    """Z(f) of a Touchstone component, interpolated linearly in real and imaginary part. Outside the
    measured band the nearest data point is used (``out_of_band`` lists the frequencies it was asked for)."""

    def __init__(self, path: str | Path, mount: str = "Series"):
        f, S, z0 = read(path)
        if S.shape[1] > 2:
            raise ValueError(f"{Path(path).name}: a component needs a 1-port or 2-port file")
        order = np.argsort(f)
        self.f = f[order]
        self.z = impedance(f, S, z0, mount)[order]
        self.out_of_band: list[float] = []

    def __call__(self, freq: float) -> complex:
        if freq < self.f[0] or freq > self.f[-1]:
            self.out_of_band.append(float(freq))
        return complex(np.interp(freq, self.f, self.z.real), np.interp(freq, self.f, self.z.imag))
