"""Monostatic RCS of a PEC sphere (Mie series, Bohren & Huffman 4.82)."""
import sys
import numpy as np
from scipy.special import spherical_jn, spherical_yn


def rcs_mono(a, f):
    x = 2 * np.pi * f / 299792458.0 * a
    n = np.arange(1, int(x + 4 * x ** (1 / 3) + 20))
    j, y = spherical_jn(n, x), spherical_yn(n, x)
    jd, yd = spherical_jn(n, x, True), spherical_yn(n, x, True)
    psi, xi = x * j, x * (j + 1j * y)
    psid, xid = j + x * jd, (j + 1j * y) + x * (jd + 1j * yd)
    an, bn = psid / xid, psi / xi
    Qb = abs(np.sum((2 * n + 1) * (-1) ** n * (an - bn))) ** 2 / x ** 2
    return Qb * np.pi * a ** 2, x


if __name__ == "__main__":
    a, f = float(sys.argv[1]), float(sys.argv[2])
    s, x = rcs_mono(a, f)
    print(f"ka={x:.4f} sigma={s:.6f} m^2 = {10 * np.log10(s):.3f} dBsm, sigma/(pi a^2)={s / (np.pi * a * a):.5f}")
    for xx in (0.1, 50.0):
        print(" check", xx, rcs_mono(xx / (2 * np.pi * 1e9 / 299792458.0), 1e9)[0] / (np.pi * (xx / (2 * np.pi * 1e9 / 299792458.0)) ** 2))
