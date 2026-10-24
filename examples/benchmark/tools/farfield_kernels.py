"""Far field from stored boundary fields: Hertzon's multi-port kernel against EMerge's Rust kernel.

    python farfield_kernels.py HUYGENS_NPZ [STEP_DEG ...]

For every angular step the far field of all ports is computed once with
hertzon.solver.farfield_post.stratton_chu_multi (phase matrix once for all ports) and once with
emr.sc.stratton_chu_ff per port, as EMerge's MWField.farfield does. Prints the times and the largest
difference relative to the peak field.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main(path: str, steps: list[float]) -> None:
    from emr.sc import stratton_chu_ff
    from hertzon.solver.farfield_post import stratton_chu_multi
    z = np.load(path)
    vis, wns = z["vis"].astype(np.float64), z["wns"].astype(np.float64)
    k0 = 2 * np.pi * float(z["freq_0"][0]) / 299792458.0
    ports = sorted({k.split("_")[2] for k in z.files if k.startswith("E_0_")}, key=int)
    Es = [z[f"E_0_{p}"].astype(np.complex128) for p in ports]
    Hs = [z[f"H_0_{p}"].astype(np.complex128) for p in ports]
    print(f"{len(ports)} ports, {vis.shape[1]} surface points")
    stratton_chu_multi(Es[:1], Hs[:1], vis, wns, np.array([0.1]), np.array([0.1]), k0)   # compile
    for st in steps:
        th = np.radians(np.arange(0, 180 + 1e-9, st))
        ph = np.radians(np.arange(0, 360 + 1e-9, st))
        T, P = (a.ravel() for a in np.meshgrid(th, ph, indexing="ij"))
        t = time.perf_counter()
        A = stratton_chu_multi(Es, Hs, vis, wns, T, P, k0)
        ta = time.perf_counter() - t
        t = time.perf_counter()
        B = [stratton_chu_ff(E, H, vis, wns, np.array([T, P]), np.float64(k0))[0] for E, H in zip(Es, Hs)]
        tb = time.perf_counter() - t
        d = max(np.max(np.abs(a - b)) / np.max(np.abs(b)) for a, b in zip(A, B))
        print(f"step {st:g} deg, {T.size} directions: Hertzon multi-port {ta:.3f} s, "
              f"EMerge Rust per port {tb:.3f} s ({tb / ta:.1f}x), max rel. difference {d:.1e}")


if __name__ == "__main__":
    main(sys.argv[1], [float(x) for x in sys.argv[2:]] or [5.0, 2.0, 1.0])
