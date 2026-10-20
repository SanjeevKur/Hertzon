import sys, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mie_pec import rcs_mono
# sphere radius in m (RCS_RADIUS, default the 25 cm sphere) and frequency
ref, _ = rcs_mono(float(os.environ.get("RCS_RADIUS", "0.25")), float(os.environ.get("RCS_FREQ", "1e9")))
for d in sys.argv[1:]:
    ff = np.load(d + "/farfield_Infinite Sphere1_LastAdaptive.npz")
    th, ph = [float(x) for x in ff["mono_dir"]]
    E0 = float(np.ravel(ff["E0"])[0]) if "E0" in ff else 1.0
    ks = sorted(int(k.split("_")[1]) for k in ff.files if k.startswith("Emono_"))
    for k in ks:
        E = ff[f"Emono_{k}"]
        s = 4 * np.pi * np.sum(np.abs(E) ** 2) / E0 ** 2
        print(f"{d:22s} f={float(ff[f'freq_{k}'][0])/1e9:.3f} GHz  RCS={s:.5f} m2 = {10*np.log10(s):7.3f} dBsm   Mie {10*np.log10(ref):7.3f} dBsm   err {10*np.log10(s/ref):+.3f} dB")
