import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))  # noqa: E401
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from t_stress import setup
shape, hc = sys.argv[1], float(sys.argv[2])
g, hcur = setup(shape, hc)
c = g.X[len(g.X) // 3]
zone = np.linalg.norm(g.X - c, axis=1) < 0.5
h = np.where(zone, 0.5 * hcur, 1.5 * hcur)
with open(sys.argv[3], "wb") as f:
    np.array([len(g.X), len(g.T), len(g.tri), len(g.seg)], np.int32).tofile(f)
    g.X.astype(np.float64).tofile(f); g.vdim.astype(np.int32).tofile(f); g.vtag.astype(np.int32).tofile(f)
    g.T.astype(np.int32).tofile(f); g.tvol.astype(np.int32).tofile(f)
    g.tri.astype(np.int32).tofile(f); g.ftag.astype(np.int32).tofile(f)
    g.seg.astype(np.int32).tofile(f); g.stag.astype(np.int32).tofile(f)
    h.astype(np.float64).tofile(f); zone.astype(np.uint8).tofile(f)
print("written", sys.argv[3])
