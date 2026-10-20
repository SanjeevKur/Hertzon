"""Repeat a failed hzmesh refinement from its dump: python -m hertzon.mesher.replay hzmesh_failure_*.npz

The dump (adapt.save_failure) holds the input mesh, the size field, the parameters and the CAD
projections of the failed run, so the refinement runs the same way without the CAD model."""
from __future__ import annotations

import json
import sys

import numpy as np

from .core import Mesh


def load(path: str):
    d = np.load(path, allow_pickle=False)
    m = Mesh(d["X"], d["vdim"], d["vtag"], d["T"], d["tvol"], d["tri"], d["ftag"], d["seg"], d["stag"])
    for dim, tag in d["curved"]:
        m.set_curved(int(dim), int(tag))
    table = {tuple(k): tuple(v) for k, v in zip(d["proj_in"].tolist(), d["proj_out"].tolist())}
    missing = []

    def proj(dim, tag, p):
        q = table.get((float(dim), float(tag), *map(float, p)))
        if q is None:
            missing.append((dim, tag, p))
            return p
        return q
    if len(d["curved"]):
        m.set_projector(proj)
    return m, d, json.loads(str(d["params"])), missing


def replay(path: str, **override) -> tuple[int, str, dict]:
    base, d, params, missing = load(path)
    mm = base.clone()
    mm.set_size(d["h"], d["zone"])
    st = mm.refine(**dict(params, **override))
    n, msg = mm.check()
    st["projections_missing"] = len(missing)
    return n, msg, st


def main() -> int:
    n, msg, st = replay(sys.argv[1])
    print(f"stored failure: {str(np.load(sys.argv[1])['msg']).strip()[:300]}")
    print(f"replay: {n} problems{':' if n else ''} {msg.strip()[:600]}")
    print({k: st[k] for k in ("ins", "seg", "face", "tet", "fail", "newtets", "projections_missing") if k in st})
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
