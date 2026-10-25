"""Far field surface of a scattering problem: the box between scatterer and boundary stays a closed,
outward oriented surface over the adaptive passes (the passes refine the mesh locally)."""
import json
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_far_field_box_outward_after_local_refinement(tmp_path):
    from hertzon.solver import bridge
    d = json.loads((EXAMPLES / "SphereRCS.hzn").read_text(encoding="utf-8"))
    design = d["designs"][0]
    setup = design["setups"][0]
    setup["props"].update(RefinementMethod="AURA (EMerge)", MaximumPasses="3", MinimumPasses="3",
                          MinimumConvergedPasses="3", MaximumDeltaEnergy="1e-9")
    proj = tmp_path / "sphere.hzn"
    proj.write_text(json.dumps(d), encoding="utf-8")
    assert bridge.run(str(proj), design["name"], setup["name"], str(tmp_path / "out")) == 0
    h = np.load(tmp_path / "out" / "huygens_LastAdaptive.npz")
    wns, vis = h["wns"], h["vis"]
    side = np.ptp(vis, axis=1)
    area = 2 * (side[0] * side[1] + side[1] * side[2] + side[0] * side[2])
    # closed surface: the area vectors add up to zero; outward: they point away from the box centre
    assert np.linalg.norm(wns.sum(axis=1)) < 1e-6 * area
    c = (vis.min(axis=1) + vis.max(axis=1)) / 2
    assert np.all(np.sum(wns * (vis - c[:, None]), axis=0) > 0)
    # lossless scatterer: the scattered power leaves the box
    assert float(h["Prad_0_1"][0]) > 0
