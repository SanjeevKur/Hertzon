"""Run a Hertzon setup headless with timers around the expensive phases."""
import sys, time, json, collections
import os
# Hertzon root: 5th argument, else three folders up (examples/benchmark/tools)
sys.path.insert(0, sys.argv[5] if len(sys.argv) > 5 else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
T = collections.defaultdict(float)
C = collections.defaultdict(int)
PASS = []

def wrap(obj, name, label):
    f = getattr(obj, name)
    def w(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            dt = time.perf_counter() - t
            T[label] += dt; C[label] += 1
            PASS.append((label, dt))
    setattr(obj, name, w)

from hertzon.solver import emerge_patches
emerge_patches.apply()          # before wrapping: the patch reads the source of _pre_update
import gmsh
wrap(gmsh.model.mesh, "generate", "gmsh.generate")
from emerge._emerge import mesh3d
wrap(mesh3d.Mesh3D, "_pre_update", "emerge._pre_update")
from emerge._emerge.physics.microwave import microwave_data
wrap(microwave_data.MWField, "_solution_quality", "error_estimate")
from emerge._emerge.physics.microwave import microwave_3d
wrap(microwave_3d.Microwave3D, "_run_adaptive_mesh", "solve")
from hertzon.mesher import adapt
wrap(adapt, "refine_model", "hzmesh.refine_model")
from hertzon.solver import local_refine
wrap(local_refine, "push_to_gmsh", "push_to_gmsh")
from hertzon.solver import bridge
wrap(bridge.Bridge, "save_mesh", "save_mesh")
wrap(bridge.Bridge, "save_fields", "save_fields")
t0 = time.perf_counter()
rc = bridge.run(*sys.argv[1:5])
tot = time.perf_counter() - t0
print("@@PROF " + json.dumps({"total": tot, "time": T, "count": C, "seq": PASS}))
