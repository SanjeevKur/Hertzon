"""Export a design as a stand-alone Python script driving EMerge.

The script embeds the design (as Hertzon JSON), rebuilds the EMerge model with
the same translation the GUI uses and then either runs the setup or leaves the
``Simulation`` object ``m`` for custom scripting.
"""
from __future__ import annotations

import json

from ..model.design import Project

TEMPLATE = '''#!/usr/bin/env python3
"""EMerge script exported from Hertzon: project {project!r}, design {design!r}.

Usage:  python {{this file}} [SETUP] [--prepare-only]
Requires: pip install emerge hertzon (or run with Hertzon on PYTHONPATH).
"""
import json
import sys
import tempfile
from pathlib import Path

import emerge as em
from hertzon.model.design import project_from_dict
from hertzon.solver.bridge import Bridge
from hertzon.solver.results import results_dir

PROJECT = json.loads(r"""{data}""")
DESIGN = {design!r}
SETUP = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else {setup!r}
PREPARE_ONLY = "--prepare-only" in sys.argv

proj = project_from_dict(PROJECT)
proj.path = str(Path(__file__).with_suffix(".hzn").resolve())
des = proj.design(DESIGN)
setup = des.setup(SETUP)
out = results_dir(proj.path, DESIGN, SETUP)
out.mkdir(parents=True, exist_ok=True)

with em.Simulation("hertzon_script", save_file=False) as m:
    bridge = Bridge(proj, des, setup, out)
    bridge.prepare(m)        # geometry, materials, boundaries, ports, initial mesh
    # --- the EMerge model is ready here: m, m.mw.bc, m.mesher ... -------------------
    # e.g. m.view(); data = m.mw.run_sweep()
    if not PREPARE_ONLY:
        bridge.solve()       # adaptive passes + sweeps exactly like the GUI
        print("Results written to", out)
'''


def design_to_script(project: Project, design: str) -> str:
    des = project.design(design)
    data = json.dumps(project.to_dict(), indent=1)
    setup = des.setups[0].name if des.setups else "Setup1"
    return TEMPLATE.format(project=project.name, design=design, data=data, setup=setup)
