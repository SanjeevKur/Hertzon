"""A wave port on a face of a PEC solid (the bottom of a coax bore in a metal body) is backed by the metal:
the solid becomes a void for the solve, so the port face keeps its field and the coax mode is found."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

em = pytest.importorskip("emerge")

from hertzon.kernel import occ as K  # noqa: E402
from hertzon.model.design import Assignment, Command, FaceRef, Project, Setup  # noqa: E402
from hertzon.model.facetrack import track_faces  # noqa: E402
from hertzon.solver import bridge as B  # noqa: E402


def test_port_at_the_bottom_of_a_bore_finds_the_coax_mode(tmp_path):
    p = Project()
    d = p.designs[0]
    d.add_object("CreateBox", "Body", Position="0,0,0", XSize="10mm", YSize="10mm", ZSize="6mm").material = "pec"
    d.add_object("CreateCylinder", "Bore", CenterPosition="5mm,5mm,3mm", Axis="Z", Radius="1mm", Height="3mm")
    d.add_object("CreateCylinder", "Pin", CenterPosition="5mm,5mm,3mm", Axis="Z", Radius="0.43mm", Height="6mm")
    d.obj("Bore").consumed_by = d.obj("Pin").consumed_by = "Body"
    d.obj("Body").history += [Command.new("Subtract", Tools="Bore", KeepOriginals="False"),
                              Command.new("Unite", Tools="Pin", KeepOriginals="False")]
    d.add_object("CreateBox", "Air", Position="-5mm,-5mm,-5mm", XSize="20mm", YSize="20mm", ZSize="20mm")
    K.ensure_gmsh("port_on_metal")
    t = K.tessellate(K.build_design(d, d.evaluator(p)))
    tb = t["Body"]
    c = np.asarray(tb.face_center) * 1e3        # the kernel works in metres
    bottom = [i for i in range(len(c)) if np.allclose(c[i], (5, 5, 3), atol=1e-3) and abs(tb.face_normal[i][2]) > 0.99]
    assert len(bottom) == 1                     # the annulus between bore and pin
    a = Assignment("1", "Wave Port", {"PortNumber": "1", "NumberOfModes": "1", "ModeType": "Auto",
                                      "ImpedanceDefinition": "Zpv", "Renormalize": "False",
                                      "RenormImpedance": "50ohm", "Deembed": "False", "DeembedDistance": "0"},
                   faces=[FaceRef("Body", bottom[0])])
    d.excitations.append(a)
    track_faces(d, t)
    st = Setup("Setup1")
    st.props.update(Frequency="10GHz", LambdaRefinement="False", CurvedElements="False")
    d.setups.append(st)
    b = B.Bridge(p, d, st, tmp_path)
    with em.Simulation("port_on_metal", save_file=False) as m:
        b.prepare(m)
        m.mw._initialize_bc_data()
        m.mw._compute_modes(10e9)
        from emerge._emerge.physics.microwave.bcs.port_bcs import ModalPort
        port = m.mw.bc.oftype(ModalPort)[0]
        modes = port.get_modes(2 * np.pi * 10e9 / 299792458)
        # coax with b/a = 1/0.43 in vacuum: 60 ln(b/a) = 50.6 ohm
        assert len(modes) == 1 and abs(modes[0].Z0 - 50.6) < 5
        assert b.voided == {"Body"}
