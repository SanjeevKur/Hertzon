import copy

import numpy as np

from hertzon.kernel import occ as K
from hertzon.model.design import Project, Assignment, FaceRef
from hertzon.model.facetrack import track_faces


def test_assignment_follows_renumbered_face():
    p = Project()
    d = p.designs[0]
    d.add_object("CreateBox", Position="0,0,0", XSize="2mm", YSize="1mm", ZSize="1mm")
    K.ensure_gmsh("facetrack_test")
    t = K.tessellate(K.build_design(d, d.evaluator(p)))
    ix = int(np.argmax([c[0] for c in t["Box1"].face_center]))
    a = Assignment("P1", "Wave Port", {}, faces=[FaceRef("Box1", ix)])
    d.excitations.append(a)
    track_faces(d, t)
    t2 = copy.deepcopy(t)
    tb = t2["Box1"]
    perm = list(range(len(tb.face_area)))[::-1]
    inv = np.argsort(perm)
    for k in ("face_area", "face_normal", "face_center"):
        setattr(tb, k, [getattr(tb, k)[i] for i in perm])
    tb.tri_face = inv[tb.tri_face]
    track_faces(d, t2)
    assert a.faces[0].face == inv[ix]
