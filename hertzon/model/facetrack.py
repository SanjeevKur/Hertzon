"""Keep face assignments on their faces when the model changes (keeps face IDs).

Faces are stored as object + boundary index. A change of dimensions can renumber the faces of an
object; every reference therefore also stores a geometric signature (centre, normal, area) and is
moved to the face that matches it best after each rebuild."""
from __future__ import annotations

import math

import numpy as np


def _flatness(t, i: int) -> float:
    """1 for a plane face, about 0 for a closed curved face (cylinder side)."""
    T = t.tris[t.tri_face == i]
    if len(T) == 0:
        return 1.0
    X = t.points
    cr = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])
    return float(np.linalg.norm(cr.sum(0)) / max(np.linalg.norm(cr, axis=1).sum(), 1e-300))


def _sig(t, i: int) -> list[float]:
    c = t.face_center[i]
    n = t.face_normal[i]
    return [float(c[0]), float(c[1]), float(c[2]), float(n[0]), float(n[1]), float(n[2]), float(t.face_area[i]),
            _flatness(t, i)]


def _score(sig, t, i: int, diag: float) -> float:
    c = np.asarray(t.face_center[i], float)
    n = np.asarray(t.face_normal[i], float)
    a = float(t.face_area[i])
    dot = float(n @ np.asarray(sig[3:6], float))
    if sig[7] < 0.9:          # curved face: its mean normal says little, compare flatness instead
        dot = 1.0 - abs(_flatness(t, i) - sig[7])
    if dot < 0.98:            # different orientation: not the same face
        return math.inf
    dist = float(np.linalg.norm(c - np.asarray(sig[:3], float))) / max(diag, 1e-30)
    area = abs(math.log(max(a, 1e-300) / max(sig[6], 1e-300)))
    return dist + 0.5 * area + (1 - dot)


def face_refs(design):
    """All FaceRefs of a design that name a single face, with the owner's name."""
    for lst in (design.boundaries, design.excitations, design.mesh_ops):
        for a in lst:
            for f in a.faces:
                yield a.name, f
    for fp in getattr(design, "field_plots", []):
        for f in fp.faces:
            yield fp.name, f


def track_faces(design, tess: dict) -> list[str]:
    """Update signatures and move references whose face was renumbered. Returns messages."""
    msgs = []
    for owner, f in face_refs(design):
        if f.face < 0:
            continue
        t = tess.get(f.obj)
        if t is None or not getattr(t, "face_area", None) or len(t.face_area) == 0:
            continue
        nf = len(t.face_area)
        if len(f.sig) != 8:
            if f.face < nf:
                f.sig = _sig(t, f.face)
            continue
        b = np.asarray(t.bbox, float)
        diag = float(np.linalg.norm(b[3:] - b[:3])) or 1.0
        scores = [_score(f.sig, t, i, diag) for i in range(nf)]
        best = int(np.argmin(scores))
        cur = scores[f.face] if f.face < nf else math.inf
        if math.isinf(scores[best]):
            msgs.append(f"{owner}: face {f.face} of '{f.obj}' no longer found; check the assignment")
            continue
        # keep the stored index unless another face matches clearly better
        if f.face >= nf or cur > scores[best] + 1e-3:
            msgs.append(f"{owner}: face of '{f.obj}' renumbered {f.face} -> {best}, assignment follows it")
            f.face = best
        f.sig = _sig(t, f.face)
    return msgs
