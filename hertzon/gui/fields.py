"""Field overlays and mesh plots on top of the modeler view."""
from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk

from ..solver.results import ResultStore, results_dir
from .doc import result_dir

QUANTITIES = ["Mag_E", "ComplexMag_E", "Vector_E", "Mag_H", "ComplexMag_H", "Vector_H",
              "Mag_Jsurf", "ComplexMag_Jsurf", "Vector_Jsurf", "Mag_J", "ComplexMag_J", "Vector_J"]


def base_unit(quantity: str) -> str:
    """Unit of a field quantity: E in V/m, H and the surface current Jsurf in A/m, the volume current J in A/m^2."""
    if quantity.endswith("_E"):
        return "V/m"
    return "A/m^2" if quantity.endswith("_J") else "A/m"

_cache: dict[str, pv.UnstructuredGrid] = {}


def _bar(quantity: str, unit: str) -> dict:
    return {"title": f"{quantity} [{unit}]\n", "vertical": True, "position_x": 0.86, "position_y": 0.12,
            "width": 0.05, "height": 0.55, "fmt": "%.3g", "title_font_size": 14, "label_font_size": 12}


def _volume_points(work: pv.UnstructuredGrid, surf: pv.PolyData, nmax: int = 3000, h: float = 0.0) -> pv.PolyData:
    """Regular sample points inside a closed object surface (for vector plots in volumes)."""
    b = np.array(surf.bounds).reshape(3, 2)
    ext = np.maximum(b[:, 1] - b[:, 0], 1e-12)
    h = h or (np.prod(ext) / nmax) ** (1 / 3)
    axes = [np.linspace(lo + 0.5 * (hi - lo) / max(1, round((hi - lo) / h)), hi, max(1, round((hi - lo) / h)),
                        endpoint=False) if hi - lo > 1e-12 else np.array([lo]) for lo, hi in b]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    pts = pv.PolyData(np.c_[X.ravel(), Y.ravel(), Z.ravel()])
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inside = pts.select_enclosed_points(surf.triangulate().clean(), tolerance=1e-6, check_surface=False)
        pts = pts.extract_points(inside.point_data["SelectedPoints"].astype(bool), adjacent_cells=False)
        pts = pv.PolyData(pts.points)
    except Exception:  # noqa: BLE001
        pass
    out = pts.sample(work)
    ok = out.point_data["vtkValidPointMask"].astype(bool)
    res = pv.PolyData(out.points[ok]).sample(work)
    res.field_data["regular"] = np.array([h])
    return res


def _line_points(lines: pv.PolyData, h: float) -> pv.PolyData:
    """Points with spacing h along the polylines of lines (arc length), joined by line cells."""
    pts, conn, off = [], [], 0
    cells = np.asarray(lines.lines)
    P = np.asarray(lines.points, float)
    i = 0
    while i < len(cells):
        n = int(cells[i])
        Q = P[cells[i + 1:i + 1 + n]]
        i += n + 1
        if n < 2:
            continue
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        if s[-1] <= 0:
            continue
        k = max(1, int(round(s[-1] / h)))
        t = (np.arange(k) + 0.5) * s[-1] / k        # centred: no arrows sitting on the end points
        X = np.c_[[np.interp(t, s, Q[:, j]) for j in range(3)]].T
        pts.append(X)
        conn.append(np.r_[len(X), np.arange(off, off + len(X))])
        off += len(X)
    if not pts:
        return pv.PolyData(np.zeros((0, 3)))
    out = pv.PolyData(np.vstack(pts), lines=np.concatenate(conn))
    out.field_data["regular"] = np.array([h])
    out.field_data["fromgrid"] = np.array([1])
    return out


def _cell_lengths(lines: pv.PolyData) -> list[float]:
    """Arc length of every polyline cell (each gets its own row of arrows in _line_points)."""
    cells, P, out, i = np.asarray(lines.lines), np.asarray(lines.points, float), [], 0
    while i < len(cells):
        n = int(cells[i])
        if n > 1:
            out.append(float(np.linalg.norm(np.diff(P[cells[i + 1:i + 1 + n]], axis=0), axis=1).sum()))
        i += n + 1
    return [x for x in out if x > 0]


LINE_AUTO_ARROWS = 20


def plot_line_cells(view, objects, faces=(), plane: str = "") -> list[float]:
    """Cell lengths (m) of the polylines of a plot drawn on lines only; [] for any other plot."""
    if faces or plane or not objects:
        return []
    out = []
    for o in objects:
        a = view.actors.get(o)
        if a is None or a.mesh is not None or a.edge_mesh is None:
            return []
        out += _cell_lengths(a.edge_mesh)
    return out


def line_spacing(cells: list[float], spacing_m: float, count: int) -> float:
    """Arrow spacing (m) on lines: the given spacing, else the total length over the number of arrows."""
    if spacing_m > 0:
        return spacing_m
    return sum(cells) / max(1, count if count > 0 else LINE_AUTO_ARROWS)


def line_arrow_count(cells: list[float], h: float) -> int:
    """Number of arrows _line_points puts on these cells with spacing h."""
    return sum(max(1, int(round(c / h))) for c in cells) if h > 0 else 0


def _line_length(lines: pv.PolyData) -> float:
    cells, P, L, i = np.asarray(lines.lines), np.asarray(lines.points, float), 0.0, 0
    while i < len(cells):
        n = int(cells[i])
        L += float(np.linalg.norm(np.diff(P[cells[i + 1:i + 1 + n]], axis=0), axis=1).sum()) if n > 1 else 0.0
        i += n + 1
    return L


def _regular_points(surf: pv.DataSet, h: float) -> pv.PolyData:
    """Equidistant points with spacing h on a surface: a grid aligned to multiples of h for planar
    surfaces / cut planes, uniform surface sampling otherwise."""
    surf = surf.extract_surface() if not isinstance(surf, pv.PolyData) else surf
    tri = surf.triangulate()
    if tri.n_cells == 0:
        return pv.PolyData(np.zeros((0, 3)))
    P3 = tri.points[tri.faces.reshape(-1, 4)[:, 1:]]                      # (F, 3, 3)
    cr = np.cross(P3[:, 1] - P3[:, 0], P3[:, 2] - P3[:, 0])
    ar = np.linalg.norm(cr, axis=1)
    good = ar > 1e-9 * max(float(ar.max()), 1e-300)
    nrm = cr[good] / ar[good, None]
    n0 = nrm[np.argmax(ar[good])]
    if np.all(np.abs(nrm @ n0) > 0.999):
        n0 = n0 / np.linalg.norm(n0)
        u = np.cross(n0, [1.0, 0, 0] if abs(n0[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        # align the grid with the global axes where possible (cut planes, axis-parallel faces)
        ax = np.eye(3)[np.argsort(np.abs(n0))]
        for cand in ax:
            if abs(cand @ n0) < 1e-6:
                u = cand
                break
        v = np.cross(n0, u)
        P = tri.points
        o = P[0] @ n0
        a, b = P @ u, P @ v
        ga = np.arange(np.floor(a.min() / h) * h + h / 2, a.max(), h)
        gb = np.arange(np.floor(b.min() / h) * h + h / 2, b.max(), h)
        if len(ga) * len(gb) > 200000:
            return _regular_points(surf, h * np.sqrt(len(ga) * len(gb) / 200000))
        A, B = np.meshgrid(ga, gb, indexing="ij")
        pts = A.ravel()[:, None] * u + B.ravel()[:, None] * v + o * n0
        if not len(pts):
            return pv.PolyData(np.zeros((0, 3)))
        _, closest = tri.find_closest_cell(pts, return_closest_point=True)
        keep = np.linalg.norm(closest - pts, axis=1) < 1e-3 * h
        return pv.PolyData(pts[keep])
    smp = vtk.vtkPolyDataPointSampler()
    smp.SetInputData(tri)
    smp.SetDistance(h)
    smp.GenerateVertexPointsOff()
    smp.GenerateEdgePointsOff()
    smp.GenerateInteriorPointsOn()
    try:
        smp.SetPointGenerationModeToRegular()
    except AttributeError:
        pass
    smp.Update()
    return pv.PolyData(pv.wrap(smp.GetOutput()).points)


def store_for(doc, solution: str) -> tuple[ResultStore, str]:
    setup, _, sol = [x.strip() for x in solution.partition(":")]
    return ResultStore(result_dir(doc, setup)), sol


def field_solutions(doc) -> list[str]:
    out = []
    if not doc.project.path:
        return out
    for s in doc.design.setups:
        st = ResultStore(result_dir(doc, s.name))
        names = {p.stem[len("fields_"):] for p in st.folder.glob("fields_*.npz")}
        names |= {p.stem[len("solution_"):] for p in st.folder.glob("solution_*.npz")}   # fields on demand
        for n in sorted(n for n in names if not n.endswith(".tmp")):
            out.append(f"{s.name} : {n}")
    return out


def source_vector(doc, st, fields: dict, grid, which: str, fi: int, port: int) -> np.ndarray:
    """Field of the excitation as set in Edit Sources: port fields superposed with
    sqrt(P) exp(j phase); an incident plane wave scaled to its V/m and shown as total, scattered
    or incident field."""
    from ..model import sources as SRC, planewave
    srcs = SRC.get(doc.design)
    pmap = st.read_json("ports.json", []) or []
    pw = next((p for p in pmap if p.get("kind") == "Incident Plane Wave"), None)
    key = f"{which}_{fi}_{port}"
    if pw is not None:
        tot = fields[key].astype(np.complex128)
        s = (srcs or {}).get(pw["name"], {"mag": 1.0, "phase": 0.0})
        ev = doc.evaluator()
        try:
            a = doc.design.assignment(pw["name"])
            props = a.props
            e_solved = abs(ev.value(props.get("EMagnitude", "1") or "1")) or 1.0
        except Exception:  # noqa: BLE001
            props, e_solved = {}, 1.0
        rot = np.exp(1j * np.radians(s["phase"]))
        tot = tot * (s["mag"] / e_solved) * rot
        ftype = SRC.field_type(doc.design)
        if ftype == "Total":
            return tot
        k, e, origin = planewave.vectors(props, ev)
        f = float(fields[f"freq_{fi}"][0])
        k0 = 2 * np.pi * f / 299792458.0
        r = np.asarray(grid.points, float) - origin
        phase = np.exp(-1j * k0 * (r @ k))[:, None]
        E = s["mag"] * rot * phase * e[None, :]
        inc = E if which == "E" else np.cross(k[None, :], E) / 376.730313668
        return inc if ftype == "Incident" else tot - inc
    if srcs is None:
        return fields[key].astype(np.complex128)
    out = None
    for p in pmap:
        s = srcs.get(p.get("name"))
        kk = f"{which}_{fi}_{p.get('number')}"
        if not s or s["mag"] == 0 or kk not in fields:
            continue
        v = fields[kk].astype(np.complex128) * np.sqrt(s["mag"]) * np.exp(1j * np.radians(s["phase"]))
        out = v if out is None else out + v
    return out if out is not None else 0 * fields[key].astype(np.complex128)


def load_grid(st: ResultStore, sol: str) -> tuple[pv.UnstructuredGrid, dict]:
    """Solver mesh with the field values. New results store the field at the 4 corners of every
    tetrahedron (discontinuous across materials), old ones at the mesh nodes. Curved tetrahedra
    (curved elements at curved surfaces) get their midpoints on the curved edges."""
    m = st.npz("mesh.npz")
    f = st.npz(f"fields_{sol}.npz")      # computes the fields from the saved solution when needed
    key = f"{st.folder}|{sol}|{(st.folder / f'fields_{sol}.npz').stat().st_mtime if (st.folder / f'fields_{sol}.npz').exists() else 0}"
    if m is None or f is None:
        raise RuntimeError("No field data saved for this solution (enable 'Save fields' in the setup)")
    if key not in _cache:
        nodes, tets = m["nodes"], m["tets"]
        npt = 10 if "tet10" in f else (4 if "tet4" in f else 0)
        if npt and len(f[next(k for k in f if k.startswith("E_"))]) != npt * len(tets):
            raise RuntimeError("Field data does not match the saved mesh (the solve is still writing); try again")
        if npt:
            C = nodes[tets]                                            # (N, 4, 3)
            if npt == 10:
                C = np.concatenate([C, np.stack([(C[:, i] + C[:, j]) / 2 for i, j in _EDGES], axis=1)], axis=1)
                cv = _curv(m)
                if cv is not None:
                    ct = np.where(cv[0] >= 0)[0]
                    C[ct, 4:] += cv[1][cv[0][ct]]
            pts = C.reshape(-1, 3)
            conn = np.arange(npt * len(tets)).reshape(-1, npt)
            ctype = vtk.VTK_QUADRATIC_TETRA if npt == 10 else vtk.VTK_TETRA
        else:
            pts, conn, ctype = nodes, tets, vtk.VTK_TETRA
        cells = np.hstack([np.full((len(conn), 1), conn.shape[1]), conn]).ravel()
        grid = pv.UnstructuredGrid(cells, np.full(len(conn), ctype, np.uint8), pts)
        grid.field_data["npt"] = np.array([npt])
        _cache.clear()
        _cache[key] = (grid, m)
    return _cache[key][0], f


def _mesh_of(grid) -> dict | None:
    for v in _cache.values():
        if v[0] is grid:
            return v[1]
    return None


_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]          # VTK quadratic tetra mid-edge order
_MID = np.full((4, 4), -1)
for _k, (_i, _j) in enumerate(_EDGES):
    _MID[_i, _j] = _MID[_j, _i] = 4 + _k
_FROM_PAIRS = [0, 3, 1, 2, 4, 5]     # mesh.npz "curv_disp" edge order (0,1)(0,2)(0,3)(1,2)(1,3)(2,3) -> _EDGES


def _curv(mesh) -> tuple[np.ndarray, np.ndarray] | None:
    """Curved tetrahedra of a saved mesh: (row of every tet in disp or -1, edge midpoint displacements
    (nc, 6, 3) in _EDGES order); None for a straight mesh."""
    if mesh is None or "curv_tets" not in mesh:
        return None
    if "_curv" not in mesh:
        ct = np.asarray(mesh["curv_tets"], np.int64)
        pos = np.full(len(mesh["tets"]), -1, np.int64)
        pos[ct] = np.arange(len(ct))
        mesh["_curv"] = (pos, np.asarray(mesh["curv_disp"], float).reshape(-1, 6, 3)[:, _FROM_PAIRS]) \
            if len(ct) else None
    return mesh["_curv"]


def _face_keys(tri: np.ndarray) -> np.ndarray:
    t = np.sort(tri, axis=1).astype(np.int64)
    return (t[:, 0] << 42) | (t[:, 1] << 21) | t[:, 2]


def _all_faces(tets: np.ndarray):
    """Every tet face: (sorted key, owning tet, local corner indices)."""
    n = len(tets)
    tri = tets[:, _FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(n), 4)
    local = np.tile(_FACES, (n, 1))
    return _face_keys(tri), owner, local


def _face_disp(mesh, owner, local) -> np.ndarray | None:
    """Midpoint displacements (nf, 3, 3) of the face edges ab, bc, ac (local corners a b c of the owning
    tet); None when no face is curved."""
    cv = _curv(mesh)
    if cv is None:
        return None
    p = cv[0][owner]
    s = p >= 0
    if not s.any():
        return None
    out = np.zeros((len(owner), 3, 3))
    a, b, c = local[s, 0], local[s, 1], local[s, 2]
    for j, (u, v) in enumerate(((a, b), (b, c), (a, c))):
        out[s, j] = cv[1][p[s], _MID[u, v] - 4]
    return out


def _face_six(mesh, owner, local) -> tuple[np.ndarray, np.ndarray]:
    """Geometry of mesh faces as quadratic triangles: nodes (nf, 6, 3) in the order a b c ab bc ac
    (midpoints on the curved edges) and a mask of the curved faces."""
    tri = np.take_along_axis(mesh["tets"][owner], local, axis=1)
    X = np.asarray(mesh["nodes"], float)[tri]
    P6 = np.concatenate([X, (X[:, [0, 1, 0]] + X[:, [1, 2, 2]]) / 2], axis=1)
    d = _face_disp(mesh, owner, local)
    if d is None:
        return P6, np.zeros(len(P6), bool)
    P6[:, 3:] += d
    return P6, np.abs(d).sum((1, 2)) > 0


_SUB: dict = {}


def _subdiv(n: int):
    """Reference triangle split into n*n triangles: quadratic shape functions N and their u, v derivatives
    (m, 6) at the m sub-points (node order a b c ab bc ac), and the sub-triangles (n*n, 3)."""
    if n not in _SUB:
        ij = [(i, j) for j in range(n + 1) for i in range(n + 1 - j)]
        idx = {p: k for k, p in enumerate(ij)}
        tris = []
        for j in range(n):
            for i in range(n - j):
                tris.append((idx[i, j], idx[i + 1, j], idx[i, j + 1]))
                if i + j < n - 1:
                    tris.append((idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]))
        u = np.array([p[0] for p in ij], float) / n
        v = np.array([p[1] for p in ij], float) / n
        a, b, c = 1 - u - v, u, v
        N = np.stack([a * (2 * a - 1), b * (2 * b - 1), c * (2 * c - 1), 4 * a * b, 4 * b * c, 4 * a * c], 1)
        Nu = np.stack([1 - 4 * a, 4 * b - 1, 0 * u, 4 * (a - b), 4 * c, -4 * c], 1)
        Nv = np.stack([1 - 4 * a, 0 * u, 4 * c - 1, -4 * b, 4 * b, 4 * (a - c)], 1)
        _SUB[n] = (N, Nu, Nv, np.array(tris, np.int64))
    return _SUB[n]


CURVED_SPLIT = 4      # a curved face is drawn as CURVED_SPLIT^2 triangles


def _patches(P6: np.ndarray, vals: dict, curved: np.ndarray, quad: bool) -> pv.PolyData:
    """Triangles of quadratic faces P6 (nf, 6, 3) with values (nf, 6, ...) at the same nodes. Curved faces
    are split into CURVED_SPLIT^2 triangles, flat ones into 4 (quad: values quadratic) or 1; every point
    carries the exact normal of its patch ("Normals", so the smooth shading shows the round surface).
    Cell data "_face": face index of every triangle. The face edges go to field data "grid_*"."""
    pts, tris, nrm, fidx = [], [], [], []
    out = {k: [] for k in vals}
    off = 0
    for sel, n in ((~curved, 2 if quad else 1), (curved, CURVED_SPLIT)):
        f = np.where(sel)[0]
        if not len(f):
            continue
        N, Nu, Nv, T = _subdiv(n)
        m = len(N)
        X = np.einsum("mk,fkd->fmd", N, P6[f])
        nv = np.cross(np.einsum("mk,fkd->fmd", Nu, P6[f]), np.einsum("mk,fkd->fmd", Nv, P6[f]))
        nv /= np.maximum(np.linalg.norm(nv, axis=2, keepdims=True), 1e-300)
        pts.append(X.reshape(-1, 3))
        nrm.append(nv.reshape(-1, 3))
        tris.append((T[None] + off + m * np.arange(len(f))[:, None, None]).reshape(-1, 3))
        fidx.append(np.repeat(f, len(T)))
        for k, v in vals.items():
            out[k].append(np.einsum("mk,fk...->fm...", N, v[f]).reshape((-1,) + v.shape[2:]))
        off += m * len(f)
    if not pts:
        return pv.PolyData()
    tris = np.vstack(tris)
    pts, nrm = np.vstack(pts), np.vstack(nrm)
    if curved.any():
        nrm = _smooth_normals(pts, nrm)
    poly = pv.PolyData(pts, np.hstack([np.full((len(tris), 1), 3), tris]).ravel())
    for k, v in out.items():
        poly.point_data[k] = np.concatenate(v)
    poly.point_data["Normals"] = nrm
    poly.point_data.active_normals_name = "Normals"
    poly.cell_data["_face"] = np.concatenate(fidx)
    gp, gl = _grid_lines(P6, curved)
    poly.field_data["grid_pts"] = gp
    poly.field_data["grid_lines"] = gl
    return poly


def _smooth_normals(pts: np.ndarray, nrm: np.ndarray, max_deg: float = 25.0) -> np.ndarray:
    """Average the normals of coincident points (neighbouring patches of a curved surface) unless they
    meet at a real edge; the patches carry their own orientation, so the signs are aligned first."""
    span = float(np.ptp(pts, axis=0).max()) or 1.0
    _, inv = np.unique(np.round(pts / (1e-7 * span)), axis=0, return_inverse=True)
    inv = inv.ravel()
    ref = np.zeros((inv.max() + 1, 3))
    ref[inv] = nrm                                        # one representative per point
    sgn = np.sign(np.einsum("id,id->i", nrm, ref[inv]))
    sgn[sgn == 0] = 1
    acc = np.zeros_like(ref)
    np.add.at(acc, inv, nrm * sgn[:, None])
    avg = acc[inv] * sgn[:, None]
    avg /= np.maximum(np.linalg.norm(avg, axis=1, keepdims=True), 1e-300)
    ok = np.einsum("id,id->i", avg, nrm) > np.cos(np.radians(max_deg))
    out = nrm.copy()
    out[ok] = avg[ok]
    return out


def _grid_lines(P6: np.ndarray, curved: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Edges of the faces as polylines (points, VTK line connectivity); curved edges with k segments."""
    pts, lines = [], []
    off = 0
    for sel, n in ((~curved, 1), (curved, k)):
        f = np.where(sel)[0]
        if not len(f):
            continue
        t = np.linspace(0, 1, n + 1)
        w = np.stack([(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)], 1)     # end a, end b, middle
        for a, b, c in ((0, 1, 3), (1, 2, 4), (0, 2, 5)):
            X = np.einsum("tk,fkd->ftd", w, P6[f][:, [a, b, c]])
            pts.append(X.reshape(-1, 3))
            ids = off + (n + 1) * np.arange(len(f))[:, None] + np.arange(n + 1)
            lines.append(np.hstack([np.full((len(f), 1), n + 1), ids]).ravel())
            off += (n + 1) * len(f)
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    return np.vstack(pts), np.concatenate(lines).astype(np.int64)


def grid_mesh(surf) -> pv.PolyData | None:
    """Solver mesh lines (element edges, curved where the elements are) of a plotted surface."""
    if surf is None or "grid_pts" not in surf.field_data:
        return None
    gp = np.asarray(surf.field_data["grid_pts"]).reshape(-1, 3)
    return pv.PolyData(gp, lines=np.asarray(surf.field_data["grid_lines"], np.int64)) if len(gp) else None


def _surface(work, mesh: dict, keys_sel, owner, local, both: bool, toward_owner: bool = True) -> pv.PolyData:
    """Triangles with the values of their owning tetrahedra (6-point faces split into 4 for quadratic data,
    curved faces finer). both=True averages the two tets sharing a face (sheet objects). The normals point
    into the owning tet (toward_owner) or out of it, so J = n x H has one orientation on the whole surface."""
    tets = mesh["tets"]
    npt = int(work.field_data["npt"][0])
    # order the corners of every face by global node id, so both owners of a face list them alike
    g = np.take_along_axis(tets[owner], local, axis=1)
    local = np.take_along_axis(local, np.argsort(g, axis=1), axis=1)
    order = np.argsort(keys_sel, kind="stable")
    k, o, l = keys_sel[order], owner[order], local[order]
    idx = np.where(np.r_[True, k[1:] != k[:-1]])[0]

    def point_ids(oo, ll):
        a, b, c = ll[:, 0], ll[:, 1], ll[:, 2]
        cols = [a, b, c] if npt != 10 else [a, b, c, _MID[a, b], _MID[b, c], _MID[a, c]]
        return npt * oo[:, None] + np.stack(cols, axis=1)

    ids = point_ids(o[idx], l[idx])
    names = [n for n in ("re", "qu", "cmag") if n in work.point_data]
    vals = {n: work.point_data[n][ids].astype(float) for n in names}
    if both:
        nxt = np.minimum(idx + 1, len(k) - 1)
        has2 = (idx + 1 < len(k)) & (k[nxt] == k[idx])
        # field jump across the sheet: a PEC sheet carries J = n x (H1 - H2); averaging both sides
        # would cancel the currents on top and bottom (blue triangles along strip edges)
        jumps = {"re": "jump", "qu": "jumpq"}
        for n, j in jumps.items():
            if n in vals:
                vals[j] = vals[n].copy()
        if has2.any():
            ids2 = point_ids(o[nxt[has2]], l[nxt[has2]])
            for n, j in jumps.items():
                if n in vals:
                    vals[j][has2] = vals[n][has2] - work.point_data[n][ids2]
            for n in names:
                vals[n][has2] = 0.5 * (vals[n][has2] + work.point_data[n][ids2])
    P6, curved = _face_six(mesh, o[idx], l[idx])
    if npt != 10:      # linear values: the midpoints take the mean of their corners
        for n, v in vals.items():
            vals[n] = np.concatenate([v, (v[:, [0, 1, 0]] + v[:, [1, 2, 2]]) / 2], axis=1)
    C = np.asarray(mesh["nodes"], float)[tets[o[idx]]].mean(axis=1)
    ng = np.cross(P6[:, 1] - P6[:, 0], P6[:, 2] - P6[:, 0])
    flip = (np.einsum("fd,fd->f", C - P6[:, :3].mean(axis=1), ng) < 0) == toward_owner
    if flip.any():
        swap = [0, 2, 1, 5, 4, 3]                  # a c b ac bc ab: the same face, the other way round
        P6[flip] = P6[flip][:, swap]
        for n, v in vals.items():
            v[flip] = v[flip][:, swap]
    return _patches(P6, vals, curved, quad=npt == 10)


def _object_tets(mesh: dict, obj_index: int) -> np.ndarray:
    """Tetrahedra inside an object's geometry. Where volumes overlap they belong to every object there,
    so a plot shows the whole object; tet_obj holds only the one that sets the material (old results)."""
    if "obj_ptr" in mesh and "obj_tets" in mesh:
        p = mesh["obj_ptr"]
        return np.asarray(mesh["obj_tets"][p[obj_index]:p[obj_index + 1]], np.int64)
    return np.where(mesh["tet_obj"] == obj_index)[0]


def _object_faces(mesh: dict, obj_index: int, outside: bool = False):
    """Boundary faces of an object (keys, owners, local corners). outside=True takes the neighbouring
    tets as owners (conductors: the field inside a PEC is zero, the surface current lives on the outer side)."""
    tets = mesh["tets"]
    sel = _object_tets(mesh, obj_index)
    keys, owner, local = _all_faces(tets[sel])
    owner = sel[owner]
    uk, cnt = np.unique(keys, return_counts=True)
    outer = np.isin(keys, uk[cnt == 1])
    keys, owner, local = keys[outer], owner[outer], local[outer]
    if outside:
        ka, oa, la = _all_faces(tets)
        inside = np.zeros(len(tets), bool)
        inside[sel] = True
        nb = np.isin(ka, keys) & ~inside[oa]
        if nb.any():
            # faces on the outer domain boundary keep the inner side
            rest = ~np.isin(keys, ka[nb])
            keys = np.r_[ka[nb], keys[rest]]
            owner = np.r_[oa[nb], owner[rest]]
            local = np.r_[la[nb], local[rest]]
    return keys, owner, local


def _object_surface(work, mesh: dict, obj_index: int, outside: bool = False) -> pv.PolyData:
    """Boundary triangles of an object with the field values (see _object_faces)."""
    return _surface(work, mesh, *_object_faces(mesh, obj_index, outside), both=False)


def _cad_cells(cad: pv.PolyData):
    """Unit normals, sizes and curvature angles (normal turn from the face to its corners) of the
    triangles of a display mesh."""
    try:
        tri = np.asarray(cad.regular_faces)
    except Exception:  # noqa: BLE001
        tri = np.asarray(cad.faces).reshape(-1, 4)[:, 1:]
    X = np.asarray(cad.points, float)[tri]
    cn = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    cn /= np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-300)
    hd = np.linalg.norm(X - X[:, [1, 2, 0]], axis=2).max(1)
    th = np.zeros(len(tri))
    if "Normals" in cad.point_data:
        pn = np.asarray(cad.point_data["Normals"], float)[tri]
        cos = np.abs(np.einsum("tkd,td->tk", pn, cn)).min(1)
        th = np.arccos(np.clip(cos, 0.0, 1.0))
    return cn, hd, th


def _on_cad(mesh: dict, owner, local, cad: pv.PolyData, face: int = -1) -> np.ndarray:
    """Mask of the mesh faces lying on the display surface cad (face >= 0: on that CAD face of it).
    The display mesh and straight solver faces are chords of a curved surface, so the distance along
    the surface normal may be as large as their sag; along the surface it has to be tiny."""
    if len(owner) == 0 or cad.n_cells == 0:
        return np.zeros(len(owner), bool)
    P6, _ = _face_six(mesh, owner, local)
    cent = (4 * P6[:, 3:].sum(1) - P6[:, :3].sum(1)) / 9           # centre of the (curved) face
    h = np.linalg.norm(P6[:, :3] - P6[:, [1, 2, 0]], axis=2).max(1)
    b = np.array(cad.bounds).reshape(3, 2)
    diag = float(np.linalg.norm(b[:, 1] - b[:, 0])) or 1.0
    near = np.all((cent >= b[:, 0] - h[:, None] - 1e-3 * diag) & (cent <= b[:, 1] + h[:, None] + 1e-3 * diag), axis=1)
    keep = np.zeros(len(owner), bool)
    idx = np.where(near)[0]
    if not len(idx):
        return keep
    cid, cp = cad.find_closest_cell(cent[idx], return_closest_point=True)
    cid = np.asarray(cid, np.int64)
    cn, hd, th = _cad_cells(cad)
    v = cent[idx] - cp
    dn = np.abs(np.einsum("id,id->i", v, cn[cid]))
    dt = np.linalg.norm(v - np.einsum("id,id->i", v, cn[cid])[:, None] * cn[cid], axis=1)
    t, hh, hc = th[cid], h[idx], hd[cid]
    sag = 1.3 * (hc * t / 4 + hh * hh * t / (3 * np.maximum(hc, 1e-300)))
    tol = 1e-4 * diag
    ok = (dn < tol + sag) & (dt < tol + np.where(t > 1e-3, 0.02 * hh, 0.0))
    if face >= 0 and "face" in cad.cell_data:
        ok &= np.asarray(cad.cell_data["face"])[cid] == face
    keep[idx[ok]] = True
    return keep


def _sheet_faces(mesh: dict, cad: pv.PolyData, solid: bool = False, face: int = -1):
    """Mesh faces lying on a CAD surface: a sheet, or the walls of an unmeshed solid (PEC)."""
    keys, owner, local = _all_faces(mesh["tets"])
    if solid:     # walls of a hole: faces with a tet on one side only
        uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        one = cnt[inv] == 1
        keys, owner, local = keys[one], owner[one], local[one]
    on = _on_cad(mesh, owner, local, cad, face)
    return keys[on], owner[on], local[on]


def _sheet_surface(work, mesh: dict, cad: pv.PolyData, solid: bool = False, face: int = -1) -> pv.PolyData:
    """Solver mesh triangles lying on a sheet object or on the walls of an unmeshed solid (values
    averaged over both sides of a sheet)."""
    return _surface(work, mesh, *_sheet_faces(mesh, cad, solid, face), both=True)


def _restrict_faces(mesh: dict, sel, cad: pv.PolyData, face: int):
    """The faces of sel lying on CAD face 'face' of the object display mesh cad."""
    keys, owner, local = sel
    on = _on_cad(mesh, owner, local, cad, face)
    return keys[on], owner[on], local[on]


# Plot attributes like the reference tool's "Modify Attributes" dialog (stored as strings in FieldPlot.props)
ATTR_DEFAULTS = {
    # Color map
    "ColorMap": "Rainbow", "Invert": False, "Colors": 0,
    # Scale
    "Divisions": 10, "Scale": "Auto", "Min": 0.0, "Max": 1.0, "dB": False, "Units": "", "Log": False,
    "LimitPrecision": False, "Precision": 4, "AnimScale": False, "Format": "Automatic",
    # Marker / Arrow
    "ArrowSize": 1.0, "ArrowThickness": 1.0, "MapSize": False, "ArrowTail": True, "ArrowCount": 400, "ArrowSpacing": 0.0,
    "LineArrows": 0,                # arrows on a polyline plot (0: auto); ArrowCount is for surfaces / volumes
    # Plots
    "Transparency": 0.0, "PlotType": "Smooth Shaded", "ShowGrid": False,
}
COLOR_MAPS = {"Rainbow": "jet", "Temperature": "hot", "Magenta": "RdPu", "Gray": "gray", "Turbo": "turbo",
              "Viridis": "viridis", "Cool-Warm": "coolwarm", "Uniform": None}
PLOT_TYPES = ["Flat Shaded", "Smooth Shaded", "Iso Lines", "Shaded + Iso Lines"]
UNITS = {"V/m": [("V/m", 1.0), ("mV/m", 1e3), ("uV/m", 1e6), ("kV/m", 1e-3)],
         "A/m": [("A/m", 1.0), ("mA/m", 1e3), ("uA/m", 1e6), ("kA/m", 1e-3)],
         "A/m^2": [("A/m^2", 1.0), ("mA/m^2", 1e3), ("kA/m^2", 1e-3), ("A/mm^2", 1e-6)]}


def attributes(props: dict | None) -> dict:
    """Typed plot attributes from FieldPlot.props (missing keys take the defaults)."""
    out = {}
    props = props or {}
    for k, d in ATTR_DEFAULTS.items():
        v = props.get(k, d)
        try:
            if isinstance(d, bool):
                v = v if isinstance(v, bool) else str(v) == "True"
            elif isinstance(d, int):
                v = int(float(v))
            elif isinstance(d, float):
                v = float(v)
            else:
                v = str(v)
        except (TypeError, ValueError):
            v = d
        out[k] = v
    if "log" in props and "Log" not in props:        # older projects
        out["Log"] = str(props["log"]) == "True"
    return out


def _qvals(vecs, nrm, at: dict, ufac: float):
    """Colour values of the arrows (magnitude in display units, dB if requested)."""
    v = nrm * ufac
    return 20 * np.log10(np.maximum(v, 1e-30)) if at["dB"] else v


def _unit(name: str, base: str) -> tuple[float, str]:
    for lab, fac in UNITS.get(base, []):
        if lab == name:
            return fac, lab
    return 1.0, base


def _cmap(a: dict):
    name = COLOR_MAPS.get(a["ColorMap"], "jet")
    if name is None:                                  # uniform colour
        from matplotlib.colors import ListedColormap
        return ListedColormap(["#d0342c"])
    return name + "_r" if a["Invert"] else name


def frequencies(fields: dict) -> list[float]:
    return [float(fields[k][0]) for k in sorted((k for k in fields if k.startswith("freq_")),
                                                key=lambda k: int(k.split("_")[1]))]


def ports(fields: dict) -> list[int]:
    return sorted({int(k.split("_")[2]) for k in fields if k.startswith("E_")})


def _refined(cad: pv.PolyData, mesh) -> pv.PolyData:
    """CAD surface subdivided to about the local solver element size, so the field interpolated from the
    solution mesh shows its variation."""
    surf = cad.extract_surface(algorithm="dataset_surface").triangulate().clean()
    try:
        X = np.asarray(mesh["nodes"] if "nodes" in mesh else mesh["points"], float)
        T = np.asarray(mesh["tets"], np.int64)
        if X.shape[0] == 3 and X.shape[1] != 3:
            X = X.T
        if T.shape[0] == 4 and T.shape[1] != 4:
            T = T.T
        c = X[T].mean(axis=1)
        e = np.linalg.norm(X[T[:, 0]] - X[T[:, 1]], axis=1)
        b = surf.bounds
        inside = np.all((c >= np.array(b[::2]) - e.max()) & (c <= np.array(b[1::2]) + e.max()), axis=1)
        h = float(np.median(e[inside] if inside.any() else e)) / 2
    except Exception:  # noqa: BLE001
        h = surf.length / 60
    # never more than about 20 000 triangles: finer than that only slows the plot down (the field
    # comes from the solver mesh anyway, so smaller triangles add no information)
    h = max(h, surf.length / 400, (2.31 * max(surf.area, 0.0) / 8000) ** 0.5)
    try:
        return surf.subdivide_adaptive(max_edge_len=h, max_n_passes=12)
    except Exception:  # noqa: BLE001
        return surf


def _conductivity(doc, grid, freq: float) -> np.ndarray:
    """Conductivity (S/m) at every grid point, for J = sigma E: the bulk conductivity of the object's material
    plus the dielectric loss (omega eps0 eps_r tan d). Perfect conductors carry no volume current (their
    current is Jsurf on the surface)."""
    npt = int(grid.field_data["npt"][0]) if "npt" in grid.field_data else 0
    mesh = _mesh_of(grid)
    if not npt or mesh is None or "tet_obj" not in mesh:
        raise RuntimeError("J needs the per-element field format: analyze again")
    from ..model.materials import is_conductor
    ev = doc.evaluator()
    w = 2 * np.pi * freq * 8.8541878128e-12
    sig = []
    for name in (str(n) for n in mesh["obj_names"]):
        val = 0.0
        ob = doc.design.obj(name) if name and doc.design.has_obj(name) else None
        if ob is not None:
            try:
                mat = doc.project.material(ob.material)
                if mat.pec or (is_conductor(mat) and not ob.solve_inside):
                    val = 0.0
                else:
                    val = float(ev.value(mat.conductivity or "0"))
                    try:
                        val += w * float(ev.value(mat.permittivity or "1")) * float(ev.value(mat.loss_tangent or "0"))
                    except Exception:  # noqa: BLE001
                        pass            # anisotropic / dispersive inputs: the bulk conductivity alone
            except Exception:  # noqa: BLE001
                val = 0.0
        sig.append(val)
    sig = np.r_[np.asarray(sig, float), 0.0]                     # tet_obj -1 (unassigned): index -1 -> 0
    return np.repeat(sig[np.asarray(mesh["tet_obj"], np.int64)], npt)


def make_overlay(doc, view, quantity: str, solution: str, freq_index: int = 0, port: int = 1, phase_deg: float = 0.0,
                 faces=None, objects=None, plane: str = "", log_scale: bool = False, vmin=None, vmax=None,
                 arrow_scale: float = 1.0, attrs: dict | None = None, animating: bool = False):
    """Return (meshes for the viewer, info text). attrs: plot attributes (see ATTR_DEFAULTS)."""
    at = attributes(attrs)
    from ..model.expr import LENGTH_UNITS
    spacing_m = at["ArrowSpacing"] * LENGTH_UNITS.get(doc.design.units, 1e-3) if at["ArrowSpacing"] > 0 else 0.0
    log_scale = log_scale or at["Log"]
    if at["Scale"] == "Use Limits":
        vmin, vmax = at["Min"], at["Max"]
    st, sol = store_for(doc, solution)
    grid, fields = load_grid(st, sol)
    jsurf = "Jsurf" in quantity
    which = "E" if quantity.endswith(("_E", "_J")) else "H"
    key = f"{which}_{freq_index}_{port}"
    if key not in fields:
        raise RuntimeError(f"No {which} field for port {port} at frequency index {freq_index}")
    vec = source_vector(doc, st, fields, grid, which, freq_index, port)
    if quantity.endswith("_J"):
        vec = vec * _conductivity(doc, grid, float(fields[f"freq_{freq_index}"][0]))[:, None]
    work = grid.copy(deep=False)
    ph = np.exp(1j * np.radians(phase_deg))
    work.point_data["re"] = np.real(vec * ph)
    if jsurf and quantity.startswith("ComplexMag"):
        work.point_data["qu"] = np.real(1j * vec * ph)   # the field a quarter period later
    work.point_data["cmag"] = np.sqrt((np.abs(vec) ** 2).sum(axis=1))
    # target surfaces
    targets = []
    mesh = _mesh_of(grid)
    tet4 = int(grid.field_data["npt"][0]) > 0 if "npt" in grid.field_data else False
    names = [str(n) for n in mesh["obj_names"]] if mesh is not None and "obj_names" in mesh else []

    def on_solver_mesh(obj: str, full: pv.PolyData, face: int = -1):
        """Field on the solver mesh triangles of an object or of one of its faces (per-object values),
        else sampled on the CAD mesh."""
        nonlocal note
        cad = full if face < 0 else full.extract_cells(np.where(full.cell_data["face"] == face)[0]).extract_surface()
        if tet4 and names:
            try:
                pec = False
                ob = doc.design.obj(obj) if doc.design.has_obj(obj) else None
                if ob is not None:
                    from ..model.materials import is_conductor
                    try:
                        pec = is_conductor(doc.project.material(ob.material)) and not ob.solve_inside
                    except Exception:  # noqa: BLE001
                        pec = str(ob.material).lower() == "pec"
                if obj in names:
                    sel = _object_faces(mesh, names.index(obj), outside=pec)
                    if face >= 0:
                        sel = _restrict_faces(mesh, sel, full, face)
                    surf = _surface(work, mesh, *sel, both=False, toward_owner=pec)
                elif ob is not None and not ob.model:
                    return _refined(cad, mesh)      # non-model: interpolate onto a fine surface
                else:
                    # sheets, and the walls of unmeshed solids (PEC): faces with one tet only
                    solid = ob is not None and getattr(ob, "dim", 2) == 3
                    sel = _sheet_faces(mesh, full, solid, face)
                    if solid and not len(sel[0]):
                        sel = _sheet_faces(mesh, full, False, face)
                    surf = _surface(work, mesh, *sel, both=True)
                if surf.n_cells:
                    return surf
            except Exception as e:  # noqa: BLE001
                note = f"  (solver-mesh surface failed: {e}; sampled on the CAD surface)"
        return cad

    note = "" if tet4 else "  (old result format: analyze again for an element-accurate plot)"
    line_cells = plot_line_cells(view, objects, faces, plane)
    for f in faces or []:
        a = view.actors.get(f.obj)
        if a is not None and a.mesh is not None:
            targets.append(on_solver_mesh(f.obj, a.mesh, f.face))
    for o in objects or []:
        a = view.actors.get(o)
        if a is not None and a.mesh is None and a.edge_mesh is not None:
            # polyline (or a 1D object): arrows / a coloured line along it
            if jsurf:
                raise RuntimeError(f"'{o}' is a line; {quantity} needs faces or objects with a surface")
            L = _line_length(a.edge_mesh)
            if L <= 0:
                continue
            if quantity.startswith("Vector"):
                # the arrow spacing, else the number of arrows on all lines of the plot (Arrows on line)
                h = line_spacing(line_cells or _cell_lengths(a.edge_mesh), spacing_m, int(at["LineArrows"]))
            else:
                h = L / 600
            targets.append(_line_points(a.edge_mesh, h))
            continue
        if a is not None and a.mesh is not None:
            if quantity.startswith("Vector") and not jsurf and doc.design.obj(o).dim == 3:
                targets.append(_volume_points(work, a.mesh, nmax=max(10, int(at["ArrowCount"])), h=spacing_m))
            else:
                targets.append(on_solver_mesh(o, a.mesh))
    if plane:
        ax, _, val = plane.partition("@")
        normal = {"XY": (0, 0, 1), "YZ": (1, 0, 0), "ZX": (0, 1, 0), "XZ": (0, 1, 0)}[ax]
        b = work.bounds
        center = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
        i = int(np.argmax(np.abs(normal)))
        if val.strip():
            center[i] = float(doc.evaluator().length(val))
        sl = work.slice(normal=normal, origin=center)
        sl.field_data["fromgrid"] = np.array([1])
        targets.append(sl)
    if not targets:
        raise RuntimeError("Select faces, objects or polylines, or give a cut plane")
    meshes = []
    allvals = []
    sampled = []
    for t in targets:
        s = t if "re" in t.point_data else t.sample(work)
        if jsurf:
            s = s.compute_normals(cell_normals=False, auto_orient_normals=False) if "Normals" not in s.point_data else s
            n = s.point_data["Normals"]
            H = s.point_data["jump"] if "jump" in s.point_data else s.point_data["re"]
            J = np.cross(n, H)
            mag = np.linalg.norm(J, axis=1)
            if quantity.startswith("ComplexMag"):
                Hq = s.point_data["jumpq"] if "jumpq" in s.point_data else s.point_data["qu"]
                mag = np.sqrt(mag ** 2 + (np.cross(n, Hq) ** 2).sum(axis=1))
            s.point_data["re"] = J                      # Vector_Jsurf: the arrows show J = n x H
        elif quantity.startswith("ComplexMag"):
            mag = s.point_data["cmag"]
        else:
            mag = np.linalg.norm(s.point_data["re"], axis=1)
        if at["PlotType"] != "Flat Shaded" and s.n_points:
            # local smooth plot: average the per-element values at shared nodes
            key = np.round(np.asarray(s.points) / (1e-9 + 1e-7 * float(np.ptp(s.points, axis=0).max())))
            _, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.ravel()
            good = np.isfinite(mag)
            acc = np.bincount(inv[good], mag[good], minlength=inv.max() + 1)
            cnt = np.bincount(inv[good], minlength=inv.max() + 1)
            mag = np.where(good, acc[inv] / np.maximum(cnt[inv], 1), mag)
        s.point_data[quantity] = mag
        sampled.append(s)
        allvals.append(mag[np.isfinite(mag)])
    unit = base_unit(quantity)
    ufac, ulabel = _unit(at["Units"], unit)
    title = quantity
    for s in sampled:
        v = s.point_data[quantity] * ufac
        if at["dB"]:
            v = 20 * np.log10(np.maximum(v, 1e-30))
        s.point_data[quantity] = v
    if at["dB"]:
        ulabel = f"dB {ulabel}"
    allvals = [s.point_data[quantity][np.isfinite(s.point_data[quantity])] for s in sampled]
    vals = np.concatenate(allvals) if allvals else np.array([0.0])
    lo = float(vmin) if vmin is not None else float(np.nanmin(vals))
    hi = float(vmax) if vmax is not None else float(np.nanmax(vals))
    if (at["AnimScale"] or animating) and quantity in ("Mag_E", "Mag_H", "Vector_E", "Vector_H", "Mag_J", "Vector_J") \
            and at["Scale"] != "Use Limits":
        # fixed scale over the whole period: the peak over all phases is the complex magnitude
        cv = np.concatenate([x.point_data["cmag"] for x in sampled]) * ufac
        cv = cv[np.isfinite(cv)]
        if len(cv):
            if at["dB"]:
                cv = 20 * np.log10(np.maximum(cv, 1e-30))
                lo, hi = float(cv.min()), float(cv.max())
            elif log_scale:
                # lower end: the smallest field magnitude on the plot (not an arbitrary 6 decades down)
                pos = cv[cv > 0]
                lo, hi = float(max(pos.min() if len(pos) else cv.max() * 1e-6, cv.max() * 1e-9)), float(cv.max())
            else:
                lo, hi = 0.0, float(cv.max())
    if at["Precision"] and at["LimitPrecision"]:
        lo, hi = float(f"{lo:.{at['Precision']}g}"), float(f"{hi:.{at['Precision']}g}")
    if hi <= lo:
        hi = lo + (abs(lo) * 1e-6 or 1e-12)
    clim = (max(lo, 1e-12 * max(hi, 1e-30)) if log_scale else lo, hi)
    cmap = _cmap(at)
    bar = _bar(title, ulabel)
    bar["n_labels"] = min(5, max(2, int(at["Divisions"]) + 1))    # at most 5 values: readable side by side
    bar["fmt"] = {"Decimal": f"%.{at['Precision']}f", "Scientific": f"%.{at['Precision']}e"}.get(at["Format"], "%.4g")
    opacity = max(0.0, min(1.0, 1.0 - at["Transparency"]))
    for k, s in enumerate(sampled):
        common = dict(scalars=quantity, cmap=cmap, clim=clim, log_scale=log_scale, opacity=opacity,
                      n_colors=int(at["Colors"]) if at["Colors"] else 256,
                      show_scalar_bar=k == 0, scalar_bar_args=bar if k == 0 else None)
        if quantity.startswith("Vector"):
            # equidistant arrows with a fixed spacing (reference tool style)
            if "regular" in s.field_data:
                ok = np.asarray(s.point_data["vtkValidPointMask"]).astype(bool) \
                    if "vtkValidPointMask" in s.point_data else np.ones(s.n_points, bool)
                sub = pv.PolyData(np.asarray(s.points)[ok])
                for nme in ("re", quantity):
                    sub.point_data[nme] = np.asarray(s.point_data[nme])[ok]
                hh = spacing_m or float(s.field_data["regular"][0])
            else:
                area = float((s if isinstance(s, pv.PolyData) else s.extract_surface()).area) if s.n_cells else 0.0
                hh = spacing_m or (np.sqrt(area / max(10, int(at["ArrowCount"]))) if area > 0 else 1e-3)
                rp = _regular_points(s, hh)
                sub = rp
                if rp.n_points:
                    # values from the plotted surface itself (keeps per-object values); 2D cells need a tolerance
                    smp = rp.sample(work) if "fromgrid" in s.field_data else rp.sample(s, tolerance=1e-2 * hh)
                    ok = smp.point_data["vtkValidPointMask"].astype(bool)
                    if ok.sum() < 0.5 * rp.n_points and "fromgrid" not in s.field_data and not jsurf:
                        smp = rp.sample(work)
                        ok = smp.point_data["vtkValidPointMask"].astype(bool)
                    sub = pv.PolyData(np.asarray(smp.points)[ok]) if ok.any() else pv.PolyData()
                    if ok.any():
                        sub.point_data["re"] = np.asarray(smp.point_data["re"])[ok]
            if sub.n_points == 0:
                continue
            vecs = np.nan_to_num(np.asarray(sub.point_data["re"]))
            nrm = np.linalg.norm(vecs, axis=1)
            sub = pv.PolyData(np.asarray(sub.points))
            sub.point_data["vec"] = vecs / np.maximum(nrm, 1e-300)[:, None]
            sub.point_data[quantity] = _qvals(vecs, nrm, at, ufac)
            spacing = hh
            size = 0.9 * spacing * arrow_scale * at["ArrowSize"]
            th = max(0.1, at["ArrowThickness"])
            geom = pv.Arrow(tip_length=0.3, tip_radius=min(0.45, 0.17 * th), shaft_radius=min(0.25, 0.075 * th),
                            tip_resolution=24, shaft_resolution=16) if at["ArrowTail"] else \
                pv.Cone(direction=(1, 0, 0), height=1.0, radius=min(0.6, 0.32 * th), center=(0.5, 0, 0), resolution=24)
            if at["MapSize"]:
                # proportional arrows ("Map size"): length follows the magnitude
                sub.point_data["len"] = nrm / max(float(nrm.max()), 1e-300)
                glyphs = sub.glyph(orient="vec", scale="len", factor=size * 1.6, geom=geom)
            else:
                glyphs = sub.glyph(orient="vec", scale=False, factor=size, geom=geom)
            meshes.append((glyphs, common))
        elif s.n_cells and getattr(s, "n_lines", 0) == s.n_cells:
            # field along a polyline: a thick coloured line
            meshes.append((s, dict(common, line_width=6, render_lines_as_tubes=True)))
        else:
            style = at["PlotType"]
            if style != "Iso Lines":
                # solver mesh lines: the element edges (curved ones as arcs), not the drawing triangles
                gm = grid_mesh(s) if at["ShowGrid"] else None
                meshes.append((s, dict(common, smooth_shading=style == "Smooth Shaded",
                                       show_edges=at["ShowGrid"] and gm is None, edge_color="#303030",
                                       line_width=0.5)))
                if gm is not None:
                    meshes.append((gm, dict(color="#303030", line_width=1, opacity=opacity)))
            if style in ("Iso Lines", "Shaded + Iso Lines"):
                try:
                    lev = np.linspace(clim[0], clim[1], max(2, int(at["Divisions"])) + 1)[1:-1]
                    cl = s.contour(isosurfaces=lev, scalars=quantity) if len(lev) else None
                    if cl is not None and cl.n_points:
                        kw = dict(common, line_width=2, show_scalar_bar=style == "Iso Lines" and k == 0)
                        if style != "Iso Lines":
                            kw.update(color="#202020", scalars=None, cmap=None, clim=None, n_colors=256,
                                      log_scale=False, scalar_bar_args=None)
                            kw = {kk: vv for kk, vv in kw.items() if vv is not None or kk == "scalars"}
                        meshes.append((cl, kw))
                except Exception:  # noqa: BLE001
                    pass
    freqs = frequencies(fields)
    fr = freqs[freq_index] if freq_index < len(freqs) else 0
    return meshes, f"{quantity}  port {port}  {fr / 1e9:.6g} GHz  phase {phase_deg:g} deg  max {hi:.4g} {ulabel}{note}"


MESH_QUALITY = ["None", "Mean Ratio", "Aspect Ratio", "Max Edge Length", "Volume"]


def tet_quality(nodes: np.ndarray, tets: np.ndarray, kind: str) -> np.ndarray:
    """Element quality per tetrahedron. Mean Ratio: 1 = regular tet, -> 0 degenerate.
    Aspect Ratio: longest edge / (2 sqrt(6) inradius), 1 = regular, large = bad."""
    P = nodes[tets]
    E = [P[:, j] - P[:, i] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    L2 = np.stack([(e ** 2).sum(1) for e in E], 1)
    V = np.abs(np.einsum("ij,ij->i", E[0], np.cross(E[1], E[2]))) / 6
    if kind == "Volume":
        return V
    if kind == "Max Edge Length":
        return np.sqrt(L2.max(1))
    if kind == "Mean Ratio":
        return 12 * (3 * V) ** (2 / 3) / np.maximum(L2.sum(1), 1e-300)
    A = sum(np.linalg.norm(np.cross(P[:, b] - P[:, a], P[:, c] - P[:, a]), axis=1) / 2
            for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
    r = 3 * V / np.maximum(A, 1e-300)
    return np.sqrt(L2.max(1)) / np.maximum(2 * np.sqrt(6) * r, 1e-300)


def mesh_overlay(doc, view, solution_setup: str, objects: list[str], quality: str = "None"):
    """Surface triangles of the solver mesh lying on the given objects, optionally coloured by the
    quality of the tetrahedron each triangle belongs to (mesh plot with element quality).
    Curved elements are drawn curved: their faces as smooth patches, their edges as arcs."""
    st = ResultStore(result_dir(doc, solution_setup))
    m = st.npz("mesh.npz")
    if m is None:
        raise RuntimeError("No mesh available; analyze the setup first")
    nodes, tets = m["nodes"], m["tets"]
    qv = tet_quality(nodes, tets, quality) if quality and quality != "None" else None
    names = [str(n) for n in m["obj_names"]] if "obj_names" in m and "tet_obj" in m else []

    def style(mesh, vals):
        if qv is None:
            return dict(color="#ffffff", show_edges=False, smooth_shading=True, opacity=1.0)
        mesh.cell_data[quality] = vals
        return dict(scalars=quality, cmap="jet_r" if quality == "Mean Ratio" else "jet", show_edges=False,
                    smooth_shading=True, opacity=1.0, show_scalar_bar=True, scalar_bar_args={"title": quality})
    line_kw = dict(color="#1a3d8f" if qv is None else "#202020", line_width=1)
    out, cuts = [], []
    for o in list(objects):
        t = doc.tess.get(o)
        ob = doc.design.obj(o) if doc.design.has_obj(o) else None
        if t is not None and t.dim == 2 and ob is not None and not getattr(ob, "model", True) \
                and len(t.face_area) == 1:
            # non-model plane (cut plane): show the volume mesh sliced by the plane
            cuts.append(o)
            objects = [x for x in objects if x != o]
            ug = pv.UnstructuredGrid({pv.CellType.TETRA: tets.astype(np.int64)}, nodes)
            ug.cell_data["_q"] = qv if qv is not None else np.zeros(len(tets))
            pts = np.asarray(t.points, float)
            sl = ug.slice(normal=np.asarray(t.face_normal[0], float), origin=pts.mean(0))
            b = np.c_[pts.min(0), pts.max(0)]
            pad = 1e-6 * float(np.linalg.norm(b[:, 1] - b[:, 0]) or 1)
            sl = sl.clip_box(list((b + [[-pad, pad]]).ravel()), invert=False)
            if sl.n_cells:
                st_kw = style(sl, np.asarray(sl.cell_data["_q"]))
                st_kw.update(show_edges=True, edge_color=line_kw["color"], line_width=1, smooth_shading=False)
                out.append((sl, st_kw))
    sel_k, sel_o, sel_l = [], [], []
    for o in objects:
        a = view.actors.get(o)
        ob = doc.design.obj(o) if doc.design.has_obj(o) else None
        if o in names:                      # meshed object: its boundary faces
            k, ow, lc = _object_faces(m, names.index(o))
        elif a is not None and a.mesh is not None:
            # sheets and the walls of unmeshed solids (PEC): faces on the CAD surface; the distance test
            # allows for the sag of straight faces and of the display mesh on curved surfaces
            solid = ob is not None and ob.dim == 3
            k, ow, lc = _sheet_faces(m, a.mesh, solid)
            if solid and not len(k):
                k, ow, lc = _sheet_faces(m, a.mesh, False)
        else:
            continue
        sel_k.append(k)
        sel_o.append(ow)
        sel_l.append(lc)
    nsel = 0
    if sel_k:
        k, ow, lc = np.concatenate(sel_k), np.concatenate(sel_o), np.concatenate(sel_l)
        _, first = np.unique(k, return_index=True)          # a face shared by two objects once
        ow, lc = ow[first], lc[first]
        nsel = len(ow)
        if nsel:
            P6, curved = _face_six(m, ow, lc)
            poly = _patches(P6, {}, curved, quad=False)
            out.append((poly, style(poly, qv[ow][poly.cell_data["_face"]] if qv is not None else None)))
            gm = grid_mesh(poly)
            if gm is not None:
                out.append((gm, dict(line_kw)))
    if nsel == 0 and not out:
        raise RuntimeError("No mesh faces found on the selected objects")
    txt = ""
    if qv is not None:
        txt = f"   {quality}: min {qv.min():.4g}, mean {qv.mean():.4g}, max {qv.max():.4g} (all tetrahedra)"
    cv = _curv(m)
    if cv is not None:
        txt += f"   {int((cv[0] >= 0).sum())} curved tetrahedra"
    return out, f"Mesh: {nsel} surface triangles" + (f", cut on {', '.join(cuts)}" if cuts else "") + \
        f", {len(tets)} tetrahedra in total" + txt
