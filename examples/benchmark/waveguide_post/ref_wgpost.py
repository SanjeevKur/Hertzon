"""Reference S11/S21 of the WR90 full-height post (wg_*.hzn) from an exact 2D reduction.

The post spans the full waveguide height, so with TE10 incident the field is z-invariant: Ez(x, y) solves
the 2D Helmholtz equation with Ez = 0 on the side walls and on the post. Isoparametric P2 FEM, single-
mode port conditions (the post is centred, so only TE10/TE30/... are excited; TE30 has decayed to 1e-4
at the ports). Run with several mesh sizes to see the reference converge."""
import sys
import numpy as np
import gmsh
import scipy.sparse as sp
import scipy.sparse.linalg as spl

a, L, r, f = 22.86e-3, 50e-3, 1.5e-3, 10e9
c0 = 299792458.0
k = 2 * np.pi * f / c0
beta = np.sqrt(k ** 2 - (np.pi / a) ** 2)


def phi(x):
    return np.sqrt(2 / a) * np.sin(np.pi * x / a)


# P2 triangle shape functions in gmsh node order (0,1,2 corners; 3: 0-1, 4: 1-2, 5: 2-0)
def shp(u, v):
    w = 1 - u - v
    N = np.array([w * (2 * w - 1), u * (2 * u - 1), v * (2 * v - 1), 4 * w * u, 4 * u * v, 4 * v * w])
    dNu = np.array([-(4 * w - 1), 4 * u - 1, 0 * u, 4 * (w - u), 4 * v, -4 * v])
    dNv = np.array([-(4 * w - 1), 0 * u, 4 * v - 1, -4 * u, 4 * u, 4 * (w - v)])
    return N, dNu, dNv


# 6-point degree-4 rule on the reference triangle
qa, qb, wa = 0.445948490915965, 0.091576213509771, 0.223381589678011
wb = 0.109951743655322
QP = np.array([[qa, qa], [1 - 2 * qa, qa], [qa, 1 - 2 * qa], [qb, qb], [1 - 2 * qb, qb], [qb, 1 - 2 * qb]])
QW = np.array([wa, wa, wa, wb, wb, wb]) / 2


def solve(h):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    o = gmsh.model.occ
    o.addRectangle(0, 0, 0, a, L)
    if NPOLY:      # polygonal post with vertices on the circle (faceted CAD approximation)
        t = 2 * np.pi * (np.arange(NPOLY) + 0.5) / NPOLY
        pts = [o.addPoint(a / 2 + r * np.cos(q), L / 2 + r * np.sin(q), 0) for q in t]
        ls = [o.addLine(pts[i], pts[(i + 1) % NPOLY]) for i in range(NPOLY)]
        o.addPlaneSurface([o.addCurveLoop(ls)], 2)
    else:
        o.addDisk(a / 2, L / 2, 0, r, r)
    if EPS is None:
        o.cut([(2, 1)], [(2, 2)])
    else:
        o.fragment([(2, 1)], [(2, 2)])
    o.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h / 4)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    # finer on the post
    post = [c for d, c in gmsh.model.getEntities(1)
            if np.hypot(*(np.array(gmsh.model.occ.getCenterOfMass(1, c)[:2]) - [a / 2, L / 2])) < 1.01 * r
            and gmsh.model.occ.getMass(1, c) < 2 * np.pi * r * 1.01]
    fld = gmsh.model.mesh.field
    fld.add("Distance", 1)
    fld.setNumbers(1, "CurvesList", [abs(c) for c in post])
    fld.add("MathEval", 2)
    fld.setString(2, "F", f"min({h}, {h / 6} + 0.3 * F1)")
    fld.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    tags, X, _ = gmsh.model.mesh.getNodes()
    X = X.reshape(-1, 3)[:, :2]
    idx = np.zeros(int(tags.max()) + 1, int)
    idx[tags.astype(int)] = np.arange(len(tags))
    T, epsel = [], []
    for _, sf in gmsh.model.getEntities(2):              # 6-node triangles per surface, with their eps
        _, en = gmsh.model.mesh.getElementsByType(9, sf)
        Ts = idx[en.astype(int)].reshape(-1, 6)
        # the post is the small surface (the outer one has its centre of mass at the post centre too)
        inside = gmsh.model.occ.getMass(2, sf) < 1.01 * np.pi * r ** 2
        T.append(Ts)
        epsel.append(np.full(len(Ts), EPS if (EPS is not None and inside) else 1.0, complex))
    T = np.vstack(T)
    epsel = np.concatenate(epsel)
    # boundary: classify 3-node lines by entity
    dirichlet, port1, port2 = set(), [], []
    for d, c in gmsh.model.getEntities(1):
        _, ln = gmsh.model.mesh.getElementsByType(8, c)
        E = idx[ln.astype(int)].reshape(-1, 3)
        cy = X[E].reshape(-1, 2)[:, 1]
        if np.allclose(cy, 0):
            port1.append(E)
        elif np.allclose(cy, L):
            port2.append(E)
        elif EPS is not None and c in post:
            continue                                     # dielectric post: inner interface, no condition
        else:
            dirichlet.update(E.ravel().tolist())
    gmsh.finalize()
    n = len(X)
    # assemble K - k^2 M
    rows, cols, vals = [], [], []
    for (u, v), w in zip(QP, QW):
        N, dNu, dNv = shp(u, v)
        P = X[T]                                         # (ne, 6, 2)
        J = np.stack([np.einsum("i,eik->ek", dNu, P), np.einsum("i,eik->ek", dNv, P)], axis=1)  # (ne,2,2)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        Ji = np.stack([np.stack([J[:, 1, 1], -J[:, 0, 1]], -1), np.stack([-J[:, 1, 0], J[:, 0, 0]], -1)], 1) / det[:, None, None]
        G = np.einsum("ekl,li->eik", Ji, np.stack([dNu, dNv]))   # gradients (ne, 6, 2)
        Ke = np.einsum("eik,ejk->eij", G, G) - k ** 2 * epsel[:, None, None] * np.einsum("i,j->ij", N, N)[None]
        Ke *= (w * np.abs(det))[:, None, None]
        rows.append(np.repeat(T, 6, axis=1).ravel())
        cols.append(np.tile(T, (1, 6)).ravel())
        vals.append(Ke.ravel())
    A = sp.csr_matrix((np.concatenate(vals).astype(complex), (np.concatenate(rows), np.concatenate(cols))), (n, n))

    def port_vec(Es):
        cvec = np.zeros(n)
        g, gw = np.polynomial.legendre.leggauss(6)
        for E in Es:
            for s, w in zip(g, gw):
                t = (s + 1) / 2
                N = np.array([(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)])   # gmsh: ends, mid
                dN = np.array([4 * t - 3, 4 * t - 1, 4 - 8 * t])
                x = N @ X[E].transpose(1, 0, 2)[:, :, 0]
                dx = dN @ X[E].transpose(1, 0, 2)[:, :, 0]
                np.add.at(cvec, E, (N[None, :] * (phi(x) * np.abs(dx) * w / 2)[:, None]))
        return cvec

    c1, c2 = port_vec(port1), port_vec(port2)
    def rank1(c):
        nz = np.nonzero(c)[0]
        return sp.csr_matrix((np.outer(c[nz], c[nz]).ravel(), (np.repeat(nz, len(nz)), np.tile(nz, len(nz)))), (n, n))

    A = A + 1j * beta * (rank1(c1) + rank1(c2))
    b = 2j * beta * c1
    free = np.ones(n, bool)
    free[list(dirichlet)] = False
    u = np.zeros(n, complex)
    Af = A[free][:, free].tocsc()
    u[free] = spl.spsolve(Af, b[free])
    S11 = c1 @ u - 1
    S21 = c2 @ u
    return len(T), S11, S21


NPOLY = 0
EPS = None
args = sys.argv[1:]
if args and args[0].startswith("eps="):
    e, _, tand = args.pop(0)[4:].partition(",")
    EPS = complex(float(e) * (1 - 1j * float(tand or 0)))
if args and args[0].startswith("poly="):
    NPOLY = int(args.pop(0)[5:])
if args and args[0].startswith("r="):
    r = float(args.pop(0)[2:])
for h in [float(x) for x in (args or ["2e-3", "1e-3", "0.5e-3"])]:
    nt, S11, S21 = solve(h)
    print(f"poly={NPOLY} r={r * 1e3:.4f} h={h * 1e3:.2f} mm  tris {nt:7d}  S11={S11.real:+.6f}{S11.imag:+.6f}j |S11|={abs(S11):.6f} "
          f"ang={np.angle(S11, deg=True):8.3f}  S21={S21.real:+.6f}{S21.imag:+.6f}j  "
          f"|S11|^2+|S21|^2={abs(S11) ** 2 + abs(S21) ** 2:.8f}", flush=True)
