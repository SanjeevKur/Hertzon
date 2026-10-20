// hzmesh - local constrained Delaunay refinement of tetrahedral meshes for adaptive FEM.
//
// The mesh is refined in place: every existing node stays where it is, new nodes are inserted
// with a constrained Bowyer-Watson step (cavity of the tets whose circumsphere contains the new
// point, bounded by the CAD surfaces and curves), so only a small neighbourhood of each new point
// changes and consecutive adaptive meshes are (nearly) nested. Points are chosen as in Delaunay
// refinement (Shewchuk): curve segments are split at their midpoints, surface triangles at their
// circumcentres, tetrahedra at their circumcentres; a circumcentre that would encroach on a
// surface triangle or curve segment splits that one instead. Elements are refined when they are
// larger than the target size or badly shaped (radius-edge ratio), inside the refinement zone only.
// Afterwards slivers are removed by 2-3 / 3-2 flips and smoothing of the new nodes.
//
// Copyright (C) 2026 Hertzon contributors. GPL-2.0-or-later.
#include "hzmesh.h"
#include "predicates.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

std::string g_err;

// faces opposite vertex i, ordered so that orient(face, v_i) > 0 (normal points into the tet)
const int FV[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
const int EV[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

inline uint64_t ekey(int a, int b) {
    if (a > b) std::swap(a, b);
    return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
}

struct K3 {
    int a, b, c;
    bool operator==(const K3 &o) const { return a == o.a && b == o.b && c == o.c; }
};
inline K3 k3(int a, int b, int c) {
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return {a, b, c};
}
struct K3H {
    size_t operator()(const K3 &k) const {
        uint64_t h = (uint64_t)(uint32_t)k.a * 0x9E3779B97F4A7C15ULL;
        h ^= (uint64_t)(uint32_t)k.b * 0xC2B2AE3D27D4EB4FULL + (h << 6) + (h >> 2);
        h ^= (uint64_t)(uint32_t)k.c * 0x165667B19E3779F9ULL + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};

inline void vsub(const double *a, const double *b, double *r) { r[0] = a[0] - b[0]; r[1] = a[1] - b[1]; r[2] = a[2] - b[2]; }
inline double vdot(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline void vcross(const double *a, const double *b, double *r) {
    r[0] = a[1] * b[2] - a[2] * b[1];
    r[1] = a[2] * b[0] - a[0] * b[2];
    r[2] = a[0] * b[1] - a[1] * b[0];
}
inline double dist2(const double *a, const double *b) {
    double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
    return x * x + y * y + z * z;
}
// > 0 for a positively oriented tet (gmsh convention: d lies on the normal side of a, b, c)
inline double orient(const double *a, const double *b, const double *c, const double *d) { return -orient3d(a, b, c, d); }

bool circumcenter_tet(const double *a, const double *b, const double *c, const double *d, double *cc) {
    double u[3], v[3], w[3], vw[3], wu[3], uv[3];
    vsub(b, a, u); vsub(c, a, v); vsub(d, a, w);
    vcross(v, w, vw); vcross(w, u, wu); vcross(u, v, uv);
    double den = 2.0 * vdot(u, vw);
    if (!(std::fabs(den) > 1e-300)) return false;
    double uu = vdot(u, u), vv = vdot(v, v), ww = vdot(w, w);
    for (int k = 0; k < 3; k++) cc[k] = a[k] + (uu * vw[k] + vv * wu[k] + ww * uv[k]) / den;
    return std::isfinite(cc[0]) && std::isfinite(cc[1]) && std::isfinite(cc[2]);
}
bool circumcenter_tri(const double *a, const double *b, const double *c, double *cc) {
    double u[3], v[3], n[3], vxn[3], nxu[3];
    vsub(b, a, u); vsub(c, a, v); vcross(u, v, n);
    double nn = vdot(n, n);
    if (!(nn > 1e-300)) return false;
    double uu = vdot(u, u), vv = vdot(v, v);
    vcross(v, n, vxn); vcross(n, u, nxu);
    for (int k = 0; k < 3; k++) cc[k] = a[k] + (uu * vxn[k] + vv * nxu[k]) / (2.0 * nn);
    return std::isfinite(cc[0]) && std::isfinite(cc[1]) && std::isfinite(cc[2]);
}

// mean ratio quality, 1 = regular tet, <= 0 for inverted
double mean_ratio(const double *a, const double *b, const double *c, const double *d) {
    double V = orient(a, b, c, d) / 6.0;
    if (V <= 0) return V < 0 ? -1.0 : 0.0;
    double L2 = dist2(a, b) + dist2(a, c) + dist2(a, d) + dist2(b, c) + dist2(b, d) + dist2(c, d);
    return 12.0 * std::cbrt(9.0 * V * V) / L2;
}
// minimum dihedral angle in degrees
double min_dihedral(const double *p0, const double *p1, const double *p2, const double *p3) {
    const double *p[4] = {p0, p1, p2, p3};
    double n[4][3], len[4];
    for (int i = 0; i < 4; i++) {
        double u[3], v[3];
        vsub(p[FV[i][1]], p[FV[i][0]], u);
        vsub(p[FV[i][2]], p[FV[i][0]], v);
        vcross(u, v, n[i]);
        len[i] = std::sqrt(vdot(n[i], n[i]));
        if (len[i] <= 0) return 0.0;
    }
    double cmax = -1.0;   // cos of the smallest dihedral angle
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++) {
            double c = -vdot(n[i], n[j]) / (len[i] * len[j]);
            if (c > cmax) cmax = c;
        }
    cmax = std::min(1.0, std::max(-1.0, cmax));
    return std::acos(cmax) * 57.29577951308232;
}

struct Params {
    double alpha = 1.0;        // refine when the element size exceeds alpha * target size
    double btet = 2.0;         // radius-edge bound for tets
    double bface = 1.3;        // radius-edge bound for surface triangles
    double beta = 0.35;        // shape refinement only for elements larger than beta * target size
    double budget = 0;         // maximum number of new tets (<= 0: unlimited)
    double maxins = 0;         // maximum number of insertions (<= 0: automatic)
    double improve = 1;        // flips + smoothing afterwards
    double sliver = 15.0;      // tets with a smaller dihedral angle (degrees) are improved
    double smooth = 3;         // smoothing sweeps
    double verbose = 0;
    double qmin_new = 0.02;    // new tets must have at least this mean ratio (else the cavity shrinks)
    double improve_all = 0;    // flips everywhere, not only in the zone (first pass on a mesh from gmsh)
    double sag = 0.0;          // > 0: curved CAD entities: split edges whose chord deviates from the geometry by
                               // more than sag * target size (the error estimator cannot see faceting errors)
};

struct Stats {
    double ins = 0, seg = 0, face = 0, tet = 0, enc = 0, fail = 0, f23 = 0, f32 = 0, smooth = 0;
    double newtets = 0, t_refine = 0, t_improve = 0, qmin = 0, dmin = 0, skipped = 0, f44 = 0;
    double fr[16] = {0};   // failed insertion attempts by reason
};

enum { INS_OK = 0, INS_ENC_SEG = 1, INS_ENC_FACE = 2, INS_FAIL = 3 };

struct Enc {
    int a = -1, b = -1, c = -1, tag = 0;   // encroached segment (a, b) or subface (a, b, c)
};

struct SubF {   // a surface triangle and the tets on both sides
    int a, b, c, tag;
    int t0, i0, t1, i1;
};

struct NewTet {
    int v[4];
    int vol;
};

// sag splits stop at edges shorter than this fraction of the target size
static const double SAG_LMIN = 0.02;

struct QE {
    double prio;
    int v[4];
    int tag;
    int slot;
    bool operator<(const QE &o) const { return prio < o.prio; }
};

struct Mesh {
    std::vector<double> X;
    std::vector<int> vdim, vtag, vt;
    std::vector<double> hs;
    std::vector<uint8_t> vz, vnew;
    std::vector<int> T, N, F, vol;
    std::vector<uint8_t> alive, tnew;
    std::vector<int> freeT;
    int nalive = 0;
    std::unordered_map<uint64_t, int> seg;
    std::unordered_set<int> curvedS, curvedC;
    mutable std::unordered_map<uint64_t, double> sagc;   // edge -> distance of its midpoint to the geometry
    std::unordered_map<int, double> offg;   // node inserted at the chord midpoint (projected one failed) -> its distance to the geometry
    std::unordered_map<int, double> hzS, hzC;   // curved surface / curve touched by the zone -> its zone size
    std::unordered_map<int, int> sideVol;   // surface -> volume its triangles' normals point into (-1: outward)
    hzm_project_cb proj = nullptr;
    std::vector<uint32_t> tm, tm2;
    uint32_t tep = 0, tep2 = 0;
    std::vector<uint32_t> vm;
    uint32_t vep = 0;
    Params prm;
    Stats st;

    const double *P(int v) const { return &X[3 * (size_t)v]; }
    int nv() const { return (int)vdim.size(); }
    int nslots() const { return (int)vol.size(); }

    // ---------------------------------------------------------------------------------------------
    int alloc_tet() {
        int t;
        if (!freeT.empty()) {
            t = freeT.back();
            freeT.pop_back();
        } else {
            t = (int)vol.size();
            T.resize(T.size() + 4, -1);
            N.resize(N.size() + 4, -1);
            F.resize(F.size() + 4, 0);
            vol.push_back(0);
            alive.push_back(0);
            tnew.push_back(0);
            tm.push_back(0);
            tm2.push_back(0);
        }
        alive[t] = 1;
        nalive++;
        return t;
    }
    void free_tet(int t) {
        alive[t] = 0;
        nalive--;
        freeT.push_back(t);
    }
    int add_vertex(const double *p, int dim, int tag, double h) {
        int v = nv();
        X.push_back(p[0]); X.push_back(p[1]); X.push_back(p[2]);
        vdim.push_back(dim); vtag.push_back(tag); vt.push_back(-1);
        hs.push_back(h); vz.push_back(1); vnew.push_back(1); vm.push_back(0);
        return v;
    }
    uint32_t new_tep() {
        if (++tep == 0) { std::fill(tm.begin(), tm.end(), 0); tep = 1; }
        return tep;
    }
    uint32_t new_tep2() {
        if (++tep2 == 0) { std::fill(tm2.begin(), tm2.end(), 0); tep2 = 1; }
        return tep2;
    }
    uint32_t new_vep() {
        if (++vep == 0) { std::fill(vm.begin(), vm.end(), 0); vep = 1; }
        return vep;
    }
    int face_of(int n, int a, int b, int c) const {
        for (int j = 0; j < 4; j++) {
            int v = T[4 * n + j];
            if (v != a && v != b && v != c) return j;
        }
        return -1;
    }
    bool has_vertex(int t, int v) const {
        const int *q = &T[4 * t];
        return q[0] == v || q[1] == v || q[2] == v || q[3] == v;
    }
    void face_verts(int t, int i, int &a, int &b, int &c) const {
        a = T[4 * t + FV[i][0]];
        b = T[4 * t + FV[i][1]];
        c = T[4 * t + FV[i][2]];
    }
    double tet_orient(int t) const { return orient(P(T[4 * t]), P(T[4 * t + 1]), P(T[4 * t + 2]), P(T[4 * t + 3])); }
    double tet_quality(int t) const { return mean_ratio(P(T[4 * t]), P(T[4 * t + 1]), P(T[4 * t + 2]), P(T[4 * t + 3])); }
    double tet_dihedral(int t) const { return min_dihedral(P(T[4 * t]), P(T[4 * t + 1]), P(T[4 * t + 2]), P(T[4 * t + 3])); }
    // p inside the circumsphere of tet t (exact)
    bool in_sphere(int t, const double *p) const {
        const int *q = &T[4 * t];
        return insphere(P(q[1]), P(q[0]), P(q[2]), P(q[3]), p) > 0;   // swap 0/1: Shewchuk orientation
    }

    // ---------------------------------------------------------------------------------------------
    // stars
    void vertex_star(int v, std::vector<int> &out) {
        out.clear();
        int t0 = vt[v];
        if (t0 < 0 || !alive[t0] || !has_vertex(t0, v)) return;
        uint32_t ep = new_tep2();
        out.push_back(t0);
        tm2[t0] = ep;
        for (size_t k = 0; k < out.size(); k++) {
            int t = out[k];
            for (int i = 0; i < 4; i++) {
                if (T[4 * t + i] == v) continue;
                int n = N[4 * t + i];
                if (n >= 0 && tm2[n] != ep) {
                    tm2[n] = ep;
                    out.push_back(n);
                }
            }
        }
    }
    int tet_with_edge(int a, int b) {
        static thread_local std::vector<int> st;
        vertex_star(a, st);
        for (int t : st)
            if (has_vertex(t, b)) return t;
        return -1;
    }
    void edge_star(int a, int b, int t0, std::vector<int> &out) {
        out.clear();
        uint32_t ep = new_tep2();
        out.push_back(t0);
        tm2[t0] = ep;
        for (size_t k = 0; k < out.size(); k++) {
            int t = out[k];
            for (int i = 0; i < 4; i++) {
                int x = T[4 * t + i];
                if (x == a || x == b) continue;
                int n = N[4 * t + i];
                if (n >= 0 && tm2[n] != ep) {
                    tm2[n] = ep;
                    out.push_back(n);
                }
            }
        }
    }
    bool find_face(int a, int b, int c, int &t, int &i) {
        static thread_local std::vector<int> st;
        vertex_star(a, st);
        for (int s : st)
            if (has_vertex(s, b) && has_vertex(s, c)) {
                t = s;
                i = face_of(s, a, b, c);
                return true;
            }
        return false;
    }
    SubF subface(int t, int i) const {
        SubF s;
        face_verts(t, i, s.a, s.b, s.c);
        s.tag = F[4 * t + i];
        s.t0 = t; s.i0 = i;
        s.t1 = N[4 * t + i];
        s.i1 = s.t1 >= 0 ? face_of(s.t1, s.a, s.b, s.c) : -1;
        return s;
    }
    // neighbouring triangle of surface `tag` across edge (u, v) of triangle (u, v, w) seen from tet t
    bool surf_neighbor(int u, int v, int w, int tag, int t, int &tn, int &in) {
        static thread_local std::vector<int> st;
        edge_star(u, v, t, st);
        for (int s : st) {
            for (int j = 0; j < 4; j++) {
                int x = T[4 * s + j];
                if (x == u || x == v) continue;
                if (F[4 * s + j] != tag) continue;
                int a, b, c;
                face_verts(s, j, a, b, c);
                if (a != w && b != w && c != w) {
                    tn = s;
                    in = j;
                    return true;
                }
            }
        }
        return false;
    }
    void project(int dim, int tag, double *p) const {
        if (!proj) return;
        if ((dim == 2 && curvedS.count(tag)) || (dim == 1 && curvedC.count(tag))) {
            double q[3];
            proj(dim, tag, p, q);
            if (std::isfinite(q[0]) && std::isfinite(q[1]) && std::isfinite(q[2])) { p[0] = q[0]; p[1] = q[1]; p[2] = q[2]; }
        }
    }
    bool is_curved(int dim, int tag) const {
        return proj && ((dim == 2 && curvedS.count(tag)) || (dim == 1 && curvedC.count(tag)));
    }
    double off_geometry(int a) const {
        auto it = offg.find(a);
        return it == offg.end() ? 0.0 : it->second;
    }
    // chord error of edge (a, b) on curved entity (dim, tag): distance of its midpoint to the geometry,
    // less the mean distance of its end points (a node that had to be inserted off the geometry would
    // otherwise keep every edge around it above the tolerance, however short it gets)
    double sagitta(int a, int b, int dim, int tag) const {
        uint64_t key = ekey(a, b);
        auto it = sagc.find(key);
        if (it != sagc.end()) return it->second;
        const double *pa = P(a), *pb = P(b);
        double m[3] = {0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2])};
        double q[3] = {m[0], m[1], m[2]};
        project(dim, tag, q);
        double d = std::max(0.0, std::sqrt(dist2(m, q)) - 0.5 * (off_geometry(a) + off_geometry(b)));
        sagc.emplace(key, d);
        return d;
    }
    // Worst chord error of a surface triangle relative to sag * target size (> 1: split edge (ea, eb)).
    // A curved surface that the zone touches is refined as a whole down to the zone's size (the target
    // there is the smaller of the local size and the mean zone size on that surface), so its faceting
    // error does not stay behind in the parts the zone did not reach. Curve segments are left to
    // seg_need, which projects onto the curve.
    double sag_ratio(const int *v, int tag, int &ea, int &eb) const {
        double best = 0;
        if (prm.sag <= 0 || !is_curved(2, tag)) return 0;
        auto hz = hzS.find(tag);
        if (hz == hzS.end()) return 0;
        for (int i = 0; i < 3; i++) {
            int a = v[i], b = v[(i + 1) % 3];
            if (seg.count(ekey(a, b))) continue;
            double ht = std::min(hz->second, 0.5 * (hs[a] + hs[b]));
            if (dist2(P(a), P(b)) < SAG_LMIN * SAG_LMIN * ht * ht) continue;     // depth limit
            double r = sagitta(a, b, 2, tag) / (prm.sag * ht);
            if (r > best) { best = r; ea = a; eb = b; }
        }
        return best;
    }
    double sag_ratio_seg(int a, int b, int tag) const {
        if (prm.sag <= 0 || !is_curved(1, tag)) return 0;
        auto hz = hzC.find(tag);
        if (hz == hzC.end()) return 0;
        double ht = std::min(hz->second, 0.5 * (hs[a] + hs[b]));
        if (dist2(P(a), P(b)) < SAG_LMIN * SAG_LMIN * ht * ht) return 0;
        return sagitta(a, b, 1, tag) / (prm.sag * ht);
    }
    // zone size of every curved surface and curve the zone touches (mean target size of its zone nodes);
    // a curve on the rim of such a surface gets at most the surface's size
    void zone_sizes() {
        hzS.clear();
        hzC.clear();
        if (prm.sag <= 0 || !proj) return;
        std::unordered_map<int, std::pair<double, int>> accS, accC;
        for (int t = 0; t < nslots(); t++) {
            if (!alive[t]) continue;
            for (int j = 0; j < 4; j++) {
                int tag = F[4 * t + j];
                if (tag <= 0 || !is_curved(2, tag)) continue;
                int a, b, c;                  // inner surfaces are seen from both sides: same weights
                face_verts(t, j, a, b, c);
                for (int x : {a, b, c})
                    if (vz[x]) { auto &q = accS[tag]; q.first += hs[x]; q.second++; }
            }
        }
        for (auto &kv : accS) hzS[kv.first] = kv.second.first / kv.second.second;
        for (auto &s2 : seg) {
            int tag = s2.second;
            if (!is_curved(1, tag)) continue;
            int a = (int)(s2.first >> 32), b = (int)(s2.first & 0xffffffffu);
            for (int x : {a, b})
                if (vz[x]) { auto &q = accC[tag]; q.first += hs[x]; q.second++; }
        }
        for (auto &kv : accC) hzC[kv.first] = kv.second.first / kv.second.second;
        if (hzS.empty()) return;
        for (int t = 0; t < nslots(); t++) {
            if (!alive[t]) continue;
            for (int j = 0; j < 4; j++) {
                int tag = F[4 * t + j];
                auto hz = tag > 0 ? hzS.find(tag) : hzS.end();
                if (hz == hzS.end()) continue;
                int v[3];
                face_verts(t, j, v[0], v[1], v[2]);
                for (int i = 0; i < 3; i++) {
                    auto it = seg.find(ekey(v[i], v[(i + 1) % 3]));
                    if (it == seg.end() || !is_curved(1, it->second)) continue;
                    auto c = hzC.find(it->second);
                    if (c == hzC.end() || c->second > hz->second) hzC[it->second] = hz->second;
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // locate p by a visibility walk from tet t; stops at constrained faces
    // returns 0 = inside tet t, 1 = blocked by subface (t, face), 2 = failure
    int locate(const double *p, int &t, int &face) {
        for (int step = 0; step < 20000; step++) {
            int base = (int)((step * 2654435761u + (uint32_t)t) >> 7) & 3;
            bool moved = false;
            for (int k = 0; k < 4; k++) {
                int i = (base + k) & 3;
                int a, b, c;
                face_verts(t, i, a, b, c);
                if (orient(P(a), P(b), P(c), p) < 0) {
                    if (F[4 * t + i] != 0 || N[4 * t + i] < 0) {
                        face = i;
                        return 1;
                    }
                    t = N[4 * t + i];
                    moved = true;
                    break;
                }
            }
            if (!moved) return 0;
        }
        return 2;
    }

    // walk over the triangles of a surface towards p; returns 0 found, 1 blocked by segment (a, b), 2 failure
    int locate2(const double *p, int tag, int &t, int &i, int &ea, int &eb) {
        for (int step = 0; step < 20000; step++) {
            int v[3];
            face_verts(t, i, v[0], v[1], v[2]);
            double u1[3], u2[3], n[3];
            vsub(P(v[1]), P(v[0]), u1);
            vsub(P(v[2]), P(v[0]), u2);
            vcross(u1, u2, n);
            int worst = -1;
            double wv = 0;
            for (int k = 0; k < 3; k++) {
                const double *a = P(v[k]), *b = P(v[(k + 1) % 3]);
                double ab[3], ap[3], cr[3];
                vsub(b, a, ab);
                vsub(p, a, ap);
                vcross(ab, ap, cr);
                double s = vdot(cr, n);
                if (s < wv) { wv = s; worst = k; }
            }
            if (worst < 0) return 0;
            int a = v[worst], b = v[(worst + 1) % 3], w = v[(worst + 2) % 3];
            if (seg.count(ekey(a, b))) {
                ea = a;
                eb = b;
                return 1;
            }
            int tn, in;
            if (!surf_neighbor(a, b, w, tag, t, tn, in)) return 2;
            t = tn;
            i = in;
        }
        return 2;
    }

    // ---------------------------------------------------------------------------------------------
    // Replace the tets `old` by `nts`. Faces on the region boundary keep their neighbours and surface
    // tags; new internal faces get the tag from `newSub` (0 when absent). Faces of `old` listed in
    // `removed` (surface triangles destroyed by the insertion) may disappear. With dry = true only
    // the topology is validated.
    std::vector<int> created;
    K3 bad_face{-1, -1, -1};   // new face that made the last dry run fail (hole / untagged interface)
    bool replace(const std::vector<int> &old, const std::vector<NewTet> &nts,
                 const std::unordered_map<K3, int, K3H> &newSub, const std::unordered_set<K3, K3H> &removed, bool dry) {
        struct Ext { int n, f, used; };
        std::unordered_map<K3, Ext, K3H> ext;
        ext.reserve(old.size() * 4);
        uint32_t ep = new_tep();
        for (int t : old) tm[t] = ep;
        for (int t : old)
            for (int i = 0; i < 4; i++) {
                int n = N[4 * t + i];
                if (n >= 0 && tm[n] == ep) continue;
                int a, b, c;
                face_verts(t, i, a, b, c);
                ext[k3(a, b, c)] = {n, F[4 * t + i], 0};
            }
        struct Int { int t, j, cnt, t2, j2; };
        std::unordered_map<K3, Int, K3H> in;
        in.reserve(nts.size() * 4);
        for (size_t k = 0; k < nts.size(); k++)
            for (int j = 0; j < 4; j++) {
                const int *v = nts[k].v;
                K3 key = k3(v[FV[j][0]], v[FV[j][1]], v[FV[j][2]]);
                auto e = ext.find(key);
                if (e != ext.end()) {
                    if (e->second.used++) { st.fr[8]++; return false; }
                    continue;
                }
                auto it = in.find(key);
                if (it == in.end())
                    in[key] = {(int)k, j, 1, -1, -1};
                else {
                    if (it->second.cnt >= 2) { st.fr[9]++; return false; }
                    it->second.cnt = 2;
                    it->second.t2 = (int)k;
                    it->second.j2 = j;
                }
            }
        for (auto &e : ext)
            if (!e.second.used && !removed.count(e.first)) { st.fr[10]++; return false; }
        for (auto &e : in) {
            bool sub = newSub.count(e.first) > 0;
            if (e.second.cnt == 1 && !sub) { st.fr[11]++; bad_face = e.first; return false; }   // hole
            if (e.second.cnt == 2 && !sub && nts[e.second.t].vol != nts[e.second.t2].vol) {   // untagged interface
                st.fr[12]++;
                bad_face = e.first;
                if (prm.verbose > 0 && st.fr[12] <= prm.verbose) {
                    const K3 &k = e.first;
                    int last = nv() - 1;
                    std::fprintf(stderr, "iface: face (%d %d %d) vols %d/%d, new node %d dim %d tag %d; dims %d/%d %d/%d %d/%d; newSub %zu removed %zu\n",
                                 k.a, k.b, k.c, nts[e.second.t].vol, nts[e.second.t2].vol, last, dbg_dim, dbg_tag,
                                 k.a < (int)vdim.size() ? vdim[k.a] : -9, k.a < (int)vdim.size() ? vtag[k.a] : -9,
                                 k.b < (int)vdim.size() ? vdim[k.b] : -9, k.b < (int)vdim.size() ? vtag[k.b] : -9,
                                 k.c < (int)vdim.size() ? vdim[k.c] : -9, k.c < (int)vdim.size() ? vtag[k.c] : -9,
                                 newSub.size(), removed.size());
                }
                return false;
            }
        }
        for (auto &e : newSub)
            if (!in.count(e.first)) { st.fr[13]++; return false; }     // the new surface triangles must all exist
        if (dry) return true;
        for (int t : old) free_tet(t);
        created.clear();
        for (size_t k = 0; k < nts.size(); k++) {
            int t = alloc_tet();
            created.push_back(t);
            for (int j = 0; j < 4; j++) {
                T[4 * t + j] = nts[k].v[j];
                N[4 * t + j] = -1;
                F[4 * t + j] = 0;
                vt[nts[k].v[j]] = t;
            }
            vol[t] = nts[k].vol;
            tnew[t] = 1;
        }
        for (size_t k = 0; k < nts.size(); k++) {
            int t = created[k];
            for (int j = 0; j < 4; j++) {
                const int *v = nts[k].v;
                int a = v[FV[j][0]], b = v[FV[j][1]], c = v[FV[j][2]];
                K3 key = k3(a, b, c);
                auto e = ext.find(key);
                if (e != ext.end()) {
                    N[4 * t + j] = e->second.n;
                    F[4 * t + j] = e->second.f;
                    if (e->second.n >= 0) N[4 * e->second.n + face_of(e->second.n, a, b, c)] = t;
                    continue;
                }
                auto it = in.find(key);
                int tag = 0;
                auto s = newSub.find(key);
                if (s != newSub.end()) tag = s->second;
                F[4 * t + j] = tag;
                if (it->second.cnt == 2) {
                    int other = (it->second.t == (int)k && it->second.j == j) ? created[it->second.t2] : created[it->second.t];
                    N[4 * t + j] = other;
                } else
                    N[4 * t + j] = -1;
            }
        }
        return true;
    }

    // ---------------------------------------------------------------------------------------------
    // Insert point p (classified on entity (pdim, ptag)).
    //   pdim 3: seed = tet to start the walk from
    //   pdim 2: seed subface (seed_t, seed_i) on surface ptag
    //   pdim 1: split segment (sa, sb) of curve ptag
    // check_enc: reject points that encroach on segments (pdim 2) / subfaces (pdim 3).
    // grow2d: remove the surface triangles whose diametral sphere contains p (else only the seed).
    int last_vertex = -1;
    int dbg_dim = 0, dbg_tag = 0;
    int insert(const double *p0, int pdim, int ptag, int seed_t, int seed_i, int sa, int sb, double h,
               bool check_enc, bool grow2d, Enc &enc) {
        double p[3] = {p0[0], p0[1], p0[2]};
        dbg_dim = pdim; dbg_tag = ptag;
        static thread_local std::vector<int> star, C;
        std::vector<int> mand0;            // mandatory tets that do not come from surface triangles
        std::vector<SubF> core, sfs;       // surface triangles that must go / all removed ones
        std::unordered_set<K3, K3H> sfset, excl;
        std::unordered_set<int> banned;    // tets kept out of the cavity
        int vol_tag = 0;
        if (pdim == 1) {
            int t0 = tet_with_edge(sa, sb);
            if (t0 < 0) { st.fr[0]++; return INS_FAIL; }
            edge_star(sa, sb, t0, star);
            std::unordered_set<K3, K3H> seen;
            for (int t : star) {
                mand0.push_back(t);
                for (int j = 0; j < 4; j++) {
                    int x = T[4 * t + j];
                    if (x == sa || x == sb || F[4 * t + j] == 0) continue;
                    int a, b, c;
                    face_verts(t, j, a, b, c);
                    if (seen.insert(k3(a, b, c)).second) core.push_back(subface(t, j));
                }
            }
            vol_tag = vol[t0];
        } else if (sa >= 0) {
            // midpoint of edge (sa, sb) on surface ptag (pdim 2) or inside a volume (pdim 3): all tets
            // around the edge must go, and on a surface its two triangles
            int t0 = seed_t >= 0 && has_vertex(seed_t, sa) && has_vertex(seed_t, sb) ? seed_t : tet_with_edge(sa, sb);
            if (t0 < 0) { st.fr[0]++; return INS_FAIL; }
            edge_star(sa, sb, t0, star);
            std::unordered_set<K3, K3H> seen;
            for (int t : star) {
                mand0.push_back(t);
                for (int j = 0; j < 4; j++) {
                    int x = T[4 * t + j];
                    if (x == sa || x == sb || F[4 * t + j] == 0) continue;
                    if (pdim == 3 || F[4 * t + j] != ptag) { st.fr[0]++; return INS_FAIL; }   // edge is on another surface
                    int a, b, c;
                    face_verts(t, j, a, b, c);
                    if (seen.insert(k3(a, b, c)).second) core.push_back(subface(t, j));
                }
            }
            vol_tag = vol[t0];
        } else if (pdim == 2) {
            int ea, eb;
            int r = locate2(p, ptag, seed_t, seed_i, ea, eb);
            if (r == 1) {
                if (!check_enc) { st.fr[1]++; return INS_FAIL; }
                enc.a = ea; enc.b = eb; enc.tag = seg[ekey(ea, eb)];
                return INS_ENC_SEG;
            }
            if (r != 0) { st.fr[1]++; return INS_FAIL; }
            core.push_back(subface(seed_t, seed_i));
            vol_tag = vol[seed_t];
        } else {
            int face;
            int r = locate(p, seed_t, face);
            if (r == 1) {
                if (!check_enc) { st.fr[2]++; return INS_FAIL; }
                int a, b, c;
                face_verts(seed_t, face, a, b, c);
                enc.a = a; enc.b = b; enc.c = c; enc.tag = F[4 * seed_t + face];
                return INS_ENC_FACE;
            }
            if (r != 0) { st.fr[2]++; return INS_FAIL; }
            mand0.push_back(seed_t);
            vol_tag = vol[seed_t];
        }
        const size_t ncore = core.size();
        bool curved = is_curved(pdim, ptag);
        std::vector<NewTet> nts;
        std::unordered_map<K3, int, K3H> newSub;
        for (int attempt = 0; attempt < 24; attempt++) {
            // surface cavities (Delaunay on each surface, bounded by curves)
            sfs = core;
            sfset.clear();
            for (const SubF &s : sfs) sfset.insert(k3(s.a, s.b, s.c));
            for (size_t k = 0; k < sfs.size(); k++) {
                SubF s = sfs[k];
                int v[3] = {s.a, s.b, s.c};
                for (int e = 0; e < 3; e++) {
                    int u = v[e], w = v[(e + 1) % 3], o = v[(e + 2) % 3];
                    if (seg.count(ekey(u, w))) {
                        if (pdim == 2 && check_enc) {
                            double m[3] = {0.5 * (P(u)[0] + P(w)[0]), 0.5 * (P(u)[1] + P(w)[1]), 0.5 * (P(u)[2] + P(w)[2])};
                            if (dist2(p, m) < 0.25 * dist2(P(u), P(w)) * (1 - 1e-9)) {
                                enc.a = u; enc.b = w; enc.tag = seg[ekey(u, w)];
                                return INS_ENC_SEG;
                            }
                        }
                        continue;
                    }
                    if (!grow2d) continue;
                    int tn, in;
                    if (!surf_neighbor(u, w, o, s.tag, s.t0, tn, in)) continue;
                    int a, b, c;
                    face_verts(tn, in, a, b, c);
                    K3 key = k3(a, b, c);
                    if (sfset.count(key) || excl.count(key)) continue;
                    double cc[3];
                    if (!circumcenter_tri(P(a), P(b), P(c), cc)) continue;
                    if (dist2(p, cc) < dist2(cc, P(a)) * (1 - 1e-9)) {
                        sfset.insert(key);
                        sfs.push_back(subface(tn, in));
                    }
                }
            }
            // owner of each mandatory tet: index of the surface triangle (-1 = cannot be given up)
            std::unordered_map<int, int> owner;
            for (int t : mand0) owner[t] = -1;
            for (size_t k = 0; k < sfs.size(); k++) {
                for (int t : {sfs[k].t0, sfs[k].t1}) {
                    if (t < 0) continue;
                    auto it = owner.find(t);
                    if (it == owner.end()) owner[t] = (int)k;
                    else if (it->second >= 0 && k < ncore) it->second = -1;
                    else if (it->second >= 0 && (int)k > it->second) it->second = (int)k;
                }
            }
            for (auto &o : owner)
                if (o.second >= 0 && o.second < (int)ncore) o.second = -1;
            // 3D cavity
            uint32_t cep = new_tep();
            C.clear();
            for (auto &o : owner) {
                tm[o.first] = cep;
                C.push_back(o.first);
            }
            std::sort(C.begin(), C.end());   // deterministic order
            for (size_t k = 0; k < C.size(); k++) {
                int t = C[k];
                for (int i = 0; i < 4; i++) {
                    int n = N[4 * t + i];
                    if (n >= 0 && tm[n] == cep) continue;
                    if (F[4 * t + i] != 0) {
                        if (check_enc && pdim == 3) {
                            int a, b, c;
                            face_verts(t, i, a, b, c);
                            double cc[3];
                            if (circumcenter_tri(P(a), P(b), P(c), cc) && dist2(p, cc) < dist2(cc, P(a)) * (1 - 1e-9)) {
                                enc.a = a; enc.b = b; enc.c = c; enc.tag = F[4 * t + i];
                                return INS_ENC_FACE;
                            }
                        }
                        continue;
                    }
                    if (n >= 0 && !banned.count(n) && in_sphere(n, p)) {
                        tm[n] = cep;
                        C.push_back(n);
                    }
                }
            }
            // shrink until the cavity is star-shaped from p, keeps every old node and the new tets are
            // not too flat; a mandatory tet that is in the way removes its surface triangle instead
            int give_up = -2;   // >= 0: surface triangle to exclude, -1: fail
            for (int iter = 0; iter < 400 && give_up == -2; iter++) {
                nts.clear();
                std::vector<int> drop;
                uint32_t vp = new_vep();
                for (int t : C) {
                    for (int i = 0; i < 4; i++) {
                        int n = N[4 * t + i];
                        if (n >= 0 && tm[n] == cep) continue;      // internal face (incl. removed subfaces)
                        int a, b, c;
                        face_verts(t, i, a, b, c);
                        if (F[4 * t + i] != 0 && sfset.count(k3(a, b, c))) continue;   // removed boundary subface
                        vm[a] = vp; vm[b] = vp; vm[c] = vp;
                        double o = orient(P(a), P(b), P(c), p);
                        bool inval = !(o > 0);
                        auto ow = owner.find(t);
                        // new tets must not be flat; a tet that has to go may only produce tets
                        // that are not much worse than itself
                        double qlim = ow == owner.end() ? prm.qmin_new : std::min(prm.qmin_new, 0.5 * tet_quality(t));
                        bool flat = !inval && mean_ratio(P(a), P(b), P(c), p) < qlim;
                        if (inval || flat) {
                            if (ow == owner.end()) {
                                drop.push_back(t);
                                break;
                            }
                            if (ow->second < 0) { give_up = -1; break; }
                            give_up = ow->second;
                            break;
                        }
                        nts.push_back({{a, b, c, -1}, vol[t]});
                    }
                    if (give_up != -2) break;
                }
                if (give_up != -2) break;
                // every node of the cavity must stay on its boundary
                if (drop.empty()) {
                    for (int t : C) {
                        for (int j = 0; j < 4; j++) {
                            int v = T[4 * t + j];
                            if (vm[v] == vp) continue;
                            bool found = false;
                            for (int s : C)
                                if (!owner.count(s) && has_vertex(s, v)) {
                                    drop.push_back(s);
                                    found = true;
                                }
                            if (found) continue;
                            // only surface-triangle tets around v: give up a removed triangle at v
                            int best = -1;
                            for (int s : C) {
                                auto ow = owner.find(s);
                                if (ow != owner.end() && ow->second >= 0 && has_vertex(s, v)) best = std::max(best, ow->second);
                            }
                            give_up = best;
                            break;
                        }
                        if (give_up != -2) break;
                    }
                    if (give_up != -2) break;
                }
                // every curve segment of the cavity must stay on its boundary, and no surface triangle
                // that is not being re-triangulated may end up inside: a cavity can wrap around a free
                // curve (edge of a sheet, embedded curve), which then lost its segments
                if (drop.empty()) {
                    static thread_local std::unordered_set<uint64_t> bedges;
                    bedges.clear();
                    for (auto &x : nts) {
                        bedges.insert(ekey(x.v[0], x.v[1]));
                        bedges.insert(ekey(x.v[1], x.v[2]));
                        bedges.insert(ekey(x.v[0], x.v[2]));
                    }
                    static const int EV[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
                    const uint64_t split = pdim == 1 ? ekey(sa, sb) : ~(uint64_t)0;
                    for (int t : C) {
                        for (auto &e : EV) {
                            int u = T[4 * t + e[0]], w = T[4 * t + e[1]];
                            uint64_t ke = ekey(u, w);
                            if (ke == split || bedges.count(ke) || !seg.count(ke)) continue;
                            bool found = false;
                            for (int s : C)
                                if (!owner.count(s) && has_vertex(s, u) && has_vertex(s, w)) {
                                    drop.push_back(s);
                                    found = true;
                                }
                            if (found) continue;
                            int best = -1;
                            for (int s : C) {
                                auto ow = owner.find(s);
                                if (ow != owner.end() && ow->second >= 0 && has_vertex(s, u) && has_vertex(s, w))
                                    best = std::max(best, ow->second);
                            }
                            give_up = best;
                            break;
                        }
                        if (give_up != -2) break;
                        for (int i = 0; i < 4; i++) {
                            int n = N[4 * t + i];
                            if (n < 0 || tm[n] != cep || F[4 * t + i] == 0) continue;
                            int a, b, c;
                            face_verts(t, i, a, b, c);
                            if (sfset.count(k3(a, b, c))) continue;
                            bool found = false;
                            for (int s : {t, n})
                                if (!owner.count(s)) { drop.push_back(s); found = true; }
                            if (found) continue;
                            int best = -1;
                            for (int s : {t, n}) {
                                auto ow = owner.find(s);
                                if (ow->second >= 0) best = std::max(best, ow->second);
                            }
                            give_up = best;
                            break;
                        }
                        if (give_up != -2) break;
                    }
                    if (give_up != -2) break;
                }
                if (drop.empty()) break;
                // remove the dropped tets, keep the part connected to the mandatory tets
                std::unordered_set<int> keep(C.begin(), C.end());
                for (int t : drop) keep.erase(t);
                uint32_t cep2 = new_tep();
                std::vector<int> C2;
                for (auto &o : owner)
                    if (tm[o.first] != cep2) {
                        tm[o.first] = cep2;
                        C2.push_back(o.first);
                    }
                std::sort(C2.begin(), C2.end());
                for (size_t k = 0; k < C2.size(); k++) {
                    int t = C2[k];
                    for (int i = 0; i < 4; i++) {
                        int n = N[4 * t + i];
                        if (n < 0 || tm[n] == cep2 || !keep.count(n)) continue;
                        if (F[4 * t + i] != 0) {
                            int a, b, c;
                            face_verts(t, i, a, b, c);
                            if (!sfset.count(k3(a, b, c))) continue;
                        }
                        tm[n] = cep2;
                        C2.push_back(n);
                    }
                }
                C.swap(C2);
                cep = cep2;
                if (iter == 399) give_up = -1;
            }
            if (give_up == -1) { st.fr[3]++; return INS_FAIL; }
            if (give_up >= 0) {
                if (give_up < (int)ncore) { st.fr[4]++; return INS_FAIL; }
                const SubF &s = sfs[give_up];
                excl.insert(k3(s.a, s.b, s.c));
                continue;
            }
            // tags of the new surface triangles (p, u, v): boundary edges of each surface cavity
            std::unordered_map<uint64_t, int> bedge;   // boundary edge -> surface tag
            {
                std::unordered_map<uint64_t, int> cnt;
                auto tkey = [](uint64_t ke, int tag) { return ke ^ ((uint64_t)(uint32_t)tag * 0x9E3779B97F4A7C15ULL); };
                for (const SubF &s : sfs) {
                    int v[3] = {s.a, s.b, s.c};
                    for (int e = 0; e < 3; e++) cnt[tkey(ekey(v[e], v[(e + 1) % 3]), s.tag)]++;
                }
                bool conflict = false;
                for (const SubF &s : sfs) {
                    int v[3] = {s.a, s.b, s.c};
                    for (int e = 0; e < 3; e++) {
                        uint64_t ke = ekey(v[e], v[(e + 1) % 3]);
                        if (cnt[tkey(ke, s.tag)] != 1) continue;
                        if (pdim == 1 && ke == ekey(sa, sb)) continue;
                        auto it = bedge.find(ke);
                        if (it == bedge.end()) bedge[ke] = s.tag;
                        else if (it->second != s.tag) conflict = true;
                    }
                }
                if (conflict) { st.fr[5]++; return INS_FAIL; }
            }
            int pv = nv();
            for (auto &t : nts) t.v[3] = pv;
            newSub.clear();
            for (auto &e : bedge) {
                int u = (int)(e.first >> 32), w = (int)(e.first & 0xffffffffu);
                newSub[k3(u, w, pv)] = e.second;
            }
            // validation: volume (when nothing moved off a plane) and topology
            if (!curved) {
                double vo = 0, vn = 0;
                for (int t : C) vo += tet_orient(t);
                for (auto &t : nts) vn += orient(P(t.v[0]), P(t.v[1]), P(t.v[2]), p);
                if (std::fabs(vn - vo) > 1e-9 * std::fabs(vo)) { st.fr[6]++; return INS_FAIL; }
            }
            X.push_back(p[0]); X.push_back(p[1]); X.push_back(p[2]);   // replace() reads the new node
            bad_face = {-1, -1, -1};
            bool ok = replace(C, nts, newSub, sfset, true);
            X.resize(X.size() - 3);
            if (!ok && bad_face.a >= 0) {
                // a new face (x, y, p) would cut through a surface: keep the tets at edge (x, y) out
                int xy[3], nxy = 0;
                for (int v : {bad_face.a, bad_face.b, bad_face.c})
                    if (v != pv) xy[nxy++] = v;
                if (nxy == 2) {
                    bool any = false;
                    int best = -1;
                    for (int t : C) {
                        if (!has_vertex(t, xy[0]) || !has_vertex(t, xy[1])) continue;
                        auto ow = owner.find(t);
                        if (ow == owner.end()) { banned.insert(t); any = true; }
                        else if (ow->second >= (int)ncore) best = std::max(best, ow->second);
                    }
                    if (any) continue;
                    if (best >= 0) {
                        const SubF &s = sfs[best];
                        excl.insert(k3(s.a, s.b, s.c));
                        continue;
                    }
                }
            }
            if (!ok) { st.fr[7]++; return INS_FAIL; }
            add_vertex(p, pdim, pdim == 3 ? vol_tag : ptag, h);
            replace(C, nts, newSub, sfset, false);
            if (pdim == 1) {
                seg.erase(ekey(sa, sb));
                seg[ekey(sa, pv)] = ptag;
                seg[ekey(pv, sb)] = ptag;
            }
            last_vertex = pv;
            return INS_OK;
        }
        { st.fr[3]++; return INS_FAIL; }
    }

    // ---------------------------------------------------------------------------------------------
    // refinement criteria
    double hmean(const int *v, int n) const {
        double s = 0;
        for (int k = 0; k < n; k++) s += hs[v[k]];
        return s / n;
    }
    // An element may be refined when one of its old nodes is in the zone or when all its nodes are
    // new. Elements that only touch the refined region (new nodes plus frozen old nodes) are left as
    // they are, so refinement cannot creep into the frozen part of the mesh.
    bool in_zone(const int *v, int n) const {
        bool allnew = true;
        for (int k = 0; k < n; k++) {
            if (!vnew[v[k]]) {
                if (vz[v[k]]) return true;
                allnew = false;
            }
        }
        return allnew;
    }
    // touches the zone or the refined region (for the local improvement)
    bool touch_zone(const int *v, int n) const {
        for (int k = 0; k < n; k++)
            if (vz[v[k]] || vnew[v[k]]) return true;
        return false;
    }
    // shape refinement only well inside the zone, so that it cannot spread over the frozen part
    bool all_zone(const int *v, int n) const {
        for (int k = 0; k < n; k++)
            if (!vz[v[k]]) return false;
        return true;
    }
    // Priority > 0 if the element should be refined: > 1 for size (longest edge above alpha * target
    // size; the longest edge, not the circumradius, so that flat elements in thin layers are not
    // refined down to the layer thickness), <= 1 for shape (radius-edge ratio, only for elements that
    // are not already much smaller than the target size).
    double tet_need(int t) const {
        const int *v = &T[4 * t];
        if (!in_zone(v, 4)) return 0;
        double lmin = 1e300, lmax = 0;
        for (int e = 0; e < 6; e++) {
            double l = dist2(P(v[EV[e][0]]), P(v[EV[e][1]]));
            lmin = std::min(lmin, l);
            lmax = std::max(lmax, l);
        }
        lmin = std::sqrt(lmin);
        lmax = std::sqrt(lmax);
        double h = hmean(v, 4);
        double size = lmax / (prm.alpha * h);
        if (size > 1.0) return 1.0 + size;
        if (lmax < prm.beta * h || !all_zone(v, 4)) return 0;
        double cc[3];
        if (!circumcenter_tet(P(v[0]), P(v[1]), P(v[2]), P(v[3]), cc)) return 0;
        double rho = std::sqrt(dist2(cc, P(v[0]))) / lmin / prm.btet;
        return rho > 1.0 ? std::min(rho, 1.0) : 0.0;
    }
    double face_need(int a, int b, int c, int tag = 0) const {
        int v[3] = {a, b, c};
        int ea, eb;
        double sr = sag_ratio(v, tag, ea, eb);       // also outside the zone (see sag_ratio)
        if (!in_zone(v, 3)) return sr > 1.0 ? 1.0 + sr : 0.0;
        double l1 = dist2(P(a), P(b)), l2 = dist2(P(b), P(c)), l3 = dist2(P(a), P(c));
        double lmin = std::sqrt(std::min(l1, std::min(l2, l3))), lmax = std::sqrt(std::max(l1, std::max(l2, l3)));
        double h = hmean(v, 3);
        double size = std::max(lmax / (prm.alpha * h), sr);
        if (size > 1.0) return 1.0 + size;
        if (lmax < prm.beta * h || !all_zone(v, 3)) return 0;
        double cc[3];
        if (!circumcenter_tri(P(a), P(b), P(c), cc)) return 0;
        double rho = std::sqrt(dist2(cc, P(a))) / lmin / prm.bface;
        return rho > 1.0 ? std::min(rho, 1.0) : 0.0;
    }
    double seg_need(int a, int b, int tag = 0) const {
        int v[2] = {a, b};
        double sr = sag_ratio_seg(a, b, tag);
        if (!in_zone(v, 2)) return sr > 1.0 ? 1.0 + sr : 0.0;
        double L = std::sqrt(dist2(P(a), P(b)));
        double size = std::max(L / (prm.alpha * hmean(v, 2)), sr);
        return size > 1.0 ? 1.0 + size : 0.0;
    }
    // may an encroached face / segment be split? Not when it is already well below the target size
    // (protects small angles and thin layers against cascading splits)
    bool may_split_face(int a, int b, int c) const {
        int v[3] = {a, b, c};
        double lmax = std::sqrt(std::max(dist2(P(a), P(b)), std::max(dist2(P(b), P(c)), dist2(P(a), P(c)))));
        return lmax > prm.beta * hmean(v, 3);
    }
    bool may_split_seg(int a, int b) const {
        int v[2] = {a, b};
        return std::sqrt(dist2(P(a), P(b))) > prm.beta * hmean(v, 2);
    }

    std::priority_queue<QE> qs, qf, qt;
    std::unordered_map<uint64_t, int> tries;

    void push_tet(int t) {
        double pr = tet_need(t);
        if (pr <= 0) return;
        QE e;
        e.prio = pr;
        for (int k = 0; k < 4; k++) e.v[k] = T[4 * t + k];
        e.tag = 0;
        e.slot = t;
        qt.push(e);
    }
    void push_face(int a, int b, int c, int tag, double bonus = 0) {
        if (tag <= 0) return;
        double pr = face_need(a, b, c, tag);
        if (pr <= 0 && bonus <= 0) return;
        QE e;
        e.prio = pr + bonus;
        e.v[0] = a; e.v[1] = b; e.v[2] = c; e.v[3] = -1;
        e.tag = tag;
        e.slot = -1;
        qf.push(e);
    }
    void push_seg(int a, int b, int tag, double bonus = 0) {
        double pr = seg_need(a, b, tag);
        if (pr <= 0 && bonus <= 0) return;
        QE e;
        e.prio = pr + bonus;
        e.v[0] = a; e.v[1] = b; e.v[2] = -1; e.v[3] = -1;
        e.tag = tag;
        e.slot = -1;
        qs.push(e);
    }
    void push_created() {
        for (int t : created) {
            push_tet(t);
            for (int j = 0; j < 4; j++) {
                if (F[4 * t + j] == 0) continue;
                int n = N[4 * t + j];
                if (n >= 0 && n < t && tnew[n] && alive[n]) {
                    // pushed from the other side (avoid duplicates when both sides are new)
                    bool other_created = false;
                    for (int c : created)
                        if (c == n) { other_created = true; break; }
                    if (other_created) continue;
                }
                int a, b, c;
                face_verts(t, j, a, b, c);
                if (a == last_vertex || b == last_vertex || c == last_vertex) push_face(a, b, c, F[4 * t + j]);
            }
        }
        if (last_vertex >= 0) {
            // new segments
            static thread_local std::vector<int> stv;
            vertex_star(last_vertex, stv);
            std::unordered_set<int> seen;
            for (int t : stv)
                for (int k = 0; k < 4; k++) {
                    int x = T[4 * t + k];
                    if (x == last_vertex || !seen.insert(x).second) continue;
                    auto it = seg.find(ekey(x, last_vertex));
                    if (it != seg.end()) push_seg(x, last_vertex, it->second);
                }
        }
    }

    bool budget_left() const {
        if (prm.budget > 0 && nalive - nt_start >= prm.budget) return false;
        if (st.ins >= max_ins) return false;
        return true;
    }
    int nt_start = 0;
    double max_ins = 0;

    bool too_many(uint64_t key) {
        int &c = tries[key];
        return ++c > 3;
    }

    // split a curve segment at its (projected) midpoint
    int split_segment(int a, int b, int tag, double h) {
        Enc enc;
        const double *pa = P(a), *pb = P(b);
        double m[3] = {0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2])};
        double mp[3] = {m[0], m[1], m[2]};
        project(1, tag, mp);
        bool moved = mp[0] != m[0] || mp[1] != m[1] || mp[2] != m[2];
        int r = insert(mp, 1, tag, -1, -1, a, b, h, false, true, enc);
        bool off = false;
        if (r != INS_OK && moved) off = (r = insert(m, 1, tag, -1, -1, a, b, h, false, true, enc)) == INS_OK;
        if (r != INS_OK) r = insert(mp, 1, tag, -1, -1, a, b, h, false, false, enc);
        if (r != INS_OK && moved) off = (r = insert(m, 1, tag, -1, -1, a, b, h, false, false, enc)) == INS_OK;
        if (off) offg[nv() - 1] = std::sqrt(dist2(m, mp));
        if (r == INS_OK) { st.ins++; st.seg++; } else st.fail++;
        return r;
    }
    // split edge (a, b) of tet t at its midpoint, on its curve, its surface or inside the volume
    int split_edge(int a, int b, int t, double h) {
        auto it = seg.find(ekey(a, b));
        if (it != seg.end()) return split_segment(a, b, it->second, h);
        Enc enc;
        const double *pa = P(a), *pb = P(b);
        double m[3] = {0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2])};
        static thread_local std::vector<int> es;
        if (!(has_vertex(t, a) && has_vertex(t, b))) t = tet_with_edge(a, b);
        if (t < 0) { st.fail++; return INS_FAIL; }
        edge_star(a, b, t, es);
        int tag = 0;
        for (int s : es)
            for (int j = 0; j < 4; j++) {
                int x = T[4 * s + j];
                if (x == a || x == b || F[4 * s + j] == 0) continue;
                if (F[4 * s + j] < 0 || (tag && F[4 * s + j] != tag)) { st.fail++; return INS_FAIL; }
                tag = F[4 * s + j];
            }
        int r;
        if (!tag) {
            r = insert(m, 3, 0, t, -1, a, b, h, false, true, enc);
            if (r == INS_OK) { st.ins++; st.tet++; }
        } else {
            double mp[3] = {m[0], m[1], m[2]};
            project(2, tag, mp);
            bool moved = mp[0] != m[0] || mp[1] != m[1] || mp[2] != m[2];
            r = insert(mp, 2, tag, t, -1, a, b, h, false, true, enc);
            if (r != INS_OK && moved && (r = insert(m, 2, tag, t, -1, a, b, h, false, true, enc)) == INS_OK)
                offg[nv() - 1] = std::sqrt(dist2(m, mp));
            if (r != INS_OK) r = insert(mp, 2, tag, t, -1, a, b, h, false, false, enc);
            if (r == INS_OK) { st.ins++; st.face++; }
        }
        if (r != INS_OK) st.fail++;
        return r;
    }
    int split_longest(const int *v, int n, int t) {
        int ea = v[0], eb = v[1];
        double lm = -1;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                double l = dist2(P(v[i]), P(v[j]));
                if (l > lm) { lm = l; ea = v[i]; eb = v[j]; }
            }
        return split_edge(ea, eb, t, 0.5 * (hs[ea] + hs[eb]));
    }

    void refine() {
        nt_start = nalive;
        sagc.clear();          // nodes may have been moved by the smoothing of an earlier run
        zone_sizes();
        max_ins = prm.maxins > 0 ? prm.maxins : 20.0 * nalive + 1000;
        tries.clear();
        // initial queues
        for (int t = 0; t < nslots(); t++)
            if (alive[t]) push_tet(t);
        for (int t = 0; t < nslots(); t++) {
            if (!alive[t]) continue;
            for (int j = 0; j < 4; j++) {
                if (F[4 * t + j] == 0) continue;
                int n = N[4 * t + j];
                if (n >= 0 && n < t) continue;
                int a, b, c;
                face_verts(t, j, a, b, c);
                push_face(a, b, c, F[4 * t + j]);
            }
        }
        for (auto &s : seg) push_seg((int)(s.first >> 32), (int)(s.first & 0xffffffffu), s.second);

        while (budget_left()) {
            Enc enc;
            if (!qs.empty()) {
                QE e = qs.top();
                qs.pop();
                auto it = seg.find(ekey(e.v[0], e.v[1]));
                if (it == seg.end()) continue;
                if (too_many(ekey(e.v[0], e.v[1]) * 0x9E3779B97F4A7C15ull ^ 2)) { st.skipped++; continue; }
                int r = split_segment(e.v[0], e.v[1], it->second, 0.5 * (hs[e.v[0]] + hs[e.v[1]]));
                if (r == INS_OK) push_created();
                continue;
            }
            if (!qf.empty()) {
                QE e = qf.top();
                qf.pop();
                int t, i;
                if (!find_face(e.v[0], e.v[1], e.v[2], t, i) || F[4 * t + i] != e.tag) continue;
                K3 key = k3(e.v[0], e.v[1], e.v[2]);
                uint64_t tk = ((uint64_t)(uint32_t)key.a * 73856093u) ^ ((uint64_t)(uint32_t)key.b << 21) ^ ((uint64_t)(uint32_t)key.c << 42) ^ 1;
                if (too_many(tk)) { st.skipped++; continue; }
                double pr = face_need(e.v[0], e.v[1], e.v[2], e.tag);
                if (pr > 1.0 || (pr <= 0 && e.prio > 1.0)) {
                    // too large (or pushed by an encroaching point): halve the longest edge, or the edge
                    // whose chord is too far from the curved surface
                    int ea = -1, eb = -1;
                    double sr = pr > 1.0 ? sag_ratio(e.v, e.tag, ea, eb) : 0.0;
                    const double *a = P(e.v[0]), *b = P(e.v[1]), *c = P(e.v[2]);
                    int vv[3] = {e.v[0], e.v[1], e.v[2]};
                    double lmax = std::sqrt(std::max(dist2(a, b), std::max(dist2(b, c), dist2(a, c))));
                    int r = sr > 1.0 && sr >= lmax / (prm.alpha * hmean(vv, 3))
                                ? split_edge(ea, eb, t, 0.5 * (hs[ea] + hs[eb]))
                                : split_longest(e.v, 3, t);
                    if (r == INS_OK) push_created();
                    continue;
                }
                const double *a = P(e.v[0]), *b = P(e.v[1]), *c = P(e.v[2]);
                double cc[3];
                if (!circumcenter_tri(a, b, c, cc)) continue;
                double h = (hs[e.v[0]] + hs[e.v[1]] + hs[e.v[2]]) / 3.0;
                project(2, e.tag, cc);
                int r = insert(cc, 2, e.tag, t, i, -1, -1, h, true, true, enc);
                if (r == INS_ENC_SEG && may_split_seg(enc.a, enc.b)) {
                    st.enc++;
                    push_seg(enc.a, enc.b, enc.tag, 10.0);
                    qf.push(e);
                    continue;
                }
                if (r == INS_OK) {
                    st.ins++; st.face++;
                    push_created();
                } else
                    st.skipped++;    // shape only: leave it
                continue;
            }
            if (!qt.empty()) {
                QE e = qt.top();
                qt.pop();
                int t = e.slot;
                if (!alive[t] || T[4 * t] != e.v[0] || T[4 * t + 1] != e.v[1] || T[4 * t + 2] != e.v[2] || T[4 * t + 3] != e.v[3]) continue;
                uint64_t tk = ((uint64_t)(uint32_t)e.v[0] * 2654435761u) ^ ((uint64_t)(uint32_t)e.v[1] << 20) ^
                              ((uint64_t)(uint32_t)e.v[2] << 40) ^ ((uint64_t)(uint32_t)e.v[3] * 0x9E3779B97F4A7C15ULL);
                if (too_many(tk)) { st.skipped++; continue; }
                int v[4] = {T[4 * t], T[4 * t + 1], T[4 * t + 2], T[4 * t + 3]};
                if (e.prio > 1.0) {
                    // too large: halve the longest edge (terminates: every split halves an edge)
                    if (split_longest(v, 4, t) == INS_OK) push_created();
                    continue;
                }
                // bad shape: Delaunay refinement with the circumcentre (it must lie near the tet)
                double h = hmean(v, 4);
                double cc[3], gc[3], lmax = 0;
                for (int k = 0; k < 3; k++) gc[k] = 0.25 * (P(v[0])[k] + P(v[1])[k] + P(v[2])[k] + P(v[3])[k]);
                for (int k = 0; k < 6; k++) lmax = std::max(lmax, dist2(P(v[EV[k][0]]), P(v[EV[k][1]])));
                int r = INS_FAIL;
                if (circumcenter_tet(P(v[0]), P(v[1]), P(v[2]), P(v[3]), cc) && dist2(cc, gc) <= 0.36 * lmax)
                    r = insert(cc, 3, 0, t, -1, -1, -1, h, true, true, enc);
                if (r == INS_ENC_FACE && enc.tag > 0 && may_split_face(enc.a, enc.b, enc.c)) {
                    st.enc++;
                    push_face(enc.a, enc.b, enc.c, enc.tag, 10.0);
                    qt.push(e);
                    continue;
                }
                if (r == INS_OK) {
                    st.ins++; st.tet++;
                    push_created();
                } else
                    st.skipped++;
                continue;
            }
            break;
        }
        // drop remaining queue entries
        qs = {}; qf = {}; qt = {};
    }

    // ---------------------------------------------------------------------------------------------
    // local improvement: flips and smoothing of new nodes
    double tq(const int *v) const { return mean_ratio(P(v[0]), P(v[1]), P(v[2]), P(v[3])); }

    bool flip23(int t, int i) {
        int n = N[4 * t + i];
        if (n < 0 || F[4 * t + i] != 0 || vol[n] != vol[t]) return false;
        int a, b, c;
        face_verts(t, i, a, b, c);
        int d = T[4 * t + i];
        int e = T[4 * n + face_of(n, a, b, c)];
        NewTet nt[3] = {{{a, b, e, d}, vol[t]}, {{b, c, e, d}, vol[t]}, {{c, a, e, d}, vol[t]}};
        double qold = std::min(tq(&T[4 * t]), tq(&T[4 * n]));
        double qnew = 1e300;
        for (auto &x : nt) {
            if (!(orient(P(x.v[0]), P(x.v[1]), P(x.v[2]), P(x.v[3])) > 0)) return false;
            qnew = std::min(qnew, tq(x.v));
        }
        if (qnew <= qold * 1.05) return false;
        std::vector<int> old = {t, n};
        std::vector<NewTet> nts(nt, nt + 3);
        static const std::unordered_map<K3, int, K3H> none;
        static const std::unordered_set<K3, K3H> nonr;
        if (!replace(old, nts, none, nonr, true)) return false;
        replace(old, nts, none, nonr, false);
        st.f23++;
        return true;
    }
    // Edge removal: the 3..7 tets around an inner edge (a, b) are replaced by the best triangulation
    // of the ring of their other vertices (Klincsek's dynamic programming); n = 3 is the 3-2 flip.
    bool edge_remove(int t, int ea, int eb) {
        int a = T[4 * t + ea], b = T[4 * t + eb];
        if (seg.count(ekey(a, b))) return false;
        static thread_local std::vector<int> es;
        edge_star(a, b, t, es);
        int n = (int)es.size();
        if (n < 3 || n > 7) return false;
        for (int s : es) {
            if (vol[s] != vol[t]) return false;
            for (int j = 0; j < 4; j++) {
                int x = T[4 * s + j];
                if (x == a || x == b) continue;
                if (F[4 * s + j] != 0 || N[4 * s + j] < 0) return false;   // face containing the edge
            }
        }
        // ring of the other vertices, ordered so that orient(a, b, r_i, r_i+1) > 0
        int ring[8], x = -1, y = -1;
        for (int k = 0; k < 4; k++) {
            int v = T[4 * t + k];
            if (v == a || v == b) continue;
            if (x < 0) x = v; else y = v;
        }
        if (!(orient(P(a), P(b), P(x), P(y)) > 0)) std::swap(x, y);
        ring[0] = x; ring[1] = y;
        int m = 2, cur = t, pv = x, cv = y;
        for (;;) {
            int j = -1;
            for (int k = 0; k < 4; k++) if (T[4 * cur + k] == pv) j = k;
            int nx = j >= 0 ? N[4 * cur + j] : -1;
            if (nx < 0) return false;
            int w = -1;
            for (int k = 0; k < 4; k++) {
                int v = T[4 * nx + k];
                if (v != a && v != b && v != cv) w = v;
            }
            if (w == ring[0]) break;
            if (m >= n || w < 0) return false;
            ring[m++] = w;
            pv = cv; cv = w; cur = nx;
        }
        if (m != n) return false;
        double qold = 1e300;
        for (int s : es) qold = std::min(qold, tq(&T[4 * s]));
        double Q[8][8];
        int K[8][8];
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++) { Q[i][j] = 1e300; K[i][j] = -1; }
        for (int len = 2; len < n; len++)
            for (int i = 0; i + len < n; i++) {
                int j = i + len;
                double best = -2;
                int bk = -1;
                for (int k = i + 1; k < j; k++) {
                    double q = std::min(Q[i][k], Q[k][j]);
                    if (q <= best) continue;
                    int va[4] = {ring[i], ring[k], ring[j], b};
                    int vb[4] = {ring[i], ring[j], ring[k], a};
                    q = std::min(q, std::min(tq(va), tq(vb)));
                    if (q > best) { best = q; bk = k; }
                }
                Q[i][j] = best;
                K[i][j] = bk;
            }
        double qnew = Q[0][n - 1];
        if (!(qnew > 0) || qnew <= qold * 1.05) return false;
        std::vector<NewTet> nts;
        int stk[16][2], sp = 0;
        stk[sp][0] = 0; stk[sp][1] = n - 1; sp++;
        while (sp) {
            sp--;
            int i = stk[sp][0], j = stk[sp][1];
            if (j - i < 2) continue;
            int k = K[i][j];
            if (k < 0) return false;
            nts.push_back({{ring[i], ring[k], ring[j], b}, vol[t]});
            nts.push_back({{ring[i], ring[j], ring[k], a}, vol[t]});
            stk[sp][0] = i; stk[sp][1] = k; sp++;
            stk[sp][0] = k; stk[sp][1] = j; sp++;
        }
        std::vector<int> old(es.begin(), es.end());
        static const std::unordered_map<K3, int, K3H> none;
        static const std::unordered_set<K3, K3H> nonr;
        if (!replace(old, nts, none, nonr, true)) return false;
        replace(old, nts, none, nonr, false);
        if (n == 3) st.f32++; else st.f44++;
        return true;
    }

    // Move a new node to raise the worst quality of its tets: centroid of the neighbours, then steps
    // along the gradient of the worst tet. Nodes on flat surfaces move in their plane; nodes on curves
    // and curved surfaces stay where they are.
    bool smooth_vertex(int v) {
        if (!vnew[v]) return false;
        int d = vdim[v];
        double nrm[3] = {0, 0, 0};
        static thread_local std::vector<int> stv;
        vertex_star(v, stv);
        if (stv.empty()) return false;
        bool onc = d == 2 && is_curved(2, vtag[v]);   // curved surface: move in the tangent plane, project
        if (d == 2) {
            // mean normal of the node's surface triangles (signs aligned with the first one)
            double ref[3] = {0, 0, 0};
            for (int t : stv)
                for (int j = 0; j < 4; j++) {
                    if (F[4 * t + j] != vtag[v] || T[4 * t + j] == v) continue;
                    int a, b, c;
                    face_verts(t, j, a, b, c);
                    double u[3], w[3], n[3];
                    vsub(P(b), P(a), u); vsub(P(c), P(a), w); vcross(u, w, n);
                    if (ref[0] == 0 && ref[1] == 0 && ref[2] == 0) std::memcpy(ref, n, sizeof ref);
                    double sg = vdot(n, ref) < 0 ? -1.0 : 1.0;
                    for (int e = 0; e < 3; e++) nrm[e] += sg * n[e];
                }
            double l = std::sqrt(vdot(nrm, nrm));
            if (!(l > 0)) return false;
            for (double &c : nrm) c /= l;
        } else if (d != 3)
            return false;
        auto starq = [&](double floor) {
            double q = 1e300;
            for (int t : stv) {
                q = std::min(q, tet_quality(t));
                if (q <= floor) break;
            }
            return q;
        };
        auto inplane = [&](double *g) {
            if (d != 2) return;
            double s = vdot(g, nrm);
            for (int k = 0; k < 3; k++) g[k] -= s * nrm[k];
        };
        double *x = &X[3 * (size_t)v];
        double orig[3] = {x[0], x[1], x[2]}, best[3] = {x[0], x[1], x[2]};
        double q0 = starq(-1e300), qbest = q0;
        double c[3] = {0, 0, 0}, L = 0;
        int cnt = 0;
        uint32_t vp = new_vep();
        for (int t : stv)
            for (int k = 0; k < 4; k++) {
                int u = T[4 * t + k];
                if (u == v || vm[u] == vp) continue;
                vm[u] = vp;
                for (int e = 0; e < 3; e++) c[e] += P(u)[e];
                L += std::sqrt(dist2(P(u), orig));
                cnt++;
            }
        if (!cnt) return false;
        L /= cnt;
        for (int e = 0; e < 3; e++) c[e] = c[e] / cnt - orig[e];
        inplane(c);
        auto place = [&]() {
            if (onc) project(2, vtag[v], x);
        };
        for (double w : {1.0, 0.5, 0.25}) {
            for (int e = 0; e < 3; e++) x[e] = orig[e] + w * c[e];
            place();
            double q = starq(qbest);
            if (q > qbest) {
                qbest = q;
                std::memcpy(best, x, sizeof best);
                break;
            }
        }
        for (int it = 0; it < 12; it++) {
            std::memcpy(x, best, sizeof best);
            int wt = -1;
            double wq = 1e300;
            for (int t : stv) {
                double q = tet_quality(t);
                if (q < wq) { wq = q; wt = t; }
            }
            double g[3], h = 1e-4 * L;
            for (int e = 0; e < 3; e++) {
                x[e] = best[e] + h;
                double qp = tet_quality(wt);
                x[e] = best[e] - h;
                double qm = tet_quality(wt);
                x[e] = best[e];
                g[e] = (qp - qm) / (2 * h);
            }
            inplane(g);
            double gl = std::sqrt(vdot(g, g));
            if (!(gl > 0)) break;
            bool moved = false;
            for (double s : {0.3, 0.1, 0.03, 0.01}) {
                for (int e = 0; e < 3; e++) x[e] = best[e] + s * L * g[e] / gl;
                place();
                double q = starq(qbest);
                if (q > qbest * 1.0001) {
                    qbest = q;
                    std::memcpy(best, x, sizeof best);
                    moved = true;
                    break;
                }
            }
            if (!moved) break;
        }
        if (qbest > q0 * 1.02) {
            std::memcpy(x, best, sizeof best);
            st.smooth++;
            return true;
        }
        std::memcpy(x, orig, sizeof orig);
        return false;
    }
    void improve() {
        double thr = prm.sliver;
        for (int sweep = 0; sweep < (int)std::max(1.0, prm.smooth); sweep++) {
            std::vector<int> cand;
            for (int t = 0; t < nslots(); t++)
                if (alive[t] && (prm.improve_all > 0 || touch_zone(&T[4 * t], 4)) && (tet_dihedral(t) < thr || tet_quality(t) < 0.3))
                    cand.push_back(t);
            if (cand.empty()) break;
            int changes = 0;
            for (int t : cand) {
                if (!alive[t] || (tet_dihedral(t) >= thr && tet_quality(t) >= 0.3)) continue;
                bool done = false;
                for (int i = 0; i < 4 && !done; i++) done = flip23(t, i);
                for (int e = 0; e < 6 && !done && alive[t]; e++) done = edge_remove(t, EV[e][0], EV[e][1]);
                if (!done && alive[t])
                    for (int k = 0; k < 4 && !done; k++) done = smooth_vertex(T[4 * t + k]);
                if (done) changes++;
            }
            if (!changes) break;
        }
    }

    // ---------------------------------------------------------------------------------------------
    int check(std::string &msg) {
        int bad = 0;
        char buf[256];
        auto rep = [&](const char *s) {
            if (bad < 10) { msg += s; msg += "\n"; }
            bad++;
        };
        for (int t = 0; t < nslots(); t++) {
            if (!alive[t]) continue;
            if (!(tet_orient(t) > 0)) { std::snprintf(buf, sizeof buf, "tet %d not positive", t); rep(buf); }
            for (int i = 0; i < 4; i++) {
                int n = N[4 * t + i];
                int a, b, c;
                face_verts(t, i, a, b, c);
                if (n < 0) {
                    if (F[4 * t + i] == 0) { std::snprintf(buf, sizeof buf, "boundary face of tet %d untagged", t); rep(buf); }
                    continue;
                }
                if (!alive[n]) { std::snprintf(buf, sizeof buf, "tet %d neighbour %d dead", t, n); rep(buf); continue; }
                int j = face_of(n, a, b, c);
                if (j < 0 || N[4 * n + j] != t) { std::snprintf(buf, sizeof buf, "tet %d/%d neighbour link broken", t, n); rep(buf); continue; }
                int x, y, z;
                face_verts(n, j, x, y, z);
                if (!(k3(a, b, c) == k3(x, y, z))) { std::snprintf(buf, sizeof buf, "tet %d/%d shared face mismatch", t, n); rep(buf); }
                if (F[4 * n + j] != F[4 * t + i]) { std::snprintf(buf, sizeof buf, "tet %d/%d face tag mismatch", t, n); rep(buf); }
                if (vol[n] != vol[t] && F[4 * t + i] == 0) { std::snprintf(buf, sizeof buf, "volume interface %d/%d untagged", t, n); rep(buf); }
            }
            for (int k = 0; k < 4; k++) {
                int v = T[4 * t + k];
                if (vt[v] < 0 || !alive[vt[v]] || !has_vertex(vt[v], v)) { std::snprintf(buf, sizeof buf, "vertex %d has a stale tet", v); rep(buf); }
            }
        }
        for (auto &s : seg) {
            int a = (int)(s.first >> 32), b = (int)(s.first & 0xffffffffu);
            if (tet_with_edge(a, b) < 0) { std::snprintf(buf, sizeof buf, "segment %d-%d missing", a, b); rep(buf); }
        }
        return bad;
    }
};

std::vector<int> g_dummy;

}   // namespace

// =================================================================================================
extern "C" {

HZM_API const char *hzm_last_error(void) { return g_err.c_str(); }

HZM_API void *hzm_create(int nv, const double *X, const int *vdim, const int *vtag,
                       int nt, const int *T, const int *tvol,
                       int nf, const int *tri, const int *ftag,
                       int ns, const int *seg, const int *stag) {
    g_err.clear();
    Mesh *m = new Mesh();
    m->X.assign(X, X + 3 * (size_t)nv);
    m->vdim.assign(vdim, vdim + nv);
    m->vtag.assign(vtag, vtag + nv);
    m->vt.assign(nv, -1);
    m->hs.assign(nv, 1e300);
    m->vz.assign(nv, 0);
    m->vnew.assign(nv, 0);
    m->vm.assign(nv, 0);
    for (int t = 0; t < nt; t++) {
        int s = m->alloc_tet();
        for (int k = 0; k < 4; k++) {
            int v = T[4 * t + k];
            if (v < 0 || v >= nv) {
                g_err = "tet with an invalid node";
                delete m;
                return nullptr;
            }
            m->T[4 * s + k] = v;
        }
        m->vol[s] = tvol[t];
        m->tnew[s] = 0;
        double o = m->tet_orient(s);
        if (o < 0) std::swap(m->T[4 * s + 2], m->T[4 * s + 3]);
        else if (!(o > 0)) {
            g_err = "degenerate (flat) tetrahedron in the input mesh";
            delete m;
            return nullptr;
        }
        for (int k = 0; k < 4; k++) m->vt[m->T[4 * s + k]] = s;
    }
    // face adjacency by sorting
    struct FR { K3 k; int t, i; };
    std::vector<FR> fr;
    fr.reserve(4 * (size_t)nt);
    for (int t = 0; t < nt; t++)
        for (int i = 0; i < 4; i++) {
            int a, b, c;
            m->face_verts(t, i, a, b, c);
            fr.push_back({k3(a, b, c), t, i});
        }
    auto lt = [](const FR &x, const FR &y) {
        if (x.k.a != y.k.a) return x.k.a < y.k.a;
        if (x.k.b != y.k.b) return x.k.b < y.k.b;
        return x.k.c < y.k.c;
    };
    std::sort(fr.begin(), fr.end(), lt);
    for (size_t k = 0; k < fr.size();) {
        size_t e = k + 1;
        while (e < fr.size() && fr[e].k == fr[k].k) e++;
        if (e - k == 2) {
            m->N[4 * fr[k].t + fr[k].i] = fr[k + 1].t;
            m->N[4 * fr[k + 1].t + fr[k + 1].i] = fr[k].t;
        } else if (e - k > 2) {
            g_err = "non-manifold face in the input mesh";
            delete m;
            return nullptr;
        }
        k = e;
    }
    // surface tags
    int missing = 0;
    for (int f = 0; f < nf; f++) {
        FR key;
        key.k = k3(tri[3 * f], tri[3 * f + 1], tri[3 * f + 2]);
        auto it = std::lower_bound(fr.begin(), fr.end(), key, lt);
        if (it == fr.end() || !(it->k == key.k)) {
            missing++;
            continue;
        }
        int t = it->t, i = it->i;
        m->F[4 * t + i] = ftag[f];
        int n = m->N[4 * t + i];
        if (n >= 0) m->F[4 * n + m->face_of(n, key.k.a, key.k.b, key.k.c)] = ftag[f];
        // orientation reference for the output: the volume the triangle normal points into
        const double *a = m->P(tri[3 * f]), *b = m->P(tri[3 * f + 1]), *c = m->P(tri[3 * f + 2]);
        int apex = m->T[4 * t + i];
        int side = orient(a, b, c, m->P(apex)) > 0 ? m->vol[t] : (n >= 0 ? m->vol[n] : -1);
        if (!m->sideVol.count(ftag[f])) m->sideVol[ftag[f]] = side;
    }
    if (missing) {
        g_err = std::to_string(missing) + " surface triangles are not faces of the tet mesh";
        delete m;
        return nullptr;
    }
    // faces on the domain boundary or between volumes without a surface tag (should not happen)
    int untagged = 0;
    for (int t = 0; t < nt; t++)
        for (int i = 0; i < 4; i++) {
            int n = m->N[4 * t + i];
            if (m->F[4 * t + i] == 0 && (n < 0 || m->vol[n] != m->vol[t])) {
                m->F[4 * t + i] = -1;   // constrained, but not written out
                untagged++;
            }
        }
    if (untagged) g_err = std::to_string(untagged) + " untagged boundary faces (kept fixed)";
    for (int s = 0; s < ns; s++) m->seg[ekey(seg[2 * s], seg[2 * s + 1])] = stag[s];
    return m;
}

HZM_API void *hzm_clone(void *p) { return new Mesh(*(Mesh *)p); }
HZM_API void hzm_free(void *p) { delete (Mesh *)p; }

HZM_API void hzm_set_curved(void *p, int dim, int tag, int curved) {
    Mesh *m = (Mesh *)p;
    auto &s = dim == 1 ? m->curvedC : m->curvedS;
    if (curved) s.insert(tag);
    else s.erase(tag);
}
HZM_API void hzm_set_projector(void *p, hzm_project_cb cb) { ((Mesh *)p)->proj = cb; }

HZM_API int hzm_set_size(void *p, const double *h, const unsigned char *zone) {
    Mesh *m = (Mesh *)p;
    for (int v = 0; v < m->nv(); v++) {
        m->hs[v] = h[v];
        m->vz[v] = zone[v] ? 1 : 0;
        m->vnew[v] = 0;
    }
    for (int t = 0; t < m->nslots(); t++) m->tnew[t] = 0;
    return 0;
}

HZM_API int hzm_refine(void *p, const double *params, double *stats) {
    Mesh *m = (Mesh *)p;
    double *pp = (double *)&m->prm;
    int np = (int)(sizeof(Params) / sizeof(double));
    for (int k = 0; k < np; k++)
        if (params && std::isfinite(params[k]) && params[k] != -12345.0) pp[k] = params[k];
    m->st = Stats();
    auto t0 = std::chrono::steady_clock::now();
    m->refine();
    auto t1 = std::chrono::steady_clock::now();
    if (m->prm.improve > 0) m->improve();
    auto t2 = std::chrono::steady_clock::now();
    m->st.newtets = m->nalive - m->nt_start;
    m->st.t_refine = std::chrono::duration<double, std::milli>(t1 - t0).count();
    m->st.t_improve = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double qmin = 1e300, dmin = 1e300;
    for (int t = 0; t < m->nslots(); t++)
        if (m->alive[t]) {
            qmin = std::min(qmin, m->tet_quality(t));
            if (m->tnew[t]) dmin = std::min(dmin, m->tet_dihedral(t));
        }
    m->st.qmin = qmin;
    m->st.dmin = dmin;
    if (stats) std::memcpy(stats, &m->st, sizeof(Stats));
    return 0;
}

HZM_API int hzm_grade(void *p, const double *h0, const double *hcap, double slope, double *hout) {
    // hout = min(hcap, min_z h0[z] + slope * (edge-path distance to z)), multi-source Dijkstra
    Mesh *m = (Mesh *)p;
    int nv = m->nv();
    std::vector<int> deg(nv + 1, 0);
    std::vector<uint64_t> ed;
    ed.reserve(7 * (size_t)m->nalive);
    for (int t = 0; t < m->nslots(); t++) {
        if (!m->alive[t]) continue;
        for (int e = 0; e < 6; e++) ed.push_back(ekey(m->T[4 * t + EV[e][0]], m->T[4 * t + EV[e][1]]));
    }
    std::sort(ed.begin(), ed.end());
    ed.erase(std::unique(ed.begin(), ed.end()), ed.end());
    for (uint64_t k : ed) { deg[(k >> 32) + 1]++; deg[(k & 0xffffffffu) + 1]++; }
    for (int v = 0; v < nv; v++) deg[v + 1] += deg[v];
    std::vector<int> adj(deg[nv]), pos(deg.begin(), deg.end() - 1);
    for (uint64_t k : ed) {
        int a = (int)(k >> 32), b = (int)(k & 0xffffffffu);
        adj[pos[a]++] = b;
        adj[pos[b]++] = a;
    }
    typedef std::pair<double, int> HV;
    std::priority_queue<HV, std::vector<HV>, std::greater<HV>> q;
    for (int v = 0; v < nv; v++) {
        hout[v] = std::min(h0[v], hcap[v]);
        if (h0[v] < hcap[v]) q.push({hout[v], v});
    }
    while (!q.empty()) {
        HV x = q.top();
        q.pop();
        int v = x.second;
        if (x.first > hout[v]) continue;
        for (int k = deg[v]; k < deg[v + 1]; k++) {
            int u = adj[k];
            double c = hout[v] + slope * std::sqrt(dist2(m->P(u), m->P(v)));
            if (c < hout[u]) {
                hout[u] = c;
                q.push({c, u});
            }
        }
    }
    return 0;
}

HZM_API int hzm_check(void *p) {
    std::string msg;
    int n = ((Mesh *)p)->check(msg);
    g_err = msg;
    return n;
}

HZM_API void hzm_sizes(void *p, int *out) {
    Mesh *m = (Mesh *)p;
    int nf = 0;
    for (int t = 0; t < m->nslots(); t++) {
        if (!m->alive[t]) continue;
        for (int i = 0; i < 4; i++) {
            if (m->F[4 * t + i] <= 0) continue;
            int n = m->N[4 * t + i];
            if (n >= 0 && n < t) continue;
            nf++;
        }
    }
    out[0] = m->nv();
    out[1] = m->nalive;
    out[2] = nf;
    out[3] = (int)m->seg.size();
}

HZM_API void hzm_get(void *p, double *X, int *vdim, int *vtag, int *T, int *tvol,
                   int *tri, int *ftag, int *seg, int *stag) {
    Mesh *m = (Mesh *)p;
    std::memcpy(X, m->X.data(), sizeof(double) * m->X.size());
    std::memcpy(vdim, m->vdim.data(), sizeof(int) * m->vdim.size());
    std::memcpy(vtag, m->vtag.data(), sizeof(int) * m->vtag.size());
    int k = 0, f = 0;
    for (int t = 0; t < m->nslots(); t++) {
        if (!m->alive[t]) continue;
        for (int j = 0; j < 4; j++) T[4 * k + j] = m->T[4 * t + j];
        tvol[k] = m->vol[t];
        k++;
        for (int i = 0; i < 4; i++) {
            int tag = m->F[4 * t + i];
            if (tag <= 0) continue;
            int n = m->N[4 * t + i];
            if (n >= 0 && n < t) continue;
            int a, b, c;
            m->face_verts(t, i, a, b, c);
            // FV order: normal points into t; flip unless t is the side the surface's normals face
            auto sv = m->sideVol.find(tag);
            int want = sv != m->sideVol.end() ? sv->second : m->vol[t];
            bool into_t = (want == m->vol[t]) && !(n >= 0 && m->vol[n] == m->vol[t] && want != m->vol[t]);
            if (!into_t) std::swap(b, c);
            tri[3 * f] = a; tri[3 * f + 1] = b; tri[3 * f + 2] = c;
            ftag[f] = tag;
            f++;
        }
    }
    int s = 0;
    std::vector<std::pair<uint64_t, int>> sv(m->seg.begin(), m->seg.end());
    std::sort(sv.begin(), sv.end());
    for (auto &e : sv) {
        seg[2 * s] = (int)(e.first >> 32);
        seg[2 * s + 1] = (int)(e.first & 0xffffffffu);
        stag[s] = e.second;
        s++;
    }
}

HZM_API void hzm_quality(void *p, double *q, double *dih) {
    Mesh *m = (Mesh *)p;
    int k = 0;
    for (int t = 0; t < m->nslots(); t++) {
        if (!m->alive[t]) continue;
        if (q) q[k] = m->tet_quality(t);
        if (dih) dih[k] = m->tet_dihedral(t);
        k++;
    }
}

}   // extern "C"
