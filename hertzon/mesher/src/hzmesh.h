/* hzmesh - local constrained Delaunay refinement of tetrahedral meshes for adaptive FEM.
 *
 * C interface (used from Python through ctypes). All indices are 0-based.
 * Copyright (C) 2026 Hertzon contributors. GPL-2.0-or-later.
 */
#ifndef HZMESH_H
#define HZMESH_H

#ifdef _WIN32
#define HZM_API __declspec(dllexport)
#else
#define HZM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Projection of a point onto a curved CAD entity: dim 1 = curve, dim 2 = surface. */
typedef void (*hzm_project_cb)(int dim, int tag, const double *xyz_in, double *xyz_out);

/* Create a mesh. X: 3*nv, vdim/vtag: classification of every node (0 point, 1 curve, 2 surface,
 * 3 volume, with the entity tag), T: 4*nt node ids, tvol: volume tag per tet, tri: 3*nf surface
 * triangles with ftag (> 0), seg: 2*ns curve segments with stag. Returns NULL on error
 * (hzm_last_error() has the reason). */
HZM_API void *hzm_create(int nv, const double *X, const int *vdim, const int *vtag,
                       int nt, const int *T, const int *tvol,
                       int nf, const int *tri, const int *ftag,
                       int ns, const int *seg, const int *stag);
HZM_API void *hzm_clone(void *m);
HZM_API void hzm_free(void *m);
HZM_API const char *hzm_last_error(void);

/* Entities with curved geometry get their new nodes projected through the callback. */
HZM_API void hzm_set_curved(void *m, int dim, int tag, int curved);
HZM_API void hzm_set_projector(void *m, hzm_project_cb cb);

/* Target element size (edge length) per node and the refinement zone flag (nonzero = node may be
 * refined around). Nodes created later inherit both from the refined element. */
HZM_API int hzm_set_size(void *m, const double *h, const unsigned char *zone);

/* Graded size field on the current mesh: hout = min(hcap, min_z h0[z] + slope * path distance to z). */
HZM_API int hzm_grade(void *m, const double *h0, const double *hcap, double slope, double *hout);

/* Refine: params (see Params in hzmesh.cpp for the order), stats out (32 doubles, see Stats). */
HZM_API int hzm_refine(void *m, const double *params, double *stats);

/* Topology/geometry consistency check; returns the number of problems (message in hzm_last_error). */
HZM_API int hzm_check(void *m);

/* Sizes of the current mesh: out[0..3] = nv, nt, nf, ns. */
HZM_API void hzm_sizes(void *m, int *out);
HZM_API void hzm_get(void *m, double *X, int *vdim, int *vtag, int *T, int *tvol,
                   int *tri, int *ftag, int *seg, int *stag);
/* Per tet quality (mean ratio, 1 = regular) and min dihedral angle in degrees. */
HZM_API void hzm_quality(void *m, double *q, double *dih);

#ifdef __cplusplus
}
#endif
#endif
