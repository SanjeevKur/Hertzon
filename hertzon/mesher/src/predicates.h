/* Shewchuk's robust adaptive geometric predicates (public domain). */
#ifndef HZM_PREDICATES_H
#define HZM_PREDICATES_H
#ifdef __cplusplus
extern "C" {
#endif
double orient2d(const double *pa, const double *pb, const double *pc);
double orient3d(const double *pa, const double *pb, const double *pc, const double *pd);
double incircle(const double *pa, const double *pb, const double *pc, const double *pd);
double insphere(const double *pa, const double *pb, const double *pc, const double *pd, const double *pe);
double orient3dfast(const double *pa, const double *pb, const double *pc, const double *pd);
#ifdef __cplusplus
}
#endif
#endif
