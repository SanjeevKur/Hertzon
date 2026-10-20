/* Loads hzmesh as a shared library (DLL on Windows), refines a mesh read from a file and prints a
 * summary; used to check the Windows build under wine and that both builds give the same mesh. */
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#define LOADLIB(n) (void *)LoadLibraryA(n)
#define SYM(h, n) (void *)GetProcAddress((HMODULE)h, n)
#else
#include <dlfcn.h>
#define LOADLIB(n) dlopen(n, RTLD_NOW)
#define SYM(h, n) dlsym(h, n)
#endif

typedef void *(*create_t)(int, const double *, const int *, const int *, int, const int *, const int *, int,
                          const int *, const int *, int, const int *, const int *);
typedef int (*setsize_t)(void *, const double *, const unsigned char *);
typedef int (*refine_t)(void *, const double *, double *);
typedef int (*check_t)(void *);
typedef void (*sizes_t)(void *, int *);
typedef void (*get_t)(void *, double *, int *, int *, int *, int *, int *, int *, int *, int *);
typedef const char *(*err_t)(void);

static void *rd(FILE *f, size_t n) {
    void *p = malloc(n ? n : 1);
    if (n && fread(p, 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(2); }
    return p;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: hzmtest LIB MESHFILE\n"); return 1; }
    void *L = LOADLIB(argv[1]);
    if (!L) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    create_t create = (create_t)SYM(L, "hzm_create");
    setsize_t setsize = (setsize_t)SYM(L, "hzm_set_size");
    refine_t refine = (refine_t)SYM(L, "hzm_refine");
    check_t check = (check_t)SYM(L, "hzm_check");
    sizes_t sizes = (sizes_t)SYM(L, "hzm_sizes");
    get_t get = (get_t)SYM(L, "hzm_get");
    err_t err = (err_t)SYM(L, "hzm_last_error");
    FILE *f = fopen(argv[2], "rb");
    if (!f) return 1;
    int hd[4];
    if (fread(hd, sizeof(int), 4, f) != 4) return 1;
    int nv = hd[0], nt = hd[1], nf = hd[2], ns = hd[3];
    double *X = rd(f, 24 * (size_t)nv);
    int *vdim = rd(f, 4 * (size_t)nv), *vtag = rd(f, 4 * (size_t)nv);
    int *T = rd(f, 16 * (size_t)nt), *tvol = rd(f, 4 * (size_t)nt);
    int *tri = rd(f, 12 * (size_t)nf), *ftag = rd(f, 4 * (size_t)nf);
    int *sg = rd(f, 8 * (size_t)ns), *stag = rd(f, 4 * (size_t)ns);
    double *h = rd(f, 8 * (size_t)nv);
    unsigned char *z = rd(f, (size_t)nv);
    fclose(f);
    void *m = create(nv, X, vdim, vtag, nt, T, tvol, nf, tri, ftag, ns, sg, stag);
    if (!m) { fprintf(stderr, "create: %s\n", err()); return 1; }
    setsize(m, h, z);
    double st[32];
    refine(m, NULL, st);
    int bad = check(m);
    int sz[4];
    sizes(m, sz);
    double *X2 = malloc(24 * (size_t)sz[0]);
    int *i1 = malloc(4 * (size_t)sz[0]), *i2 = malloc(4 * (size_t)sz[0]);
    int *T2 = malloc(16 * (size_t)sz[1]), *tv2 = malloc(4 * (size_t)sz[1]);
    int *F2 = malloc(12 * (size_t)sz[2] + 4), *ft2 = malloc(4 * (size_t)sz[2] + 4);
    int *S2 = malloc(8 * (size_t)sz[3] + 4), *st2 = malloc(4 * (size_t)sz[3] + 4);
    get(m, X2, i1, i2, T2, tv2, F2, ft2, S2, st2);
    double cs = 0;
    long long ct = 0;
    for (int k = 0; k < 3 * sz[0]; k++) cs += X2[k] * (1 + k % 7);
    for (int k = 0; k < 4 * sz[1]; k++) ct += (long long)T2[k] * (1 + k % 5);
    printf("nodes %d tets %d tris %d segs %d | inserted %.0f failed %.0f | check %d | checksum %.12e %lld\n",
           sz[0], sz[1], sz[2], sz[3], st[0], st[5], bad, cs, ct);
    return bad != 0;
}
