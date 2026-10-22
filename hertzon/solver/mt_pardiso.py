"""Parallel sweeps with PARDISO: several frequencies at once, each on a share of the cores.

EMerge solves the frequencies of a parallel sweep in worker threads and allows only solvers that
"release the GIL" there, which leaves SuperLU: much slower than PARDISO and limited to 32 bit fill-in
("Not enough memory to perform factorization" on large models). PARDISO is safe in threads as long as
every thread has its own solver handle (each worker thread gets its own SolveRoutine with its own
PardisoInterface) and the ctypes call releases the GIL. Each worker thread limits MKL to its share of
the cores with mkl_set_num_threads_local, so N workers x (cores / N) threads keep the machine busy
without oversubscription. PARDISO's efficiency drops with many threads on one matrix, so on many
cores a few frequencies at once finish a sweep sooner than one at a time with all cores.

EMerge assembles all frequencies of a parallel sweep one after another and only then hands them to the
worker threads, so the workers wait for the whole assembly. run_sweep is changed to submit every
frequency as soon as it is assembled: the next assembly runs while the first ones are solved. EMerge
also drops the solvers of the worker threads without freeing PARDISO's memory (only the main solver
is reset), so every parallel batch left its factorizations allocated; they are freed now.
"""
from __future__ import annotations

import ctypes
import re
import threading
import time

WORKER_THREADS = 0      # MKL threads per sweep worker; 0: EMerge's own choice (SuperLU in threads)
_DONE = False
TIMES = {"assembly": 0.0, "solve": 0.0}     # seconds, the solve time summed over the worker threads
_LOCK = threading.Lock()

ROUTINES: list = []     # solve routines of the worker threads of the running parallel sweep

_PIPELINE = [   # (old, new) in the dedented source of Microwave3D.run_sweep, multi-threaded branch only
    ('                "MT", thread_nr=worker_nr\n            )\n        return thread_local.routine\n',
     '                "MT", thread_nr=worker_nr\n            )\n'
     '            _hzn_mt.ROUTINES.append(thread_local.routine)\n        return thread_local.routine\n'),
    ("    if parallel:\n        thread_local.__dict__.clear()\n",
     "    if parallel:\n        _hzn_mt.release()\n        thread_local.__dict__.clear()\n"),
    ("                jobs = []\n                ## Assemble jobs\n                for freq in fgroup:\n",
     "                jobs = []\n                futs = []\n                ## Assemble jobs\n"
     "                for freq in fgroup:\n"),
    ("                    jobs.append(job)\n                    material_set.append(mats)\n\n"
     "                # Solve all problems using the executor.\n",
     "                    jobs.append(job)\n                    material_set.append(mats)\n"
     "                    futs.append(executor.submit(run_job, job))\n\n"
     "                # Solve all problems using the executor.\n"),
    ("                group_results = list(executor.map(run_job, jobs))\n",
     "                group_results = [f.result() for f in futs]\n"),
]


def _main() -> bool:
    return threading.current_thread() is threading.main_thread()


def _set_threads(lib) -> None:
    """MKL threads of the calling thread only (the C name; the lowercase one is the Fortran entry point
    and takes a pointer)."""
    fn = getattr(lib, "MKL_Set_Num_Threads_Local", None) if lib is not None else None
    if fn is None:
        return
    fn.argtypes = [ctypes.c_int]
    fn.restype = ctypes.c_int
    fn(int(WORKER_THREADS))


def apply() -> None:
    global _DONE
    if _DONE:
        return
    _DONE = True
    from emerge._emerge import solver as S

    pick = S.AutomaticRoutine.pick_solver

    def pick_solver(self, A, b, matrix_type, direct: bool = True):
        if (WORKER_THREADS > 0 and self.parallel == "MT" and direct and A.shape[0] >= 10_000
                and S._PARDISO_AVAILABLE and S.EMSolver.PARDISO in self.solvers
                and not (matrix_type is S.MatrixType.SPD and S._SKSP_AVAILABLE)):
            return self.solvers[S.EMSolver.PARDISO]       # this worker thread's own instance
        return pick(self, A, b, matrix_type, direct)

    S.AutomaticRoutine.pick_solver = pick_solver
    if "__new__" in vars(S.AutomaticRoutine):
        S.SolveRoutine.duplicate = _duplicate


    solve = S.SolverPardiso.solve

    def pardiso_solve(self, A, b, precon, id: int = -1):  # noqa: A002
        if WORKER_THREADS > 0 and not _main():
            _set_threads(getattr(self.solver, "libmkl", None))
        t = time.perf_counter()
        try:
            return solve(self, A, b, precon, id)
        finally:
            with _LOCK:
                TIMES["solve"] += time.perf_counter() - t

    pardiso_solve._hertzon_fast = getattr(solve, "_hertzon_fast", False)
    S.SolverPardiso.solve = pardiso_solve

    from emerge._emerge.physics.microwave.assembly.assembler import Assembler
    assemble = Assembler.assemble_freq_matrix

    def assemble_freq_matrix(self, *a, **k):
        t = time.perf_counter()
        try:
            return assemble(self, *a, **k)
        finally:
            TIMES["assembly"] += time.perf_counter() - t

    assemble_freq_matrix._hertzon_fast = getattr(assemble, "_hertzon_fast", False)
    Assembler.assemble_freq_matrix = assemble_freq_matrix
    _pipeline()


def _duplicate(self):
    """A routine of its own for a worker thread. EMerge 3 made AutomaticRoutine a singleton, so its
    duplicate() returned the same routine: the worker threads shared it, replaced each other's solvers
    and now and then factorized in the same PARDISO handle at the same time (segmentation fault in
    MKL). Forced solvers are mapped to the copies as well, not shared with the main routine."""
    from emerge._emerge import solver as S
    new = object.__new__(type(self))
    S.SolveRoutine.__init__(new)
    keep = ("solvers", "forced_solver", "pre")
    new.__dict__.update({k: v for k, v in vars(self).items() if k not in keep})
    copies = {}
    for tpe, solver in self.solvers.items():
        copies[id(solver)] = new.solvers[tpe] = solver.duplicate()
    new.forced_solver = [copies.get(id(s), s) for s in self.forced_solver]
    return new


def _pipeline() -> None:
    """Solve the frequencies of a parallel sweep while the next ones are assembled (see above)."""
    from emerge._emerge.physics.microwave import microwave_3d as M
    from .fast_assembly import _patched
    fn = M.Microwave3D.run_sweep
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(fn))
    if any(src.count(old) != 1 for old, _ in _PIPELINE):
        return                                  # another EMerge version: keep its code
    new, n = _patched(fn, [(re.compile(re.escape(old)), rep.replace("\\", "\\\\")) for old, rep in _PIPELINE])
    if new is not None and n == len(_PIPELINE):
        import sys
        new.__globals__["_hzn_mt"] = sys.modules[__name__]
        M.Microwave3D.run_sweep = new


def release() -> None:
    """Free the PARDISO memory of the worker threads' solvers (the threads have finished)."""
    while ROUTINES:
        r = ROUTINES.pop()
        try:
            r.reset()
        except Exception:  # noqa: BLE001
            pass
