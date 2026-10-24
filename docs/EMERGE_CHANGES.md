# Where Hertzon's EMerge changes go

Hertzon still changes parts of [EMerge](https://github.com/FennisRobert/EMerge) at run time
(see [EMERGE_INTEGRATION.md](EMERGE_INTEGRATION.md) for what each change does and what it is worth in
speed). Long term, none of this should stay as a run-time patch in Hertzon. This page decides for each
item where it goes instead, and why.

## The rules

The EMerge maintainer set the rules in the
[`CLAUDE.md` of EMerge's `hertzon` branch](https://github.com/FennisRobert/EMerge/blob/hertzon/CLAUDE.md).
In short:

| Rule | Content |
|---|---|
| R1 | A bug fix in EMerge itself (wrong results, crashes, broken API) is **always a PR to `main`**, one fix per PR, with a test. |
| R2 | A change that only matters for the Hertzon GUI stays on `hertzon`, never a PR to `main`. |
| R3 | Code written by an AI is marked `# [Written by Claude]` (Python) or `// [Written by Claude]` (Rust). |
| R4 | New technical features in EMerge's core (elements, solvers, assembly, EM formulations) stay on `hertzon`. A PR to `main` is discouraged: the maintainer wants to write this code himself. |
| R5 | Code already marked as AI written (AURA refinement, adaptive frequency sweep) may be improved freely, also on `main`. |
| R6 | On `hertzon`, add new code next to the existing code instead of editing it ("add, don't rewrite"), so `main` keeps merging into `hertzon` without conflicts. |

Hertzon adds two rules of its own:

| Rule | Content |
|---|---|
| H1 | Every change to EMerge is agreed with the Hertzon maintainer first. EMerge changes are made in EMerge (on `hertzon` or as a PR to `main`), not as new run-time patches in Hertzon. |
| H2 | Use of the public EMerge API that only the GUI needs stays in Hertzon. It is no patch and needs no change in EMerge. |

`main` is protected: changes reach it only as pull requests that the maintainer merges. Fix branches for
these pull requests start from `main`, never from `hertzon`.

## Decisions

### Bugs: pull request to `main` (R1)

| Item | Hertzon code today | Why `main` | Status |
|---|---|---|---|
| Parallel sweep threads share one solver (`AutomaticRoutine` singleton) | – | Crashes in MKL; affects every user of parallel sweeps | Fixed on `main` (`e80e840`); Hertzon's workaround removed |
| Factorizations of sweep worker threads are never released | – | Memory leak in every threaded sweep: scipy's SuperLU frees a factorization only in the thread that created it, and EMerge drops the worker routines in the main thread (+350 MB per batch of 4 frequencies on a 15k unknown matrix) | Fixed on `main` (#72, `821607b`); Hertzon's workaround removed |
| Thin conductor DoF numbering differs between assembly and post-processing | – | Fields and surface currents at every thin conductor | Fixed by the maintainer on `main` (`77beeb6`): both sides of the sheet are now solved separately. Hertzon's patch removed |
| Port mode solve ends the whole solve on `ArpackNoConvergence` | `emerge_patches._robust_eigs` | – | No PR: since 3.0.0b5 the port mode solvers factorize the shift once, use an oversampled Krylov space and skip a failing shift. Three waveport runs without Hertzon's retry and all benchmark runs: no failure. Dropped from Hertzon |
| Error estimate counts the current jump at finite-conductivity sheets as error | `bridge._with_conductor_sheets` | Wrong refinement at conductor sheets; may overlap with the maintainer's thin trace fix (`c951924`) | Needs a model that shows the effect first; none of the benchmark models does |
| gmsh fails on trimmed B-spline faces of CAD files | `emerge_patches.apply_mesh_fallback` | Meshing crash for imported CAD, not GUI specific | To do, low priority |
| Far field with symmetry planes: radiated power scaled by `factor**2` although only the simulated part is integrated | – (Hertzon does not mirror yet) | Possible wrong radiated power for every user of `syms` | Open. EMerge's horn demo (quarter model) is no reference: 2.25 W cross its radiation boundary at the PML for 0.99 W into the port, so the check needs a model with the integration surface inside the air, as a full and a half model |

### Speed features in the solver core: `hertzon` branch (R4, R6)

These are new technical capability in EMerge's core, so they go to `hertzon` as new code next to the
existing code, switched on by a setting, and marked as AI written (R3).

| Item | Hertzon code today | Gain (EMERGE_INTEGRATION.md) |
|---|---|---|
| PARDISO complex symmetric LDL^T (mtype 6) | `fast_assembly._pardiso_solve` | 7–34 % per model |
| Parallel removal of PEC rows/columns | `fast_assembly.submatrix` | 5–10 % |
| PARDISO per worker thread in parallel sweeps (EMerge uses SuperLU there) | `mt_pardiso` | parallel sweeps 5× faster, no out-of-memory |
| Far field for many excitations (phases once) as a new Rust kernel | `farfield_post.stratton_chu_multi` | 3.7–4.2× on 11 ports |
| Error estimate for many excitations in one pass | – | up to about 9 % with many ports, low priority |
| Curved (isoparametric) elements, natively in the Rust assembly | `curvilinear.py`, `ned2_kernels.py`, `emerge_patches.apply_curved` | accuracy on curved surfaces; the maintainer asked for this on `hertzon` |

The maintainer may later take any of them into `main` in his own way.

### AI-written EMerge code: free to improve (R5)

| Item | Decision |
|---|---|
| Hertzon's reduced-basis interpolating sweep (`hybrid_sweep`, 30–48 % of the run time) | First compare with EMerge's adaptive sweep (`adaptive_sweep.py`, AI written). If Hertzon's approach is better, improve `adaptive_sweep.py` (PR to `main` allowed) and switch Hertzon to it; otherwise Hertzon drops its own sweep. |
| Hertzon's refinement methods (local Delaunay, bisection, Mmg, remesh) | Stay in Hertzon for now. Compare with AURA (AI written); improvements to AURA that are clearly better and more stable may go to `main`. |

### GUI only: `hertzon` branch (R2) or Hertzon itself (H2)

| Item | Decision |
|---|---|
| Per-frequency "solved" callback for progress bars (wraps `SimJob.submit_solution`) | Small hook on `hertzon` |
| Solve hook for the reduced-basis sweep (wraps `SolveRoutine.solve`) | Disappears with the sweep decision above |
| Private API used by Hertzon (`Mesher._set_size_on_edge`, `Simulation._reset_mesh`, `MATHLIB`, `Assembler.assemble_*`) | Ask for stable names; until then Hertzon keeps using them |
| Plane wave angle convention (`definition="EA"`), initial mesh size (`set_global_max_size`) | Public API, stays in Hertzon |
| Hard process exit on Windows after the results are saved | Process handling, stays in Hertzon |
| Lazy fields, stored boundary fields for far fields after the solve, far field quantities (realized gain, efficiency, port superposition), face matching | Built on public results, stays in Hertzon |
| `MWField.interpolate(mapping=...)` instead of replacing `interpolate_index` | Done: Hertzon uses EMerge's API |

### Not EMerge: emsutil

`EHFieldFF` in [emsutil](https://github.com/FennisRobert/emsutil) (axial ratio sign, `gain`/`dir` with a
`kind` that cannot be passed, RCS without θ/φ split) are small bugs there, so pull requests to emsutil.

### Removed from Hertzon

Fixed in EMerge 3.0.0b5 or dead on EMerge 3, nothing to send: thin conductor split after a re-mesh
(`_thin_conductor_new_mesh`), the ARPACK retry (`_robust_eigs`), the welcome prompt workaround in `setup.ps1`, and the replacements for
EMerge 2.8.9 (`fast_error.py`, vectorised `_pre_update`, 2.8.9 assembly, `AMRPoints` patch).

## How Hertzon depends on EMerge

Hertzon installs EMerge from a pinned commit, normally the head of EMerge's `hertzon` branch, instead of a
PyPI release:

1. `packaging/emerge-commit` holds the commit.
2. When it changes, `.github/workflows/emerge-wheels.yml` builds the wheels (Windows x64, Linux x86_64,
   abi3, version `<EMerge version>.post<commit time>`) and publishes them as the pre-release
   `emerge-<short sha>` of the Hertzon repository, with a link to the EMerge source commit (GPL).
3. The `emerge` entries of `pyproject.toml` point at these wheels; on other platforms pip builds the same
   commit from source (needs Rust).

Updating EMerge is therefore one Hertzon pull request: new commit, wheel URLs, benchmark
(`examples/benchmark/tools/emerge_bench.py`) against the previous commit, results in
[EMERGE_INTEGRATION.md](EMERGE_INTEGRATION.md).
