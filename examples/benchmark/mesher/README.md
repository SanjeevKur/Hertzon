# Mesher comparison: time to solution

Hertzon's local Delaunay refinement (hzmesh, `Local (Delaunay)`) against EMerge's AURA (`AURA (EMerge,
experimental)`) on the same models. Both runs of a case use the same error estimate (EMerge's, unchanged),
the same convergence criterion, the same growth per pass (30 %) and curved elements. Only the refinement
step between two passes differs.

```
python examples/benchmark/mesher/compare.py <out_dir>
python examples/benchmark/mesher/compare.py <out_dir> --cases post_pec,microstrip --methods aura
```

Every case is solved headless in a fresh solver process (`examples/benchmark/tools/profrun.py`), adaptive
passes only, no frequency sweep. A short warm-up run first compiles the numba kernels, so that this one-time
cost is not counted in the first case. Results: `<out_dir>/results.md`, per pass in `results.json`.

## Cases

| Case | Project | Reference |
|---|---|---|
| post_pec | WR90 waveguide, metal post r = 1.5 mm, 10 GHz | exact 2D reference (`waveguide_post/ref_wgpost.py`), S11 = 0.18037 + 0.93542j |
| post_dielectric | WR90 waveguide, post eps_r 9.8, r = 2 mm | exact 2D reference, S11 = -0.33672 + 0.77887j |
| microstrip | 50 Ω microstrip, two lumped ports, 4 GHz (`scaling/small_microstrip.hzn`) | none |
| patch | 2.45 GHz patch antenna (`patch_antenna/patch_local.hzn`) | none |

The convergence settings are those of each project (waveguide: Delta S <= 0.0005 in 2 passes, at most 12
passes; microstrip: Delta S <= 0.01 once; patch: Delta S <= 0.01 in 2 passes, at most 12).

## Results

4 cores, 16 GB (development container), EMerge 3.0.0b5 at commit `2811e9a`. "Time to error" adds up the
pass times until |S11 - S11,ref| is first at most 0.001. "Total" is the whole run, "Mesh" the refinement
plus mesh import. Full numbers: [results_4core.json](results_4core.json).

| Case | Method | Passes | Tets | Converged | Final Delta S | Error vs. reference | Time to error <= 0.001 | Total | Mesh |
|---|---|---|---|---|---|---|---|---|---|
| post_pec | local | 12 | 26694 | no | 0.0004 | 0.00004 | 13 s (8439 tets) | 56 s | 2.6 s |
| post_pec | aura | 10 | 12483 | yes | 0.0003 | 0.00013 | 11 s (7208 tets) | 20 s | 0.2 s |
| post_dielectric | local | 9 | 11723 | yes | 0.0001 | 0.00117 | not reached | 24 s | 1.3 s |
| post_dielectric | aura | 10 | 14660 | yes | 0.0002 | 0.00051 | 21 s (11139 tets) | 31 s | 0.2 s |
| microstrip | local | 8 | 27428 | yes | 0.0073 | no reference | - | 27 s | 4.0 s |
| microstrip | aura | 5 | 13455 | yes | 0.0067 | no reference | - | 11 s | 0.3 s |
| patch | local | 12 | 31563 | no | 0.0027 | no reference | - | 60 s | 3.3 s |
| patch | aura | 10 | 21239 | yes | 0.0060 | no reference | - | 35 s | 0.1 s |

What it shows:

- AURA reaches the convergence criterion faster in three of four cases (0.4 to 0.6 times the time), mostly
  because it needs fewer passes and fewer tetrahedra. The refinement step itself is 0.1-0.4 s per run
  against 1.3-4 s for hzmesh plus the import through gmsh; the solve dominates both.
- Where an exact reference exists, AURA is more accurate at the same tet count. With hzmesh the dielectric
  post stops at an error of 0.0012 although Delta S is only 0.0001: Delta S stalls before the result is
  right. The metal post with hzmesh ends more accurate (0.00004), but only after 12 passes, 26.7k tets and
  without meeting the criterion, because Delta S jumps in passes 10 and 11.
- Microstrip and patch have no converged reference yet. Longer runs (12 passes, Delta S limit off) in the
  same container: on the microstrip AURA at 31k tets is where hzmesh is at 83k (first order, Delta S at the
  end 0.0006 against 0.0021); on the patch S11 still changes with both at 40-50k tets. A fine reference
  needs more memory than 16 GB.
- No pass failed and no pass fell back to a full re-mesh with either method.

## Open

- Fine references for microstrip and patch on a larger computer, then the error column for them.
- More structures (periodic cell, coax, filter with thin traces).
