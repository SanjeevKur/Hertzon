# Scaling benchmarks

The projects of Tools > Solver Benchmark: how the solve time changes with the number of cores on the
computer, and whether parallel sweep processes help. They are the Microstrip, PatchAntenna and
HairpinFilter examples with at most 50 adaptive passes (they stop on Maximum Delta S long before),
no saved fields and no fixed solver threads.

Headless:

```
python -m hertzon.solver.benchmark examples/benchmark/scaling/large_patch.hzn Patch245 Setup1 <out_dir> --cores 1,2,4,8 --workers 1,2,4
```

Every core count solves the whole setup in a fresh solver process (1 core first); the
`--workers` runs solve only the sweeps again, with `Parallel` on, on the adaptive mesh of the fastest run. The times per phase are in `<out_dir>/benchmark.json`.

Results on a 32-core workstation, with the swept bands plotted: [docs/BENCHMARKS.md](../../../docs/BENCHMARKS.md).

Size and time with 4 cores and hzAURA refinement (development container, for orientation):

| Case | Design | Final mesh | Passes | Total (passes, sweep) |
|---|---|---|---|---|
| small_microstrip.hzn | Microstrip50 | 13.5k tetrahedra | 5 | 32 s (11 s, 18 s) |
| large_patch.hzn | Patch245 | 19.9k tetrahedra | 6 | 104 s (29 s, 73 s) |
| medium_hairpin.hzn | Hairpin_N3 | 41.8k tetrahedra | 8 | 222 s (87 s, 132 s) |

On the 32-core Windows workstation of [docs/BENCHMARKS.md](../../../docs/BENCHMARKS.md) (25 October 2026) the
same projects end on 13.4k (microstrip, 5 passes), 34.5k (patch, 8 passes) and 18.5k tetrahedra (hairpin,
5 passes). The `benchmark.json` of every run there, and of the earlier runs of 19 October, is in
[`results`](results).

The file names come from the earlier refinement, with which the patch needed the largest mesh; with
hzAURA the hairpin filter does.

All three have a 101-point interpolating sweep.
