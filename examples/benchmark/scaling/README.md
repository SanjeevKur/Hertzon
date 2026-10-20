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

Size and time with 4 cores (development container, for orientation):

| Case | Design | Final mesh | Passes | Total (passes, sweep) |
|---|---|---|---|---|
| small_microstrip.hzn | Microstrip50 | 18.2k tetrahedra | 6 | 74 s (23 s, 48 s) |
| medium_hairpin.hzn | Hairpin_N3 | 20.9k tetrahedra | 6 | 269 s (51 s, 214 s) |
| large_patch.hzn | Patch245 | 43.6k tetrahedra | 9 | 330 s (78 s, 249 s) |

All three have a 101-point interpolating sweep.
