# Benchmarks

Timings of **Tools > Solver Benchmark** on a workstation. Every core count solves the whole setup in a fresh
solver process: initial mesh, adaptive passes and a 101-point interpolating frequency sweep. The projects are
in [`examples/benchmark/scaling`](../examples/benchmark/scaling), the raw results of every run in
[`examples/benchmark/scaling/results`](../examples/benchmark/scaling/results). Run the same benchmark on your
own computer to find the best number of cores and sweep processes for it.

## Computer

| | |
|---|---|
| CPU | AMD Zen 2 (reported as "AMD64 Family 23 Model 49"), 32 cores, 64 threads |
| Memory | 256 GB |
| System | Windows 10, Python 3.13 |

Measured on 25 October 2026 with version 0.2.3 (EMerge 3.0.0b5, hzAURA refinement), one run after the
other on an otherwise idle computer. The earlier runs on the same computer (18 and 19 October, with the
refinement of that time) are compared [below](#compared-with-the-runs-of-19-october).

## Broadband sweeps

The cases sweep a wide band, not a few frequencies around one point: the patch antenna from 1.225 to
3.675 GHz (3:1, half to one and a half times its 2.45 GHz resonance), the microstrip from 1 to 6 GHz (6:1),
the hairpin filter from 2 to 2.9 GHz. Each sweep has 101 points. The interpolating sweep solves only some of
them (circles) and fits a rational model through them, adding frequencies until the model changes by less
than the tolerance between steps. For the patch, 6 of its 21 solved frequencies come from the reduced basis
of the earlier solutions instead of a new factorization, so 15 factorizations cover the whole 3:1 band.

![S-parameters of the sweeps with the solved frequencies](benchmark/sweeps.png)

*Results of the 64-core runs. Line: the 101 sweep points. Circles: the frequencies that were solved.
Dashed: S11 of the runs of 19 October.*

## Patch antenna (examples/benchmark/scaling/large_patch.hzn)

2.45 GHz patch antenna, radiation boundary, second-order elements. 8 adaptive passes up to 34.5k
tetrahedra (319,488 unknowns, 2.7 GB per factorization, 7.2 GB peak memory), then the 1.225–3.675 GHz sweep.

| Cores | Total | Adaptive passes | Sweep | Speedup |
|---|---|---|---|---|
| 1 | 370 s | 83 s | 284 s | 1.0 |
| 2 | 228 s | 58 s | 167 s | 1.6 |
| 4 | 146 s | 42 s | 101 s | 2.5 |
| 8 | 103 s | 33 s | 66 s | 3.6 |
| 16 | 85 s | 30 s | 52 s | 4.3 |
| 32 | 81 s | 30 s | 48 s | 4.6 |
| 64 | 82 s | 31 s | 49 s | 4.5 |

Sweep with parallel processes (32 cores shared): 1 process 51 s, **2 processes 34 s**, 4 processes 34 s,
8 processes 38 s.

## Small microstrip (examples/benchmark/scaling/small_microstrip.hzn)

50 Ω microstrip with two lumped ports. 5 adaptive passes up to 13.4k tetrahedra (90,042 unknowns, 1.7 GB peak
memory), then the 1–6 GHz sweep with 17 solved frequencies.

| Cores | Total | Adaptive passes | Sweep | Speedup |
|---|---|---|---|---|
| 1 | 50 s | 17 s | 29 s | 1.0 |
| 2 | 33 s | 12 s | 18 s | 1.5 |
| 4 | 25 s | 10 s | 13 s | 2.0 |
| 8 | 22 s | 9 s | 10 s | 2.3 |
| 16 | 21 s | 9 s | 9 s | 2.4 |
| 32 | 21 s | 9 s | 9 s | 2.4 |
| 64 | 21 s | 9 s | 9 s | 2.4 |

Sweep with parallel processes (16 cores shared): 1 process 10 s, 2 processes 8.8 s, **4 processes 7.8 s**,
8 or more processes 10 s.

## Hairpin filter (examples/benchmark/scaling/medium_hairpin.hzn)

Band-pass filter with three hairpin resonators and two ports. 5 adaptive passes up to 18.5k tetrahedra
(190,587 unknowns, 1.8 GB per factorization, 4.4 GB peak memory), then the 2–2.9 GHz sweep with 17 solved
frequencies (7 factorizations). Not part of the earlier runs.

| Cores | Total | Adaptive passes | Sweep | Speedup |
|---|---|---|---|---|
| 1 | 145 s | 45 s | 97 s | 1.0 |
| 2 | 93 s | 32 s | 58 s | 1.6 |
| 4 | 65 s | 26 s | 36 s | 2.2 |
| 8 | 49 s | 21 s | 24 s | 3.0 |
| 16 | 43 s | 20 s | 20 s | 3.4 |
| 32 | 42 s | 20 s | 19 s | 3.5 |
| 64 | 42 s | 20 s | 19 s | 3.5 |

Sweep with parallel processes (32 cores shared): 1 process 21 s, 2 processes 15 s, **4 processes 15 s**,
8 processes 17 s.

## Compared with the runs of 19 October

Same computer, same projects and the same benchmark settings (cores 1 to 64, 101-point sweeps). The runs of
18 and 19 October used the mesh refinement of that time; with hzAURA the adaptive passes end on a different
mesh, so the mesh and the solve times changed together.

**Patch antenna**: before 66.9k tetrahedra in 10 passes (612,762 unknowns, 15.4 GB peak memory), now 34.5k in
8 passes (319,488 unknowns, 7.2 GB). Both stop on the same Maximum Delta S; the swept S11 agrees within
0.011 (resonance at 2.47 GHz, −25.1 dB, in both), see the plot above.

| Cores | Total before | Total now | Passes before | Passes now | Sweep before | Sweep now | Faster |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1375 s | 370 s | 315 s | 83 s | 1043 s | 284 s | 3.7× |
| 2 | 777 s | 228 s | 173 s | 58 s | 601 s | 167 s | 3.4× |
| 4 | 475 s | 146 s | 103 s | 42 s | 368 s | 101 s | 3.3× |
| 8 | 303 s | 103 s | 68 s | 33 s | 231 s | 66 s | 2.9× |
| 16 | 230 s | 85 s | 51 s | 30 s | 174 s | 52 s | 2.7× |
| 32 | 207 s | 81 s | 48 s | 30 s | 155 s | 48 s | 2.6× |
| 64 | 203 s | 82 s | 44 s | 31 s | 154 s | 49 s | 2.5× |

Fastest sweep with parallel processes: before 122 s (2 processes, 64 cores), now 34 s (2 processes, 32 cores).

**Small microstrip**: before 11.6k tetrahedra in 4 passes (2.0 GB peak memory), now 13.4k in 5 passes (1.7 GB),
17 instead of 19 solved frequencies. The mesh is a little larger and every run is still faster. S11 stays
below −21 dB in both runs; at that level the two curves differ by up to 0.024.

| Cores | Total before | Total now | Passes before | Passes now | Sweep before | Sweep now | Faster |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 78 s | 50 s | 20 s | 17 s | 54 s | 29 s | 1.6× |
| 2 | 53 s | 33 s | 16 s | 12 s | 33 s | 18 s | 1.6× |
| 4 | 39 s | 25 s | 14 s | 10 s | 21 s | 13 s | 1.5× |
| 8 | 32 s | 22 s | 13 s | 9 s | 15 s | 10 s | 1.5× |
| 16 | 30 s | 21 s | 13 s | 9 s | 13 s | 9 s | 1.4× |
| 32 | 29 s | 21 s | 12 s | 9 s | 13 s | 9 s | 1.4× |
| 64 | 29 s | 21 s | 12 s | 9 s | 13 s | 9 s | 1.4× |

Fastest sweep with parallel processes: before 10.5 s (8 processes, 64 cores), now 7.8 s (4 processes, 16 cores).

## What this means

* The patch antenna is solved 2.5 to 3.7 times faster than on 19 October, with the same result. Most of it
  comes from the smaller mesh that hzAURA converges to (half the unknowns); the microstrip, whose mesh grew a
  little, is still 1.4 to 1.6 times faster.
* The sweep is half to three quarters of the time of the patch and the filter. Its cost is the number of
  factorizations, not the number of sweep points.
* The models scale well up to 8 cores and gain little above 16, most likely because the sparse factorization
  is limited by memory bandwidth rather than by the number of cores.
* On a many-core computer, 2 to 4 parallel sweep processes cut the sweep by another 20 to 33 %.
* The **Fast** sweep factorizes once at the band centre (a few times for very wide bands) and could cut
  the sweep further; it was not part of this benchmark.
