# Waiting on EMerge

Hertzon carries a few workarounds for EMerge bugs that are being fixed upstream. This list says what each one
waits for, and how to check that it can go once the fix is in an EMerge release. Background on every
replacement is in `docs/EMERGE_INTEGRATION.md` and `docs/EMERGE_CHANGES.md`.

**When Hertzon moves to a new EMerge version:** go through this list. For every entry whose upstream fix is in
that version, run the check with the workaround switched off. Most workarounds can be switched off without a
code change with `HERTZON_EMERGE_NATIVE` (comma separated, e.g. `HERTZON_EMERGE_NATIVE=thin_conductor`). Only
when the check passes, remove the workaround in its own PR and move the entry to "Done" below.

## Open

| Upstream fix | Hertzon workaround | Waiting for | Check before removing |
|---|---|---|---|
| Release the factorizations of sweep worker threads on their own thread (memory grew with every batch of a threaded sweep) | `mt_pardiso.release` | Merged into EMerge `main` (FennisRobert/EMerge#72), not yet in a release | Threaded sweep with `HERTZON_EMERGE_NATIVE=mt_pardiso`: memory stays flat over many batches; same S-parameters as with the workaround |
| Parallel sweep threads share one solver (`AutomaticRoutine` singleton) | `mt_pardiso._duplicate` | Fixed on EMerge `main` (`e80e840`), not yet in a release | Parallel sweep of the benchmark models with `HERTZON_EMERGE_NATIVE=mt_pardiso`: no crash, same S-parameters |
| ThinConductor DoF split (fields at a conductor sheet read DoFs the solver does not use) | `emerge_patches.apply_thin_conductor` | The maintainer's rework on EMerge `main` (`77beeb6`, "Separate the two sides of a ThinConductor sheet in the solve"), not yet in a release | `HERTZON_EMERGE_NATIVE=thin_conductor` on the microstrip benchmark: no noise in field and surface current plots at the trace, S-parameters compared with the workaround (they may change, since the sheet may then be two-sided), adaptive meshing with a thin conductor still converges |

## Done

| Upstream fix | Removed workaround | EMerge version |
|---|---|---|
| Port mode solve stops on `ArpackNoConvergence` | `emerge_patches._robust_eigs` | 3.0.0b5 |
| Thin conductor split after a new mesh | `_thin_conductor_new_mesh` | 3.0.0b5 |
