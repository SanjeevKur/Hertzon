# Waiting on EMerge

Hertzon is built on a pinned commit of EMerge's `hertzon` branch (`packaging/emerge-commit`). This list
says which EMerge fixes Hertzon is waiting for, what in Hertzon depends on each one, and what to check once
it arrives. Background on every EMerge change is in `docs/EMERGE_INTEGRATION.md` and
`docs/EMERGE_CHANGES.md`.

**When `packaging/emerge-commit` moves to a new EMerge commit:** go through this list. For every entry whose
fix is in that commit, run its check. Only when the check passes, remove the workaround (or enable the
feature) in its own PR and move the entry to "Done" below.

## Open: workarounds waiting for an EMerge fix

(none)

## Open: Hertzon features waiting for an EMerge fix

| EMerge fix | Hertzon feature that needs it | Waiting for | Check before enabling |
|---|---|---|---|
| Far field with symmetry planes: radiated power halved per plane, directivity 3 dB too high per plane | Symmetry planes (half and quarter models); Hertzon does not mirror far fields yet | PR FennisRobert/EMerge#75 merged and in `packaging/emerge-commit` | A model solved in full and as a half model: same radiated power, gain and directivity |

## Done

| EMerge fix | Removed Hertzon workaround | EMerge version or commit |
|---|---|---|
| Parallel sweep threads share one solver (`AutomaticRoutine` singleton) | `mt_pardiso._duplicate` | `hertzon` branch `2811e9a` |
| Factorizations of sweep worker threads are released on their own thread (FennisRobert/EMerge#72) | `mt_pardiso.release` | `hertzon` branch `2811e9a` |
| ThinConductor sheet split on both sides in the solve | `emerge_patches.apply_thin_conductor` | `hertzon` branch `2811e9a` |
| Port mode solve stops on `ArpackNoConvergence` | `emerge_patches._robust_eigs` | 3.0.0b5 |
| Thin conductor split after a new mesh | `_thin_conductor_new_mesh` | 3.0.0b5 |
