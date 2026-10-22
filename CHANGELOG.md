# Changelog

All notable changes of each release. The release workflow copies the section of the released version into
the GitHub release notes.

## Unreleased

### Changed
- **Solver core EMerge 3.0.0a25** (was 2.8.9): matrix assembly in EMerge's Rust core, about 4x faster
  sweep assembly; the first solve after the installation no longer waits for numba compiles of EMerge
  (cold start of the microstrip benchmark 12 min -> 2.5 min). Python 3.10 to 3.13.
- Hertzon's own extensions (re-mesh refinement, curved elements, thin conductors, PARDISO LDLT, parallel
  sweeps) ported; the curved-element kernels are now part of Hertzon (`ned2_kernels.py`, from EMerge 2.8.9).

### New
- **EMerge material library** (emsutil): 255 more materials (dielectrics, metals, semiconductors, foams,
  solder masks, liquids) in the Materials dialog, with a "Libraries" filter; frequency dependent data is
  used by the solver.
- **Markers in the polar Radiation Pattern**: double-click or right-click > Marker (max/min, next max/min,
  delta), drag a marker around the circle, marker table under the plot.
- **Beamwidth / side lobes** (right-click > Marker): opening angle of the main lobe at a level below the
  maximum (default -3 dB, e.g. -10 dB), its edges and the side lobe level, in the polar and the rectangular
  pattern.
- **Create Far Fields Report > Rectangular Plot**: the pattern cut over the angle as its own menu entry.
- **Save fields** of a sweep also for Fast sweeps: far fields, gain patterns and antenna parameters over
  frequency. Far field and antenna reports name the solved sweeps without fields (Interpolating, or Save
  fields off) instead of leaving them out silently.

### Fixed
- Plane wave excitation on EMerge 3 (its new default angle convention sent the wave in the wrong direction).
- Thin conductors after a re-mesh pass (EMerge 3 did not split the new mesh).
- Rare crash of parallel sweeps on EMerge 3 (the worker threads shared one solver routine).

## 0.1.21 (2026-10-22)

### New
- **Subtract** dialog as in HFSS: arrow buttons **→ / ←** move the selected objects between Blank Parts and
  Tool Parts; several objects are selected by dragging with the mouse, Ctrl or Shift. Several blank parts are
  each cut by all tool parts. Swap exchanges all blank and tool parts.
- **Radiation Pattern** of a sweep with saved fields: the report shows the far field of the chosen solution
  (no longer always LastAdaptive) and has a **Frequency** selector
  ([#5](https://github.com/c8501024/Hertzon/issues/5)); the 3D Polar Plot too.
- Radiation Pattern **Plot: Rectangular**: the cut over the angle with markers (max/min, delta, bandwidth),
  axis properties and the marker table.

### Fixed
- View: the projection button is now **Orthographic** and shows its state (pressed: orthographic, not
  pressed: perspective), also in View > Orthographic Projection; switching keeps the model size on screen
  ([#4](https://github.com/c8501024/Hertzon/issues/4)).

## 0.1.20 (2026-10-22)

### New
- **File > Export > Gerber...**: manufacturing data from the real CAD geometry, as in the layout tools of
  commercial simulators ([docs/GERBER_EXPORT.md](https://github.com/c8501024/Hertzon/blob/main/docs/GERBER_EXPORT.md),
  [tutorial](https://github.com/c8501024/Hertzon/blob/main/docs/TUTORIAL.md#6-gerber-export-of-a-printed-circuit-board)).
  - One Gerber X2 file per copper layer (`TF.FileFunction Copper,Ln,Top/Inr/Bot`), the board outline
    (`Profile,NP`) and Excellon drill files, split into plated (vias) and non-plated holes.
  - Board plane: XY, YZ or ZX plane of any coordinate system; one origin for all files (coordinate system
    origin, lower left corner of the board outline or a point); mm or inch; nothing is mirrored.
  - Copper layers are found from the elevations of conductor sheets and thin conductor bodies; the layer
    assignment is suggested per object and can be changed. Bodies are cut at half height, never projected.
  - Overlapping copper is merged, cut-outs are written with clear polarity and islands inside them stay;
    lines and circular arcs are exact, ellipses and splines within an adjustable tolerance.
  - Air region, ports and radiation/PML/symmetry boundaries are never exported. Ambiguous geometry is
    reported with the object names; errors block the export, nothing is written then.
  - Dialog with folder, units, plane, origin, layer table, assignment, preview and a message after the
    export ("Export finished!" with Open Folder, or the reason it failed). Ribbon: Import / Export > Export Gerber.
- `examples/GerberTestBoard.hzn`: two-layer test board with copper shapes, cut-out with island, via and holes.

### Changed
- The exports are grouped in **File > Export** (Geometry (STEP), Touchstone, EMerge Python Script, Gerber).
- `CHANGELOG.md` (this file); the GitHub release notes are taken from it.

## 0.1.19 (2026-10-22)
- The solver process ends with TerminateProcess on Windows (the exit still crashed with 0xC0000005 now and
  then); the warm-up counts a finished solve as done.

## 0.1.18 (2026-10-22)
- Setup warm-up also solves a coax with wave ports, so the first pass of a wave port model no longer
  compiles for about a minute after the installation; a hanging warm-up solve is stopped after a time limit
  and logged to `hertzon_warmup.log`.
- Roadmap: openEMS, time domain, low frequency, circuits and photonics.

## 0.1.17 (2026-10-22)
- Alt + left drag rotates the view; navigation table in the tutorial.

## 0.1.16 (2026-10-22)
- Benchmarks: workstation timings of the solver benchmark and its broadband sweeps.

## 0.1.15 (2026-10-21)
- Tutorial screenshots of every model, field plots, S-parameters, patterns, RCS and the MCP dialog.

## 0.1.14 (2026-10-21)
- Step-by-step tutorial (waveguide, coax, dipole, sphere RCS, MCP) with its example projects; the lumped
  port size of a sheet ignores gmsh's bounding box tolerance.

## 0.1.13 (2026-10-21)
- No-warranty notice in the README and the About dialog; all shipped examples listed.

## 0.1.11 / 0.1.12 (2026-10-21)
- EMerge credit and citation texts (Robert Fennis, Zenodo DOI), README badges, the window opens maximized
  without a cut-off edge.

## 0.1.9 (2026-10-21)
- First release archived on Zenodo with a DOI; `CITATION.cff`.

Earlier versions: see the git history.
