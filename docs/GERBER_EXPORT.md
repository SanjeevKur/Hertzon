# Gerber export

**File > Export > Gerber...** (also *Project > Import / Export > Export Gerber* in the ribbon) writes the
planar conductor geometry of the active design as manufacturing data:

| File | Content |
|---|---|
| `<prefix>-L<n>_<name>.gbr` | one Gerber X2 file per copper layer, `%TF.FileFunction,Copper,L<n>,Top/Inr/Bot*%` |
| `<prefix>-Profile.gbr` | board outline, `%TF.FileFunction,Profile,NP*%`, 0.1 mm line, aperture function `Profile` |
| `<prefix>-PTH_L<i>-L<j>.drl` | Excellon, plated holes (vias) connecting layers i..j (`TF.FileFunction,Plated,i,j,PTH`) |
| `<prefix>-NPTH.drl` | Excellon, non-plated holes (`TF.FileFunction,NonPlated,1,n,NPTH`) |
| `<prefix>-export-report.txt` | units, plane, origin, layers with their objects, drills, messages |

All files carry `TF.GenerationSoftware`, `TF.CreationDate`, `TF.SameCoordinates` and `TF.FilePolarity`.
Gerber coordinates are format 4.6 in mm (`%MOMM*%`) or 2.6 in inch (`%MOIN*%`); Excellon uses decimal
coordinates (`METRIC` / `INCH`).

## Using the dialog

1. **Output:** folder, file name prefix, units (mm or inch), and whether the board outline and drill files
   are written.
2. **Board plane and origin:** the board lies in the *XY*, *YZ* or *ZX* plane of a coordinate system (Global
   or any relative CS of the design). The plane's first axis is board X, the second board Y, the third the
   board normal (top = +normal). The **origin** is the CS origin, the lower left corner of the board outline,
   or a point (X, Y in the board plane, design units unless written with a unit). All layers and drill files
   use this one frame; nothing is mirrored, the bottom layer is written as seen from the top.
3. **Geometry:** *Curve tolerance* is the largest chord deviation for curves that are neither straight
   lines nor circular arcs (ellipses, splines). *Max. copper thickness* separates layer copper from other
   conductors (see below).
4. **Layer assignment:** every object of the design with what it was detected as, the suggested target and
   a note. Change the target per object: a copper layer, *Board outline*, *Plated hole (via)* or *Do not
   export*. *Use suggested assignment* resets all changes. Simulation helpers cannot be assigned.
5. **Layers:** detected copper layers from top to bottom with their elevation and objects; rename them or
   switch a layer off.
6. **Preview:** the exported geometry in file coordinates (after origin and units): copper layers, outline,
   PTH (white) and NPTH (red) holes and the origin marker. Show all layers or one.
7. **Messages:** errors (red) block the export, warnings (orange) name the objects concerned, and the
   status line counts layers and holes. **Export** generates every file in memory first and only writes
   when nothing failed; existing files are overwritten after a confirmation.

## How the geometry is obtained

* The design is built with the normal OpenCASCADE kernel into a separate gmsh model (real CAD geometry,
  exact units), moved rigidly into the board frame and removed again after the analysis.
* **Classification:** copper = sheets (or faces of solids) with a *Perfect E* or *Finite Conductivity*
  boundary, and solids of a conducting material (PEC or ≥ 1e5 S/m). Board = dielectric solids. Never
  exported: the air region, non-model objects, port sheets (any excitation), sheets with radiation, PML,
  symmetry, master/slave, Perfect H, impedance or RLC boundaries, vacuum/air bodies.
* **Layers** come from the copper elevations: a sheet parallel to the board has one elevation, a conducting
  solid not thicker than *Max. copper thickness* an interval; touching or overlapping elevations form one
  layer. Layers are numbered from the top (L1 = Top).
* **Sheets** are taken with their own edges (2D contours). **Conducting solids** on a layer are cut with a
  plane parallel to the board at half their height; a thicker solid assigned to a layer by hand is cut in
  the layer plane. Sheets that are not planar or not parallel to the board are reported, never projected.
* **Union and polarity:** all faces of a layer are fragmented together (OCC boolean). Edges shared by two
  fragments are internal; the others are chained into closed contours, oriented counter-clockwise for
  copper and clockwise for holes, and grouped by nesting. Each copper area is written as a dark region
  (`%LPD*%`), its holes and slots as clear regions (`%LPC*%`), and copper islands inside a hole afterwards
  as dark regions again (sorted by nesting depth), so overlaps merge and cut-outs and islands stay correct.
* **Curves:** lines are `G01`, circular arcs and full circles `G02`/`G03` with `I`/`J` centre offsets in
  multi-quadrant mode (`G75`), from the exact OCC geometry. A curve is recognised as an arc when 65 samples
  lie on one circle within 1e-9 of the radius. Other curves become polylines whose chords deviate at most
  the curve tolerance (adaptive bisection); the dialog lists the objects concerned.
* **Drills:** plated holes only from *via* bodies: conducting solids whose curved faces are coaxial
  cylinders along the board normal and that reach two or more copper layers (drill = outer diameter).
  Non-plated holes only from circular holes through a board body (outline) that contain no via, up to
  6.5 mm; larger round openings stay in the profile. Cut-outs in copper never become drills. A via without
  copper on a layer it passes and a non-plated hole through copper are reported as warnings.

## Checks performed

`tests/test_gerber.py` exports a two-layer test board (also in `examples/GerberTestBoard.hzn`): rectangle,
circle, a plane with a round cut-out and an island inside it, an overlapping rectangle and circle, an
ellipse, a thin copper body (35 µm), a via with a pad, a ground plane with clearances, a mounting hole and
a milled slot in the substrate, plus a port sheet, a radiation sheet and the air region. The files are read
back with two independent parsers:

* **gerbonara:** file attributes, units, bounding boxes of each layer (mm and inch), number of dark and
  clear regions, exact arc centres and radii, the clear region being the cut-out (not the island), profile
  segments of the slot, drill positions, diameters and plating of the PTH and NPTH files.
* **pygerber:** renders the copper files at 40 px/mm; the copper area agrees with the analytic area of the
  model within 1 % and sample pixels confirm the island, the clear ring around it and the merged overlap.

Further tests: a rotated and shifted coordinate system with a custom origin (copper, arcs and drills move
together), origin at the outline corner in inch, the chord deviation of the ellipse for two tolerances,
errors that block the export (no copper, a helper object forced onto a layer, nothing written), the patch
antenna and hairpin filter examples, and the dialog itself (suggested assignment, manual change, renamed
layer, export).

## Limitations

* Only conductor geometry parallel to the board plane is exported; tilted or curved conductor sheets and
  thick conductors that are no via (e.g. a coax shield) are reported and skipped unless assigned by hand.
* No solder mask, paste, silkscreen, component or netlist data and no Gerber job file (`.gbrjob`).
* Vias must be cylinders along the board normal; slots and non-circular plated holes are not written as
  drills (non-plated cut-outs of the board body go into the profile).
* Copper is written as regions only (no traces with apertures); a CAM tool sees polygons, not tracks.
* Ellipses and splines are approximated within the curve tolerance; lines and circular arcs are exact up to
  the file resolution (1 nm in mm, 1 µin in inch).
* Copper areas that only touch at a single point are written as separate contours sharing that point.
