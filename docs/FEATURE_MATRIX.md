# Hertzon feature overview

What Hertzon covers of a typical 3D frequency-domain FEM workflow (no asymptotic, integral-equation or
transient solvers), and how each step is done with the EMerge FEM engine underneath.

Legend: **✔** implemented and tested · **◐** implemented with limitations · **✖** not yet

## Project and design

| Function | In Hertzon | Status | Notes |
|---|---|---|---|
| Project with several designs | Project Manager: Project > Design | ✔ | Insert / copy / rename / delete design |
| Project and design variables (`$var`, `var`) | Project Variables, Design Properties | ✔ | Expressions with units (`2*w+1mm`, `c0/f`), functions `sin cos sqrt …` |
| Solution Type Modal / Terminal / Eigenmode | Simulation > Solution Type | ◐ | Terminal is solved as Modal (port S-parameters, no terminal transform) |
| Auto Open region | Solution Type dialog: Auto Open | ✔ | Radiation or PML picked automatically (PML when the design radiates: far-field sphere, plane wave or "Radiating structure") |
| — | **Auto Open - Filter** | ✔ | Air box with a fixed margin and radiation boundary on all outer faces (filter workflow) |
| Undo / Redo, copy / paste | Edit menu, Ctrl+Z / Ctrl+Y / Ctrl+C / Ctrl+V | ✔ | |
| Validation Check | Simulation > Validation Check | ✔ | Design, 3D model, boundaries/excitations, mesh, setups |
| Overlapping objects (material override) | Design setting Material override; Validation Check > Resolve Overlap / Resolve All Overlaps | ✔ | Contained objects allowed (inner wins); partial overlaps of different materials are errors that block the analysis, except metal over dielectric with Material override (default on); Resolve subtracts the winner (clone) |
| Message Manager, Progress window with Abort | Docks at the bottom | ✔ | |
| Save / Save As / Recent | File menu (`.hzn` JSON file, results in `.hznresults/`) | ✔ | |

## 3D modeler

| Function | In Hertzon | Status | Notes |
|---|---|---|---|
| Box, Cylinder, Sphere, Cone, Torus, Regular Polyhedron | Draw menu, interactive (click-click-height) or by dialog | ✔ | Snaps to grid and vertices, X/Y/Z entry in the status bar |
| Rectangle, Circle, Ellipse, Regular Polygon | Draw menu | ✔ | Working plane XY / YZ / ZX |
| Polyline (lines, closed and covered) | Draw > Polyline | ✔ | Straight, 3 point arc, center point arc (AngularArc), spline segments |
| Arc (3 point, center point) | Draw > Arc | ✔ | A polyline of one arc segment |
| Region (padding % or absolute) | Draw > Region | ✔ | |
| Unite, Subtract (blank/tool, clone), Intersect, Split | Modeler > Boolean | ✔ | |
| Move, Rotate, Mirror, Scale, Duplicate along line / around axis / mirror | Edit > Arrange / Duplicate | ✔ | |
| Thicken sheet, Sweep along vector / around axis, Cover lines | Draw / Modeler | ✔ | Sweep along path ✖ |
| Fillet | Modeler > Fillet | ◐ | Edges of solids; chamfer ✖ |
| Import STEP / IGES / BREP, Export STEP | File / Modeler | ✔ | |
| History tree with editable commands, suppress | Model tree + Properties | ✔ | Every command parameter accepts expressions |
| Object attributes: material, solve inside, model, color, transparency, wireframe | Properties > Attribute | ✔ | |
| Selection modes Object (O), Face (F), Edge (E), Vertex (V), select by name, Ctrl+A | ✔ | ✔ | |
| Units, grid, measure, fit all / fit selection, views, ortho/perspective | View / Modeler | ✔ | |
| Material library (Edit Materials, add/clone) | Tools > Edit Materials | ✔ | εr, μr, σ, tan δ, magnetic tan δ; frequency-dependent materials ✖ |
| User coordinate systems | — | ✖ | Global CS only |

## Boundaries

| Function | In Hertzon | EMerge | Status |
|---|---|---|---|
| Perfect E | Perfect E | `PEC` | ✔ |
| Perfect H | Perfect H | `PMC` | ✔ |
| Finite Conductivity (material / σ, roughness, thickness) | Finite Conductivity | `SurfaceImpedance` / `ThinConductor` | ✔ |
| Impedance (R + jX) | Impedance | `SurfaceImpedance` | ✔ |
| Lumped RLC (parallel or series RLC, Touchstone component) | Lumped RLC | `LumpedElement` with Z(f) | ✔ |
| SMD components between two traces (0201 to 1206, vendor series, pads, optional 3D terminations, component library of Touchstone files) | Lumped Elements tab | `LumpedElement` | ✔ |
| Radiation (first order) | Radiation | `AbsorbingBoundary` | ✔ (second order ◐) |
| PML | PML (on region faces) and Auto Open | `pmlbox` | ✔ |
| Symmetry (Perfect E / Perfect H) | Symmetry | `PEC` / `PMC` | ✔ (impedance multiplier ✖) |
| Master / Slave (Primary / Secondary) | Primary / Secondary | — | ✖ (can be assigned, solver warns) |
| Layered Impedance | — | — | ✖ |

## Excitations

| Function | In Hertzon | Status | Notes |
|---|---|---|---|
| Wave Port (modes, Zpi/Zpv/Zvi, renormalize, de-embed) | Wave Port | ✔ | De-embedding uses the solved port propagation constant |
| Lumped Port (impedance, integration line) | Lumped Port | ✔ | Integration line placed automatically between the touching conductors, or drawn in the 3D view |
| Floquet Port | Floquet Port | ◐ | Needs periodic boundaries, which are not available yet |
| Incident Plane Wave | Incident Plane Wave | ✖ | Can be assigned, solver warns |

## Mesh operations

| Function | In Hertzon | Status |
|---|---|---|
| Length Based (on selection / inside selection) | ✔ |
| Length Based on edges | ✖ (use On Selection on the adjacent faces) |
| Skin Depth Based | ◐ (mapped to a surface length of skin depth × layers) |
| Initial mesh: Lambda refinement, lambda target, free-space lambda | ✔ (Setup > Options; default target follows the basis order: 0.1 / 0.3333 / 0.6667) |
| Plot Mesh | ✔ |

## Analysis setup (driven)

| Setting | In Hertzon | Status |
|---|---|---|
| Solution frequency: Single / Multi-Frequencies / Broadband | ✔ |
| Maximum Number of Passes | ✔ |
| Maximum Delta S | ✔ (port-mode sign aligned) |
| Use Matrix Convergence (ΔMag, ΔPhase) | ✔ |
| Minimum Number of Passes, Minimum Converged Passes | ✔ |
| Percent Refinement per pass | ✔ (enforced, adaptive error-driven refinement) |
| Maximum Refinement (tets) | ✔ |
| Basis function order | ✔ Zero Order (6), First Order (20, default), Second Order (complete quadratic, 30); no Mixed Order |
| Solver: Auto / Direct (PARDISO, MUMPS, UMFPACK, SuperLU) | ✔ |
| Save fields | ✔ |
| Eigenmode setup (min frequency, number of modes) | ✔ |

## Frequency sweeps

| Function | In Hertzon | Status |
|---|---|---|
| Discrete | ✔ |
| Interpolating (error tolerance, max solutions) | ✔ (rational fit with adaptive validation points; validation points from a reduced basis of the direct solutions, factorized only when its residual is too large) |
| Fast (ALPS) | ✔ (one centre factorization, LDLᵀ, reduced basis with residual control per point) |
| Linear Step / Linear Count / Log Scale / Single Points | ✔ |
| Save fields (all frequencies) | ✔ Discrete and Fast: field plots, far fields, gain patterns and antenna parameters at every sweep frequency; Interpolating stores S-parameters only |

## Results and post-processing

| Function | In Hertzon | Status |
|---|---|---|
| Solution Data: Profile, Convergence (table + plot), Matrix Data S/Y/Z (mag/phase, dB, re/im), Mesh Statistics | ✔ |
| Rectangular plot, Smith chart, Data table, markers | ✔ |
| Expressions `dB(S(1,1))`, `VSWR`, `GroupDelay`, `Z`, `Y`, `ang_deg`, `re`, `im` | ✔ |
| Radiation: Infinite Sphere, 2D pattern cuts, 3D polar plot, Gain / Directivity / Realized gain, Eθ/Eφ | ✔ |
| Field overlays on faces, objects and cut planes: Mag, Complex Mag and Vector of E, H, Jsurf and J (volume current sigma E), phase animation | ✔ |
| Export Touchstone, CSV, images | ✔ |
| Export Gerber X2 (copper layers, profile) and Excellon (PTH / NPTH), File > Export > Gerber ([details](GERBER_EXPORT.md)) | ✔ |
| Optimization: conjugate gradient (projected Polak-Ribiere+), goals on report expressions, error-area or worst-case cost, filter mask, parallel gradient evaluations | ✔ |
| Parametric sweep | ✖ (next step) |
| Filter design toolkit: Tools > Filter Design > Tapped Hairpin (N = 2 to 9 resonators, design frequency, hfsynpy microstrip synthesis, parametrized geometry, setup, sweep and conjugate gradient optimization on the filter mask) | ✔ |
| Antenna design toolkit: Tools > Antenna Design > Rectangular Patch (frequency and substrate in; transmission line model, inset feed, setup, sweep, far-field sphere and optimization of length and inset on the return loss) | ✔ |
| Fields calculator | ✖ |
| Solve with HPC / distributed | ◐ (threads per solve, parallel sweep points) |

## Extras

* **File > Export > EMerge Python Script**: writes a stand-alone script that rebuilds the same EMerge model
  and runs the setup (or stops after meshing for custom scripting).
* Plain-text project files (JSON), easy to diff and generate.
