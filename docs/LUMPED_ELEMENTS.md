# Lumped elements

Chip components between two traces, from an ideal R, L or C to a part measured by its vendor.

## Lumped RLC boundary

A Lumped RLC boundary on a sheet that bridges the gap between two conductors. The integration line (the
arrow) runs from one conductor to the other; it sets the length of the element, the width is the sheet area
divided by it. Hertzon places it between the two conductors that touch the sheet; right click the boundary
in the project tree to draw it in the 3D view or to place it again.

| Model | Impedance |
|---|---|
| Parallel RLC | R, L and C in parallel (the default, as in older projects) |
| Series RLC | R + jωL + 1/(jωC), a chip component with its ESR and ESL |
| Touchstone | Z(f) from a .s1p file, or from a .s2p file of the part in series between the two ports (Z = 2 Z0 (1 − S21)/S21) or as a shunt to ground (Z = Z0 S21 / (2 (1 − S21))) |

Touchstone data are interpolated linearly; outside the measured band the nearest point is used and the
solver warns. A relative path is relative to the project file.

## Lumped Elements tab

1. Select the two traces (**O**) the component connects. They have to lie in one plane with a gap along X,
   Y or Z.
2. **Resistor**, **Inductor**, **Capacitor** or **Touchstone**.
3. Choose the value or the Touchstone file, the package and the geometry:
   - **Planar**: the element sheet lies between the pads in the trace plane. *Typical ESL and ESR* adds
     the mounted inductance and loss of the size in series (0201 0.15 nH, 0402 0.3 nH, 0603 0.45 nH,
     0805 0.6 nH, 1206 0.9 nH; 0.05 Ω).
   - **3D**: the body gets metal terminations of its height, the element sheet lies on top between them.
     The current loop through the body is then part of the field solution, so no ESL is added.

Hertzon adds, in a group named after the component (`C1`, `L1`, `R1`, `U1` ...):

- **Pads** from each trace end to the pad gap G of the land pattern, as wide as the pad width X. They are
  made like the first trace: of its metal, or added to its Perfect E / Finite Conductivity boundary.
  When the gap already is G there are no pads.
- the **element sheet** and its **Lumped RLC** boundary `C1_RLC` with the integration line,
- with 3D the **terminations** (PEC boxes).

The land pattern follows IPC-7351 (nominal density) from the body dimensions: G = L − 2e, outer extent
L + 2 × 0.10 mm below size 0603 and L + 2 × 0.35 mm from it on, pad width W. G and X can be changed in the
dialog. Draw the traces with a gap of about the outer extent of the land pattern; a larger gap gives long
pads, which Hertzon reports.

## Packages

| Package | L × W × T (mm) | Terminal e (mm) |
|---|---|---|
| 0201 | 0.60 × 0.30 × 0.30 | 0.15 |
| 0402 | 1.00 × 0.50 × 0.50 | 0.25 |
| 0603 | 1.60 × 0.80 × 0.80 | 0.35 |
| 0805 | 2.00 × 1.25 × 0.85 | 0.50 |
| 1206 | 3.20 × 1.60 × 0.85 | 0.50 |


The S-parameter files of the vendors are free to download for circuit simulation but may not be passed on,
so Hertzon does not ship them. Choose the downloaded .s1p/.s2p file in the dialog and the size of the part.
