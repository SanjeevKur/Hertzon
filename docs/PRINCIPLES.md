# Hertzon design principles

The rules every part of Hertzon follows. New features and changes are checked against them.

## WYSIWYG for every export

**What you see is what you get.** Every export reproduces exactly what the user sees in Hertzon at the
moment of the export. An export never re-renders, re-scales or re-styles the content with settings of its
own.

This holds for every kind of export:

- **Images and screenshots** (3D view, field overlays, reports): the same pixel size, the same view and
  camera, the same colours, colour map and colour scale, the same legend, labels and fonts as on screen.
- **Animations (GIF)**: every frame exactly as the view shows it, in its pixel size, played at the speed the
  user sees (the frame rate measured during playback, not a nominal setting), with colours that stay the
  same from frame to frame.
- **Plots and reports**: the same traces, axes, axis ranges and units, scales (dB, linear, polar),
  markers, title and legend as the report window shows.
- **Data (CSV, tables)**: the same rows and columns, in the same units and with the same quantities as the
  table or plot they come from. When a file format prescribes its own units (Touchstone, Gerber, STEP),
  the format wins and the export follows the format exactly.

When an export cannot match the screen exactly (for example a file format that cannot store a value as
precisely as it is shown), the export comes as close as the format allows and tells the user what differs
and why, in the message after the export.

Options that change the result (higher resolution, other size, transparent background) are allowed only
as an explicit choice of the user in the export dialog. The default is always the screen.
