# Backlog

Small improvements collected for a future release. Items move to `CHANGELOG.md` once they are released.

## Open

### Subtract dialog: move single objects between Blank Parts and Tool Parts
- Today the dialog (`MainWindow.boolean`, `hertzon/gui/main_window.py`) only has a global **Swap** button,
  which exchanges the first tool with the blank.
- Wanted, as in Ansys HFSS: arrow buttons **→** and **←** between the two lists that move the selected
  object(s) from Blank Parts to Tool Parts and back.
- Probably also: multi-selection in both lists; OK only when Blank Parts is not empty (several blank parts,
  each subtracted by all tools, as in HFSS).

### Select several objects by dragging with the mouse
- As in Ansys HFSS: press and hold the left mouse button and drag over the entries to select several
  objects at once, in the lists of the Subtract dialog (`QAbstractItemView.ExtendedSelection`, also with
  Ctrl/Shift click) and ideally in the model tree as well.
