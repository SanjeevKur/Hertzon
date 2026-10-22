"""File > Export > Gerber: layer assignment, preview and export of Gerber X2 / Excellon files."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QLabel, QPushButton,
                               QComboBox, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
                               QWidget, QGroupBox, QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
                               QSplitter, QAbstractItemView)

from ..export import gerber as G

LAYER_COLORS = ["#c8892b", "#2b7bc8", "#3a9a5b", "#9b59b6", "#c0392b", "#16a085", "#7f8c8d", "#d35400"]
ROLE_TEXT = {"skip": "Do not export", "outline": "Board outline", "via": "Plated hole (via)"}
CATEGORY_TEXT = {"copper": "Copper sheet", "conductor": "Conductor body", "via": "Via", "board": "Board body",
                 "excluded": "Simulation helper", "ambiguous": "Ambiguous", "other": "Other", "error": "Error"}
MSG_COLORS = {"error": "#c0392b", "warning": "#b9770e", "info": "#1f618d"}
MSG_ICONS = {"error": "✖", "warning": "⚠", "info": "ℹ"}
ORIGINS = [("cs", "Origin of the coordinate system"), ("outline", "Lower left corner of the board outline"),
           ("point", "Point (X, Y in the board plane)")]


class GerberExportDialog(QDialog):
    def __init__(self, parent, doc):
        super().__init__(parent)
        self.doc = doc
        self.project = doc.project
        self.des = doc.design
        self.ev = doc.evaluator()
        self.res: G.Result | None = None
        self.roles: dict[str, str] = {}             # only the assignments the user changed
        self.layer_names: dict[str, str] = {}
        self.layer_enabled: dict[str, bool] = {}
        self._filling = False
        self.setWindowTitle(f"Export Gerber: {self.project.name} - {self.des.name}")
        lay = QVBoxLayout(self)

        # ---- settings -------------------------------------------------------------------------
        g = QGroupBox("Output")
        f = QFormLayout(g)
        row = QHBoxLayout()
        base = Path(self.project.path).resolve().parent if self.project.path else Path.home()
        self.folder = QLineEdit(str(base / f"{_clean(self.des.name)}_gerber"))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        f.addRow("Folder:", row)
        self.base = QLineEdit(_clean(self.des.name))
        f.addRow("File name prefix:", self.base)
        self.units = QComboBox()
        self.units.addItems(["mm", "inch"])
        f.addRow("Units:", self.units)
        opts = QHBoxLayout()
        self.outline = QCheckBox("Board outline (Profile)")
        self.outline.setChecked(True)
        self.drills = QCheckBox("Drill files (Excellon, PTH / NPTH)")
        self.drills.setChecked(True)
        opts.addWidget(self.outline)
        opts.addWidget(self.drills)
        opts.addStretch(1)
        f.addRow("", opts)

        h = QGroupBox("Board plane and origin")
        f2 = QFormLayout(h)
        row = QHBoxLayout()
        self.cs = QComboBox()
        self.cs.addItems(self.des.cs_names())
        if self.des.working_cs in self.des.cs_names():
            self.cs.setCurrentText(self.des.working_cs)
        self.plane = QComboBox()
        self.plane.addItems(list(G.PLANES))
        row.addWidget(QLabel("Coordinate system"))
        row.addWidget(self.cs, 1)
        row.addWidget(QLabel("Plane"))
        row.addWidget(self.plane)
        f2.addRow("Reference:", row)
        row = QHBoxLayout()
        self.origin = QComboBox()
        for key, text in ORIGINS:
            self.origin.addItem(text, key)
        self.origin.setCurrentIndex(1)
        self.opoint = QLineEdit("0, 0")
        self.opoint.setToolTip(f"X, Y in the board plane, design units ({self.des.units}) unless given")
        row.addWidget(self.origin, 1)
        row.addWidget(self.opoint)
        f2.addRow("Origin:", row)
        row = QHBoxLayout()
        self.tol = QLineEdit("1um")
        self.tol.setToolTip("Maximum chord deviation for curves that are neither lines nor circular arcs "
                            "(ellipses, splines). Lines and arcs are written exactly.")
        self.tmax = QLineEdit("0.2mm")
        self.tmax.setToolTip("Conducting bodies up to this thickness (along the board normal) are layer copper; "
                             "thicker coaxial cylinders that connect layers are vias")
        row.addWidget(QLabel("Curve tolerance"))
        row.addWidget(self.tol)
        row.addWidget(QLabel("Max. copper thickness"))
        row.addWidget(self.tmax)
        row.addStretch(1)
        f2.addRow("Geometry:", row)
        top = QHBoxLayout()
        top.addWidget(g, 1)
        top.addWidget(h, 1)
        lay.addLayout(top)

        # ---- tables and preview ---------------------------------------------------------------
        split = QSplitter(Qt.Horizontal)
        tabs = QTabWidget()
        self.layers = QTableWidget(0, 5)
        self.layers.setHorizontalHeaderLabels(["Export", "Layer name", "Function", "Elevation", "Objects"])
        self.layers.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.layers.verticalHeader().hide()
        self.layers.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.layers.itemChanged.connect(self._layer_edited)
        tabs.addTab(self.layers, "Layers")
        objw = QWidget()
        ol = QVBoxLayout(objw)
        ol.setContentsMargins(0, 0, 0, 0)
        self.objects = QTableWidget(0, 4)
        self.objects.setHorizontalHeaderLabels(["Object", "Detected as", "Export as", "Note"])
        self.objects.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.objects.verticalHeader().hide()
        self.objects.setEditTriggers(QAbstractItemView.NoEditTriggers)
        ol.addWidget(self.objects, 1)
        reset = QPushButton("Use suggested assignment")
        reset.clicked.connect(self._reset_roles)
        rr = QHBoxLayout()
        rr.addStretch(1)
        rr.addWidget(reset)
        ol.addLayout(rr)
        tabs.addTab(objw, "Layer assignment")
        tabs.setCurrentIndex(1)
        split.addWidget(tabs)
        pv = QWidget()
        pl = QVBoxLayout(pv)
        pl.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview:"))
        self.show_layer = QComboBox()
        self.show_layer.currentIndexChanged.connect(lambda *_: self._draw())
        row.addWidget(self.show_layer, 1)
        pl.addLayout(row)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(5, 4.5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        pl.addWidget(NavigationToolbar2QT(self.canvas, pv))
        pl.addWidget(self.canvas, 1)
        split.addWidget(pv)
        split.setSizes([560, 520])
        lay.addWidget(split, 1)
        self.msgs = QListWidget()
        self.msgs.setMaximumHeight(110)
        self.msgs.setWordWrap(True)
        lay.addWidget(self.msgs)
        row = QHBoxLayout()
        self.status = QLabel("")
        row.addWidget(self.status, 1)
        self.export_btn = QPushButton("Export")
        self.export_btn.setDefault(True)
        self.export_btn.clicked.connect(self.export)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addWidget(self.export_btn)
        row.addWidget(close)
        lay.addLayout(row)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.recompute)
        for w in (self.cs, self.plane, self.origin, self.units):
            w.currentIndexChanged.connect(self.timer.start)
        for w in (self.opoint, self.tol, self.tmax):
            w.editingFinished.connect(self.timer.start)
        for w in (self.outline, self.drills):
            w.toggled.connect(self.timer.start)
        self.origin.currentIndexChanged.connect(lambda *_: self.opoint.setEnabled(self.origin.currentData() == "point"))
        self.opoint.setEnabled(False)
        self.resize(1180, 820)
        self.recompute()

    # ---- options ----------------------------------------------------------------------------------
    def options(self) -> G.Options:
        x, _, y = self.opoint.text().partition(",")
        return G.Options(units=self.units.currentText(), cs=self.cs.currentText(), plane=self.plane.currentText(),
                         origin=self.origin.currentData(),
                         origin_point=(self.ev.length(x or "0"), self.ev.length(y or "0")),
                         tolerance=self.ev.length(self.tol.text()),
                         max_copper_thickness=self.ev.length(self.tmax.text()),
                         outline=self.outline.isChecked(), drills=self.drills.isChecked(),
                         roles=dict(self.roles), layer_names=dict(self.layer_names),
                         layer_enabled=dict(self.layer_enabled))

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Gerber output folder", self.folder.text())
        if d:
            self.folder.setText(d)

    # ---- computation ------------------------------------------------------------------------------
    def recompute(self) -> None:
        self.setCursor(Qt.WaitCursor)
        try:
            self.res = G.run(self.project, self.des, self.options())
        except Exception as e:  # noqa: BLE001
            self.res = None
            self._messages([G.Msg("error", f"Export analysis failed: {e}")])
            self.export_btn.setEnabled(False)
            self.status.setText("")
            self._draw()
            return
        finally:
            self.unsetCursor()
        self._fill()
        self._draw()

    def _fill(self) -> None:
        res = self.res
        self._filling = True
        try:
            self.layers.setRowCount(len(res.layers))
            u = res.options.units
            s = G.UNITS[u]
            for r, L in enumerate(res.layers):
                chk = QTableWidgetItem()
                chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                chk.setCheckState(Qt.Checked if L.enabled else Qt.Unchecked)
                chk.setData(Qt.UserRole, L.id)
                self.layers.setItem(r, 0, chk)
                nm = QTableWidgetItem(L.name)
                nm.setData(Qt.UserRole, L.id)
                nm.setForeground(QBrush(QColor(LAYER_COLORS[(L.number - 1) % len(LAYER_COLORS)])))
                self.layers.setItem(r, 1, nm)
                for c, text in ((2, f"Copper L{L.number} ({L.side})"),
                                (3, f"{L.z0 * s:.5g}" + ("" if abs(L.z1 - L.z0) < 1e-12 else f" .. {L.z1 * s:.5g}")
                                 + f" {u}"),
                                (4, ", ".join(res.item(k).label for k in L.items) or "-")):
                    it = QTableWidgetItem(text)
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self.layers.setItem(r, c, it)
            self.layers.resizeColumnsToContents()
            self.layers.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)

            self.objects.setRowCount(len(res.items))
            for r, it in enumerate(res.items):
                self.objects.setItem(r, 0, QTableWidgetItem(it.label))
                cat = QTableWidgetItem(CATEGORY_TEXT.get(it.category, it.category))
                if it.category in ("ambiguous", "error"):
                    cat.setForeground(QBrush(QColor(MSG_COLORS["warning" if it.category == "ambiguous" else "error"])))
                self.objects.setItem(r, 1, cat)
                combo = QComboBox()
                choices = [("skip", ROLE_TEXT["skip"])] + [(L.id, f"L{L.number} {L.name}") for L in res.layers] + \
                          [("outline", ROLE_TEXT["outline"]), ("via", ROLE_TEXT["via"])]
                for key, text in choices:
                    combo.addItem(text + ("  (suggested)" if key == it.suggested and key != "skip" else ""), key)
                combo.setCurrentIndex(max(0, [k for k, _ in choices].index(it.role) if it.role in
                                          [k for k, _ in choices] else 0))
                if it.category == "excluded":
                    combo.setEnabled(False)
                    combo.setToolTip("Simulation helpers (air region, ports, radiation/PML ...) are never exported")
                combo.currentIndexChanged.connect(lambda _i, key=it.key, cb=combo: self._role_changed(key, cb))
                self.objects.setCellWidget(r, 2, combo)
                note = QTableWidgetItem(it.note)
                note.setToolTip(it.note)
                self.objects.setItem(r, 3, note)
            self.objects.resizeColumnsToContents()
            self.objects.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)

            cur = self.show_layer.currentData()
            self.show_layer.blockSignals(True)
            self.show_layer.clear()
            self.show_layer.addItem("All exported layers", "all")
            for L in res.layers:
                self.show_layer.addItem(f"L{L.number} {L.name}", L.id)
            idx = self.show_layer.findData(cur)
            self.show_layer.setCurrentIndex(max(idx, 0))
            self.show_layer.blockSignals(False)
        finally:
            self._filling = False
        self._messages(res.messages)
        self._update_status()

    def _update_status(self) -> None:
        res = self.res
        ok = res is not None and res.ok and any(L.enabled and L.components for L in res.layers)
        self.export_btn.setEnabled(ok)
        if res is None:
            return
        n = sum(1 for L in res.layers if L.enabled and L.components)
        pth = sum(1 for d in res.drills if d.plated)
        npth = len(res.drills) - pth
        txt = f"{n} copper layer(s), outline: {'yes' if res.outline else 'no'}, drills: {pth} PTH / {npth} NPTH"
        if not res.ok:
            txt = f"<span style='color:{MSG_COLORS['error']}'>Export blocked: {len(res.errors)} error(s)</span>  " + txt
        self.status.setText(txt)

    def _messages(self, msgs: list) -> None:
        self.msgs.clear()
        for m in msgs:
            it = QListWidgetItem(f"{MSG_ICONS.get(m.level, '')} {m.text}")
            it.setForeground(QBrush(QColor(MSG_COLORS.get(m.level, "#000"))))
            self.msgs.addItem(it)
        if not msgs:
            self.msgs.addItem(QListWidgetItem("✔ No problems found."))

    def _role_changed(self, key: str, combo: QComboBox) -> None:
        if self._filling:
            return
        self.roles[key] = combo.currentData()
        self.timer.start()

    def _reset_roles(self) -> None:
        self.roles.clear()
        self.recompute()

    def _layer_edited(self, item: QTableWidgetItem) -> None:
        if self._filling or self.res is None:
            return
        lid = item.data(Qt.UserRole)
        L = next((x for x in self.res.layers if x.id == lid), None)
        if L is None:
            return
        if item.column() == 0:
            L.enabled = item.checkState() == Qt.Checked
            self.layer_enabled[lid] = L.enabled
        elif item.column() == 1:
            L.name = item.text().strip() or L.name
            self.layer_names[lid] = L.name
            self._filling = True
            for i in range(self.show_layer.count()):
                if self.show_layer.itemData(i) == lid:
                    self.show_layer.setItemText(i, f"L{L.number} {L.name}")
            self._filling = False
        self._update_status()
        self._draw()

    # ---- preview ----------------------------------------------------------------------------------
    def _draw(self) -> None:
        from matplotlib.path import Path as MPath
        from matplotlib.patches import PathPatch, Circle
        ax = self.ax
        ax.clear()
        res = self.res
        if res is None:
            self.canvas.draw_idle()
            return
        u = res.options.units
        s = G.UNITS[u]
        ox, oy = res.origin

        def poly(lp):
            p = lp.polygon()
            return (p[:, 0] - ox) * s, (p[:, 1] - oy) * s

        show = self.show_layer.currentData() or "all"
        handles = []
        for L in sorted(res.layers, key=lambda x: x.z):
            if (show == "all" and not L.enabled) or (show != "all" and L.id != show):
                continue
            col = LAYER_COLORS[(L.number - 1) % len(LAYER_COLORS)]
            for comp in L.components:
                verts, codes = [], []
                for lp in [comp.outer] + comp.holes:
                    x, y = poly(lp)
                    verts += list(zip(x, y)) + [(x[0], y[0])]
                    codes += [MPath.MOVETO] + [MPath.LINETO] * (len(x) - 1) + [MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(verts, codes), fc=col, ec=col, lw=0.6,
                                       alpha=0.85 if show != "all" else 0.55))
            handles.append((col, f"L{L.number} {L.name}"))
        for comp in res.outline:
            for lp in [comp.outer] + comp.holes:
                x, y = poly(lp)
                ax.plot(list(x) + [x[0]], list(y) + [y[0]], color="#333333", lw=1.0)
        for d in res.drills:
            c = Circle(((d.x - ox) * s, (d.y - oy) * s), d.d / 2 * s, fc="white" if d.plated else "none",
                       ec="#111111" if d.plated else "#c0392b", lw=0.8, zorder=5)
            ax.add_patch(c)
        ax.plot([0], [0], marker="+", color="#c0392b", ms=14, mew=1.5, zorder=6)
        ax.annotate("origin", (0, 0), xytext=(6, 6), textcoords="offset points", color="#c0392b", fontsize=8)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        leg = [Patch(fc=c, alpha=0.7, label=t) for c, t in handles]
        if res.outline:
            leg.append(Line2D([], [], color="#333333", label="Board outline"))
        if any(d.plated for d in res.drills):
            leg.append(Line2D([], [], ls="", marker="o", mfc="white", mec="#111111", label="PTH"))
        if any(not d.plated for d in res.drills):
            leg.append(Line2D([], [], ls="", marker="o", mfc="none", mec="#c0392b", label="NPTH"))
        if leg:
            ax.legend(handles=leg, fontsize=8, loc="upper right", framealpha=0.85)
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale_view()
        ax.set_xlabel(f"X [{u}]")
        ax.set_ylabel(f"Y [{u}]")
        ax.set_title(f"Plane {res.options.plane} of {res.options.cs}, seen from the top (not mirrored)", fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.5)
        self.canvas.draw_idle()

    # ---- export -----------------------------------------------------------------------------------
    def export(self, interactive: bool = True) -> list[Path]:
        """Recompute with the current settings and write the files (interactive=False: no message boxes)."""
        self.timer.stop()
        try:
            opts = self.options()
        except Exception as e:  # noqa: BLE001
            if interactive:
                QMessageBox.warning(self, "Export Gerber", f"Invalid setting: {e}")
            return []
        self.res = G.run(self.project, self.des, opts)
        self._fill()
        self._draw()
        folder = Path(self.folder.text().strip() or ".")
        base = _clean(self.base.text().strip() or self.des.name)
        try:
            plan = G.file_plan(self.res, base)
        except G.ExportError as e:
            if interactive:
                QMessageBox.critical(self, "Export Gerber", f"Nothing was written:\n{e}")
            return []
        existing = [n for n in plan if (folder / n).exists()]
        if existing and interactive:
            r = QMessageBox.question(self, "Export Gerber", "Overwrite existing files?\n\n" + "\n".join(existing))
            if r != QMessageBox.Yes:
                return []
        written = G.write(self.res, folder, base)
        log = getattr(self.parent(), "msgs", None)
        if log is not None:
            log.add("info", f"Gerber export: {len(written)} files written to {folder}")
        warn = [m for m in self.res.messages if m.level == "warning"]
        if interactive:
            QMessageBox.information(self, "Export Gerber", f"Written to {folder}:\n\n" +
                                    "\n".join(p.name for p in written) +
                                    (f"\n\n{len(warn)} warning(s), see the export report." if warn else ""))
        return written


def _clean(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s) or "board"
