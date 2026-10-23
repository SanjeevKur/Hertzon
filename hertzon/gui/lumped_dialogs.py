"""Lumped Elements tab: insert an SMD component between two traces, and the component library."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QStandardPaths, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QFormLayout, QLineEdit, QComboBox, QCheckBox, QDoubleSpinBox, QLabel,
                               QPushButton, QHBoxLayout, QVBoxLayout, QDialogButtonBox, QFileDialog, QTableWidget,
                               QTableWidgetItem, QHeaderView, QAbstractItemView, QMessageBox, QWidget, QMenu)

from ..model import lumped as LU

# official S-parameter download pages (the files are downloaded by the user, Hertzon does not ship them)
VENDOR_PAGES = (
    ("Murata capacitors (MLCC)...", "https://www.murata.com/en-us/tool/data/sparameterdata/sparameter-mlcc"),
    ("Murata inductors...", "https://www.murata.com/en-us/tool/data/sparameterdata/sparameter-inductor"),
    ("Coilcraft inductors...", "https://coilcraft.com/models.cfm"),
)
KIND_NAMES = {"R": "Resistor", "L": "Inductor", "C": "Capacitor", "S": "Touchstone component"}
VALUE_DEFAULTS = {"R": "50ohm", "L": "10nH", "C": "1pF"}


def library_folder() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or str(Path.home() / ".hertzon")
    return Path(base) / "component_library"


def open_library() -> LU.Library:
    return LU.Library(library_folder())


class ComponentDialog(QDialog):
    """Kind, value, package, model and land pattern of a component placed in a gap of ``gap`` mm."""

    def __init__(self, parent, kind: str, gap: float, library: LU.Library, part: LU.LibraryPart | None = None):
        super().__init__(parent)
        self.setWindowTitle("Insert Component")
        self.library, self.gap = library, gap
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)
        self.kind = QComboBox()
        for k, t in KIND_NAMES.items():
            self.kind.addItem(t, k)
        self.kind.setCurrentIndex(list(KIND_NAMES).index(kind))
        form.addRow("Component:", self.kind)
        self.name = QLineEdit("")
        self.name.setPlaceholderText("automatic (R1, L1, C1, U1 ...)")
        form.addRow("Name:", self.name)
        self.value = QLineEdit(VALUE_DEFAULTS.get(kind, ""))
        self.value_label = QLabel("Value:")
        form.addRow(self.value_label, self.value)
        # Touchstone file
        self.ts_row = QWidget()
        h = QHBoxLayout(self.ts_row)
        h.setContentsMargins(0, 0, 0, 0)
        self.ts_file = QLineEdit()
        b1 = QPushButton("Browse...")
        b1.clicked.connect(self._browse)
        b2 = QPushButton("Library...")
        b2.clicked.connect(self._from_library)
        h.addWidget(self.ts_file, 1)
        h.addWidget(b1)
        h.addWidget(b2)
        self.ts_label = QLabel("Touchstone file:")
        form.addRow(self.ts_label, self.ts_row)
        self.mount = QComboBox()
        self.mount.addItems(["Series", "Shunt"])
        self.mount.setToolTip("How the component sat in a 2-port test fixture (vendor files: usually series)")
        self.mount_label = QLabel("2-port measured as:")
        form.addRow(self.mount_label, self.mount)
        self.package = QComboBox()
        form.addRow("Package:", self.package)
        self.pkg_info = QLabel()
        self.pkg_info.setWordWrap(True)
        form.addRow("", self.pkg_info)
        self.geometry = QComboBox()
        self.geometry.addItem("Planar: element sheet between the pads", False)
        self.geometry.addItem("3D: body with metal terminations, element on top", True)
        self.geometry.setToolTip("3D includes the current loop through the body (its mounted inductance) in "
                                 "the field solution; it needs a finer mesh")
        form.addRow("Geometry:", self.geometry)
        self.parasitics = QCheckBox("Add the typical ESL and ESR of the size (planar geometry)")
        self.parasitics.setChecked(True)
        form.addRow("", self.parasitics)
        self.G = QDoubleSpinBox(decimals=3, minimum=0.01, maximum=100, singleStep=0.05, suffix=" mm")
        self.X = QDoubleSpinBox(decimals=3, minimum=0.01, maximum=100, singleStep=0.05, suffix=" mm")
        form.addRow("Pad gap G:", self.G)
        form.addRow("Pad width X:", self.X)
        self.gap_info = QLabel()
        self.gap_info.setWordWrap(True)
        lay.addWidget(self.gap_info)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.package.currentIndexChanged.connect(self._package_changed)
        self.geometry.currentIndexChanged.connect(self._geometry_changed)
        self.G.valueChanged.connect(self._gap_text)
        self._kind_changed()
        if part is not None:
            self.use_part(part)
        self.resize(520, 0)

    # -- state --------------------------------------------------------------------------------------
    def k(self) -> str:
        return self.kind.currentData()

    def pkg(self) -> LU.Package:
        return LU.PACKAGES[self.package.currentData()]

    def _kind_changed(self, *_):
        k = self.k()
        cur = self.package.currentData()
        self.package.blockSignals(True)
        self.package.clear()
        for p in LU.GENERIC + [s for s in LU.SERIES if k in ("S", s.kind)]:
            self.package.addItem(p.name, p.name)
        i = self.package.findData(cur) if cur else self.package.findData("0402")
        self.package.setCurrentIndex(max(i, 0))
        self.package.blockSignals(False)
        ts = k == "S"
        for w in (self.ts_label, self.ts_row, self.mount_label, self.mount):
            w.setVisible(ts)
        self.value.setVisible(not ts)
        self.value_label.setVisible(not ts)
        if not ts and not self.value.text().strip():
            self.value.setText(VALUE_DEFAULTS[k])
        self._package_changed()

    def _package_changed(self, *_):
        p = self.pkg()
        land = LU.land_pattern(p)
        self.G.setValue(land.G)
        self.X.setValue(land.X)
        self.pkg_info.setText(f"Body {p.L:g} x {p.W:g} x {p.T:g} mm, terminals {p.e:g} mm"
                              + (f" ({p.note})" if p.note else ""))
        self._geometry_changed()

    def _geometry_changed(self, *_):
        self.parasitics.setEnabled(not self.geometry.currentData() and self.k() != "S")
        self._gap_text()

    def _gap_text(self, *_):
        G = self.G.value()
        if self.gap + 1e-9 < G:
            self.gap_info.setText(f"<span style='color:#c0392b'>The gap between the traces is {self.gap:.4g} mm, "
                                  f"smaller than the pad gap G = {G:.4g} mm.</span>")
        else:
            self.gap_info.setText(f"Gap between the traces {self.gap:.4g} mm: pads of {(self.gap - G) / 2:.4g} mm "
                                  f"reach from the traces to the pad gap.")

    # -- Touchstone ---------------------------------------------------------------------------------
    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Touchstone File", "", "Touchstone (*.s1p *.s2p)")
        if f:
            self.ts_file.setText(f)
            ser = LU.series_for(Path(f).stem)
            if ser is not None:
                self.package.setCurrentIndex(max(self.package.findData(ser.name), 0))

    def _from_library(self):
        dlg = LibraryDialog(self, self.library)
        if dlg.exec() == QDialog.Accepted and dlg.selected is not None:
            self.use_part(dlg.selected, keep_kind=True)

    def use_part(self, part: LU.LibraryPart, keep_kind: bool = False):
        if not keep_kind:
            self.kind.setCurrentIndex(list(KIND_NAMES).index("S"))
        self.ts_file.setText(part.file)
        if part.series:
            i = self.package.findData(part.series)
            if i >= 0:
                self.package.setCurrentIndex(i)

    def _ok(self):
        if self.k() == "S" and not Path(self.ts_file.text().strip()).is_file():
            QMessageBox.warning(self, "Insert Component", "Choose a Touchstone file (.s1p/.s2p).")
            return
        if self.gap + 1e-9 < self.G.value():
            QMessageBox.warning(self, "Insert Component", self.gap_info.text().replace("<span style='color:#c0392b'>",
                                                                                      "").replace("</span>", ""))
            return
        self.accept()

    def result_values(self) -> dict:
        p = self.pkg()
        land = LU.land_pattern(p)
        return dict(kind=self.k(), name=self.name.text().strip(), value=self.value.text().strip(), pkg=p,
                    land=LU.Land(G=self.G.value(), Z=max(land.Z, self.G.value()), X=self.X.value()),
                    terminations=bool(self.geometry.currentData()), parasitics=self.parasitics.isChecked(),
                    touchstone=self.ts_file.text().strip(), mount=self.mount.currentText())


class LibraryDialog(QDialog):
    """Touchstone files of components imported from vendor downloads (folder or ZIP)."""

    def __init__(self, parent, library: LU.Library):
        super().__init__(parent)
        self.setWindowTitle("Component Library")
        self.library = library
        self.selected: LU.LibraryPart | None = None
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search part number, value or series (e.g. 0402HP, 10nH, GJM15)")
        self.kind = QComboBox()
        for t, k in (("All", ""), ("Inductors", "L"), ("Capacitors", "C"), ("Resistors", "R")):
            self.kind.addItem(t, k)
        top.addWidget(self.search, 1)
        top.addWidget(self.kind)
        lay.addLayout(top)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Part", "Value", "Series", "Ports", "File"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.doubleClicked.connect(self._ok)
        lay.addWidget(self.table, 1)
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        row = QHBoxLayout()
        for txt, fn in (("Import Folder...", self._import_folder), ("Import ZIP...", self._import_zip),
                        ("Import Files...", self._import_files)):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        dl = QPushButton("Vendor Downloads")
        dl.setToolTip("Opens the S-parameter download page of the vendor in the browser; import the downloaded "
                      "ZIP afterwards")
        m = QMenu(dl)
        for txt, url in VENDOR_PAGES:
            m.addAction(txt, lambda u=url: QDesktopServices.openUrl(QUrl(u)))
        dl.setMenu(m)
        row.addWidget(dl)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Use Part")
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.search.textChanged.connect(self._fill)
        self.kind.currentIndexChanged.connect(self._fill)
        self._fill()
        self.resize(820, 520)

    def _fill(self, *_):
        parts = self.library.search(self.search.text(), self.kind.currentData())
        self.table.setRowCount(len(parts))
        for r, p in enumerate(parts):
            for c, v in enumerate((p.part, p.value, p.series or "(unknown series)", str(p.ports), p.file)):
                it = QTableWidgetItem(v)
                it.setData(Qt.UserRole, p.part)
                self.table.setItem(r, c, it)
        self.info.setText(f"{len(parts)} of {len(self.library.parts)} parts. Download the S-parameter files "
                          "(.s2p/.s1p) of the vendor (Coilcraft, Murata ...) and import the folder or ZIP; the "
                          "files are copied to " + str(self.library.folder) + ". Series known to the library "
                          "get their body dimensions automatically.")

    def _imported(self, res):
        n, skipped = res
        self._fill()
        msg = f"{n} part(s) imported."
        if skipped:
            msg += "\n\nSkipped:\n" + "\n".join(skipped[:20])
        QMessageBox.information(self, "Component Library", msg)

    def _import_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with Touchstone files")
        if d:
            self._imported(self.library.import_folder(d))

    def _import_zip(self):
        f, _ = QFileDialog.getOpenFileName(self, "ZIP with Touchstone files", "", "ZIP (*.zip)")
        if f:
            self._imported(self.library.import_folder(f))

    def _import_files(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Touchstone files", "", "Touchstone (*.s1p *.s2p)")
        if fs:
            self._imported(self.library.import_files([Path(f) for f in fs]))

    def _ok(self, *_):
        r = self.table.currentRow()
        if r < 0:
            QMessageBox.information(self, "Component Library", "Select a part.")
            return
        name = self.table.item(r, 0).data(Qt.UserRole)
        self.selected = next(p for p in self.library.parts if p.part == name)
        self.accept()
