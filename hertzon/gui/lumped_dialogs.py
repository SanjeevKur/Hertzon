"""Lumped Elements tab: insert an SMD component, placed freely or between two traces."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (QDialog, QFormLayout, QLineEdit, QComboBox, QCheckBox, QDoubleSpinBox, QLabel,
                               QPushButton, QHBoxLayout, QVBoxLayout, QDialogButtonBox, QFileDialog, QMessageBox,
                               QWidget)

from ..model import lumped as LU

KIND_NAMES = {"R": "Resistor", "L": "Inductor", "C": "Capacitor", "S": "Touchstone component"}
VALUE_DEFAULTS = {"R": "50ohm", "L": "10nH", "C": "1pF"}


class ComponentDialog(QDialog):
    """Kind, value, package, model and land pattern of a component: placed freely (``gap`` None, with the
    direction of its axis in the board plane with the normal ``normal``) or in a gap of ``gap`` mm."""

    def __init__(self, parent, kind: str, gap: float | None = None, package: str | None = None, normal: int = 2):
        super().__init__(parent)
        self.setWindowTitle("Insert Component")
        self.gap, self.normal = gap, normal
        self._package0 = package
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
        h.addWidget(self.ts_file, 1)
        h.addWidget(b1)
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
        self.direction = QComboBox()
        for i in range(3):
            if i != normal:
                self.direction.addItem(f"along {'XYZ'[i]}", i)
        self.direction.setToolTip("Direction from pad to pad, in the board plane")
        if gap is None:
            form.addRow("Orientation:", self.direction)
        self.parasitics = QCheckBox("Add the typical ESL and ESR of the size (planar geometry)")
        self.parasitics.setChecked(True)
        form.addRow("", self.parasitics)
        self.G = QDoubleSpinBox(decimals=3, minimum=0.01, maximum=100, singleStep=0.05, suffix=" mm")
        self.X = QDoubleSpinBox(decimals=3, minimum=0.01, maximum=100, singleStep=0.05, suffix=" mm")
        form.addRow("Pad gap G:", self.G)
        form.addRow("Pad width X:", self.X)
        self.Z = QDoubleSpinBox(decimals=3, minimum=0.02, maximum=200, singleStep=0.05, suffix=" mm")
        self.Z.setToolTip("Outer extent of the land pattern (pad end to pad end)")
        if gap is None:
            form.addRow("Land pattern length Z:", self.Z)
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
        i = self.package.findData(cur or self._package0 or "0402")
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
        self.Z.setValue(land.Z)
        self.pkg_info.setText(f"Body {p.L:g} x {p.W:g} x {p.T:g} mm, terminals {p.e:g} mm"
                              + (f" ({p.note})" if p.note else "")
                              + f"; land pattern G {land.G:g}, Z {land.Z:g}, X {land.X:g} mm")
        self._geometry_changed()

    def _geometry_changed(self, *_):
        self.parasitics.setEnabled(not self.geometry.currentData() and self.k() != "S")
        self._gap_text()

    def _gap_text(self, *_):
        G = self.G.value()
        if self.gap is None:
            self.gap_info.setText("The component is placed with its centre at the clicked point. Its position is "
                                  "the design variables <name>_x, _y, _z: change them to move it.")
            return
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

    def _ok(self):
        if self.k() == "S" and not Path(self.ts_file.text().strip()).is_file():
            QMessageBox.warning(self, "Insert Component", "Choose a Touchstone file (.s1p/.s2p).")
            return
        if self.gap is None and self.Z.value() <= self.G.value() + 1e-6:
            QMessageBox.warning(self, "Insert Component", "The land pattern length Z has to be larger than the pad "
                                                          "gap G.")
            return
        if self.gap is not None and self.gap + 1e-9 < self.G.value():
            QMessageBox.warning(self, "Insert Component", self.gap_info.text().replace("<span style='color:#c0392b'>",
                                                                                      "").replace("</span>", ""))
            return
        self.accept()

    def result_values(self) -> dict:
        p = self.pkg()
        land = LU.land_pattern(p)
        Z = max(self.Z.value(), self.G.value()) if self.gap is None else max(land.Z, self.G.value())
        return dict(kind=self.k(), name=self.name.text().strip(), value=self.value.text().strip(), pkg=p,
                    land=LU.Land(G=self.G.value(), Z=Z, X=self.X.value()), axis=self.direction.currentData(),
                    terminations=bool(self.geometry.currentData()), parasitics=self.parasitics.isChecked(),
                    touchstone=self.ts_file.text().strip(), mount=self.mount.currentText())
