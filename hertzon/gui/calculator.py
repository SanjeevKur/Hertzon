"""Fields Calculator (Results > Fields > Fields Calculator): its own window next to the modeler.

Left the expressions of the design, the functions and a library; in the middle the expression, built from
building blocks or typed as text (the same function names as the report expressions), shown typeset with its
kind and unit, errors underlined where they are; right the result on the solution, frequency, excitation and
phase chosen at the top. An operation over a geometry (integral, maximum, ...) hides the window and asks for
the face, object or line in the 3D view. Named expressions are saved with the design and appear in Plot Fields
(fields) and in reports (values)."""
from __future__ import annotations

import io
import re

from PySide6.QtCore import QStringListModel, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPixmap, QShortcut, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QCompleter, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
                               QSizePolicy, QSplitter, QTextEdit, QToolBar, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from ..post.calc import lang as L
from ..post.calc import library as LIB
from ..post.calc import units as UN
from . import calc_bridge as CB
from . import fields as F

PREVIEW = "Fields Calculator"         # name of the live preview overlay in the 3D view
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
ERR = "#c0392b"
MUTED = "#6b7280"


def render_tex(tex: str, px: int = 18) -> QPixmap | None:
    """The formula typeset with matplotlib's mathtext as a pixmap (None if it cannot be drawn)."""
    try:
        from matplotlib import mathtext
        from matplotlib.font_manager import FontProperties
        buf = io.BytesIO()
        mathtext.math_to_image("$" + tex + "$", buf, prop=FontProperties(size=px), dpi=100, format="png",
                               color="#1f2937")
        pm = QPixmap()
        pm.loadFromData(buf.getvalue(), "PNG")
        return pm if not pm.isNull() else None
    except Exception:  # noqa: BLE001
        return None


def call_at(text: str, pos: int) -> tuple[str, int] | None:
    """(function, index of the argument) the cursor at pos is in, or None."""
    try:
        toks = L.tokenize(text[:pos])
    except L.CalcError:
        return None
    stack: list[list] = []
    for i, t in enumerate(toks):
        if t.t == "op" and t.v in "([":
            fn = toks[i - 1].v if t.v == "(" and i > 0 and toks[i - 1].t == "id" else None
            stack.append([fn, 0])
        elif t.t == "op" and t.v in ")]":
            if stack:
                stack.pop()
        elif t.t == "op" and t.v == "," and stack:
            stack[-1][1] += 1
    for fn, k in reversed(stack):
        if fn:
            return fn, k
    return None


class ExprEdit(QPlainTextEdit):
    """The expression line: Enter evaluates, errors are underlined, names complete as you type."""

    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabChangesFocus(True)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        f = QFont("Consolas")
        f.setStyleHint(QFont.Monospace)
        f.setPointSize(11)
        self.setFont(f)
        self.setFixedHeight(self.fontMetrics().lineSpacing() * 3 + 14)
        self.setPlaceholderText("e.g.  flux(Port1, re(S))   or   0.5*omega*eps0*eps_r*tand*|E|^2")
        self.completer = QCompleter(self)
        self.completer.setWidget(self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.completer.activated.connect(self._complete)
        self.model = QStringListModel(self)
        self.completer.setModel(self.model)

    def set_words(self, words: list[str]) -> None:
        self.model.setStringList(sorted(set(words), key=str.lower))

    def _prefix(self) -> str:
        c = self.textCursor()
        text = self.toPlainText()[:c.position()]
        m = re.search(r"[A-Za-z_][A-Za-z_0-9]*$", text)
        return m.group(0) if m else ""

    def _complete(self, word: str) -> None:
        c = self.textCursor()
        c.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, len(self._prefix()))
        c.insertText(word)
        self.setTextCursor(c)

    def insert(self, text: str) -> None:
        c = self.textCursor()
        c.insertText(text)
        self.setTextCursor(c)
        self.setFocus()

    def underline(self, start: int | None, end: int | None) -> None:
        if start is None:
            self.setExtraSelections([])
            return
        n = len(self.toPlainText())
        start = max(0, min(start, n))
        end = max(start + 1, min(end if end is not None else start + 1, n))
        if start >= n:
            start, end = max(0, n - 1), n
        sel = QTextEdit.ExtraSelection()
        fmt = QTextCharFormat()
        fmt.setUnderlineStyle(QTextCharFormat.WaveUnderline)
        fmt.setUnderlineColor(QColor(ERR))
        sel.format = fmt
        c = self.textCursor()
        c.setPosition(start)
        c.setPosition(end, QTextCursor.KeepAnchor)
        sel.cursor = c
        self.setExtraSelections([sel])

    def keyPressEvent(self, e):
        pop = self.completer.popup()
        if pop.isVisible() and e.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Tab, Qt.Key_Escape):
            e.ignore()
            return
        if e.key() in (Qt.Key_Enter, Qt.Key_Return):
            self.submitted.emit()
            return
        super().keyPressEvent(e)
        p = self._prefix()
        if len(p) >= 2 and e.text() and (e.text().isalnum() or e.text() == "_"):
            self.completer.setCompletionPrefix(p)
            if self.completer.completionCount():
                r = self.cursorRect()
                r.setWidth(220)
                self.completer.complete(r)
                return
        pop.hide()


STYLE = """
QMainWindow#FieldsCalculator { background:#f6f7f9; }
QWidget#Side, QWidget#Right { background:#fbfbfc; }
QLabel[role="h"] { color:#5a6476; font-size:11px; font-weight:600; letter-spacing:0.5px; }
QLabel[role="muted"] { color:#6b7280; }
QLineEdit#Name { font-size:20px; font-weight:600; border:none; border-bottom:1px solid #c3c8d2;
                 background:transparent; padding:2px; }
QLineEdit#Desc { border:none; border-bottom:1px dashed #c3c8d2; background:transparent; color:#4b5563;
                 padding:2px; }
QFrame#Group { background:#ffffff; border:1px solid #d6dae1; border-radius:4px; }
QFrame#Group QLabel { color:#7a8496; font-size:10px; font-weight:600; border:none; }
QToolButton[role="sym"] { font-family:"Cambria Math","STIX Two Math","Times New Roman",serif; font-size:15px;
                          min-width:28px; min-height:24px; border:1px solid #e1e4ea; border-radius:3px;
                          background:#fafbfc; padding:1px 4px; }
QToolButton[role="sym"]:hover { background:#e8eefc; border-color:#9fb4ee; }
QLabel#Formula { background:#ffffff; border:1px solid #c3c8d2; border-radius:4px; }
QLabel#Value { font-size:30px; font-weight:600; }
QPushButton#Primary { background:#2f5bd3; color:white; border:1px solid #2a50bb; border-radius:4px;
                      padding:5px 14px; font-weight:600; }
QPushButton#Primary:hover { background:#2a50bb; }
QLabel#Banner { background:#fff7e0; border:1px solid #e6c766; border-radius:4px; padding:6px; }
"""

# palette: (group, [(label, text inserted, tooltip)])
PALETTE = [
    ("FIELDS", [("E", "E", "electric field (V/m)"), ("H", "H", "magnetic field (A/m)"),
                ("J", "J", "volume current density (A/m²)"), ("Jₛ", "Js", "surface current density n×H (A/m)"),
                ("S", "S", "complex Poynting vector ½ E×H* (W/m²)"), ("D", "D", "electric flux density"),
                ("B", "B", "magnetic flux density")]),
    ("OPERATORS", [("|x|", "|", "magnitude |…|"), ("·", "·", "dot product"), ("×", "×", "cross product"),
                   ("x²", "^2", "square"), ("√", "sqrt(", "square root"), ("Re", "re(", "real part"),
                   ("Im", "im(", "imaginary part"), ("x*", "conj(", "complex conjugate")]),
    ("CALCULUS", [("∇×", "curl(", "curl"), ("∇·", "div(", "divergence"), ("∇", "grad(", "gradient")]),
    ("CONSTANTS", [("ε₀", "eps0", "vacuum permittivity"), ("μ₀", "mu0", "vacuum permeability"),
                   ("ω", "omega", "angular frequency 2πf"), ("k₀", "k0", "free-space wavenumber"),
                   ("εᵣ", "eps_r", "relative permittivity of the material at the point"),
                   ("tanδ", "tand", "loss tangent of the material"), ("σ", "sigma", "conductivity")]),
]
# operations over a geometry: (label, function, dimension of the geometry, tooltip)
GEOMETRY_OPS = [("∫dl", "int_line", 1, "line integral (vectors: tangential F·dl)"),
                ("∮dl", "circ", 1, "circulation around a closed line"),
                ("∬dA", "int_surf", 2, "surface integral"),
                ("∯·n", "flux", 2, "flux of a vector through a face, along its normal"),
                ("∭dV", "int_vol", 3, "volume integral"),
                ("max", "max", -1, "maximum over a geometry"), ("avg", "avg", -1, "mean value over a geometry"),
                ("@pt", "at", 0, "value at a point")]
WHAT = {0: "a point", 1: "a line", 2: "a face", 3: "a volume", -1: "a geometry"}


def _h(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("role", "h")
    return lab


class PickBar(QDialog):
    """Shown while the calculator is hidden: asks for the geometry of an integral (or any name) to be picked in
    the 3D view; Use puts it into the expression and brings the calculator back."""

    def __init__(self, calc: "CalculatorWindow", fn: str | None, dim: int):
        super().__init__(calc.win, Qt.Tool | Qt.WindowStaysOnTopHint)
        self.calc, self.fn, self.dim = calc, fn, dim
        self.doc = calc.doc
        self.setWindowTitle("Fields Calculator: pick")
        self.setAttribute(Qt.WA_DeleteOnClose)
        lay = QVBoxLayout(self)
        op = next((o for o in GEOMETRY_OPS if o[1] == fn), None)
        title = QLabel(f"<b>{op[0] + '  ' + op[3] if op else 'Insert a name'}</b>")
        lay.addWidget(title)
        how = {2: "Click a face in the 3D view (Ctrl+click: several faces).",
               3: "Click an object in the 3D view (Ctrl+click: several).",
               1: "Click a line (polyline) in the 3D view.",
               0: "Choose a point object or type a point below.",
               -1: "Click an object or a face in the 3D view."}[dim]
        lay.addWidget(QLabel(f"Select {WHAT[dim]}: {how}"))
        self.sel = QLabel("Selected: —")
        self.sel.setStyleSheet("font-family:Consolas,monospace")
        lay.addWidget(self.sel)
        row = QHBoxLayout()
        row.addWidget(QLabel("or choose"))
        self.names = QComboBox()
        self.names.setEditable(True)
        self.names.setMinimumWidth(220)
        env = CB.make_env(self.doc)
        dims = {0, 1, 2, 3} if dim < 0 else {dim}
        items = [n for n, g in env.geometry.items() if set(g.dims) & dims and (fn != "circ" or g.closed)]
        if dims & {2}:
            items += ["cut_xy(0)", "cut_yz(0)", "cut_xz(0)"]
        if dims & {0}:
            items += ["point(0, 0, 0)"]
        self.names.addItem("")
        self.names.addItems(items)
        self.names.currentTextChanged.connect(self._show)
        row.addWidget(self.names, 1)
        lay.addLayout(row)
        self.flip = QCheckBox("Reverse the normal" if dim == 2 else "Reverse the direction")
        self.flip.setVisible(dim in (1, 2))
        lay.addWidget(self.flip)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Use")
        bb.accepted.connect(self._use)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.mode0 = self.doc.selection_mode
        mode = {2: "face", 3: "object", 1: "object", 0: "object", -1: self.mode0}[dim]
        if mode != self.mode0:
            calc.win.set_mode(mode)
        self.doc.selectionChanged.connect(self._show)
        self.finished.connect(self._back)
        self._show()
        g = calc.win.geometry()
        self.adjustSize()
        self.move(g.x() + (g.width() - self.width()) // 2, g.y() + 90)

    def geometry_text(self) -> str:
        typed = self.names.currentText().strip()
        names = [typed] if typed else []
        if not names:
            tess = self.doc.tess
            for s in self.doc.selection:
                t = tess.get(s.obj)
                sheet = t is not None and getattr(t, "dim", 3) == 2
                n = f"face({s.obj}, {s.index})" if self.doc.selection_mode == "face" and s.index >= 0 \
                    and not sheet else s.obj
                if n not in names:
                    names.append(n)
        if self.flip.isChecked() and self.dim in (1, 2):
            names = [f"flip({n})" for n in names]
        return "" if not names else names[0] if len(names) == 1 else "[" + ", ".join(names) + "]"

    def _show(self, *_):
        t = self.geometry_text()
        self.sel.setText(f"Selected: {t or '—'}")

    def _use(self) -> None:
        geo = self.geometry_text()
        if not geo:
            self.sel.setText(f"<span style='color:{ERR}'>Select {WHAT[self.dim]} first</span>")
            return
        self.calc.insert_geometry(self.fn, geo)
        self.accept()

    def _back(self, *_):
        try:
            self.doc.selectionChanged.disconnect(self._show)
        except (RuntimeError, TypeError):
            pass
        if self.doc.selection_mode != self.mode0:
            self.calc.win.set_mode(self.mode0)
        self.calc.show()
        self.calc.raise_()
        self.calc.activateWindow()
        self.calc.edit.setFocus()


class CalculatorWindow(QMainWindow):
    """The Fields Calculator: its own window next to the modeler. Left the expressions of the project, the
    functions and the library; in the middle the expression with building blocks, typeset formula and text;
    right the result."""

    def __init__(self, win):
        super().__init__(win, Qt.Window)
        self.win = win
        self.doc = win.doc
        self.setObjectName("FieldsCalculator")
        self.setWindowTitle("Fields Calculator")
        self.setStyleSheet(STYLE)
        self.current: str | None = None        # name of the loaded project expression
        self.saved_text = ""
        self.info: L.Info | None = None
        self.for_plot = False
        self._last: tuple | None = None       # (text, context) of the shown result
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._analyze)
        self._toolbar()
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._side())
        split.addWidget(self._work())
        split.addWidget(self._right())
        split.setStretchFactor(1, 1)
        split.setSizes([260, 680, 300])
        self.setCentralWidget(split)
        self.status = QLabel()
        self.statusBar().addWidget(self.status, 1)
        keys = QLabel("Enter: evaluate · Ctrl+S: save · Ctrl+N: new")
        keys.setProperty("role", "muted")
        self.statusBar().addPermanentWidget(keys)
        QShortcut(QKeySequence.Save, self, self.save)
        QShortcut(QKeySequence.New, self, self.new)
        QShortcut(QKeySequence("F5"), self, self.evaluate)
        self.doc.changed.connect(self._doc_changed)
        self.resize(1240, 720)
        st = getattr(win, "settings", None)
        if st is not None and st.value("calc/geometry"):
            self.restoreGeometry(st.value("calc/geometry"))
        self.refresh()
        self.new()

    # ---- layout ----------------------------------------------------------------------------------
    def _toolbar(self) -> None:
        tb = QToolBar("Fields Calculator")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        tb.addAction("New", self.new).setToolTip("A new expression (Ctrl+N)")
        tb.addAction("Save", self.save).setToolTip("Save the expression in the design (Ctrl+S)")
        self.a_del = tb.addAction("Delete", self.delete)
        tb.addSeparator()
        ev = QPushButton("▶  Evaluate")
        ev.setObjectName("Primary")
        ev.clicked.connect(self.evaluate)
        tb.addWidget(ev)
        tb.addAction("Plot Field…", self.plot_field).setToolTip("Plot this field in the 3D view")
        tb.addAction("Add to Report", self.add_to_report).setToolTip("Show this value over frequency in a report")
        tb.addSeparator()
        tb.addAction("Pick…", lambda: self.pick(None, -1)).setToolTip(
            "Hide the calculator, click an object or face in the 3D view, and insert its name")
        tb.addAction("ƒ Functions…", self.functions_dialog).setToolTip("Your own functions: name(a, b) = ...")
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.sol = QComboBox()
        self.sol.setMinimumWidth(190)
        self.freq = QComboBox()
        self.port = QComboBox()
        self.phase = QDoubleSpinBox()
        self.phase.setRange(-360, 360)
        self.phase.setSuffix("°")
        self.phase.setToolTip("Phase for inst() and for the preview in the 3D view")
        for lab, w in (("Solution ", self.sol), ("  ", self.freq), ("  Port ", self.port), ("  Phase ", self.phase)):
            t = QLabel(lab)
            t.setProperty("role", "muted")
            tb.addWidget(t)
            tb.addWidget(w)
        self.sol.currentTextChanged.connect(self._solution_changed)
        self.freq.currentIndexChanged.connect(lambda *_: self._context_changed())
        self.port.currentIndexChanged.connect(lambda *_: self._context_changed())
        self.phase.valueChanged.connect(lambda *_: self._context_changed())

    def _side(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Side")
        w.setMinimumWidth(250)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 4, 8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search expressions")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_tree)
        lay.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(8)
        self.tree.setFrameShape(QFrame.NoFrame)
        self.tree.setStyleSheet("QTreeWidget{background:transparent} QTreeWidget::item{padding:3px 0}")
        self.tree.itemClicked.connect(self._item_clicked)
        self.tree.itemDoubleClicked.connect(self._item_double_clicked)
        lay.addWidget(self.tree, 1)
        leg = QLabel("∫ value   ◧ field   ƒ function")
        leg.setProperty("role", "muted")
        lay.addWidget(leg)
        return w

    def _group(self, title: str, buttons) -> QFrame:
        fr = QFrame()
        fr.setObjectName("Group")
        v = QVBoxLayout(fr)
        v.setContentsMargins(6, 3, 6, 5)
        v.setSpacing(2)
        v.addWidget(QLabel(title))
        row = QHBoxLayout()
        row.setSpacing(3)
        for label, slot, tip in buttons:
            b = QToolButton()
            b.setText(label)
            b.setToolTip(tip)
            b.setProperty("role", "sym")
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)
        return fr

    def _work(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 12, 16, 8)
        lay.setSpacing(10)
        self.banner = QLabel()
        self.banner.setObjectName("Banner")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        lay.addWidget(self.banner)
        nrow = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setObjectName("Name")
        self.name.setPlaceholderText("Name")
        self.name.setFixedWidth(220)
        self.name.textChanged.connect(self._dirty_changed)
        self.desc = QLineEdit()
        self.desc.setObjectName("Desc")
        self.desc.setPlaceholderText("Description (optional)")
        self.desc.textChanged.connect(self._dirty_changed)
        self.unsaved = QLabel("")
        self.unsaved.setProperty("role", "muted")
        self.unsaved.setMinimumWidth(self.unsaved.fontMetrics().horizontalAdvance("unsaved") + 6)
        nrow.addWidget(self.name)
        nrow.addSpacing(10)
        nrow.addWidget(self.desc, 1)
        nrow.addWidget(self.unsaved)
        lay.addLayout(nrow)

        pal = QGridLayout()
        pal.setHorizontalSpacing(8)
        pal.setVerticalSpacing(8)
        groups = [self._group(t, [(lab, (lambda _=False, s=ins: self.edit.insert(s)), tip) for lab, ins, tip in items])
                  for t, items in PALETTE]
        geo = self._group("OVER GEOMETRY", [(lab, (lambda _=False, f=fn, d=dim: self.pick(f, d)),
                                             tip + " — pick the geometry in the 3D view")
                                            for lab, fn, dim, tip in GEOMETRY_OPS])
        comp = self._group("COMPONENT", [("xyz", self.component_dialog,
                                          "a component of a vector in a coordinate system (x, ρ, φ, r, θ, …)")])
        pal.addWidget(groups[0], 0, 0)
        pal.addWidget(groups[1], 0, 1)
        pal.addWidget(groups[2], 0, 2)
        pal.addWidget(geo, 1, 0, 1, 2)
        pal.addWidget(groups[3], 1, 2, 1, 2)
        pal.addWidget(comp, 0, 3)
        pal.setColumnStretch(4, 1)
        lay.addLayout(pal)

        self.formula = QLabel()
        self.formula.setObjectName("Formula")
        self.formula.setAlignment(Qt.AlignCenter)
        self.formula.setMinimumHeight(150)
        self.formula.setMaximumHeight(280)
        self.formula.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.formula, 1)
        self.edit = ExprEdit()
        self.edit.setFixedHeight(self.edit.fontMetrics().lineSpacing() * 2 + 16)
        self.edit.textChanged.connect(self._text_changed)
        self.edit.cursorPositionChanged.connect(self._signature)
        self.edit.submitted.connect(self.evaluate)
        lay.addWidget(self.edit)
        self.sig = QLabel()
        self.sig.setWordWrap(True)
        self.sig.setTextFormat(Qt.RichText)
        self.sig.setProperty("role", "muted")
        self.sig.linkActivated.connect(lambda s: self.edit.insert(s))
        lay.addWidget(self.sig)
        self.chips = QLabel()
        self.chips.setWordWrap(True)
        self.chips.setTextFormat(Qt.RichText)
        lay.addWidget(self.chips)
        lay.addStretch(0)
        return w

    def _right(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Right")
        w.setMinimumWidth(280)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 12, 14, 10)
        lay.setSpacing(8)
        lay.addWidget(_h("RESULT"))
        self.result = QLabel("—")
        self.result.setObjectName("Value")
        self.result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        self.result_note = QLabel()
        self.result_note.setWordWrap(True)
        self.result_note.setProperty("role", "muted")
        lay.addWidget(self.result_note)

        # a field: reduce it to a value, show it in 3D
        self.field_box = QWidget()
        fl = QVBoxLayout(self.field_box)
        fl.setContentsMargins(0, 6, 0, 0)
        fl.addWidget(_h("REDUCE TO A VALUE"))
        rr = QHBoxLayout()
        self.red_fn = QComboBox()
        self.red_fn.addItems(["Maximum", "Minimum", "Average", "Integral", "Value at point"])
        self.red_fn.currentIndexChanged.connect(lambda *_: self._fill_reduce_geometries())
        self.red_geo = QComboBox()
        self.red_geo.setEditable(True)
        rr.addWidget(self.red_fn)
        rr.addWidget(QLabel("over"))
        rr.addWidget(self.red_geo, 1)
        fl.addLayout(rr)
        rb = QHBoxLayout()
        b = QPushButton("Evaluate")
        b.clicked.connect(self._reduce)
        u = QPushButton("Use as expression")
        u.clicked.connect(lambda: self._reduce(use=True))
        rb.addWidget(b)
        rb.addWidget(u)
        fl.addLayout(rb)
        fl.addSpacing(6)
        fl.addWidget(_h("SHOW IN 3D"))
        pr = QHBoxLayout()
        self.live = QCheckBox("Live on")
        self.live.setToolTip("Show the field in the 3D view while you type")
        self.live.toggled.connect(lambda on: self._preview() if on else self._remove_preview())
        self.target = QComboBox()
        self.target.addItems(["Cut XY", "Cut YZ", "Cut XZ", "3D selection"])
        self.target.currentIndexChanged.connect(lambda *_: self._preview())
        pr.addWidget(self.live)
        pr.addWidget(self.target, 1)
        fl.addLayout(pr)
        lay.addWidget(self.field_box)

        # a value: over frequency
        self.freq_box = QWidget()
        ql = QVBoxLayout(self.freq_box)
        ql.setContentsMargins(0, 6, 0, 0)
        hr = QHBoxLayout()
        hr.addWidget(_h("OVER FREQUENCY"))
        hr.addStretch(1)
        self.b_sweep = QPushButton("Compute")
        self.b_sweep.setToolTip("Evaluate at every frequency with saved fields of the solution")
        self.b_sweep.clicked.connect(self._over_frequency)
        hr.addWidget(self.b_sweep)
        ql.addLayout(hr)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(3, 1.5), dpi=90)
        self.fig.patch.set_facecolor("#fbfbfc")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(140)
        ql.addWidget(self.canvas)
        self.freq_note = QLabel()
        self.freq_note.setProperty("role", "muted")
        ql.addWidget(self.freq_note)
        lay.addWidget(self.freq_box)

        lay.addWidget(_h("USED BY"))
        self.used = QLabel("—")
        self.used.setWordWrap(True)
        self.used.setProperty("role", "muted")
        lay.addWidget(self.used)
        lay.addStretch(1)
        cp = QPushButton("Copy value")
        cp.clicked.connect(lambda: QApplication.clipboard().setText(self.result.text()))
        lay.addWidget(cp)
        return w

    # ---- context ---------------------------------------------------------------------------------
    def refresh(self) -> None:
        """Expressions list, geometry names for completion, solutions."""
        self._fill_tree()
        self._fill_solutions()
        env = CB.make_env(self.doc)
        words = list(L.SYMBOLS) + list(L.SIGNATURES) + list(env.named) + list(env.functions) + \
            list(env.geometry) + [c for c in env.cs_names]
        self.edit.set_words(words)
        self._fill_reduce_geometries()
        self._analyze()

    def _doc_changed(self, what: str) -> None:
        if what in ("results", "analysis", "fields", "model", "all", "design"):
            cur = self.current
            self._fill_tree()
            if what in ("results", "analysis", "all", "design"):
                self._fill_solutions()
            if cur and not any(e.name == cur for e in self.doc.design.field_exprs):
                self.current = None
                self._dirty_changed()
            self._used_by()

    def _fill_solutions(self) -> None:
        cur = self.sol.currentText()
        self.sol.blockSignals(True)
        self.sol.clear()
        sols = CB.field_solutions(self.doc) if self.doc.project.path else []
        self.sol.addItems(sols or ["(no field solution: analyze with Save fields)"])
        if cur in sols:
            self.sol.setCurrentText(cur)
        self.sol.blockSignals(False)
        self._solution_changed(self.sol.currentText())

    def _solution_changed(self, text: str) -> None:
        self.freq.blockSignals(True)
        self.port.blockSignals(True)
        f0, p0 = self.freq.currentIndex(), self.port.currentText()
        self.freq.clear()
        self.port.clear()
        try:
            fr = CB.frequencies(self.doc, text) if " : " in text else []
        except Exception:  # noqa: BLE001
            fr = []
        self.freq.addItems([f"{f / 1e9:.6g} GHz" for f in fr] or ["—"])
        if 0 <= f0 < self.freq.count():
            self.freq.setCurrentIndex(f0)
        ports = [str(int(p.props.get("PortNumber", i + 1) or i + 1)) for i, p in enumerate(self.doc.design.ports())]
        self.port.addItems(ports or ["1"])
        if p0 in ports:
            self.port.setCurrentText(p0)
        self.freq.blockSignals(False)
        self.port.blockSignals(False)
        self._context_changed()

    def context(self) -> tuple[str, int, int, float]:
        return (self.sol.currentText(), max(0, self.freq.currentIndex()), int(self.port.currentText() or 1),
                self.phase.value())

    def _context_changed(self) -> None:
        if self.result.text() not in ("—", "", "Field"):
            self.result.setText("—")
            self.result_note.setText("Context changed: Evaluate again")
        self.fig.clear()
        self.canvas.draw_idle()
        self.freq_note.setText("")
        self._preview()

    # ---- expressions list ------------------------------------------------------------------------
    def _fill_tree(self) -> None:
        self.tree.clear()
        env = CB.make_env(self.doc)
        des = self.doc.design

        def head(text):
            it = QTreeWidgetItem(self.tree, [text, ""])
            f = it.font(0)
            f.setBold(True)
            f.setPointSizeF(f.pointSizeF() * 0.85)
            it.setFont(0, f)
            it.setForeground(0, QColor("#5a6476"))
            it.setFlags(Qt.ItemIsEnabled)
            it.setFirstColumnSpanned(True)
            return it

        def row(name, text, desc, kind, unit, data):
            icon = {"Value": "∫", "Vector": "∫", "function": "ƒ"}.get(kind, "◧" if kind else "·")
            it = QTreeWidgetItem(self.tree, [f"{icon}  {name}", unit])
            it.setToolTip(0, f"{name} = {text}" + (f"\n{desc}" if desc else ""))
            it.setData(0, Qt.UserRole, data)
            it.setForeground(1, QColor("#7a8496"))
            it.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
            if self.current and data == ("expr", self.current):
                self.tree.setCurrentItem(it)
            return it

        def kind_of(text):
            try:
                i = L.analyze(text, env)
                return ("Value" if not i.field and not i.vec else "Vector" if not i.field else "field",
                        UN.label(i.unit))
            except L.CalcError:
                return "error", "⚠"
        head("PROJECT")
        for e in des.field_exprs:
            k, u = kind_of(e.expr)
            row(e.name, e.expr, e.description, k, u, ("expr", e.name))
        if not des.field_exprs:
            it = QTreeWidgetItem(self.tree, ["   (none yet)", ""])
            it.setForeground(0, QColor("#98a0ae"))
            it.setFlags(Qt.NoItemFlags)
        if des.field_funcs:
            head("MY FUNCTIONS")
            for text in des.field_funcs:
                name = text.split("(", 1)[0].strip()
                row(text.split("=", 1)[0].strip(), text, "", "function", "", ("func", name))
        head("LIBRARY")
        for name, text, desc in LIB.EXPRESSIONS:
            k, u = kind_of(text)
            row(desc.split(" (")[0].split(",")[0], text, desc, k, u, ("lib", name))
        for text, desc in LIB.FUNCTIONS:
            row(desc.split(" (")[0].split(",")[0], text, desc, "function", "", ("libfn", text))
        self.tree.resizeColumnToContents(1)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, self.tree.header().ResizeMode.Stretch)
        self._filter_tree(self.search.text())

    def _filter_tree(self, text: str) -> None:
        t = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.data(0, Qt.UserRole):
                it.setHidden(bool(t) and t not in it.text(0).lower() and t not in it.toolTip(0).lower())

    def _item_clicked(self, it: QTreeWidgetItem, _col: int) -> None:
        data = it.data(0, Qt.UserRole)
        if not data:
            return
        kind, key = data
        if kind == "expr":
            self.load(key)
        elif kind == "lib":
            name, text, desc = next(x for x in LIB.EXPRESSIONS if x[0] == key)
            self.new()
            self.name.setText(name if not any(e.name == name for e in self.doc.design.field_exprs) else "")
            self.desc.setText(desc)
            self.edit.setPlainText(text)
        elif kind in ("func", "libfn"):
            self.functions_dialog(select=key)

    def _item_double_clicked(self, it: QTreeWidgetItem, _col: int) -> None:
        """A double click on a project expression inserts its name at the cursor."""
        data = it.data(0, Qt.UserRole)
        if data and data[0] == "expr" and data[1] != self.current:
            self.edit.insert(data[1])

    def load(self, name: str) -> None:
        e = next((x for x in self.doc.design.field_exprs if x.name == name), None)
        if e is None:
            return
        self.current = name
        self.name.setText(e.name)
        self.desc.setText(e.description)
        self.edit.setPlainText(e.expr)
        self.saved_text = self._state()
        self._dirty_changed()
        self._clear_result()
        self._used_by()

    def new(self) -> None:
        self.current = None
        self.name.clear()
        self.desc.clear()
        self.edit.clear()
        self.saved_text = self._state()
        self._clear_result()
        self.result_note.setText("Type an expression, use the building blocks, or start from the Library.")
        self._dirty_changed()
        self._used_by()
        self.tree.clearSelection()
        self.edit.setFocus()

    def _clear_result(self) -> None:
        self.result.setText("—")
        self.result_note.setText("")
        self.fig.clear()
        self.canvas.draw_idle()
        self.freq_note.setText("")

    def _state(self) -> str:
        return f"{self.name.text()}\n{self.desc.text()}\n{self.edit.toPlainText()}"

    def _dirty_changed(self, *_):
        dirty = self._state() != self.saved_text and bool(self.edit.toPlainText().strip())
        self.unsaved.setText("unsaved" if dirty else ("saved" if self.current else ""))
        self.a_del.setEnabled(self.current is not None)
        self.setWindowTitle(f"Fields Calculator — {self.name.text() or 'new expression'}"
                            f"{' *' if dirty else ''}  ({self.doc.design.name})")

    def _used_by(self) -> None:
        if not self.current:
            self.used.setText("—")
            return
        des = self.doc.design
        uses = CB.dependents(des, self.current)
        uses += [f"plot {p.name}" for p in des.field_plots if p.quantity == self.current]
        uses += [f"report {r.name}" for r in des.reports
                 if any(self.current in CB.field_names_in(des, t) for t in r.traces)]
        self.used.setText(" · ".join(uses) or "not used yet")

    # ---- typing ----------------------------------------------------------------------------------
    def _text_changed(self) -> None:
        self._dirty_changed()
        self._timer.start()

    def _analyze(self) -> None:
        text = self.edit.toPlainText().strip()
        self.info = None
        self.edit.underline(None, None)
        if not text:
            self.formula.clear()
            self.formula.setText("<span style='color:#98a0ae;font-size:15px'>The formula appears here as you "
                                 "type or click the building blocks</span>")
            self.chips.setText("")
            self.status.setText("")
            self.field_box.setVisible(False)
            self.freq_box.setVisible(False)
            return
        env = CB.make_env(self.doc)
        try:
            self.info = L.analyze(self.edit.toPlainText(), env)
        except L.CalcError as e:
            self.edit.underline(e.pos, e.end)
            self.chips.setText("")
            self.status.setText(f"<span style='color:{ERR}'>⚠ {e}</span>")
            self.formula.clear()
            self.formula.setText(f"<span style='color:{ERR};font-size:14px'>{e}</span>")
            self.field_box.setVisible(False)
            self.freq_box.setVisible(False)
            return
        i = self.info
        pm = render_tex(i.tex, px=30)
        if pm is not None:
            self.formula.setPixmap(pm)
        else:
            self.formula.setText(text)

        def chip(t, bg="#e8ecf4", fg="#1d2330"):
            return f"<span style='background:{bg};color:{fg}'>&nbsp;{t}&nbsp;</span>"
        chips = [chip(f"<b>{i.kind}</b>", "#dfeee4", "#1b6b3a"), chip(UN.describe(i.unit)),
                 chip("complex" if i.cx else "real")]
        chips += [chip(c) for c in L.conventions_text(i)]
        if i.normals:
            chips.append(chip("direction: " + ", ".join(dict.fromkeys(g.label() for g in i.normals))))
        if i.deps:
            chips.append(chip("uses " + ", ".join(dict.fromkeys(i.deps))))
        if i.surf and i.field:
            chips.append(chip("faces only"))
        self.chips.setText("&nbsp;&nbsp;".join(chips))
        self.status.setText("✓ No errors")
        self.field_box.setVisible(i.field)
        self.freq_box.setVisible(not i.field and not i.vec)
        self._fill_reduce_geometries()
        if self.live.isChecked():
            self._preview()

    def _signature(self) -> None:
        self.sig.setText(self._signature_html())
        self.sig.setVisible(bool(self.sig.text()))

    def _signature_html(self) -> str:
        text, pos = self.edit.toPlainText(), self.edit.textCursor().position()
        c = call_at(text, pos)
        env = CB.make_env(self.doc)
        if not c:
            return ""
        fn, k = c
        if fn in L.SIGNATURES:
            params, doc = L.SIGNATURES[fn]
        elif fn in env.functions:
            params, doc = env.functions[fn].params, env.functions[fn].body
        else:
            return ""
        ps = [f"<b>{p}</b>" if j == k else p for j, p in enumerate(params)]
        html = f"{fn}({', '.join(ps)}) — {doc}"
        if k == 0 and fn in L.GEOM_FN:
            dims = L.GEOM_FN[fn]
            names = [n for n, g in env.geometry.items() if set(g.dims) & dims and (fn != "circ" or g.closed)]
            if names:
                html += "<br>" + " ".join(f"<a href='{n}'>{n}</a>" for n in names[:24])
        return html

    # ---- geometry from the 3D view ---------------------------------------------------------------
    def pick(self, fn: str | None, dim: int) -> None:
        """Hide the calculator and ask for the geometry in the 3D view (PickBar); it comes back afterwards."""
        self._pick_sel = self.edit.textCursor().selectedText()
        self.hide()
        bar = PickBar(self, fn, dim)
        bar.show()
        self._bar = bar

    def insert_geometry(self, fn: str | None, geo: str) -> None:
        """fn(geo, |) at the cursor; a selected text, or a whole field expression, becomes the integrand."""
        if fn is None:
            self.edit.insert(geo)
            self._analyze()
            return
        sel = getattr(self, "_pick_sel", "") or ""
        text = self.edit.toPlainText().strip()
        self._analyze()
        if not sel and text and self.info is not None and self.info.field:
            self.edit.selectAll()
            sel = text
        if sel:
            inner = sel
            if fn in ("max", "min", "avg") and self.info is not None and (self.info.vec or self.info.cx) \
                    and not (sel.startswith("|") and sel.endswith("|")):
                inner = f"|{sel}|"
            self.edit.insert(f"{fn}({geo}, {inner})")
        else:
            self.edit.insert(f"{fn}({geo}, )")
            c = self.edit.textCursor()
            c.movePosition(QTextCursor.Left)
            self.edit.setTextCursor(c)
        self._analyze()

    # ---- evaluation ------------------------------------------------------------------------------
    def evaluate(self) -> None:
        self._analyze()
        if self.info is None:
            return
        if self.info.field:
            self.result.setText("Field")
            self.result_note.setText("A value at every point: plot it (Plot Field…, or Show in 3D below), or "
                                     "reduce it to a value.")
            if self.live.isChecked():
                self._preview()
            return
        self._evaluate_text(self.edit.toPlainText())

    def _evaluate_text(self, text: str) -> None:
        sol, fi, port, phase = self.context()
        if " : " not in sol:
            self.result.setText("—")
            self.result_note.setText("No solution with saved fields: analyze with 'Save fields' in the setup.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            r = CB.evaluate(self.doc, text, sol, fi, port, phase)
        except Exception as e:  # noqa: BLE001
            self.result.setText("—")
            self.result_note.setText(f"<span style='color:{ERR}'>{e}</span>")
            pos = getattr(e, "pos", None)
            if pos is not None and text == self.edit.toPlainText():
                self.edit.underline(pos, getattr(e, "end", None))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.result.setText(r.text())
        notes = [f"{sol} · {self.freq.currentText()} · port {port}" + (f" · phase {phase:g}°" if phase else "")]
        notes += L.conventions_text(r.info) + r.notes
        self.result_note.setText("<br>".join(notes))

    def _over_frequency(self) -> None:
        text = self.edit.toPlainText().strip()
        sol, _, port, _ = self.context()
        if not text or self.info is None or " : " not in sol:
            return
        if self.freq.count() < 2:
            # one frequency: use a sweep with saved fields instead
            for s in CB.field_solutions(self.doc):
                try:
                    if len(CB.frequencies(self.doc, s)) > 1:
                        sol = s
                        break
                except Exception:  # noqa: BLE001
                    continue
            else:
                self.freq_note.setText("Needs a sweep with saved fields")
                return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            f, v = CB.over_frequency(self.doc, text, sol, port)
        except Exception as e:  # noqa: BLE001
            self.result_note.setText(f"<span style='color:{ERR}'>{e}</span>")
            return
        finally:
            QApplication.restoreOverrideCursor()
        import numpy as np
        y = np.real(v) if np.allclose(np.imag(v), 0, atol=1e-12 * max(np.abs(v).max(), 1e-30)) else np.abs(v)
        self.fig.clear()
        ax = self.fig.add_axes([0.2, 0.2, 0.76, 0.74])
        k = max(float(np.abs(y).max()), 1e-300)
        exp3 = int(np.floor(np.log10(k) / 3) * 3) if UN.label(self.info.unit) else 0
        pre = {-15: "f", -12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G"}.get(exp3, "")
        if not pre and exp3:
            exp3 = 0
        ax.plot(f / 1e9, y / 10.0 ** exp3, marker="o" if len(f) < 12 else None, color="#2f5bd3", lw=1.6, ms=3)
        ax.tick_params(labelsize=7)
        ax.margins(x=0.03, y=0.15)
        self.freq_note.setText(f"{pre}{UN.label(self.info.unit)} over GHz · {sol}")
        ax.grid(True, alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        self.canvas.draw_idle()

    def _fill_reduce_geometries(self) -> None:
        cur = self.red_geo.currentText()
        env = CB.make_env(self.doc)
        dims = [{0, 1, 2, 3}, {0, 1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {0}][self.red_fn.currentIndex()]
        names = [n for n, g in env.geometry.items() if set(g.dims) & dims]
        if 2 in dims:
            names += ["cut_xy(0)", "cut_yz(0)", "cut_xz(0)"]
        if 0 in dims:
            names += ["point(0, 0, 0)"]
        self.red_geo.blockSignals(True)
        self.red_geo.clear()
        self.red_geo.addItems(names)
        if cur:
            self.red_geo.setCurrentText(cur)
        self.red_geo.blockSignals(False)

    def _reduced(self) -> str | None:
        text = self.edit.toPlainText().strip()
        geo = self.red_geo.currentText().strip()
        if not text or not geo or self.info is None:
            return None
        i = self.red_fn.currentIndex()
        inner = text
        if i in (0, 1, 2) and (self.info.vec or self.info.cx):
            inner = f"|{text}|" if not text.startswith("|") or not text.endswith("|") else text
        if i == 3:
            env = CB.make_env(self.doc)
            try:
                items = L.geometry_items(L.parse(geo), env)
                dims = set.union(*(L.geometry_dims(g, env)[0] for g in items))
            except L.CalcError:
                dims = {3}
            fn = "int_vol" if 3 in dims else "int_surf" if 2 in dims else "int_line"
            return f"{fn}({geo}, {text})"
        return f"{['max', 'min', 'avg', '', 'at'][i]}({geo}, {inner if i != 4 else text})"

    def _reduce(self, use: bool = False) -> None:
        expr = self._reduced()
        if expr is None:
            return
        if use:
            self.edit.setPlainText(expr)
            self.current = None if self.current and not self.name.text() else self.current
            self.evaluate()
            return
        self._evaluate_text(expr)
        self.result_note.setText(f"{expr}<br>" + self.result_note.text())

    # ---- 3D preview ------------------------------------------------------------------------------
    def _preview(self) -> None:
        if not self.live.isChecked():
            return
        view = self.win.view
        i = self.info
        sol, fi, port, phase = self.context()
        if i is None or not i.field or " : " not in sol:
            view.remove_overlay(PREVIEW)
            return
        t = self.target.currentText()
        plane, faces, objs = "", [], []
        if t.startswith("Cut"):
            plane = t[4:] + "@"
        else:
            faces, objs = self.win._plot_selection()
            if not faces and not objs:
                self.result_note.setText("Select objects or faces in the 3D view for the preview")
                return
        try:
            meshes, info = F.make_overlay(self.doc, view, "=" + self.edit.toPlainText(), sol, fi, port, phase,
                                          faces, objs, plane, attrs={"PlotType": "Smooth Shaded"})
        except Exception as e:  # noqa: BLE001
            self.result_note.setText(f"<span style='color:{ERR}'>Preview: {e}</span>")
            return
        view.set_overlay(PREVIEW, meshes)
        self.win.hint.setText(info.replace("=" + self.edit.toPlainText(), self.name.text() or "expression"))

    def _remove_preview(self) -> None:
        try:
            self.win.view.remove_overlay(PREVIEW)
        except Exception:  # noqa: BLE001
            pass

    def closeEvent(self, e):
        self.live.setChecked(False)
        st = getattr(self.win, "settings", None)
        if st is not None:
            st.setValue("calc/geometry", self.saveGeometry())
        super().closeEvent(e)

    # ---- lifecycle -------------------------------------------------------------------------------
    def _check_name(self, name: str) -> str | None:
        env = CB.make_env(self.doc)
        if not _IDENT.match(name):
            return "A name starts with a letter and has letters, digits and _ only"
        if name in L.SYMBOLS or name in L.SIGNATURES or name in L.ALIASES:
            return f"'{name}' is a built-in name"
        if name in env.functions:
            return f"'{name}' is a function"
        if name in env.geometry:
            return f"'{name}' is an object of the model"
        if name in env.cs_names:
            return f"'{name}' is a coordinate system"
        if name in F.QUANTITIES:
            return f"'{name}' is a built-in field plot quantity"
        if name != self.current and any(e.name == name for e in self.doc.design.field_exprs):
            return f"An expression named '{name}' exists already"
        return None

    def save(self) -> bool:
        name, text = self.name.text().strip(), self.edit.toPlainText().strip()
        if not text:
            return False
        if not name:
            self.name.setFocus()
            self.status.setText(f"<span style='color:{ERR}'>Give the expression a name to save it</span>")
            return False
        msg = self._check_name(name)
        if msg:
            QMessageBox.warning(self, "Fields Calculator", msg)
            return False
        env = CB.make_env(self.doc)
        env.named = {k: v for k, v in env.named.items() if k != (self.current or name)}
        try:
            L.analyze(text, env)
        except L.CalcError as e:
            if "Cycle" in str(e) or (self.current and self.current in L.names_in(text)):
                QMessageBox.warning(self, "Fields Calculator", f"Cannot save: {e}")
                return False
            if QMessageBox.question(self, "Fields Calculator", f"The expression has an error:\n{e}\n\nSave anyway?") \
                    != QMessageBox.Yes:
                return False
        des = self.doc.design
        self.doc.checkpoint()
        if self.current and name != self.current:
            deps = CB.dependents(des, self.current)
            if deps and QMessageBox.question(
                    self, "Rename", f"Rename '{self.current}' to '{name}'?\nThe uses in {', '.join(deps)} and in "
                    "reports are renamed too.") != QMessageBox.Yes:
                return False
            CB.rename_everywhere(des, self.current, name)
            for fp in des.field_plots:
                if fp.quantity == self.current:
                    fp.quantity = name
        e = next((x for x in des.field_exprs if x.name == (self.current or name)), None)
        if e is None:
            from ..model.design import FieldExpression
            e = FieldExpression(name, text, self.desc.text().strip())
            des.field_exprs.append(e)
        e.name, e.expr, e.description = name, text, self.desc.text().strip()
        self.current = name
        self.doc.dirty = True
        self.doc.changed.emit("fields")
        self.saved_text = self._state()
        self.refresh()
        self._dirty_changed()
        self._used_by()
        self.status.setText(f"Saved {name}")
        self.win.msgs.add("info", f"Fields Calculator: saved {name} = {text}")
        if self.for_plot and self.info is not None and self.info.field:
            self.for_plot = False
            self.banner.setVisible(False)
            self.win.plot_fields(name)
        return True

    def delete(self) -> None:
        if not self.current:
            return
        des = self.doc.design
        deps = CB.dependents(des, self.current)
        plots = [p.name for p in des.field_plots if p.quantity == self.current]
        reps = [r.name for r in des.reports if any(self.current in CB.field_names_in(des, t) for t in r.traces)]
        more = ""
        if deps or plots or reps:
            more = "\n\nIt is used by: " + ", ".join(deps + [f"plot {p}" for p in plots] + [f"report {r}" for r in reps]) + \
                   "\nThey will show an error until you change them."
        if QMessageBox.question(self, "Delete", f"Delete '{self.current}'?{more}") != QMessageBox.Yes:
            return
        self.doc.checkpoint()
        des.field_exprs = [e for e in des.field_exprs if e.name != self.current]
        self.doc.dirty = True
        self.doc.changed.emit("fields")
        self.new()
        self.refresh()

    # ---- outputs ---------------------------------------------------------------------------------
    def _ensure_saved(self, what: str) -> str | None:
        if self.current and self._state() == self.saved_text:
            return self.current
        if not self.name.text().strip():
            self.name.setFocus()
            self.status.setText(f"<span style='color:{ERR}'>{what} needs a name: give one and Save</span>")
            return None
        return self.name.text().strip() if self.save() else None

    def plot_field(self) -> None:
        self._analyze()
        if self.info is None:
            return
        if not self.info.field:
            QMessageBox.information(self, "Plot Field", "This expression is a value. Field plots need a field "
                                                        "(for a value use Add to Report).")
            return
        name = self._ensure_saved("A field plot")
        if name:
            self.live.setChecked(False)
            self.win.plot_fields(name)

    def add_to_report(self) -> None:
        self._analyze()
        if self.info is None:
            return
        if self.info.field:
            QMessageBox.information(self, "Add to Report", "Reports show values over frequency: reduce the field "
                                                           "to a value first (Reduce to a value, right).")
            return
        name = self._ensure_saved("A report trace")
        if name:
            sol, _, port, _ = self.context()
            self.win.field_report(name, sol, port)

    def open_for_plot(self) -> None:
        self.for_plot = True
        self.banner.setText("<b>New quantity for a field plot.</b> Build the expression, give it a name and "
                            "Save: the field plot dialog opens with it.")
        self.banner.setVisible(True)
        self.new()
        self.edit.setFocus()

    # ---- dialogs ---------------------------------------------------------------------------------
    def component_dialog(self) -> None:
        dlg = ComponentDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self.edit.insert(dlg.text())

    def functions_dialog(self, select: str | None = None) -> None:
        dlg = FunctionsDialog(self, select)
        dlg.exec()
        self.refresh()


class ComponentDialog(QDialog):
    """A component of a vector field in a coordinate system of the model."""

    COMPS = {"Cartesian": [("x", "x"), ("y", "y"), ("z", "z")],
             "Cylindrical": [("ρ (rho)", "rho"), ("φ (phi)", "phi"), ("z", "z")],
             "Spherical": [("r", "r"), ("θ (theta)", "theta"), ("φ (phi)", "phi")]}

    def __init__(self, panel: CalculatorWindow):
        super().__init__(panel)
        self.panel = panel
        self.setWindowTitle("Insert Component")
        f = QFormLayout(self)
        self.vec = QComboBox()
        self.vec.setEditable(True)
        items = ["E", "H", "D", "B", "J", "S", "Js"]
        cur = panel.edit.toPlainText().strip()
        if cur and panel.info is not None and panel.info.vec and cur not in items:
            items.insert(0, cur)
        self.vec.addItems(items)
        f.addRow("Vector", self.vec)
        self.cs = QComboBox()
        self.cs.addItems(panel.doc.design.cs_names())
        f.addRow("Coordinate system", self.cs)
        self.sys = QComboBox()
        self.sys.addItems(list(self.COMPS))
        f.addRow("System", self.sys)
        self.comp = QComboBox()
        f.addRow("Component", self.comp)
        self.rep = QComboBox()
        self.rep.addItems(["Complex (peak phasor)", "Real part", "Imaginary part", "Magnitude", "Phase",
                           "Instantaneous (context phase)"])
        f.addRow("As", self.rep)
        self.preview = QLabel()
        self.preview.setStyleSheet(f"color:{MUTED}")
        f.addRow("", self.preview)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Insert")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self.sys.currentTextChanged.connect(self._comps)
        for w in (self.vec, self.cs, self.comp, self.rep):
            w.currentTextChanged.connect(lambda *_: self.preview.setText(self.text()))
        self._comps(self.sys.currentText())

    def _comps(self, s: str) -> None:
        self.comp.clear()
        for label, key in self.COMPS[s]:
            self.comp.addItem(label, key)
        self.preview.setText(self.text())

    def text(self) -> str:
        v = self.vec.currentText().strip() or "E"
        if not re.match(r"^[A-Za-z_]\w*$", v):
            v = f"({v})"
        cs = self.cs.currentText()
        c = f"{self.comp.currentData() or 'x'}({v}{'' if cs == 'Global' else ', ' + cs})"
        return [c, f"re({c})", f"im({c})", f"|{c}|", f"phase({c})", f"inst({c})"][self.rep.currentIndex()]


class FunctionsDialog(QDialog):
    """Your own functions name(a, b) = expression: expanded where they are used; the parameters may also
    be geometries."""

    def __init__(self, panel: CalculatorWindow, select: str | None = None):
        super().__init__(panel)
        self.panel = panel
        self.doc = panel.doc
        self.setWindowTitle("Functions")
        self.resize(560, 420)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Functions of this design. Parameters can be fields, values or geometries, e.g.\n"
                             "voltage(L) = int_line(L, E)      ratio(a, b) = a/b"))
        self.list = QListWidget()
        self.list.currentTextChanged.connect(lambda t: self.edit.setText(t))
        lay.addWidget(self.list, 1)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("name(a, b) = expression")
        self.edit.textChanged.connect(self._check)
        lay.addWidget(self.edit)
        self.msg = QLabel()
        self.msg.setWordWrap(True)
        lay.addWidget(self.msg)
        row = QHBoxLayout()
        add = QPushButton("Add / Update")
        add.clicked.connect(self._add)
        rem = QPushButton("Delete")
        rem.clicked.connect(self._delete)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(QLabel("Library:"))
        self.lib = QListWidget()
        for text, desc in LIB.FUNCTIONS:
            self.lib.addItem(text)
            self.lib.item(self.lib.count() - 1).setToolTip(desc)
        self.lib.itemDoubleClicked.connect(lambda it: self.edit.setText(it.text()))
        self.lib.setMaximumHeight(120)
        lay.addWidget(self.lib)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._fill()
        if select:
            hits = [i for i in range(self.list.count()) if self.list.item(i).text().split("(")[0].strip() == select
                    or self.list.item(i).text() == select]
            if hits:
                self.list.setCurrentRow(hits[0])
            else:
                self.edit.setText(select)

    def _fill(self) -> None:
        self.list.clear()
        self.list.addItems(self.doc.design.field_funcs)

    def _check(self, text: str) -> bool:
        try:
            f = L.parse_function(text)
        except L.CalcError as e:
            self.msg.setText(f"<span style='color:{ERR}'>{e}</span>" if text.strip() else "")
            return False
        env = CB.make_env(self.doc)
        if f.name in env.named or f.name in env.geometry or f.name in env.cs_names:
            self.msg.setText(f"<span style='color:{ERR}'>'{f.name}' is used by an expression, object or "
                             "coordinate system</span>")
            return False
        self.msg.setText(f"<span style='color:{MUTED}'>{f.name}({', '.join(f.params)})</span>")
        return True

    def _add(self) -> None:
        text = self.edit.text().strip()
        if not self._check(text):
            return
        f = L.parse_function(text)
        des = self.doc.design
        self.doc.checkpoint()
        des.field_funcs = [x for x in des.field_funcs if x.split("(", 1)[0].strip() != f.name] + [f.text]
        self.doc.dirty = True
        self.doc.changed.emit("fields")
        self._fill()

    def _delete(self) -> None:
        it = self.list.currentItem()
        if it is None:
            return
        name = it.text().split("(", 1)[0].strip()
        deps = CB.dependents(self.doc.design, name)
        more = f"\n\nIt is used by: {', '.join(deps)}" if deps else ""
        if QMessageBox.question(self, "Delete", f"Delete the function {name}()?{more}") != QMessageBox.Yes:
            return
        self.doc.checkpoint()
        self.doc.design.field_funcs = [x for x in self.doc.design.field_funcs if x != it.text()]
        self.doc.dirty = True
        self.doc.changed.emit("fields")
        self._fill()
