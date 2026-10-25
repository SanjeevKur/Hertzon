"""Fields Calculator panel (Results > Fields Calculator), docked next to the 3D view.

Expressions are typed as text (the same function names as the report expressions); the panel shows the
formula typeset, the kind and unit while typing, underlines errors where they are, and evaluates on the
solution, frequency, excitation and phase chosen at the top. Named expressions are saved with the design and
appear in Plot Fields (fields) and in reports (values). A click in the 3D view inserts the picked object or
face at the cursor while "Pick" is on."""
from __future__ import annotations

import io
import re

from PySide6.QtCore import QStringListModel, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPixmap, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QCompleter, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton,
                               QSizePolicy, QTextEdit, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget, QButtonGroup)

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


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color:#e5e7eb")
    return f


class CalculatorPanel(QWidget):
    """The docked Fields Calculator."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.doc = win.doc
        self.current: str | None = None        # name of the loaded project expression
        self.saved_text = ""
        self.info: L.Info | None = None
        self.for_plot = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._analyze)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # banner when the calculator was opened from Plot Fields
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("background:#fff7e0;border:1px solid #e6c766;border-radius:4px;padding:6px")
        self.banner.setVisible(False)
        lay.addWidget(self.banner)

        # context
        ctx = QGridLayout()
        ctx.setHorizontalSpacing(6)
        self.sol = QComboBox()
        self.freq = QComboBox()
        self.port = QComboBox()
        self.phase = QDoubleSpinBox()
        self.phase.setRange(-360, 360)
        self.phase.setSuffix(" deg")
        self.phase.setToolTip("Phase for inst() and for the preview in the 3D view")
        ctx.addWidget(QLabel("Solution"), 0, 0)
        ctx.addWidget(self.sol, 0, 1, 1, 3)
        ctx.addWidget(QLabel("Frequency"), 1, 0)
        ctx.addWidget(self.freq, 1, 1)
        ctx.addWidget(QLabel("Port"), 1, 2)
        ctx.addWidget(self.port, 1, 3)
        ctx.addWidget(QLabel("Phase"), 2, 0)
        ctx.addWidget(self.phase, 2, 1)
        ctx.setColumnStretch(1, 1)
        lay.addLayout(ctx)
        self.sol.currentTextChanged.connect(self._solution_changed)
        self.freq.currentIndexChanged.connect(lambda *_: self._context_changed())
        self.port.currentIndexChanged.connect(lambda *_: self._context_changed())
        self.phase.valueChanged.connect(lambda *_: self._context_changed())
        lay.addWidget(_sep())

        # named expressions
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Expression", "Kind", "Unit"])
        self.tree.setRootIsDecorated(True)
        self.tree.setMinimumHeight(110)
        self.tree.itemClicked.connect(self._item_clicked)
        self.tree.setColumnWidth(0, 150)
        lay.addWidget(self.tree, 2)

        # name, description, editor
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        nrow = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("Name to save, e.g. P_diel")
        self.name.textChanged.connect(self._dirty_changed)
        self.unsaved = QLabel("")
        self.unsaved.setMinimumWidth(self.unsaved.fontMetrics().horizontalAdvance("unsaved") + 4)
        self.unsaved.setStyleSheet(f"color:{MUTED}")
        nrow.addWidget(self.name, 1)
        nrow.addWidget(self.unsaved)
        form.addRow("Name", nrow)
        self.desc = QLineEdit()
        self.desc.setPlaceholderText("Description (optional)")
        self.desc.textChanged.connect(self._dirty_changed)
        form.addRow("Description", self.desc)
        lay.addLayout(form)

        bar = QHBoxLayout()
        bar.setSpacing(3)
        for label, ins, tip in (("|x|", "|", "magnitude |...|"), ("·", "·", "dot product"),
                                ("×", "×", "cross product"), ("x²", "^2", "power"), ("Re", "re(", "real part"),
                                ("∇×", "curl(", "curl"), ("ε₀", "eps0", "vacuum permittivity"),
                                ("μ₀", "mu0", "vacuum permeability"), ("ω", "omega", "angular frequency")):
            b = QToolButton()
            b.setText(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, s=ins: self.edit.insert(s))
            bar.addWidget(b)
        bar.addStretch(1)
        lay.addLayout(bar)
        bar2 = QHBoxLayout()
        bar2.setSpacing(4)
        for label, slot, tip in (("∫ Integral…", self.integral_dialog, "Line, face or volume integral"),
                                 ("xyz Component…", self.component_dialog,
                                  "A component of a vector in a coordinate system"),
                                 ("ƒ Functions…", self.functions_dialog, "Your own functions: name(a, b) = ...")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar2.addWidget(b)
        self.pick = QToolButton()
        self.pick.setText("Pick")
        self.pick.setCheckable(True)
        self.pick.setToolTip("While on, a click on an object or a face in the 3D view (or the model tree) "
                             "inserts its name at the cursor; press F for faces")
        bar2.addWidget(self.pick)
        bar2.addStretch(1)
        lay.addLayout(bar2)

        self.edit = ExprEdit()
        self.edit.textChanged.connect(self._text_changed)
        self.edit.cursorPositionChanged.connect(self._signature)
        self.edit.submitted.connect(self.evaluate)
        lay.addWidget(self.edit)
        self.sig = QLabel()
        self.sig.setWordWrap(True)
        self.sig.setTextFormat(Qt.RichText)
        self.sig.setStyleSheet(f"color:{MUTED}")
        self.sig.linkActivated.connect(lambda s: self.edit.insert(s))
        lay.addWidget(self.sig)

        # typeset formula and what it is
        self.formula = QLabel()
        self.formula.setAlignment(Qt.AlignCenter)
        self.formula.setMinimumHeight(48)
        self.formula.setStyleSheet("background:#ffffff;border:1px solid #e5e7eb;border-radius:4px;padding:4px")
        self.formula.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay.addWidget(self.formula)
        self.chips = QLabel()
        self.chips.setWordWrap(True)
        self.chips.setTextFormat(Qt.RichText)
        lay.addWidget(self.chips)

        # result
        box = QGroupBox("Result")
        bl = QVBoxLayout(box)
        self.result = QLabel("—")
        rf = self.result.font()
        rf.setPointSize(rf.pointSize() + 3)
        rf.setBold(True)
        self.result.setFont(rf)
        self.result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result.setWordWrap(True)
        bl.addWidget(self.result)
        self.result_note = QLabel()
        self.result_note.setWordWrap(True)
        self.result_note.setStyleSheet(f"color:{MUTED}")
        bl.addWidget(self.result_note)
        # a field reduced to a value
        self.reduce_box = QWidget()
        rl = QHBoxLayout(self.reduce_box)
        rl.setContentsMargins(0, 0, 0, 0)
        self.red_fn = QComboBox()
        self.red_fn.addItems(["Maximum", "Minimum", "Average", "Integral", "Value at point"])
        self.red_geo = QComboBox()
        self.red_geo.setEditable(True)
        self.red_geo.setMinimumWidth(110)
        rb = QPushButton("Reduce")
        rb.setToolTip("Evaluate the field reduced to a value over the geometry")
        rb.clicked.connect(self._reduce)
        ub = QPushButton("Use as expression")
        ub.clicked.connect(lambda: self._reduce(use=True))
        rl.addWidget(self.red_fn)
        rl.addWidget(QLabel("over"))
        rl.addWidget(self.red_geo, 1)
        rl.addWidget(rb)
        rl.addWidget(ub)
        self.red_fn.currentIndexChanged.connect(lambda *_: self._fill_reduce_geometries())
        bl.addWidget(self.reduce_box)
        prow = QHBoxLayout()
        self.live = QCheckBox("Show in 3D on")
        self.live.setToolTip("Field expressions: show the field in the 3D view while you type")
        self.live.toggled.connect(lambda on: self._preview() if on else self._remove_preview())
        self.target = QComboBox()
        self.target.addItems(["Cut XY", "Cut YZ", "Cut XZ", "3D selection"])
        self.target.currentIndexChanged.connect(lambda *_: self._preview())
        prow.addWidget(self.live)
        prow.addWidget(self.target, 1)
        bl.addLayout(prow)
        lay.addWidget(box)

        # actions
        act = QGridLayout()
        self.b_eval = QPushButton("Evaluate")
        self.b_eval.setDefault(True)
        self.b_eval.clicked.connect(self.evaluate)
        self.b_save = QPushButton("Save")
        self.b_save.clicked.connect(self.save)
        self.b_new = QPushButton("New")
        self.b_new.clicked.connect(self.new)
        self.b_del = QPushButton("Delete")
        self.b_del.clicked.connect(self.delete)
        self.b_plot = QPushButton("Plot Field…")
        self.b_plot.clicked.connect(self.plot_field)
        self.b_report = QPushButton("Add to Report")
        self.b_report.clicked.connect(self.add_to_report)
        for i, b in enumerate((self.b_eval, self.b_save, self.b_new, self.b_del, self.b_plot, self.b_report)):
            act.addWidget(b, i // 3, i % 3)
        lay.addLayout(act)

        self.doc.selectionChanged.connect(self._picked)
        self.doc.changed.connect(self._doc_changed)
        self.refresh()
        self.new()

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
        if self.result.text() not in ("—", ""):
            self.result.setText("—")
            self.result_note.setText("Context changed: Evaluate again")
        self._preview()

    # ---- expressions list ------------------------------------------------------------------------
    def _fill_tree(self) -> None:
        self.tree.clear()
        env = CB.make_env(self.doc)
        des = self.doc.design

        def row(parent, name, text, desc, kind="", unit="", data=None):
            it = QTreeWidgetItem(parent, [name, kind, unit])
            it.setToolTip(0, f"{name} = {text}" + (f"\n{desc}" if desc else ""))
            it.setData(0, Qt.UserRole, data)
            return it
        proj = QTreeWidgetItem(self.tree, ["Project", "", ""])
        for e in des.field_exprs:
            try:
                i = L.analyze(e.expr, env)
                kind, unit = i.kind, UN.label(i.unit)
            except L.CalcError as ex:
                kind, unit = "error", str(ex)[:40]
            row(proj, e.name, e.expr, e.description, kind, unit, ("expr", e.name))
        fn = QTreeWidgetItem(self.tree, ["Functions", "", ""])
        for text in des.field_funcs:
            name = text.split("(", 1)[0].strip()
            row(fn, text.split("=", 1)[0].strip(), text, "", "function", "", ("func", name))
        lib = QTreeWidgetItem(self.tree, ["Library", "", ""])
        for name, text, desc in LIB.EXPRESSIONS:
            try:
                i = L.analyze(text, env)
                kind, unit = i.kind, UN.label(i.unit)
            except L.CalcError:
                kind, unit = "", ""
            row(lib, name, text, desc, kind, unit, ("lib", name))
        for text, desc in LIB.FUNCTIONS:
            row(lib, text.split("=", 1)[0].strip(), text, desc, "function", "", ("libfn", text))
        proj.setExpanded(True)
        fn.setExpanded(bool(des.field_funcs))
        lib.setExpanded(not des.field_exprs)
        for it in (proj, fn, lib):
            f = it.font(0)
            f.setBold(True)
            it.setFont(0, f)

    def _item_clicked(self, it: QTreeWidgetItem, _col: int) -> None:
        data = it.data(0, Qt.UserRole)
        if not data:
            return
        kind, key = data
        if self.pick.isChecked() and kind == "expr":
            self.edit.insert(key)
            return
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
        self.result.setText("—")
        self.result_note.setText("")

    def new(self) -> None:
        self.current = None
        self.name.clear()
        self.desc.clear()
        self.edit.clear()
        self.saved_text = self._state()
        self.result.setText("—")
        self.result_note.setText("Type an expression, pick an example from the Library, or use ∫ Integral…")
        self._dirty_changed()

    def _state(self) -> str:
        return f"{self.name.text()}\n{self.desc.text()}\n{self.edit.toPlainText()}"

    def _dirty_changed(self, *_):
        dirty = self._state() != self.saved_text and bool(self.edit.toPlainText().strip())
        self.unsaved.setText("unsaved" if dirty else ("saved" if self.current else ""))
        self.b_del.setEnabled(self.current is not None)

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
            self.chips.setText("")
            self.reduce_box.setVisible(False)
            return
        env = CB.make_env(self.doc)
        try:
            self.info = L.analyze(self.edit.toPlainText(), env)
        except L.CalcError as e:
            self.edit.underline(e.pos, e.end)
            self.chips.setText(f"<span style='color:{ERR}'>{e}</span>")
            self.formula.clear()
            self.reduce_box.setVisible(False)
            return
        i = self.info
        pm = render_tex(i.tex)
        if pm is not None:
            self.formula.setPixmap(pm)
        else:
            self.formula.setText(text)
        chips = [f"<b>{i.kind}</b>", UN.describe(i.unit), "complex" if i.cx else "real"]
        chips += L.conventions_text(i)
        if i.normals:
            chips.append("direction: " + ", ".join(dict.fromkeys(g.label() for g in i.normals)))
        if i.deps:
            chips.append("uses " + ", ".join(dict.fromkeys(i.deps)))
        if i.surf and i.field:
            chips.append("faces only")
        self.chips.setText("  ·  ".join(chips))
        self.reduce_box.setVisible(i.field)
        self._fill_reduce_geometries()
        if self.live.isChecked():
            self._preview()

    def _signature(self) -> None:
        text, pos = self.edit.toPlainText(), self.edit.textCursor().position()
        c = call_at(text, pos)
        env = CB.make_env(self.doc)
        if not c:
            self.sig.setText("")
            return
        fn, k = c
        if fn in L.SIGNATURES:
            params, doc = L.SIGNATURES[fn]
        elif fn in env.functions:
            params, doc = env.functions[fn].params, env.functions[fn].body
        else:
            self.sig.setText("")
            return
        ps = [f"<b>{p}</b>" if j == k else p for j, p in enumerate(params)]
        html = f"{fn}({', '.join(ps)}) — {doc}"
        if k == 0 and fn in L.GEOM_FN:
            dims = L.GEOM_FN[fn]
            names = [n for n, g in env.geometry.items() if set(g.dims) & dims and (fn != "circ" or g.closed)]
            if names:
                html += "<br>" + " ".join(f"<a href='{n}'>{n}</a>" for n in names[:24])
        self.sig.setText(html)

    def _picked(self) -> None:
        """Pick on: the selection in the 3D view (or the model tree) goes into the expression."""
        if not self.pick.isChecked() or not self.isVisible():
            return
        sel = self.doc.selection
        if not sel:
            return
        names = []
        for s in sel:
            if self.doc.selection_mode == "face" and s.index >= 0:
                names.append(f"face({s.obj}, {s.index})")
            elif s.obj not in names:
                names.append(s.obj)
        self.edit.insert(names[0] if len(names) == 1 else "[" + ", ".join(names) + "]")

    # ---- evaluation ------------------------------------------------------------------------------
    def evaluate(self) -> None:
        self._analyze()
        if self.info is None:
            return
        if self.info.field:
            self.result.setText("Field")
            self.result_note.setText("A field has a value at every point: plot it (Plot Field…, or Show in 3D), "
                                     "or reduce it to a value below.")
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
        notes = [f"{sol}, {self.freq.currentText()}, port {port}" + (f", phase {phase:g}°" if phase else "")]
        notes += L.conventions_text(r.info) + r.notes
        self.result_note.setText("; ".join(notes))

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
        self.result_note.setText(f"{expr}: " + self.result_note.text())

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

    def hideEvent(self, e):
        self.live.setChecked(False)
        self.pick.setChecked(False)
        super().hideEvent(e)

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
            self.result_note.setText(f"<span style='color:{ERR}'>Give the expression a name to save it</span>")
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
            self.result_note.setText(f"<span style='color:{ERR}'>{what} needs a name: give one and Save</span>")
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
                                                           "to a value first (Maximum, Integral, ... below).")
            return
        name = self._ensure_saved("A report trace")
        if name:
            sol, _, port, _ = self.context()
            self.win.field_report(name, sol, port)

    def open_for_plot(self) -> None:
        self.for_plot = True
        self.banner.setText("<b>New quantity for a field plot.</b> Type or pick the expression, give it a name and "
                            "Save: the field plot dialog opens with it.")
        self.banner.setVisible(True)
        self.new()
        self.edit.setFocus()

    # ---- dialogs ---------------------------------------------------------------------------------
    def integral_dialog(self) -> None:
        dlg = IntegralDialog(self)
        dlg.show()
        self._integral = dlg

    def component_dialog(self) -> None:
        dlg = ComponentDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self.edit.insert(dlg.text())

    def functions_dialog(self, select: str | None = None) -> None:
        dlg = FunctionsDialog(self, select)
        dlg.exec()
        self.refresh()


class IntegralDialog(QDialog):
    """Builds an integral: over a line, face or volume, of a field; the geometry from the list or the 3D
    selection. Not modal, so objects and faces can be picked in the 3D view meanwhile."""

    def __init__(self, panel: CalculatorPanel):
        super().__init__(panel)
        self.panel = panel
        self.setWindowTitle("Insert Integral")
        self.setAttribute(Qt.WA_DeleteOnClose)
        f = QFormLayout(self)
        self.kind = QButtonGroup(self)
        kr = QHBoxLayout()
        for i, t in enumerate(("Line", "Face", "Volume")):
            rb = QRadioButton(t)
            self.kind.addButton(rb, i)
            kr.addWidget(rb)
        self.kind.button(1).setChecked(True)
        self.kind.idToggled.connect(lambda *_: self._fill())
        f.addRow("Integral over", kr)
        gr = QHBoxLayout()
        self.geo = QComboBox()
        self.geo.setEditable(True)
        self.geo.setMinimumWidth(180)
        pick = QPushButton("Use 3D selection")
        pick.setToolTip("Select objects or faces in the 3D view (F: faces), then click here")
        pick.clicked.connect(self._use_selection)
        gr.addWidget(self.geo, 1)
        gr.addWidget(pick)
        f.addRow("Geometry", gr)
        self.flip = QCheckBox("Reverse the normal (faces) or the direction (lines)")
        f.addRow("", self.flip)
        cur = panel.edit.toPlainText().strip()
        self.f = QLineEdit(cur if panel.info is not None and panel.info.field else "re(S)")
        f.addRow("Integrand", self.f)
        self.mode = QComboBox()
        f.addRow("Vector as", self.mode)
        self.preview = QLabel()
        self.preview.setStyleSheet(f"color:{MUTED}")
        self.preview.setWordWrap(True)
        f.addRow("", self.preview)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Insert")
        bb.accepted.connect(self._insert)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        for w in (self.geo, self.mode):
            w.currentTextChanged.connect(self._update)
        self.geo.editTextChanged.connect(self._update)
        self.f.textChanged.connect(self._update)
        self.flip.toggled.connect(self._update)
        self._fill()

    def _dim(self) -> int:
        return [1, 2, 3][self.kind.checkedId()]

    def _fill(self) -> None:
        env = CB.make_env(self.panel.doc)
        d = self._dim()
        names = [n for n, g in env.geometry.items() if d in g.dims]
        if d == 2:
            names += ["cut_xy(0)", "cut_yz(0)", "cut_xz(0)"]
        cur = self.geo.currentText()
        self.geo.clear()
        self.geo.addItems(names)
        if cur in names:
            self.geo.setCurrentText(cur)
        self.mode.clear()
        self.mode.addItems({1: ["Tangential F·dl", "Magnitude |F|"], 2: ["Flux F·n", "Per component", "Magnitude |F|"],
                            3: ["Per component", "Magnitude |F|"]}[d])
        self.flip.setEnabled(d < 3)
        self._update()

    def _use_selection(self) -> None:
        doc = self.panel.doc
        names = []
        for s in doc.selection:
            n = f"face({s.obj}, {s.index})" if doc.selection_mode == "face" and s.index >= 0 else s.obj
            if n not in names:
                names.append(n)
        if not names:
            QMessageBox.information(self, "Insert Integral", "Select objects or faces in the 3D view first.")
            return
        self.geo.setCurrentText(names[0] if len(names) == 1 else "[" + ", ".join(names) + "]")

    def text(self) -> str:
        geo = self.geo.currentText().strip()
        if self.flip.isChecked() and self._dim() < 3:
            geo = f"[{', '.join(f'flip({g.strip()})' for g in geo[1:-1].split(','))}]" if geo.startswith("[") \
                else f"flip({geo})"
        f = self.f.text().strip() or "E"
        env = CB.make_env(self.panel.doc)
        try:
            vec = L.analyze(f, env).vec
        except L.CalcError:
            vec = False
        d, m = self._dim(), self.mode.currentText()
        if vec and m.startswith("Magnitude"):
            f = f"|{f}|"
        if d == 1:
            return f"int_line({geo}, {f})"
        if d == 2:
            return f"flux({geo}, {f})" if vec and m.startswith("Flux") else f"int_surf({geo}, {f})"
        return f"int_vol({geo}, {f})"

    def _update(self, *_):
        t = self.text()
        try:
            i = L.analyze(t, CB.make_env(self.panel.doc))
            self.preview.setText(f"{t}   →   {i.kind}, {UN.describe(i.unit)}")
        except L.CalcError as e:
            self.preview.setText(f"{t}   →   <span style='color:{ERR}'>{e}</span>")

    def _insert(self) -> None:
        self.panel.edit.insert(self.text())
        self.accept()


class ComponentDialog(QDialog):
    """A component of a vector field in a coordinate system of the model."""

    COMPS = {"Cartesian": [("x", "x"), ("y", "y"), ("z", "z")],
             "Cylindrical": [("ρ (rho)", "rho"), ("φ (phi)", "phi"), ("z", "z")],
             "Spherical": [("r", "r"), ("θ (theta)", "theta"), ("φ (phi)", "phi")]}

    def __init__(self, panel: CalculatorPanel):
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

    def __init__(self, panel: CalculatorPanel, select: str | None = None):
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
