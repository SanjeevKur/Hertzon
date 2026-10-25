"""The typeset formula editor of the Fields Calculator (the Formula view).

A MathLive math field (MIT licence, bundled in gui/formula, no network) in a QWebEngineView. Python sets the
formula as LaTeX and gets every change back as LaTeX; post.calc.latex turns it into the expression text.
Without Qt WebEngine the calculator keeps its text line (the Text view) only: FormulaEditor.available()."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QVBoxLayout, QWidget

HTML = Path(__file__).with_name("formula") / "editor.html"


def available() -> bool:
    try:
        from PySide6.QtWebChannel import QWebChannel  # noqa: F401
        from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return HTML.exists()


class _Bridge(QObject):
    """The object the page sees as `py`."""

    def __init__(self, ed: "FormulaEditor"):
        super().__init__(ed)
        self.ed = ed

    @Slot()
    def ready(self) -> None:
        self.ed._ready()

    @Slot(str)
    def changed(self, tex: str) -> None:
        self.ed._changed(tex)

    @Slot()
    def submit(self) -> None:
        self.ed.submitted.emit()

    @Slot(str, int)
    def pick(self, fn: str, dim: int) -> None:
        self.ed.pick_requested.emit(fn, dim)


class FormulaEditor(QWidget):
    """latex_changed(tex) on every edit, submitted() on Enter, pick_requested(fn, dim) when an integral or a
    maximum was typed (\\iint + Space) and its geometry has to be picked."""

    latex_changed = Signal(str)
    submitted = Signal()
    pick_requested = Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWebChannel import QWebChannel
        from PySide6.QtWebEngineCore import QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView
        self.tex = ""
        self.is_ready = False
        self._queue: list[str] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = QWebEngineView(self)
        self.view.setContextMenuPolicy(Qt.NoContextMenu)
        s = self.view.settings()
        s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, False)
        s.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        self.bridge = _Bridge(self)
        self.channel = QWebChannel(self.view.page())
        self.channel.registerObject("py", self.bridge)
        self.view.page().setWebChannel(self.channel)
        self.view.load(QUrl.fromLocalFile(str(HTML)))
        self.view.loadFinished.connect(self._keep_keys)
        lay.addWidget(self.view)
        self.setFocusProxy(self.view)
        self._keep_keys()

    # ---- keys ------------------------------------------------------------------------------------
    EDIT_KEYS = (QKeySequence.Copy, QKeySequence.Cut, QKeySequence.Paste, QKeySequence.Undo, QKeySequence.Redo,
                 QKeySequence.SelectAll, QKeySequence.Delete)

    def _keep_keys(self, *_):
        """Typed letters and the editing keys go to the formula, not to shortcuts of the main window (E selects
        edges there, Ctrl+Z undoes a model change)."""
        for w in (self.view, self.view.focusProxy()):
            if w is not None and not w.property("hzKeys"):
                w.setProperty("hzKeys", True)
                w.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.ShortcutOverride:
            mods = ev.modifiers() & ~(Qt.ShiftModifier | Qt.KeypadModifier)
            if mods in (Qt.NoModifier, Qt.ControlModifier | Qt.AltModifier) or \
                    any(ev.matches(k) for k in self.EDIT_KEYS):
                ev.accept()
                return True
        return super().eventFilter(obj, ev)

    # ---- page -> Python --------------------------------------------------------------------------
    def _ready(self) -> None:
        self.is_ready = True
        for js in self._queue:
            self.view.page().runJavaScript(js)
        self._queue.clear()

    def _changed(self, tex: str) -> None:
        if tex != self.tex:
            self.tex = tex
            self.latex_changed.emit(tex)

    # ---- Python -> page --------------------------------------------------------------------------
    def _js(self, fn: str, *args) -> None:
        js = f"hz.{fn}(" + ", ".join(json.dumps(a) for a in args) + ")"
        if self.is_ready:
            self.view.page().runJavaScript(js)
        else:
            self._queue.append(js)

    def set_latex(self, tex: str) -> None:
        """Show this formula (no latex_changed for it)."""
        self.tex = tex
        self._js("setLatex", tex)

    def insert(self, template: str) -> None:
        """Insert at the caret; #@ stands for the selection (or an empty box), #? for an empty box."""
        self._js("insert", template)

    def wrap(self, template: str, whole: bool) -> None:
        """An operation over a geometry around the selection, or around the whole formula if whole."""
        self._js("wrap", template, bool(whole))

    def fill_geometry(self, tex: str) -> None:
        """The picked geometry into the first empty box below a sign."""
        self._js("fillGeometry", tex)

    def set_shortcuts(self, shortcuts: dict) -> None:
        """Words that turn into a name or function as they are typed (omega -> ω, Port1 -> Port1)."""
        self._js("setShortcuts", shortcuts)

    def focus_formula(self) -> None:
        self.view.setFocus()
        self._js("focus")
