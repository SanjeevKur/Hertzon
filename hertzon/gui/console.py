"""Python console and script recording (Tools > Run Script / Record Script / command window)."""
from __future__ import annotations

import contextlib
import io
import json
import traceback

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QPlainTextEdit, QLineEdit, QLabel


class ConsoleDock(QDockWidget):
    """Interactive Python with the project in scope: app, doc, project, des, schema, Command, commit()."""

    def __init__(self, win):
        super().__init__("Python Console", win)
        self.setObjectName("dock_console")
        self.win = win
        self.ns = win.script_namespace()
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        f = QFont("Consolas")
        f.setStyleHint(QFont.Monospace)
        self.out.setFont(f)
        self.inp = QLineEdit()
        self.inp.setFont(f)
        self.inp.setPlaceholderText("Python, e.g.  des.add_object('CreateBox', XSize='2'); commit()")
        self.inp.returnPressed.connect(self._run)
        self.inp.installEventFilter(self)
        lay.addWidget(QLabel("Names: app, doc, project, des, schema, Command, Assignment, FaceRef, np, commit()"))
        lay.addWidget(self.out, 1)
        lay.addWidget(self.inp)
        self.setWidget(w)
        self.history: list[str] = []
        self.hpos = 0
        self._buffer: list[str] = []

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if obj is self.inp and ev.type() == QEvent.KeyPress and ev.key() in (Qt.Key_Up, Qt.Key_Down):
            if self.history:
                self.hpos = max(0, min(len(self.history), self.hpos + (-1 if ev.key() == Qt.Key_Up else 1)))
                self.inp.setText(self.history[self.hpos] if self.hpos < len(self.history) else "")
            return True
        return super().eventFilter(obj, ev)

    def write(self, text: str) -> None:
        self.out.moveCursor(QTextCursor.End)
        self.out.insertPlainText(text)
        self.out.moveCursor(QTextCursor.End)

    def execute(self, line: str) -> str:
        """Run one line (multi-line blocks: end with an empty line); returns the printed output."""
        import codeop
        self._buffer.append(line)
        src = "\n".join(self._buffer)
        try:
            code = codeop.compile_command(src, "<console>", "single")
        except SyntaxError:
            self._buffer = []
            return traceback.format_exc(limit=0)
        if code is None and line.strip():
            return ""                    # incomplete block: wait for more lines
        self._buffer = []
        if code is None:
            code = compile(src, "<console>", "exec")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(code, self.ns)  # noqa: S102
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        return buf.getvalue()

    def _run(self) -> None:
        line = self.inp.text()
        self.inp.clear()
        if line.strip():
            self.history.append(line)
        self.hpos = len(self.history)
        prompt = "... " if self._buffer else ">>> "
        self.write(prompt + line + "\n")
        self.write(self.execute(line))


class ScriptRecorder:
    """Writes model changes as Python while recording (objects, operations, assignments, variables)."""

    def __init__(self, doc):
        self.doc = doc
        self.lines = ["# Hertzon recorded script: run with Tools > Run Script",
                      "# names: des (design), project, Command, Assignment, FaceRef, commit()", ""]
        self.prev = self._snap()
        doc.changed.connect(self._changed)

    def _snap(self) -> dict:
        from dataclasses import asdict
        d = asdict(self.doc.design)
        return {"objects": {o["name"]: o for o in d["objects"]},
                "assign": {a["name"]: (k, a) for k in ("boundaries", "excitations", "mesh_ops") for a in d[k]},
                "variables": dict(d.get("variables", {})), "pvars": dict(self.doc.project.variables)}

    def _changed(self, *_):
        cur = self._snap()
        p = self.prev
        L = self.lines
        for k, v in cur["pvars"].items():
            if p["pvars"].get(k) != v:
                L.append(f"project.variables[{k!r}] = {v!r}")
        for k, v in cur["variables"].items():
            if p["variables"].get(k) != v:
                L.append(f"des.variables[{k!r}] = {v!r}")
        for name, o in cur["objects"].items():
            old = p["objects"].get(name)
            hist = o["history"]
            if old is None:
                c0 = hist[0]
                L.append(f"o = des.add_object({c0['kind']!r}, name={name!r}, material={o['material']!r}, "
                         f"**{json.dumps(c0['props'])})")
                for c in hist[1:]:
                    L.append(f"o.history.append(Command({c['kind']!r}, {json.dumps(c['props'])}))")
                for attr in ("model", "solve_inside", "color", "transparency"):
                    L.append(f"o.{attr} = {o[attr]!r}")
                continue
            if old["history"][0] != hist[0]:
                L.append(f"des.obj({name!r}).history[0].props.update({json.dumps(hist[0]['props'])})")
            n = len(old["history"])
            if len(hist) > n and hist[:n] == old["history"]:
                for c in hist[n:]:
                    L.append(f"des.obj({name!r}).history.append(Command({c['kind']!r}, {json.dumps(c['props'])}))")
            for attr in ("material", "model", "solve_inside", "consumed_by"):
                if old.get(attr) != o.get(attr):
                    L.append(f"des.obj({name!r}).{attr} = {o.get(attr)!r}")
        for name in p["objects"]:
            if name not in cur["objects"]:
                L.append(f"des.objects = [x for x in des.objects if x.name != {name!r}]")
        for name, (lst, a) in cur["assign"].items():
            if p["assign"].get(name, (None, None))[1] != a:
                faces = ", ".join(f"FaceRef({f['obj']!r}, {f['face']})" for f in a["faces"])
                L.append(f"des.{lst} = [x for x in des.{lst} if x.name != {name!r}] + "
                         f"[Assignment({name!r}, {a['kind']!r}, {json.dumps(a['props'])}, faces=[{faces}], "
                         f"objects={a['objects']!r})]")
        for name, (lst, _) in p["assign"].items():
            if name not in cur["assign"]:
                L.append(f"des.{lst} = [x for x in des.{lst} if x.name != {name!r}]")
        if L and L[-1] != "commit()" and cur != p:
            L.append("commit()")
        self.prev = cur

    def stop(self) -> None:
        try:
            self.doc.changed.disconnect(self._changed)
        except (RuntimeError, TypeError):
            pass

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"
