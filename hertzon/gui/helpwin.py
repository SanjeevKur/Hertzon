"""F1 help: every property with its description, grouped like the dialogs (opens at the current context)."""
from __future__ import annotations

import html

from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

from ..model import schema
from ..model.help import help_for


def _anchor(title: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in title)


def sections() -> list[tuple[str, list]]:
    out = [("Object Attributes", schema.OBJECT_ATTRIBUTES), ("Design Settings", schema.DESIGN_SETTINGS),
           ("Solution Setup", schema.SETUP), ("Eigenmode Setup", schema.EIGEN_SETUP),
           ("Frequency Sweep", schema.SWEEP), ("Infinite Sphere", schema.INFINITE_SPHERE),
           ("Optimization", schema.OPTIMIZATION)]
    for fam, d in (("Boundary", schema.BOUNDARIES), ("Excitation", schema.EXCITATIONS),
                   ("Mesh Operation", schema.MESH_OPS), ("Geometry", schema.GEOMETRY)):
        for k, props in d.items():
            out.append((f"{fam}: {k}", props))
    return out


def build_html() -> str:
    parts = ["<style>h2{color:#1d4f91;margin-top:18px} td{padding:3px 8px;vertical-align:top} "
             "td.n{font-weight:bold;white-space:nowrap}</style><h1>Hertzon property reference</h1>",
             "<p>Hover over a property name for the same text as a tooltip. F1 opens this page at the item "
             "shown in the Properties window.</p><p>"]
    secs = sections()
    parts.append(" | ".join(f"<a href='#{_anchor(t)}'>{html.escape(t)}</a>" for t, _ in secs) + "</p>")
    for title, props in secs:
        group = title.split(": ", 1)[-1]
        parts.append(f"<a name='{_anchor(title)}'></a><h2>{html.escape(title)}</h2><table>")
        for p in props:
            if p.name.startswith("_"):
                continue
            parts.append(f"<tr><td class='n'>{html.escape(p.title)}</td><td>"
                         f"{html.escape(help_for(p, group) or '-')}</td></tr>")
        parts.append("</table>")
    return "".join(parts)


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hertzon Help")
        self.setModal(False)
        lay = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(True)
        self.browser.setHtml(build_html())
        lay.addWidget(self.browser)
        self.resize(820, 640)

    def goto(self, title: str) -> None:
        if title:
            self.browser.scrollToAnchor(_anchor(title))


def context_title(win) -> str:
    """Section matching what the Properties window shows."""
    t = getattr(win.props, "target", None)
    if not t:
        return ""
    kind = t[0]
    des = win.des
    try:
        if kind == "object":
            return "Object Attributes"
        if kind in ("boundary", "excitation", "meshop"):
            a = des.assignment(t[1])
            return {"boundary": "Boundary", "excitation": "Excitation", "meshop": "Mesh Operation"}[kind] + f": {a.kind}"
        if kind == "setup":
            return "Eigenmode Setup" if des.setup(t[1]).eigen else "Solution Setup"
        if kind == "sweep":
            return "Frequency Sweep"
        if kind == "design":
            return "Design Settings"
    except Exception:  # noqa: BLE001
        return ""
    return ""
