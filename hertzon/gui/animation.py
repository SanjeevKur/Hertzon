"""field plot animation: the frames over a swept variable (phase, frequency, excitation) are
computed once up front, then played back with a speed setting, pause/stop and a frame slider."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QLabel, QProgressDialog, QPushButton, QSlider, QSpinBox, QWidget, QApplication,
                               QStackedWidget)

from . import fields as F

VARIABLES = ("Phase", "Frequency", "Excitation")


class AnimationSetupDialog(QDialog):
    """Setup Animation - swept variable with start, stop and number of steps."""

    def __init__(self, parent, fp, freqs: list[float], ports: list[int]):
        super().__init__(parent)
        self.setWindowTitle(f"Setup Animation - {fp.name}")
        self.freqs, self.ports = freqs, ports
        lay = QFormLayout(self)
        self.var = QComboBox()
        self.var.addItems([v for v in VARIABLES if v == "Phase" or (v == "Frequency" and len(freqs) > 1)
                           or (v == "Excitation" and len(ports) > 1)])
        lay.addRow("Swept variable", self.var)
        self.stack = QStackedWidget()
        # phase
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(0, 0, 0, 0)
        self.ph0, self.ph1 = QDoubleSpinBox(), QDoubleSpinBox()
        for sb, v in ((self.ph0, 0.0), (self.ph1, 360.0)):
            sb.setRange(-3600, 3600)
            sb.setSuffix(" deg")
            sb.setValue(v)
        self.nph = QSpinBox()
        self.nph.setRange(2, 721)
        self.nph.setValue(19)
        f.addRow("Start", self.ph0)
        f.addRow("Stop", self.ph1)
        f.addRow("Steps", self.nph)
        self.stack.addWidget(w)
        # frequency
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(0, 0, 0, 0)
        self.f0, self.f1 = QComboBox(), QComboBox()
        labels = [f"{x / 1e9:.6g} GHz" for x in freqs]
        for cb in (self.f0, self.f1):
            cb.addItems(labels)
        self.f1.setCurrentIndex(max(0, len(labels) - 1))
        self.fph = QDoubleSpinBox()
        self.fph.setRange(-360, 360)
        self.fph.setSuffix(" deg")
        self.fph.setValue(float(fp.props.get("phase", 0)))
        f.addRow("From", self.f0)
        f.addRow("To", self.f1)
        f.addRow("Phase", self.fph)
        self.stack.addWidget(w)
        # excitation
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(0, 0, 0, 0)
        f.addRow(QLabel("All excitations (ports) one after the other"))
        self.stack.addWidget(w)
        lay.addRow(self.stack)
        self.var.currentTextChanged.connect(lambda t: self.stack.setCurrentIndex(VARIABLES.index(t)))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)

    def frames(self, fp) -> list[tuple[str, dict]]:
        """(label, overrides of freq index / port / phase) per frame; "swept" names what changes, "fval" is
        the frequency (Hz) of a frequency frame."""
        fi, port, ph = int(fp.props.get("freq", 0)), int(fp.props.get("port", 1)), float(fp.props.get("phase", 0))
        var = self.var.currentText()
        if var == "Phase":
            vals = np.linspace(self.ph0.value(), self.ph1.value(), self.nph.value())
            if abs((self.ph1.value() - self.ph0.value()) % 360) < 1e-9 and len(vals) > 2:
                vals = vals[:-1]                      # 0..360: the last frame equals the first
            return [(f"Phase = {v:.4g} deg", {"freq": fi, "port": port, "phase": float(v), "swept": "phase"})
                    for v in vals]
        if var == "Frequency":
            a, b = sorted((self.f0.currentIndex(), self.f1.currentIndex()))
            return [(f"Freq = {self.freqs[i] / 1e9:.6g} GHz", {"freq": i, "port": port, "phase": self.fph.value(),
                                                               "swept": "freq", "fval": self.freqs[i]})
                    for i in range(a, b + 1)]
        return [(f"Excitation = port {p}", {"freq": fi, "port": p, "phase": ph, "swept": "port"})
                for p in self.ports]


class AnimationBar(QWidget):
    """Player for precomputed frames: play, pause, stop, frame slider and speed (frames per second)."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.frames: list[tuple[str, dict]] = []     # (label, {plot name: overlay meshes})
        self.names: list[str] = []                   # the animated field plots
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        self.title = QLabel("")
        self.b_play = QPushButton("Play")
        self.b_pause = QPushButton("Pause")
        self.b_stop = QPushButton("Stop")
        self.b_close = QPushButton("Close")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumWidth(220)
        self.value = QLabel("")
        self.value.setMinimumWidth(170)
        self.speed = QSpinBox()
        self.speed.setRange(1, 60)
        self.speed.setValue(8)
        self.speed.setSuffix(" frames/s")
        for w in (self.title, self.b_play, self.b_pause, self.b_stop, self.slider, self.value,
                  QLabel("Speed"), self.speed, self.b_close):
            lay.addWidget(w)
        self.b_play.clicked.connect(self.play)
        self.b_pause.clicked.connect(self.pause)
        self.b_stop.clicked.connect(self.stop)
        self.b_close.clicked.connect(self.close_player)
        self.slider.valueChanged.connect(self.show_frame)
        self.speed.valueChanged.connect(lambda v: self.timer.isActive() and self.timer.start(int(1000 / v)))
        self.hide()

    def load(self, names: list[str], frames: list[tuple[str, dict]]) -> None:
        self.stop()
        self.names, self.frames = list(names), frames
        self.title.setText(f"<b>{', '.join(names)}</b>")
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(frames) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.show()
        self.show_frame(0)

    def show_frame(self, i: int) -> None:
        if not self.frames:
            return
        label, plots = self.frames[i]
        for name, meshes in plots.items():     # all animated plots together; the others stay as they are
            self.win.view.set_overlay(name, meshes, render=False)
        self.win.view.plotter.render()
        self.value.setText(f"{label}   ({i + 1}/{len(self.frames)})")

    def _next(self) -> None:
        self.slider.setValue((self.slider.value() + 1) % max(1, len(self.frames)))

    def play(self) -> None:
        if self.frames:
            self.timer.start(int(1000 / self.speed.value()))

    def pause(self) -> None:
        self.timer.stop()

    def stop(self) -> None:
        self.timer.stop()
        if self.frames:
            self.slider.setValue(0)

    def is_playing(self) -> bool:
        return self.timer.isActive()

    def close_player(self) -> None:
        self.timer.stop()
        self.frames, self.names = [], []
        self.hide()


def precompute(win, fps, frames: list[tuple[str, dict]]) -> list[tuple[str, dict]] | None:
    """Field overlays of every frame for every plot in fps, each plot with one common colour scale over the
    frames. Each plot keeps its own frequency, excitation and phase except the swept one; a plot
    of another solution follows a frequency sweep at its nearest frequency."""
    prog = QProgressDialog("Computing animation frames ...", "Cancel", 0, len(frames) * len(fps), win)
    prog.setWindowTitle("Animation")
    prog.setWindowModality(Qt.WindowModal)
    prog.setMinimumDuration(300)
    freqs = {}
    for fp in fps:
        try:
            st, sol = F.store_for(win.doc, fp.solution)
            freqs[fp.name] = F.frequencies(F.load_grid(st, sol)[1])
        except Exception:  # noqa: BLE001
            freqs[fp.name] = []
    out = [(label, {}) for label, _ in frames]
    for j, fp in enumerate(fps):
        static = prep = None
        fixed = fp.quantity.startswith("ComplexMag") and frames and frames[0][1].get("swept") == "phase"
        for i, (label, o) in enumerate(frames):
            if prog.wasCanceled():
                return None
            prog.setValue(j * len(frames) + i)
            prog.setLabelText(f"Computing animation frames ... {fp.name}: {label}")
            QApplication.processEvents()
            fi, port, ph = int(fp.props.get("freq", 0)), int(fp.props.get("port", 1)), float(fp.props.get("phase", 0))
            sw = o.get("swept", "phase")
            if sw == "phase":
                ph = o["phase"]
            elif sw == "port":
                port = o["port"]
            else:
                ph, fl = o["phase"], freqs[fp.name]
                if fl:
                    fi = int(np.argmin([abs(f - o.get("fval", f)) for f in fl])) if "fval" in o else o["freq"]
            if fixed and i > 0:
                if fp.name in out[0][1]:
                    out[i][1][fp.name] = out[0][1][fp.name]
                continue
            try:
                if sw == "phase":
                    # only the phase changes: the plot is built once, every frame just turns the field
                    if prep is None:
                        prep = F.prepare_overlay(win.doc, win.view, fp.quantity, fp.solution, fi, port,
                                                 fp.faces, fp.objects, fp.plane, attrs=fp.props)
                    meshes, _ = F.overlay_at(prep, ph, animating=True)
                else:
                    meshes, _ = F.make_overlay(win.doc, win.view, fp.quantity, fp.solution, fi, port, ph,
                                               fp.faces, fp.objects, fp.plane, attrs=fp.props, animating=True)
            except Exception:  # noqa: BLE001
                if len(fps) == 1:
                    raise
                if static is None:               # e.g. the port does not exist in this plot's solution
                    try:
                        static, _ = F.make_overlay(win.doc, win.view, fp.quantity, fp.solution,
                                                   int(fp.props.get("freq", 0)), int(fp.props.get("port", 1)),
                                                   float(fp.props.get("phase", 0)), fp.faces, fp.objects,
                                                   fp.plane, attrs=fp.props, animating=True)
                    except Exception:  # noqa: BLE001
                        static = []
                meshes = static
            if meshes:
                out[i][1][fp.name] = meshes
        lims = [kw["clim"] for _, ps in out for _, kw in ps.get(fp.name, []) if kw.get("clim") is not None]
        if lims and fp.props.get("Scale", "Auto") != "Use Limits":
            lo, hi = min(c[0] for c in lims), max(c[1] for c in lims)
            for _, ps in out:
                for _, kw in ps.get(fp.name, []):
                    if kw.get("clim") is not None:
                        kw["clim"] = (lo, hi)
    prog.setValue(len(frames) * len(fps))
    return out
