"""field plot animation: the frames over a swept variable (phase, frequency, excitation) are
computed once up front, then played back with a speed setting, pause/stop and a frame slider, and can be
saved as a GIF that looks and runs like the playback on screen."""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QSlider, QSpinBox,
                               QWidget, QApplication, QStackedWidget)

from . import fields as F
from .gif_export import MAX_GIF_FPS, FpsMeter, write_gif

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
    """Player for precomputed frames: play, pause, stop, frame slider and speed (frames per second). The
    frames per second actually shown are measured during playback (drawing can be slower than the setting)
    and used for the GIF export."""

    MEASURE_TIMEOUT = 20.0                       # s, longest playback to measure the rate for an export

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
        self.actual = QLabel("")
        self.actual.setMinimumWidth(110)
        self.actual.setToolTip("Frames per second actually shown, measured during playback; the GIF export "
                               "uses this rate")
        self.b_gif = QPushButton("Export GIF...")
        self.b_gif.setToolTip("Save the animation as a GIF with the size of the view and the measured speed")
        for w in (self.title, self.b_play, self.b_pause, self.b_stop, self.slider, self.value,
                  QLabel("Speed"), self.speed, self.actual, self.b_gif, self.b_close):
            lay.addWidget(w)
        self.meter = FpsMeter()
        self._shown = 0.0                            # when the measured rate was last written to the label
        self.b_play.clicked.connect(self.play)
        self.b_pause.clicked.connect(self.pause)
        self.b_stop.clicked.connect(self.stop)
        self.b_close.clicked.connect(self.close_player)
        self.b_gif.clicked.connect(lambda: self.export_gif())
        self.slider.valueChanged.connect(self.show_frame)
        self.speed.valueChanged.connect(self._speed_changed)
        self.hide()

    def _speed_changed(self, v: int) -> None:
        self.meter.reset()                           # the old measurement belongs to the old setting
        self._show_rate()
        if self.timer.isActive():
            self.timer.start(int(1000 / v))

    def _show_rate(self) -> None:
        f = self.meter.fps()
        self.actual.setText(f"Actual {f:.1f} frames/s" if f else "Actual -")

    def load(self, names: list[str], frames: list[tuple[str, dict]]) -> None:
        self.stop()
        self.names, self.frames = list(names), frames
        self.meter.reset()
        self.meter.set_window(len(frames))           # average over one loop: frames can take different times
        self._show_rate()
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
        if self.timer.isActive():                    # only playback counts, not moving the slider by hand
            t = time.perf_counter()
            self.meter.tick(t)
            if t - self._shown > 0.5:
                self._shown = t
                self._show_rate()

    def _next(self) -> None:
        self.slider.setValue((self.slider.value() + 1) % max(1, len(self.frames)))

    def play(self) -> None:
        if self.frames:
            self.timer.start(int(1000 / self.speed.value()))

    def pause(self) -> None:
        self.timer.stop()
        self.meter.pause()
        self._show_rate()

    def stop(self) -> None:
        self.timer.stop()
        self.meter.pause()
        self._show_rate()
        if self.frames:
            self.slider.setValue(0)

    def is_playing(self) -> bool:
        return self.timer.isActive()

    def close_player(self) -> None:
        self.timer.stop()
        self.meter.reset()
        self.frames, self.names = [], []
        self.hide()

    def _needed(self) -> int:
        """Frame intervals measured before the rate counts: one loop, or at least 2 s of playback."""
        return max(2, min(len(self.frames), 2 * self.speed.value()))

    def measure(self) -> float | None:
        """The measured frames per second; plays the animation briefly when there is no measurement yet."""
        if self.meter.count() < self._needed():
            playing = self.timer.isActive()
            self.actual.setText("Measuring ...")
            if not playing:
                self.play()
            loop, t0 = QEventLoop(), time.perf_counter()
            check = QTimer(self)
            check.timeout.connect(lambda: (self.meter.count() >= self._needed() or not self.timer.isActive()
                                           or time.perf_counter() - t0 > self.MEASURE_TIMEOUT) and loop.quit())
            check.start(20)
            loop.exec()
            check.stop()
            if not playing:
                self.pause()
        self._show_rate()
        return self.meter.fps()

    def export_gif(self, path: str | None = None) -> str | None:
        """Save all frames as a looping GIF: each frame exactly as the view shows it (same pixels, same
        size) and played at the frames per second measured in the viewer."""
        if not self.frames:
            return None
        if path is None:
            name = (self.names[0] if self.names else "animation").replace(" ", "_")
            path, _ = QFileDialog.getSaveFileName(self, "Export GIF", f"{name}.gif", "GIF (*.gif)")
            if not path:
                return None
            if not path.lower().endswith(".gif"):
                path += ".gif"
        fps = self.measure()
        if not fps:
            QMessageBox.warning(self, "Export GIF", "The playback speed could not be measured.")
            return None
        playing, cur = self.timer.isActive(), self.slider.value()
        self.timer.stop()
        self.meter.pause()
        prog = QProgressDialog("Capturing frames ...", "Cancel", 0, len(self.frames), self)
        prog.setWindowTitle("Export GIF")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(500)
        images = []
        try:
            for i in range(len(self.frames)):
                if prog.wasCanceled():
                    return None
                prog.setValue(i)
                self.slider.blockSignals(True)
                self.slider.setValue(i)
                self.slider.blockSignals(False)
                self.show_frame(i)
                images.append(self.win.view.plotter.screenshot(transparent_background=False, return_img=True))
            prog.setValue(len(self.frames))
            delays = write_gif(path, images, fps)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Export GIF", f"The GIF could not be saved: {e}")
            return None
        finally:
            prog.close()
            self.slider.setValue(cur)
            self.show_frame(cur)
            if playing:
                self.play()
        h, w = images[0].shape[:2]
        note = (f" (a GIF shows at most {MAX_GIF_FPS:.0f} frames/s, saved at {100 * len(delays) / sum(delays):.1f})"
                if fps > MAX_GIF_FPS else "")
        msg = (f"Saved {path}: {len(images)} frames, {w} x {h} pixels, {fps:.2f} frames/s measured "
               f"(setting {self.speed.value()}){note}")
        msgs = getattr(self.win, "msgs", None)
        if msgs is not None:
            msgs.add("info", msg)
        return path


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
