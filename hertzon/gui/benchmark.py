"""Tools > Solver Benchmark: how the solve time scales with the cores on this computer, and which HPC
settings to use (Tools > Options > HPC and Analysis)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QStandardPaths
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QLineEdit, QCheckBox,
                               QPushButton, QLabel, QProgressBar, QTableWidget, QTableWidgetItem, QSplitter,
                               QAbstractItemView, QMessageBox, QFileDialog)

from ..solver.benchmark import CASES, CASES_DIR, default_cores, machine
from ..solver.results import _safe
from .solve import solver_python

TAG = "@@HZN "
COLUMNS = ["Run", "Cores", "Sweep processes", "Total (s)", "Initial mesh (s)", "Adaptive passes (s)", "Sweep (s)",
           "Passes", "Tetrahedra", "Peak memory (MB)", "Speedup", "Efficiency"]


def benchmark_root() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation) or str(Path.home() / ".hertzon")
    return Path(base) / "benchmark"


def recommendation(result: dict) -> tuple[str, int | None, int | None]:
    """(text, cores, sweep processes) from the runs of a benchmark."""
    runs = [r for r in result.get("runs", []) if r.get("kind") == "cores"]
    if not runs:
        return "", None, None
    ncpu = result.get("machine", {}).get("logical_cores") or os.cpu_count() or 1
    runs.sort(key=lambda r: r["cores"])
    best = min(runs, key=lambda r: r["total"])
    knee = next(r for r in runs if r["total"] <= 1.1 * best["total"])
    one = next((r for r in runs if r["cores"] == 1), None)
    lines = [f"Fastest with <b>{best['cores']} cores</b>: {best['total']:.0f} s"
             + (f", {one['total'] / best['total']:.1f}x faster than 1 core." if one and best is not one else ".")]
    # the fewest cores that are as fast within the measuring accuracy are the ones to use
    same = next(r for r in runs if r["total"] <= 1.02 * best["total"])
    if same is not best:
        lines.append(f"<b>{same['cores']} cores</b> are just as fast ({same['total']:.0f} s); more cores bring nothing.")
    if knee is not same:
        slower = 100 * (knee['total'] / best['total'] - 1)
        lines.append(f"With {knee['cores']} cores it is only {slower:.0f} % slower. More cores hardly help for a "
                     f"problem of this size ({best['tets']} tetrahedra); when you solve several designs at once "
                     f"(parallel variations or evaluations), give each about {knee['cores']} cores.")
    if best["cores"] == max(r["cores"] for r in runs) and best["cores"] < ncpu:
        lines.append(f"Not tested above {best['cores']} cores; more may still be faster.")
    workers = None
    sw = [r for r in result.get("runs", []) if r.get("kind") == "sweep"]
    if sw:
        base = next((r for r in sw if r["workers"] == 1), sw[0])
        fast = min(sw, key=lambda r: r["sweep"])
        if fast["workers"] > 1 and fast["sweep"] < 0.9 * base["sweep"]:
            workers = fast["workers"]
            lines.append(f"Parallel sweep: <b>{workers} processes</b> make the sweep "
                         f"{base['sweep'] / fast['sweep']:.1f}x faster (switch on 'Parallel' in the sweep).")
        else:
            workers = 1
            lines.append("Parallel sweep processes do not make the sweep faster here: keep the sweep serial "
                         "and give it all cores.")
    return "<br>".join(lines), same["cores"], workers


class BenchmarkDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Solver Benchmark")
        self.proc: QProcess | None = None
        self.out: Path | None = None
        self.result: dict = {}
        lay = QVBoxLayout(self)
        f = QFormLayout()
        self.case = QComboBox()
        for title, file, design, setup in CASES:
            self.case.addItem(title, (str(CASES_DIR / file), design, setup, Path(file).stem))
        doc = win.doc
        if doc.project.path:
            for s in doc.design.setups:
                if not s.eigen:
                    self.case.addItem(f"This project: {doc.design_name} : {s.name}",
                                      (doc.project.path, doc.design_name, s.name,
                                       _safe(f"{Path(doc.project.path).stem}_{doc.design_name}_{s.name}")))
        f.addRow("Benchmark:", self.case)
        ncpu = os.cpu_count() or 1
        self.cores = QLineEdit(", ".join(map(str, default_cores(ncpu))))
        f.addRow("Cores to test:", self.cores)
        wr = QHBoxLayout()
        self.do_sweep = QCheckBox("Also test parallel sweep processes:")
        self.workers = QLineEdit(", ".join(str(w) for w in (1, 2, 4, 8) if w <= max(1, ncpu // 2)))
        wr.addWidget(self.do_sweep)
        wr.addWidget(self.workers, 1)
        f.addRow("", wr)
        m = machine()
        f.addRow("Computer:", QLabel(f"{m['cpu']}, {m['logical_cores']} logical cores"
                                     + (f" ({m['physical_cores']} physical)" if m.get("physical_cores") else "")
                                     + (f", {m['memory_gb']} GB memory" if m.get("memory_gb") else "")))
        lay.addLayout(f)
        hint = QLabel("Every core count solves the whole setup (adaptive passes to convergence, at most 50 in the "
                      "built-in cases, and the sweep) in a fresh solver process, starting with 1 core. "
                      "Do not run other analyses or heavy programs meanwhile; they distort the times.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        br = QHBoxLayout()
        self.b_start = QPushButton("Start")
        self.b_abort = QPushButton("Abort")
        self.b_abort.setEnabled(False)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.status = QLabel("")
        br.addWidget(self.b_start)
        br.addWidget(self.b_abort)
        br.addWidget(self.bar, 1)
        lay.addLayout(br)
        lay.addWidget(self.status)
        split = QSplitter(Qt.Vertical)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(7, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        split.addWidget(self.canvas)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        split.addWidget(self.table)
        lay.addWidget(split, 1)
        self.advice = QLabel("")
        self.advice.setWordWrap(True)
        self.advice.setTextFormat(Qt.RichText)
        lay.addWidget(self.advice)
        bl = QHBoxLayout()
        self.b_apply = QPushButton("Apply Recommended Settings")
        self.b_apply.setEnabled(False)
        self.b_export = QPushButton("Export CSV...")
        close = QPushButton("Close")
        bl.addWidget(self.b_apply)
        bl.addWidget(self.b_export)
        bl.addStretch(1)
        bl.addWidget(close)
        lay.addLayout(bl)
        self.b_start.clicked.connect(self.start)
        self.b_abort.clicked.connect(self.abort)
        self.b_apply.clicked.connect(self.apply)
        self.b_export.clicked.connect(self.export)
        close.clicked.connect(self.close)
        self.case.currentIndexChanged.connect(lambda *_: self.load())
        self.resize(980, 760)
        self.load()

    # ---- results ------------------------------------------------------------------------------
    def _folder(self) -> Path:
        return benchmark_root() / self.case.currentData()[3]

    def load(self) -> None:
        """Show the last result of the chosen case."""
        try:
            self.result = json.loads((self._folder() / "benchmark.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.result = {}
        self.show_result()

    def show_result(self) -> None:
        runs = self.result.get("runs", [])
        cr = sorted([r for r in runs if r.get("kind") == "cores"], key=lambda r: r["cores"])
        ref = cr[0] if cr else None
        rows = []
        # 1 core first, then the sweep runs from 1 process up, also for results measured largest first
        order = cr + sorted([r for r in runs if r.get("kind") == "sweep"], key=lambda r: r["workers"])
        for i, r in enumerate(order, start=1):
            if r.get("kind") == "cores":
                sp = ref["total"] * ref["cores"] / r["total"] if ref and r["total"] else 0.0
                rows.append([str(i), str(r["cores"]), "1", f"{r['total']:.1f}", f"{r['mesh']:.1f}",
                             f"{r['adaptive']:.1f}", f"{r['sweep']:.1f}", str(r["passes"]), str(r["tets"]),
                             f"{r['memory_mb']:.0f}", f"{sp:.2f}", f"{100 * sp / r['cores']:.0f} %"])
            else:
                rows.append([str(i), str(r["cores"]), str(r["workers"]), "", "", "", f"{r['sweep']:.1f}",
                             "", "", "", "", ""])
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for c, v in enumerate(row):
                self.table.setItem(i, c, QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        self.fig.clear()
        if cr:
            ax1, ax2 = self.fig.add_subplot(121), self.fig.add_subplot(122)
            x = [r["cores"] for r in cr]
            for key, lab in (("total", "Total"), ("adaptive", "Adaptive passes"), ("sweep", "Sweep"),
                             ("mesh", "Initial mesh")):
                ax1.plot(x, [r[key] for r in cr], "o-", ms=4, label=lab)
            ax1.set_xscale("log", base=2)
            ax1.set_xlabel("Cores")
            ax1.set_ylabel("Time (s)")
            ax1.set_title("Solve time", fontsize=10)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            sp = [ref["total"] * ref["cores"] / r["total"] for r in cr]
            ax2.plot(x, sp, "o-", ms=4, label="Measured (total)")
            ax2.plot(x, [ref["adaptive"] * ref["cores"] / r["adaptive"] if r["adaptive"] else 0 for r in cr],
                     "s--", ms=3, label="Adaptive passes")
            ax2.plot(x, x, ":", color="#888", label="Ideal")
            ax2.set_xscale("log", base=2)
            ax2.set_yscale("log", base=2)
            ax2.set_xlabel("Cores")
            ax2.set_ylabel("Speedup")
            ax2.set_title("Speedup", fontsize=10)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            from matplotlib.ticker import ScalarFormatter
            ax2.yaxis.set_major_formatter(ScalarFormatter())
            for ax in (ax1, ax2):
                ax.set_xticks(x)
                ax.set_xticklabels([str(v) for v in x])
        self.canvas.draw_idle()
        text, cores, workers = recommendation(self.result)
        when = ""
        if self.result.get("finished"):
            import time
            when = f"Measured {time.strftime('%Y-%m-%d %H:%M', time.localtime(self.result['finished']))}. "
        self.advice.setText((when + text) if text else ("No result yet for this benchmark." if not runs else ""))
        self.rec = (cores, workers)
        self.b_apply.setEnabled(cores is not None and self.proc is None and bool(self.result.get("finished")))

    # ---- running ------------------------------------------------------------------------------
    def start(self) -> None:
        if self.proc is not None:
            return
        if getattr(self.win.solver, "busy", False):
            QMessageBox.information(self, "Solver Benchmark", "Wait until the running analysis has finished; "
                                                              "it would distort the times.")
            return
        try:
            cores = [int(x) for x in self.cores.text().replace(";", ",").split(",") if x.strip()]
            workers = [int(x) for x in self.workers.text().replace(";", ",").split(",") if x.strip()] \
                if self.do_sweep.isChecked() else []
            if not cores or min(cores + (workers or [1])) < 1:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Solver Benchmark", "Enter the core counts as whole numbers, e.g. 1, 2, 4, 8.")
            return
        project, design, setup, _ = self.case.currentData()
        self.out = self._folder()
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "ABORT").unlink(missing_ok=True)
        args = ["-m", "hertzon.solver.benchmark", project, design, setup, str(self.out),
                "--cores", ",".join(map(str, cores))]
        if workers:
            args += ["--workers", ",".join(map(str, workers))]
        p = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        root = str(Path(__file__).resolve().parents[2])
        env.insert("PYTHONPATH", root + os.pathsep + env.value("PYTHONPATH", ""))
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        p.setProcessEnvironment(env)
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._finished)
        self._buf = b""
        self.proc = p
        self.result = {"runs": []}
        self.show_result()
        self.b_start.setEnabled(False)
        self.b_abort.setEnabled(True)
        self.case.setEnabled(False)
        self.bar.setValue(0)
        self.status.setText("Starting")
        self.win.msgs.add("info", f"Solver benchmark started: {self.case.currentText()}")
        p.start(solver_python(), args)

    def _read(self) -> None:
        self._buf += bytes(self.proc.readAllStandardOutput())
        *lines, self._buf = self._buf.split(b"\n")
        for raw in lines:
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith(TAG):
                continue
            try:
                msg = json.loads(line[len(TAG):])
            except ValueError:
                continue
            t = msg.get("type")
            if t == "progress":
                self.bar.setValue(int(msg.get("value", 0)))
                self.status.setText(msg.get("text", ""))
            elif t == "log":
                self.win.msgs.add(msg.get("level", "info"), f"Benchmark: {msg.get('msg', '')}")
                self.load_running()
            elif t == "failed":
                self.status.setText(f"<span style='color:#c0392b'>Failed: {msg.get('error')}</span>")

    def load_running(self) -> None:
        try:
            self.result = json.loads((self.out / "benchmark.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        self.show_result()

    def _finished(self, code, _status) -> None:
        self._read()
        self.proc = None
        self.b_start.setEnabled(True)
        self.b_abort.setEnabled(False)
        self.case.setEnabled(True)
        self.load()
        if code == 0:
            self.bar.setValue(100)
            self.status.setText("Done")
            self.win.msgs.add("info", "Solver benchmark finished")
        elif not self.status.text().startswith("<span"):
            self.status.setText("Aborted" if (self.out / "ABORT").exists() else f"Ended with code {code}")

    def abort(self) -> None:
        if self.proc is None or self.out is None:
            return
        (self.out / "ABORT").write_text("1", encoding="utf-8")
        self.status.setText("Aborting")
        proc = self.proc
        from PySide6.QtCore import QTimer
        QTimer.singleShot(15000, lambda: proc.kill() if self.proc is proc else None)

    def closeEvent(self, ev) -> None:
        if self.proc is not None:
            if QMessageBox.question(self, "Solver Benchmark", "Abort the running benchmark?") != QMessageBox.Yes:
                ev.ignore()
                return
            self.abort()
        super().closeEvent(ev)

    # ---- using the result -----------------------------------------------------------------------
    def apply(self) -> None:
        cores, workers = self.rec
        if cores is None:
            return
        s = self.win.settings
        ncpu = os.cpu_count() or 1
        s.setValue("hpc/cores", 0 if cores >= ncpu else cores)
        if workers:
            s.setValue("hpc/sweep_workers", workers)
        self.win.apply_hpc_settings()
        self.win.msgs.add("info", f"HPC settings: {'all' if cores >= ncpu else cores} cores"
                          + (f", {workers} parallel sweep process(es)" if workers else ""))

    def export(self) -> None:
        if not self.table.rowCount():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Benchmark", "benchmark.csv", "CSV (*.csv)")
        if not path:
            return
        lines = [";".join(COLUMNS)]
        for r in range(self.table.rowCount()):
            lines.append(";".join(self.table.item(r, c).text() if self.table.item(r, c) else ""
                                  for c in range(self.table.columnCount())))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
