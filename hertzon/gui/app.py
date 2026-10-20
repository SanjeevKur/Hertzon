"""Application entry point: ``python -m hertzon [project.hzn]``."""
from __future__ import annotations

import sys


def _crash_log():
    """Hard crashes (native code) and uncaught Python errors go to Hertzon_crash.log in the user's home,
    so a window that just disappears still leaves a trace."""
    import faulthandler
    import time
    import traceback
    from pathlib import Path
    path = Path.home() / "Hertzon_crash.log"
    try:
        f = open(path, "a", encoding="utf-8", buffering=1)
    except OSError:
        return
    f.write(f"\n=== Hertzon started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    faulthandler.enable(file=f, all_threads=True)
    old = sys.excepthook

    def hook(t, v, tb):
        f.write(f"--- uncaught {time.strftime('%H:%M:%S')} ---\n")
        traceback.print_exception(t, v, tb, file=f)
        old(t, v, tb)
    sys.excepthook = hook
    return f


def _splash(app):
    """Loading screen while the 3D viewer and the solver interface are imported."""
    from pathlib import Path
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap, QGuiApplication
    from PySide6.QtWidgets import QSplashScreen
    img = Path(__file__).with_name("splash.jpg")
    pm = QPixmap(str(img))
    if pm.isNull():
        return None
    screen = QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen else 1.0
    width = min(960.0, 0.5 * screen.availableGeometry().width()) if screen else 800.0
    pm = pm.scaledToWidth(int(width * dpr), Qt.SmoothTransformation)
    pm.setDevicePixelRatio(dpr)
    sp = QSplashScreen(pm)             # not on top: it must never hide a question or another Hertzon window
    sp.show()
    app.processEvents()
    return sp


def _status(sp, app, text: str) -> None:
    if sp is not None:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor
        sp.showMessage(text + "  ", Qt.AlignRight | Qt.AlignBottom, QColor(200, 215, 240))
        app.processEvents()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    _keep = _crash_log()  # noqa: F841
    from pathlib import Path
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon
    if sys.platform == "win32":   # own taskbar icon instead of the Python one
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Hertzon.Hertzon")
        except Exception:  # noqa: BLE001
            pass
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Hertzon")
    app.setOrganizationName("Hertzon")
    app.setStyle("Fusion")
    from .theme import apply as _theme
    _theme(app)
    ico = Path(__file__).with_name("hertzon.ico")
    if ico.exists():
        app.setWindowIcon(QIcon(str(ico)))
    sp = _splash(app)
    _status(sp, app, "Loading 3D viewer ...")
    import pyvista as pv
    pv.global_theme.allow_empty_mesh = True
    _status(sp, app, "Loading modeler and solver interface ...")
    from .main_window import MainWindow
    win = MainWindow()
    files = [a for a in argv[1:] if a.lower().endswith(".hzn")]
    win.resize(1500, 950)
    win.show()
    if sp is not None:
        sp.finish(win)
    if files:
        # opened in the visible window (own progress dialog): a question while opening (e.g. a newer
        # autosave) appears in front instead of waiting unseen behind the loading screen
        app.processEvents()
        win.file_open(files[0])
    if "--ai-link" in argv:          # started by the MCP server for an AI assistant
        win.ai_link.start(auto_register=False)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
