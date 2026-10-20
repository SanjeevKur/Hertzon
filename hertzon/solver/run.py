"""Solver process entry point: python -m hertzon.solver.run PROJECT DESIGN SETUP OUTDIR"""
import sys


def full_speed() -> None:
    """Windows 11 may run a process without a window in efficiency mode (EcoQoS: low clock, efficiency
    cores), which makes the solver many times slower; opt out of that power throttling."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class State(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        # PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, off
        st = State(1, 1, 0)
        k32 = ctypes.windll.kernel32
        k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))  # ProcessPowerThrottling
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    full_speed()
    # Windows consoles/pipes default to cp1252; EMerge logs Unicode symbols
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    if len(sys.argv) != 5:
        print("usage: python -m hertzon.solver.run PROJECT DESIGN SETUP OUTDIR", file=sys.stderr)
        return 2
    from . import bridge
    bridge.HARD_EXIT = True
    return bridge.run(*sys.argv[1:5])


if __name__ == "__main__":
    sys.exit(main())
