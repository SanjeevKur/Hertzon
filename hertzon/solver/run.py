"""Solver process entry point: python -m hertzon.solver.run PROJECT DESIGN SETUP OUTDIR"""
import sys


def main() -> int:
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
