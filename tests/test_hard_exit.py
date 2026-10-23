"""Solver process exit on Windows: TerminateProcess gets a real 64 bit process handle."""
import ctypes
from ctypes import wintypes

from hertzon.solver import bridge as B


class _Fn:
    def __init__(self, impl):
        self.impl, self.argtypes, self.restype = impl, None, ctypes.c_int

    def __call__(self, *args):
        return self.impl(*args)


def test_terminate_self_typed():
    calls = []
    k32 = type("K32", (), {})()
    k32.GetCurrentProcess = _Fn(lambda: 2 ** 64 - 1)     # pseudo handle -1 as HANDLE (c_void_p)
    k32.TerminateProcess = _Fn(lambda h, code: calls.append((h, code)) or 1)
    assert B.terminate_self(k32)
    assert calls == [(2 ** 64 - 1, 0)]
    assert k32.GetCurrentProcess.restype is wintypes.HANDLE and k32.GetCurrentProcess.argtypes == []
    assert k32.TerminateProcess.argtypes == [wintypes.HANDLE, wintypes.UINT]
    assert k32.TerminateProcess.restype is wintypes.BOOL


def test_hard_exit_only_in_solver_process(monkeypatch):
    exits = []
    monkeypatch.setattr(B.os, "_exit", exits.append)
    monkeypatch.setattr(B, "HARD_EXIT", False)
    B.hard_exit()
    assert exits == []
    monkeypatch.setattr(B, "HARD_EXIT", True)
    B.hard_exit()
    assert exits == [0]
