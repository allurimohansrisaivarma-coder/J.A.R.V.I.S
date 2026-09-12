"""Keep double-clicking the packaged app from opening competing voice sessions."""

import ctypes
from ctypes import wintypes


class SingleInstance:
    def __init__(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.CreateMutexW.restype = wintypes.HANDLE
        self._kernel = kernel
        self.handle = kernel.CreateMutexW(None, False, "Local\\JARVIS-Desktop-Assistant")
        self.already_running = ctypes.get_last_error() == 183

    def focus_existing(self):
        user = ctypes.windll.user32
        user.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        user.FindWindowW.restype = wintypes.HWND
        user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user.SetForegroundWindow.argtypes = [wintypes.HWND]
        hwnd = user.FindWindowW(None, "J.A.R.V.I.S.")
        if hwnd:
            user.ShowWindow(hwnd, 9)
            user.SetForegroundWindow(hwnd)

    def close(self):
        if self.handle:
            self._kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self._kernel.CloseHandle(self.handle)
            self.handle = None
