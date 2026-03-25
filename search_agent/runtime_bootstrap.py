"""
Windows: make stdout/stderr UTF-8 so Rich, Cyrillic, and typographic quotes render correctly.

Classic PowerShell/cmd often use a legacy code page; without UTF-8, UTF-8 bytes are shown as
``â€¦`` / ``Ð…`` (mojibake). This runs before :class:`rich.console.Console` is constructed.
"""

from __future__ import annotations

import sys


def ensure_utf8_stdio() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
