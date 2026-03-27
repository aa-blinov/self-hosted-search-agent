"""
Windows: make stdout/stderr UTF-8 so Rich, Cyrillic, and typographic quotes render correctly.

Classic PowerShell/cmd often use a legacy code page; without UTF-8, UTF-8 bytes are shown as
``â€¦`` / ``Ð…`` (mojibake). This runs before :class:`rich.console.Console` is constructed.
"""

from __future__ import annotations

import sys
import warnings

# dateutil.parser does not recognise US timezone abbreviations (PDT, EDT, CDT, MST, …)
# and emits UnknownTimezoneWarning every time htmldate/trafilatura parse article dates.
# The parsed date values are still correct (just naive instead of tz-aware), and we never
# use the timezone component ourselves.  Suppress to keep output clean.
warnings.filterwarnings(
    "ignore",
    message="tzname .* identified but not understood",
    category=RuntimeWarning,
    module="dateutil",
)


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
