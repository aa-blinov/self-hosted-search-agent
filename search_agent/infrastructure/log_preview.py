"""Shared one-line text previews for Rich logs (no markup inside snippet)."""

from __future__ import annotations

import re


def preview_snippet(text: str, max_len: int = 72) -> str:
    """Collapse whitespace, truncate; Rich markup must be escaped by the caller."""
    one = re.sub(r"\s+", " ", (text or "").strip())
    if not one:
        return "(empty)"
    if len(one) > max_len:
        return one[: max_len - 1] + "…"
    return one
