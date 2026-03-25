"""Shared URL normalization for SERP results."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

_TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "ref",
    "ref_src",
    "ref_url",
    "source",
}


def canonicalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in _TRACKING_QUERY_KEYS
        ]
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/") or "/"
        return urlunparse(
            (
                parsed.scheme.lower() or "https",
                netloc,
                path,
                "",
                urlencode(query),
                "",
            )
        )
    except Exception:
        return url
