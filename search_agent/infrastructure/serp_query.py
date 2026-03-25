"""
Shared SERP query shaping for Brave and DuckDuckGo-style backends.

Applies bang_prefixes (operator text) and engines (site: filters) from SearchProfile.
"""

from __future__ import annotations

_BANG_HINTS: dict[str, str] = {
    "scientific_publications": "(arxiv OR site:arxiv.org)",
    "it": "(programming OR software OR documentation)",
    "q&a": "(stackoverflow OR stack exchange)",
    "repos": "(github OR gitlab OR pypi OR npm OR crates.io)",
}


def build_routed_query(query: str, profile) -> str:
    bang_parts: list[str] = []
    for raw in getattr(profile, "bang_prefixes", []) or []:
        p = raw.strip()
        if not p.startswith("!"):
            continue
        key = p[1:].strip()
        hint = _BANG_HINTS.get(key)
        if hint:
            bang_parts.append(hint)
        else:
            bang_parts.append(key.replace("_", " "))
    if bang_parts:
        query = f"{' '.join(bang_parts)} {query}".strip()

    engine_prefix: list[str] = []
    for engine in getattr(profile, "engines", None) or []:
        if engine == "wikipedia":
            engine_prefix.append("site:wikipedia.org")
        elif engine == "wikidata":
            engine_prefix.append("site:wikidata.org")
        elif engine == "reddit":
            engine_prefix.append("site:reddit.com")
        elif engine == "arxiv":
            engine_prefix.append("site:arxiv.org")
    if engine_prefix:
        if len(engine_prefix) == 1:
            prefix = engine_prefix[0]
        else:
            prefix = "(" + " OR ".join(engine_prefix) + ")"
        query = f"{prefix} {query}".strip()

    return query
