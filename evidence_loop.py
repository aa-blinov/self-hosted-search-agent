"""
Backward-compatible wrapper around the new claim-level agent runtime.

The old contract returned `(sources, search_log)`. This shim preserves that API
for any legacy callers while routing execution through `run_search_agent()`.
"""

from __future__ import annotations

from agent import run_search_agent


def evidence_loop(
    query: str,
    search_fn=None,
    profile=None,
    client=None,
    log=None,
    **_: dict,
) -> tuple[list[dict], list[str]]:
    log = log or (lambda msg: None)
    if profile is None:
        raise ValueError("profile is required")

    report = run_search_agent(query, profile=profile, client=client, log=log)
    search_log: list[str] = []
    sources: list[dict] = []
    seen_urls: set[str] = set()

    for run in report.claims:
        search_log.extend([variant.query_text for variant in run.query_variants])
        bundle = run.evidence_bundle
        passages = []
        if bundle is not None:
            passages.extend(bundle.supporting_passages)
            passages.extend(bundle.contradicting_passages)
        if not passages:
            passages = run.passages[:2]
        for passage in passages:
            if passage.url in seen_urls:
                continue
            seen_urls.add(passage.url)
            sources.append({
                "title": passage.title,
                "url": passage.url,
                "snippet": passage.text,
            })

    return sources, search_log
