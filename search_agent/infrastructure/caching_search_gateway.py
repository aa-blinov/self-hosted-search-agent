from __future__ import annotations

import threading
from concurrent.futures import Future

import logfire

from search_agent import tuning
from search_agent.application.contracts import SearchGatewayPort
from search_agent.domain.models import SearchSnapshot
from search_agent.infrastructure.serp_query import build_routed_query


class CachingBudgetSearchGateway:
    """
    Wraps a SearchGateway: caches SERP by (provider, profile, routed query) and
    enforces a per-run cap on backend search calls (cache hits are free).
    Thread-safe for parallel claim workers.
    """

    def __init__(self, inner: SearchGatewayPort, *, provider_label: str) -> None:
        self._inner = inner
        self._provider_label = provider_label
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str, str], list[SearchSnapshot]] = {}
        self._inflight: dict[tuple[str, str, str], Future[list[SearchSnapshot]]] = {}
        self._search_calls = 0

    @property
    def search_calls(self) -> int:
        return self._search_calls

    def search_variant(self, query: str, profile, log=None) -> list[SearchSnapshot]:
        log = log or (lambda _msg: None)
        routed = build_routed_query(query, profile).strip().casefold()
        pname = getattr(profile, "name", "") or ""
        key = (self._provider_label, pname, routed)

        # Fast path: cache hit — return without network I/O.
        wait_for: Future[list[SearchSnapshot]] | None = None
        owner = False
        with self._lock:
            if key in self._cache:
                logfire.info(
                    "search_gateway.cache_hit",
                    provider=self._provider_label,
                    profile=pname,
                    routed_prefix=routed[:120],
                )
                return list(self._cache[key])
            if key in self._inflight:
                wait_for = self._inflight[key]
            else:
                cap = tuning.AGENT_MAX_SEARCH_CALLS_PER_RUN
                if cap > 0 and self._search_calls >= cap:
                    logfire.warning(
                        "search_gateway.budget_exhausted",
                        cap=cap,
                        provider=self._provider_label,
                    )
                    log("  [yellow]Search budget exhausted; skipping backend search.[/yellow]")
                    return []

                # Reserve a slot before releasing the lock for the network call.
                self._search_calls += 1
                wait_for = Future()
                self._inflight[key] = wait_for
                owner = True

        if not owner and wait_for is not None:
            return list(wait_for.result())

        # Network I/O without holding the lock so parallel variants run concurrently.
        try:
            out = self._inner.search_variant(query, profile, log=log)
            stored = list(out)
        except Exception as exc:
            with self._lock:
                pending = self._inflight.pop(key, None)
                if pending is not None and not pending.done():
                    pending.set_exception(exc)
            raise

        with self._lock:
            self._cache[key] = stored
            pending = self._inflight.pop(key, None)
            if pending is not None and not pending.done():
                pending.set_result(list(stored))

        return list(stored)
