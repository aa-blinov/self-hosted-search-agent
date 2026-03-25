"""Tests for CachingBudgetSearchGateway (SERP dedup + search budget)."""

import unittest
import unittest.mock

from search_agent.config.profiles import get_profile
from search_agent.domain.models import SearchSnapshot, SerpResult
from search_agent.infrastructure.caching_search_gateway import CachingBudgetSearchGateway


class _CountingInner:
    def __init__(self) -> None:
        self.calls = 0

    def search_variant(self, query: str, profile, log=None):
        self.calls += 1
        return [
            SearchSnapshot(
                query=query,
                suggestions=[],
                retrieved_at="2026-01-01T00:00:00+00:00",
                results=[
                    SerpResult(
                        result_id="r1",
                        query_variant_id="v1",
                        title="t",
                        url="https://example.com/",
                        snippet="s",
                        canonical_url="https://example.com/",
                        host="example.com",
                        position=1,
                    )
                ],
                profile_name=getattr(profile, "name", None),
            )
        ]


class CachingSearchGatewayTests(unittest.TestCase):
    def test_cache_reuses_backend_second_call(self) -> None:
        inner = _CountingInner()
        gw = CachingBudgetSearchGateway(inner, provider_label="ddgs")
        profile = get_profile("web")
        gw.search_variant("hello world", profile)
        gw.search_variant("hello world", profile)
        self.assertEqual(inner.calls, 1)
        self.assertEqual(gw.search_calls, 1)

    def test_budget_blocks_after_cap(self) -> None:
        inner = _CountingInner()
        gw = CachingBudgetSearchGateway(inner, provider_label="ddgs")
        profile = get_profile("web")
        with unittest.mock.patch(
            "search_agent.infrastructure.caching_search_gateway.tuning.AGENT_MAX_SEARCH_CALLS_PER_RUN",
            1,
        ):
            out_a = gw.search_variant("query a", profile)
            out_b = gw.search_variant("query b", profile)
        self.assertTrue(out_a)
        self.assertEqual(out_b, [])
        self.assertEqual(inner.calls, 1)
