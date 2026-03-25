import unittest
from unittest.mock import patch

from search_agent.config.profiles import get_profile
from search_agent.domain.models import SearchSnapshot, SerpResult
from search_agent.infrastructure.brave_search import search_brave_with_fallback
from search_agent.settings import AppSettings


class SearchFallbackTests(unittest.TestCase):
    def test_search_fallback_recovers_after_backend_degradation(self):
        calls: list[tuple[str, str]] = []

        def fake_search(query, profile, settings, log=None):
            calls.append((query, profile.name))
            if len(calls) == 1:
                return SearchSnapshot(
                    query=query,
                    suggestions=[],
                    results=[],
                    retrieved_at="2026-03-24T00:00:00+00:00",
                    profile_name=profile.name,
                    unresponsive_engines=["duckduckgo: timeout"],
                )
            return SearchSnapshot(
                query=query,
                suggestions=[],
                results=[
                    SerpResult(
                        result_id="serp:1",
                        query_variant_id="legacy",
                        title="Satya Nadella - Source",
                        url="https://news.microsoft.com/source/exec/satya-nadella/",
                        snippet="Satya Nadella is Chairman and Chief Executive Officer of Microsoft.",
                        canonical_url="https://news.microsoft.com/source/exec/satya-nadella/",
                        host="news.microsoft.com",
                        position=1,
                    )
                ],
                retrieved_at="2026-03-24T00:00:01+00:00",
                profile_name=profile.name,
                unresponsive_engines=[],
            )

        with patch("search_agent.infrastructure.brave_search.search_brave", side_effect=fake_search), patch(
            "search_agent.infrastructure.brave_search.time.sleep",
            return_value=None,
        ):
            snapshots = search_brave_with_fallback(
                "Who is the CEO of Microsoft?",
                get_profile("web"),
                AppSettings(brave_api_key="test"),
            )

        self.assertGreaterEqual(len(snapshots), 2)
        self.assertEqual(calls[0][1], "web")
        self.assertEqual(calls[1][1], "web")
        self.assertEqual(calls[1][0], "Who is the CEO of Microsoft")
        self.assertEqual(snapshots[-1].profile_name, "web")
        self.assertEqual(len(snapshots[-1].results), 1)


if __name__ == "__main__":
    unittest.main()
