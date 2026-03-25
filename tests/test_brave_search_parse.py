"""Brave response parsing (no HTTP)."""

from __future__ import annotations

import unittest

from search_agent.config.profiles import get_profile
from search_agent.infrastructure.brave_search import _parse_news_results, _parse_web_results


class BraveNewsParseTests(unittest.TestCase):
    def test_news_top_level_results_key(self) -> None:
        """Brave /res/v1/news/search returns results[] at document root."""
        data = {
            "type": "news",
            "results": [
                {
                    "title": "Hello",
                    "url": "https://example.com/n",
                    "description": "Snippet",
                    "page_age": "2026-03-23T15:30:46",
                }
            ],
        }
        results, unresponsive = _parse_news_results(data, get_profile("news"), 10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/n")
        self.assertEqual(results[0].published_at, "2026-03-23T15:30:46")
        self.assertFalse(unresponsive)

    def test_news_nested_news_results_fallback(self) -> None:
        data = {
            "news": {
                "results": [{"title": "Legacy", "url": "https://legacy.test/", "description": ""}],
            }
        }
        results, _ = _parse_news_results(data, get_profile("news"), 10)
        self.assertEqual(len(results), 1)
        self.assertIn("legacy.test", results[0].url)


class BraveWebParseTests(unittest.TestCase):
    def test_web_under_web_key(self) -> None:
        data = {"web": {"results": [{"title": "W", "url": "https://w.example/", "description": "d"}]}}
        results, _ = _parse_web_results(data, get_profile("web"), 10)
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
