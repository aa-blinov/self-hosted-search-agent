import unittest

from search_agent.infrastructure.source_handlers import (
    SHALLOW_SOURCE_HANDLERS,
    dispatch_shallow_fetch,
    is_reddit_post_url,
)


class SourceHandlersTests(unittest.TestCase):
    def test_reddit_url_detection(self) -> None:
        u = "https://www.reddit.com/r/Python/comments/abc123/title_here/"
        self.assertTrue(is_reddit_post_url(u))

    def test_dispatch_shallow_unknown_yields_none(self) -> None:
        """No specialized handler → caller uses generic HTTP in extractor."""
        out = dispatch_shallow_fetch(
            "https://example.com/article",
            max_chars=4000,
            timeout=8,
            log=lambda _m: None,
        )
        self.assertIsNone(out)

    def test_registry_has_expected_ids(self) -> None:
        ids = [h.id for h in SHALLOW_SOURCE_HANDLERS]
        self.assertEqual(
            ids,
            ["reddit", "arxiv", "crossref", "semantic_scholar", "github"],
        )


if __name__ == "__main__":
    unittest.main()
