"""Regression: vc.ru-style hosts use browser-only shallow path."""

import unittest

from search_agent.infrastructure.extractor import _url_shallow_browser_first


class ExtractorBrowserHostTests(unittest.TestCase):
    def test_vc_ru_matches(self) -> None:
        self.assertTrue(_url_shallow_browser_first("https://vc.ru/dev/123"))
        self.assertTrue(_url_shallow_browser_first("https://www.vc.ru/dev/123"))
        self.assertTrue(_url_shallow_browser_first("https://m.vc.ru/dev/123"))

    def test_other_hosts_not_browser_first(self) -> None:
        self.assertFalse(_url_shallow_browser_first("https://example.com/"))
        self.assertFalse(_url_shallow_browser_first("https://docs.python.org/3/"))


if __name__ == "__main__":
    unittest.main()
