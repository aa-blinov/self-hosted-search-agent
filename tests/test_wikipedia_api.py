import unittest

from search_agent.infrastructure.wikipedia_api import parse_wikipedia_article_url


class WikipediaApiTests(unittest.TestCase):
    def test_parse_en_article(self) -> None:
        u = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        p = parse_wikipedia_article_url(u)
        self.assertEqual(p, ("en", "Python (programming language)"))

    def test_parse_ru_percent_encoded(self) -> None:
        u = "https://ru.wikipedia.org/wiki/%D0%9F%D0%B8%D1%82%D0%BE%D0%BD"
        self.assertEqual(parse_wikipedia_article_url(u), ("ru", "Питон"))

    def test_parse_ru_article(self) -> None:
        u = "https://ru.wikipedia.org/wiki/Python"
        self.assertEqual(parse_wikipedia_article_url(u), ("ru", "Python"))

    def test_parse_mobile_host(self) -> None:
        u = "https://en.m.wikipedia.org/wiki/Hello_World"
        self.assertEqual(parse_wikipedia_article_url(u), ("en", "Hello World"))

    def test_parse_zh_min_nan(self) -> None:
        u = "https://zh-min-nan.wikipedia.org/wiki/Chit-chāi"
        self.assertEqual(parse_wikipedia_article_url(u), ("zh-min-nan", "Chit-chāi"))

    def test_rejects_non_wiki(self) -> None:
        self.assertIsNone(parse_wikipedia_article_url("https://en.wiktionary.org/wiki/foo"))
        self.assertIsNone(parse_wikipedia_article_url("https://example.com/wiki/Foo"))


if __name__ == "__main__":
    unittest.main()
