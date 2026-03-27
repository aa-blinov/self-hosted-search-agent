import unittest

from search_agent.infrastructure.extractor import _legacy_shallow_payload


class _Response:
    url = "https://example.com/story"


class ExtractorHtmlTests(unittest.TestCase):
    def test_legacy_shallow_payload_collects_structured_html_signals(self) -> None:
        html = """
        <html>
          <head>
            <title>Example Story</title>
            <meta name="description" content="Example description for readers." />
            <meta property="article:published_time" content="2026-03-27" />
            <script type="application/ld+json">
              {
                "@type": "NewsArticle",
                "headline": "Example Story",
                "datePublished": "2026-03-27",
                "author": {"name": "Jane Doe"}
              }
            </script>
          </head>
          <body>
            <h1>Main Heading</h1>
            <h2>Context Heading</h2>
            <p>This is the first sufficiently long paragraph for the legacy extractor path to keep.</p>
            <p>This is the second sufficiently long paragraph, and it should also be preserved.</p>
          </body>
        </html>
        """

        payload = _legacy_shallow_payload(html, _Response(), max_chars=4000)

        self.assertEqual(payload["title"], "Example Story")
        self.assertEqual(payload["meta_description"], "Example description for readers.")
        self.assertEqual(payload["author"], "Jane Doe")
        self.assertEqual(payload["published_at"], "2026-03-27")
        self.assertIn("Main Heading", payload["headings"])
        self.assertIn("Context Heading", payload["headings"])
        self.assertEqual(len(payload["first_paragraphs"]), 2)
        self.assertIn("Example description for readers.", payload["content"])


if __name__ == "__main__":
    unittest.main()
