import unittest
from datetime import datetime

from search_agent.application.text_heuristics import (
    extract_entities,
    extract_region_hint,
    extract_time_scope,
    is_news_digest_query,
    normalize_relative_time_references,
    should_decompose,
    tokenize,
)


class TextHeuristicsTests(unittest.TestCase):
    def test_tokenize_preserves_versions_and_entities(self):
        tokens = tokenize("OpenAI released GPT-4.1 in Python 3.13.")

        self.assertIn("openai", tokens)
        self.assertIn("gpt-4.1", tokens)
        self.assertIn("3.13", tokens)

    def test_extract_entities_uses_lexical_sequences(self):
        entities = extract_entities('What did OpenAI announce about "GPT-4.1" in March 2026?')

        self.assertIn("OpenAI", entities)
        self.assertIn("GPT-4.1", entities)

    def test_extract_time_scope_handles_iso_and_named_dates(self):
        self.assertEqual(extract_time_scope("events on 2026-03-25 in Astana"), "2026-03-25")
        self.assertEqual(extract_time_scope("released on 7 October 2024"), "7 October 2024")

    def test_extract_region_hint_requires_entity_like_region(self):
        self.assertEqual(extract_region_hint("what happened in Astana today"), "Astana")
        self.assertEqual(extract_region_hint("что произошло в Астане сегодня"), "Астане")
        self.assertIsNone(extract_region_hint("latest developments in artificial intelligence"))

    def test_normalize_relative_time_references_replaces_word_tokens(self):
        normalized = normalize_relative_time_references(
            "What happened today in Astana?",
            now=datetime(2026, 3, 25),
        )

        self.assertEqual(normalized, "What happened 2026-03-25 in Astana?")

    def test_news_digest_and_decompose_use_phrase_rules(self):
        query = "What are the latest developments in artificial intelligence this week?"

        self.assertTrue(is_news_digest_query(query, freshness=True))
        self.assertTrue(should_decompose("Compare Python 3.11 and 3.12 features"))


if __name__ == "__main__":
    unittest.main()
