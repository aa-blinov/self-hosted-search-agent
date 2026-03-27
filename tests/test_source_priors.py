import unittest

from search_agent.domain.source_priors import lookup_source_prior


class SourcePriorsTests(unittest.TestCase):
    def test_lookup_source_prior_matches_host_segments_without_regex(self) -> None:
        prior = lookup_source_prior("press.example.com")

        self.assertEqual(prior.domain_type_override, "official")
        self.assertIn("official_subdomain", prior.labels)
        self.assertGreater(prior.primary_boost, 0.0)

    def test_lookup_source_prior_matches_forum_segments_without_regex(self) -> None:
        prior = lookup_source_prior("community.example.com")

        self.assertEqual(prior.domain_type_override, "forum")
        self.assertIn("community_forum", prior.labels)
        self.assertLess(prior.primary_boost, 0.0)


if __name__ == "__main__":
    unittest.main()
