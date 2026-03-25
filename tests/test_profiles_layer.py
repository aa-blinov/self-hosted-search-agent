"""Unit tests for search profiles (no network; no Brave API calls)."""

from __future__ import annotations

import unittest

from search_agent.config.profiles import PROFILES, SearchProfile, get_profile
from search_agent.infrastructure.brave_search import _merge_goggles, _news_mode
from search_agent.infrastructure.ddgs_gateway import _ddgs_region, _ddgs_safesearch, _ddgs_timelimit
from search_agent.infrastructure.serp_query import build_routed_query
from search_agent.settings import AppSettings


_NEWS_PROFILE_NAMES = frozenset({"news", "news_fresh", "ru_news", "ru_news_fresh"})


class ProfilesLayerTests(unittest.TestCase):
    def test_every_profile_loads_and_name_matches_key(self) -> None:
        for key, profile in PROFILES.items():
            self.assertIs(get_profile(key), profile)
            self.assertEqual(profile.name, key)
            self.assertTrue(profile.description.strip())
            self.assertTrue(profile.categories)

    def test_unknown_profile_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_profile("not_a_profile")
        self.assertIn("not_a_profile", str(ctx.exception))

    def test_news_mode_matches_news_vertical_profiles(self) -> None:
        for name in PROFILES:
            p = get_profile(name)
            want = name in _NEWS_PROFILE_NAMES
            self.assertEqual(_news_mode(p), want, msg=f"profile={name}")

    def test_build_routed_query_plain_web_unchanged(self) -> None:
        q = "test query"
        self.assertEqual(build_routed_query(q, get_profile("web")), q)

    def test_build_routed_query_site_filters(self) -> None:
        q = "hello"
        self.assertIn("site:reddit.com", build_routed_query(q, get_profile("reddit")))
        self.assertIn("site:arxiv.org", build_routed_query(q, get_profile("arxiv")))
        wiki = build_routed_query(q, get_profile("wikipedia"))
        self.assertIn("site:wikipedia.org", wiki)
        self.assertIn("site:wikidata.org", wiki)
        self.assertIn("(", wiki)
        self.assertIn(" OR ", wiki)
        ref = build_routed_query(q, get_profile("reference"))
        self.assertIn("site:wikipedia.org", ref)
        self.assertIn("site:wikidata.org", ref)

    def test_build_routed_query_science_bangs(self) -> None:
        out = build_routed_query("quantum", get_profile("science"))
        self.assertIn("arxiv", out.casefold())
        self.assertIn("quantum", out)

    def test_merge_goggles_env_then_profile_deduped(self) -> None:
        s = AppSettings(brave_api_key="x", brave_goggles="https://example.com/extra.goggle")
        p = get_profile("tech")
        merged = _merge_goggles(p, s)
        self.assertIn("https://example.com/extra.goggle", merged)
        self.assertTrue(any("goggles-quickstart" in g and "tech_blogs" in g for g in merged))

    def test_ddgs_timelimit_maps_time_range(self) -> None:
        p = SearchProfile(
            name="t",
            description="x",
            categories=["general"],
            language="auto",
            time_range="week",
            fetch_top_n=1,
            max_results=5,
        )
        self.assertEqual(_ddgs_timelimit(p), "w")
        p2 = SearchProfile(
            name="t2",
            description="x",
            categories=["general"],
            language="auto",
            time_range=None,
            fetch_top_n=1,
            max_results=5,
        )
        self.assertIsNone(_ddgs_timelimit(p2))

    def test_ddgs_timelimit_profile_override(self) -> None:
        p = SearchProfile(
            name="t",
            description="x",
            categories=["general"],
            language="auto",
            time_range="week",
            fetch_top_n=1,
            max_results=5,
            ddgs_timelimit="m",
        )
        self.assertEqual(_ddgs_timelimit(p), "m")

    def test_ddgs_region_fallback_to_settings(self) -> None:
        s = AppSettings(ddgs_region="wt-wt")
        self.assertEqual(_ddgs_region(get_profile("web"), s), "wt-wt")
        self.assertEqual(_ddgs_region(get_profile("ru"), s), "ru-ru")

    def test_ddgs_safesearch_fallback_to_settings(self) -> None:
        s = AppSettings(ddgs_safesearch="off")
        self.assertEqual(_ddgs_safesearch(get_profile("web"), s), "off")

    def test_profiles_count_stable(self) -> None:
        self.assertEqual(len(PROFILES), 15)


class CliProfileHeuristicTests(unittest.TestCase):
    def test_suggest_ru_news_fresh_for_today_cyrillic(self) -> None:
        from search_agent.cli import _suggest_profile

        self.assertEqual(_suggest_profile("что сегодня было в Астане"), "ru_news_fresh")

    def test_suggest_news_fresh_english_today(self) -> None:
        from search_agent.cli import _suggest_profile

        self.assertEqual(_suggest_profile("What happened in London today"), "news_fresh")


if __name__ == "__main__":
    unittest.main()
