import unittest
from unittest.mock import patch

from search_agent.config.profiles import get_profile
from search_agent.infrastructure.ddgs_gateway import DDGSSearchGateway
from search_agent.infrastructure.gateway_factory import build_search_gateway
from search_agent.infrastructure.search_gateway import SearxngSearchGateway
from search_agent.settings import AppSettings


class SearchGatewayLayerTests(unittest.TestCase):
    def test_gateway_factory_selects_ddgs(self):
        gateway = build_search_gateway(AppSettings(search_provider="ddgs"))
        self.assertIsInstance(gateway, DDGSSearchGateway)

    def test_gateway_factory_defaults_to_searxng(self):
        gateway = build_search_gateway(AppSettings(search_provider="searxng"))
        self.assertIsInstance(gateway, SearxngSearchGateway)

    def test_ddgs_gateway_converts_results_to_snapshot(self):
        gateway = DDGSSearchGateway(
            AppSettings(
                search_provider="ddgs",
                ddgs_region="wt-wt",
                ddgs_safesearch="moderate",
                ddgs_timeout=10,
            )
        )

        with patch("search_agent.infrastructure.ddgs_gateway.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = [
                {
                    "title": "Satya Nadella - Microsoft",
                    "href": "https://news.microsoft.com/source/exec/satya-nadella/?utm_source=test",
                    "body": "Satya Nadella is Chairman and Chief Executive Officer of Microsoft.",
                }
            ]
            snapshots = gateway.search_variant("Who is the CEO of Microsoft?", get_profile("web"))

        self.assertEqual(len(snapshots), 1)
        snapshot = snapshots[0]
        self.assertEqual(snapshot.profile_name, "ddgs:web")
        self.assertEqual(len(snapshot.results), 1)
        self.assertEqual(snapshot.results[0].canonical_url, "https://news.microsoft.com/source/exec/satya-nadella")
        self.assertEqual(snapshot.results[0].host, "news.microsoft.com")


if __name__ == "__main__":
    unittest.main()
