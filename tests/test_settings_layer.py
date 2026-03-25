import unittest

from search_agent.settings import AppSettings


class SettingsLayerTests(unittest.TestCase):
    def test_resolved_search_provider_override(self):
        self.assertEqual(
            AppSettings(search_provider="brave", search_provider_override="ddgs").resolved_search_provider(),
            "ddgs",
        )
        self.assertEqual(
            AppSettings(search_provider="ddgs", search_provider_override=None).resolved_search_provider(),
            "ddgs",
        )

    def test_resolved_send_to_logfire_parses_common_values(self):
        self.assertEqual(AppSettings(logfire_send_to_logfire="if-token-present").resolved_send_to_logfire(), "if-token-present")
        self.assertTrue(AppSettings(logfire_send_to_logfire="true").resolved_send_to_logfire())
        self.assertFalse(AppSettings(logfire_send_to_logfire="false").resolved_send_to_logfire())

    def test_resolved_brave_goggles_parses_json_and_semicolon(self):
        self.assertEqual(AppSettings(brave_goggles=None).resolved_brave_goggles(), [])
        self.assertEqual(
            AppSettings(brave_goggles='["https://a.example/x.goggle","https://b.example/y.goggle"]').resolved_brave_goggles(),
            ["https://a.example/x.goggle", "https://b.example/y.goggle"],
        )
        self.assertEqual(
            AppSettings(brave_goggles="https://a.example/x;\nhttps://b.example/y").resolved_brave_goggles(),
            ["https://a.example/x", "https://b.example/y"],
        )


if __name__ == "__main__":
    unittest.main()
