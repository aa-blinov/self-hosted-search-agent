import unittest

from search_agent.settings import AppSettings


class SettingsLayerTests(unittest.TestCase):
    def test_app_settings_env_surface_excludes_internal_tuning(self) -> None:
        """Tuning lives in ``search_agent.tuning``, not AppSettings."""
        fields = set(AppSettings.model_fields)
        self.assertIn("llm_api_key", fields)
        self.assertIn("logfire_token", fields)
        self.assertNotIn("shallow_fetch_timeout", fields)
        self.assertNotIn("agent_max_claim_iterations", fields)
        self.assertNotIn("eval_case_delay_sec", fields)

    def test_resolved_search_provider_override(self):
        self.assertEqual(
            AppSettings(search_provider="brave", search_provider_override="ddgs").resolved_search_provider(),
            "ddgs",
        )
        self.assertEqual(
            AppSettings(search_provider="ddgs", search_provider_override=None).resolved_search_provider(),
            "ddgs",
        )

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
