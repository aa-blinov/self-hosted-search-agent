import unittest

from search_agent import tuning
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

    def test_resolved_extract_max_chars_defaults_to_tuning(self) -> None:
        self.assertEqual(AppSettings().resolved_extract_max_chars(), tuning.EXTRACT_MAX_CHARS)
        self.assertEqual(AppSettings(extract_max_chars=900).resolved_extract_max_chars(), 900)

    def test_resolved_compose_and_rag_max_tokens(self) -> None:
        self.assertEqual(AppSettings().resolved_compose_answer_max_tokens(), tuning.COMPOSE_ANSWER_MAX_TOKENS)
        self.assertEqual(AppSettings(compose_answer_max_tokens=512).resolved_compose_answer_max_tokens(), 512)
        self.assertEqual(AppSettings().resolved_rag_analysis_max_tokens(), tuning.RAG_ANALYSIS_MAX_TOKENS)
        self.assertEqual(AppSettings(rag_analysis_max_tokens=700).resolved_rag_analysis_max_tokens(), 700)

    def test_resolved_intelligence_max_tokens(self) -> None:
        self.assertEqual(AppSettings().resolved_claim_decompose_max_tokens(), tuning.CLAIM_DECOMPOSE_MAX_TOKENS)
        self.assertEqual(AppSettings(claim_decompose_max_tokens=400).resolved_claim_decompose_max_tokens(), 400)
        self.assertEqual(AppSettings().resolved_verify_claim_max_tokens(), tuning.VERIFY_CLAIM_MAX_TOKENS)
        self.assertEqual(AppSettings(verify_claim_max_tokens=900).resolved_verify_claim_max_tokens(), 900)
        self.assertEqual(AppSettings().resolved_time_normalize_max_tokens(), tuning.TIME_NORMALIZE_MAX_TOKENS)
        self.assertEqual(AppSettings(time_normalize_max_tokens=64).resolved_time_normalize_max_tokens(), 64)

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
