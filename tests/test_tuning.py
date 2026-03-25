"""Sanity checks for ``search_agent.tuning`` (non-env agent limits)."""

import unittest

from search_agent import tuning


class TuningModuleTests(unittest.TestCase):
    def test_serp_gate_bounds(self) -> None:
        self.assertGreater(tuning.SERP_GATE_MIN_URLS, 0)
        self.assertGreaterEqual(tuning.SERP_GATE_MAX_URLS, tuning.SERP_GATE_MIN_URLS)

    def test_claim_loop_and_variants(self) -> None:
        self.assertGreater(tuning.AGENT_MAX_CLAIM_ITERATIONS, 0)
        self.assertGreater(tuning.AGENT_MAX_PARALLEL_CLAIMS, 0)
        self.assertGreaterEqual(tuning.AGENT_MAX_SEARCH_CALLS_PER_RUN, 0)
        self.assertGreater(tuning.AGENT_MAX_QUERY_VARIANTS, 0)
        self.assertGreaterEqual(tuning.AGENT_MAX_REFINE_VARIANTS, tuning.AGENT_MAX_QUERY_VARIANTS)

    def test_intelligence_token_caps(self) -> None:
        self.assertGreater(tuning.CLAIM_DECOMPOSE_MAX_TOKENS, 0)
        self.assertGreater(tuning.VERIFY_CLAIM_MAX_TOKENS, 0)
        self.assertGreater(tuning.TIME_NORMALIZE_MAX_TOKENS, 0)

    def test_compose_answer_limits(self) -> None:
        self.assertGreater(tuning.COMPOSE_ANSWER_MAX_TOKENS, 0)
        self.assertGreater(tuning.RAG_ANALYSIS_MAX_TOKENS, 0)
        self.assertGreater(tuning.COMPOSE_ANSWER_MAX_SPAN_CHARS, 500)
        self.assertGreater(tuning.COMPOSE_ANSWER_DIGEST_LINE_CHARS, 0)
        self.assertGreater(tuning.TRAIFILATURA_MIN_MAIN_CHARS, 0)

    def test_fetch_limits_non_negative(self) -> None:
        self.assertGreater(tuning.SHALLOW_FETCH_TIMEOUT, 0)
        self.assertGreaterEqual(tuning.SHALLOW_FETCH_HTTP_ATTEMPTS, 1)
        self.assertGreaterEqual(tuning.SHALLOW_FETCH_RETRY_BACKOFF_SEC, 0.0)
        self.assertGreater(tuning.EXTRACT_MAX_CHARS, 0)
        self.assertGreater(tuning.CRAWL4AI_TIMEOUT, 0)
        self.assertGreater(tuning.FETCH_SHALLOW_CONCURRENCY, 0)
        self.assertGreater(tuning.FETCH_DEEP_CONCURRENCY, 0)

    def test_eval_harness_delay(self) -> None:
        self.assertGreaterEqual(tuning.EVAL_CASE_DELAY_SEC, 0.0)

    def test_logfire_constants(self) -> None:
        self.assertTrue(tuning.LOGFIRE_SERVICE_NAME)
        self.assertTrue(tuning.LOGFIRE_ENVIRONMENT)


if __name__ == "__main__":
    unittest.main()
