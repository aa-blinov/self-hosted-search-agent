import unittest
from unittest.mock import patch

from search_agent.application.text_heuristics import normalize_relative_time_references
from search_agent.domain.models import Claim, Passage, QueryClassification
from search_agent.infrastructure.intelligence import (
    PydanticAIQueryIntelligence,
    _ClaimDecompositionOutput,
    _ClaimDraft,
    _ClaimProfileOutput,
    _IntentOutput,
    _NormalizedQueryOutput,
    _VerificationOutput,
    _EvidenceQuote,
)
from search_agent.settings import AppSettings


class IntelligenceLayerTests(unittest.TestCase):
    def test_classify_query_uses_deterministic_relative_date_normalization(self):
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))

        with patch("search_agent.infrastructure.intelligence.normalize_relative_time_references", return_value="OpenAI revenue 2026-03-25"):
            with patch.object(service._normalize_agent, "run_sync") as normalize_call:
                with patch.object(service, "_classify_intent_llm", return_value=_IntentOutput(intent="factual", complexity="single_hop")):
                    classification = service.classify_query("OpenAI revenue today")

        self.assertEqual(classification.normalized_query, "OpenAI revenue 2026-03-25")
        self.assertEqual(classification.time_scope, "2026-03-25")
        normalize_call.assert_not_called()

    def test_classify_query_marks_news_digest_intent(self):
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))

        with patch("search_agent.infrastructure.intelligence.normalize_relative_time_references", return_value="что 2026-03-25 было в Астане"):
            with patch.object(service, "_classify_intent_llm", return_value=_IntentOutput(intent="news_digest", complexity="single_hop")):
                classification = service.classify_query("что сегодня было в Астане")

        self.assertEqual(classification.intent, "news_digest")
        self.assertEqual(classification.time_scope, "2026-03-25")
        self.assertEqual(classification.region_hint, "Астане")

    def test_normalize_relative_time_references_supports_explicit_now(self):
        normalized = normalize_relative_time_references(
            "что сегодня было в Астане",
            now=__import__("datetime").datetime(2026, 3, 25),
        )
        self.assertEqual(normalized, "что 2026-03-25 было в Астане")

    def test_decompose_claims_maps_structured_output_to_claims(self):
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        classification = QueryClassification(
            query="Compare OpenAI revenue and Microsoft revenue in 2025",
            normalized_query="Compare OpenAI revenue and Microsoft revenue in 2025",
            intent="comparison",
            complexity="multi_hop",
            needs_freshness=False,
            time_scope="2025",
        )
        result = type(
            "Result",
            (),
            {
                "output": _ClaimDecompositionOutput(
                    claims=[
                        _ClaimDraft(
                            claim_text="OpenAI revenue in 2025",
                            priority=1,
                            needs_freshness=False,
                            entity_set=["OpenAI"],
                            time_scope="2025",
                            claim_profile=_ClaimProfileOutput(
                                answer_shape="exact_number",
                                required_dimensions=["number"],
                                focus_terms=["revenue", "2025"],
                            ),
                        ),
                        _ClaimDraft(
                            claim_text="Microsoft revenue in 2025",
                            priority=2,
                            needs_freshness=False,
                            entity_set=["Microsoft"],
                            time_scope="2025",
                            claim_profile=_ClaimProfileOutput(
                                answer_shape="exact_number",
                                required_dimensions=["number"],
                                focus_terms=["revenue", "2025"],
                            ),
                        ),
                    ]
                )
            },
        )()

        with patch.object(service._claim_agent, "run_sync", return_value=result):
            claims = service.decompose_claims(classification)

        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0].entity_set, ["OpenAI"])
        self.assertEqual(claims[1].entity_set, ["Microsoft"])
        self.assertEqual(claims[0].claim_profile.focus_terms, ["revenue", "2025"])

    def test_decompose_claims_preserves_factual_proposition_with_time_anchor(self):
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        classification = QueryClassification(
            query="Was Python 3.13.0 released on October 1, 2024?",
            normalized_query="Was Python 3.13.0 released on October 1, 2024?",
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
            time_scope="October 1, 2024",
        )
        result = type(
            "Result",
            (),
            {
                "output": _ClaimDecompositionOutput(
                    claims=[
                        _ClaimDraft(
                            claim_text="Python 3.13.0 release date",
                            priority=1,
                            needs_freshness=False,
                            entity_set=["Python"],
                            time_scope=None,
                            claim_profile=_ClaimProfileOutput(
                                answer_shape="exact_date",
                                required_dimensions=["release_date"],
                                focus_terms=["release date"],
                            ),
                        )
                    ]
                )
            },
        )()

        with patch.object(service._claim_agent, "run_sync", return_value=result):
            claims = service.decompose_claims(classification)

        self.assertEqual(claims[0].claim_text, "Was Python 3.13.0 released on October 1, 2024?")

    def test_decompose_claims_rejects_hallucinated_exact_date_answer_in_claim_text(self):
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        classification = QueryClassification(
            query="When was Python 3.13.0 released?",
            normalized_query="When was Python 3.13.0 released?",
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
        )
        result = type(
            "Result",
            (),
            {
                "output": _ClaimDecompositionOutput(
                    claims=[
                        _ClaimDraft(
                            claim_text="Python 3.13.0 was released on 2025-10-07.",
                            priority=1,
                            needs_freshness=False,
                            entity_set=["Python"],
                            time_scope=None,
                            claim_profile=_ClaimProfileOutput(
                                answer_shape="exact_date",
                                required_dimensions=["release_date"],
                                focus_terms=["release date"],
                            ),
                        )
                    ]
                )
            },
        )()

        with patch.object(service._claim_agent, "run_sync", return_value=result):
            claims = service.decompose_claims(classification)

        self.assertEqual(claims[0].claim_text, "When was Python 3.13.0 released?")

    def test_decompose_claims_keeps_atomic_subclaims_for_multi_hop_factual_queries(self):
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        classification = QueryClassification(
            query="Who is the CEO of Microsoft and when was Python 3.13.0 released?",
            normalized_query="Who is the CEO of Microsoft and when was Python 3.13.0 released?",
            intent="factual",
            complexity="multi_hop",
            needs_freshness=False,
        )
        result = type(
            "Result",
            (),
            {
                "output": _ClaimDecompositionOutput(
                    claims=[
                        _ClaimDraft(
                            claim_text="Who is the CEO of Microsoft?",
                            priority=1,
                            needs_freshness=False,
                            entity_set=["Microsoft"],
                            time_scope=None,
                            claim_profile=_ClaimProfileOutput(
                                answer_shape="fact",
                                required_dimensions=["role"],
                                focus_terms=["CEO", "Microsoft"],
                                preferred_domain_types=["official"],
                            ),
                        ),
                        _ClaimDraft(
                            claim_text="When was Python 3.13.0 released?",
                            priority=2,
                            needs_freshness=False,
                            entity_set=["Python"],
                            time_scope=None,
                            claim_profile=_ClaimProfileOutput(
                                answer_shape="exact_date",
                                required_dimensions=["release_date"],
                                focus_terms=["release date"],
                            ),
                        ),
                    ]
                )
            },
        )()

        with patch.object(service._claim_agent, "run_sync", return_value=result):
            claims = service.decompose_claims(classification)

        self.assertEqual(claims[0].claim_text, "Who is the CEO of Microsoft?")
        self.assertEqual(claims[1].claim_text, "When was Python 3.13.0 released?")

    def test_claim_profile_normalizes_open_ended_contracts(self) -> None:
        classification = QueryClassification(
            query="How does asyncio work in Python?",
            normalized_query="How does asyncio work in Python?",
            intent="synthesis",
            complexity="single_hop",
            needs_freshness=False,
        )

        profile = PydanticAIQueryIntelligence._claim_profile_from_output(
            _ClaimProfileOutput(
                answer_shape="overview",
                primary_source_required=True,
                min_independent_sources=1,
                routing_bias="short_path",
                allow_synthesis_without_primary=False,
                strict_contract=True,
                focus_terms=["event details", "asyncio"],
            ),
            classification,
        )

        self.assertEqual(profile.answer_shape, "overview")
        self.assertFalse(profile.primary_source_required)
        self.assertGreaterEqual(profile.min_independent_sources, 2)
        self.assertEqual(profile.routing_bias, "iterative_loop")
        self.assertTrue(profile.allow_synthesis_without_primary)
        self.assertFalse(profile.strict_contract)

    def test_claim_profile_normalizes_event_exact_number_contracts(self) -> None:
        classification = QueryClassification(
            query="What was the exact room temperature when Satya Nadella was named CEO?",
            normalized_query="What was the exact room temperature when Satya Nadella was named CEO?",
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
        )

        profile = PydanticAIQueryIntelligence._claim_profile_from_output(
            _ClaimProfileOutput(
                answer_shape="exact_number",
                primary_source_required=True,
                min_independent_sources=2,
                routing_bias="targeted_retrieval",
                required_dimensions=["time", "number", "source"],
                strict_contract=False,
                focus_terms=["room temperature", "event details"],
            ),
            classification,
        )

        self.assertEqual(profile.answer_shape, "exact_number")
        self.assertEqual(profile.routing_bias, "iterative_loop")
        self.assertIn("number", profile.required_dimensions)
        self.assertTrue(profile.strict_contract)

    def test_claim_profile_normalizes_event_context_exact_number_contracts(self) -> None:
        classification = QueryClassification(
            query="What was the room temperature when Satya Nadella was named CEO?",
            normalized_query="What was the room temperature when Satya Nadella was named CEO?",
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
        )

        profile = PydanticAIQueryIntelligence._claim_profile_from_output(
            _ClaimProfileOutput(
                answer_shape="exact_number",
                min_independent_sources=1,
                routing_bias=None,
                required_dimensions=["temperature value", "event context", "number"],
                strict_contract=False,
                focus_terms=["room temperature", "temperature"],
            ),
            classification,
        )

        self.assertGreaterEqual(profile.min_independent_sources, 2)
        self.assertEqual(profile.routing_bias, "iterative_loop")
        self.assertTrue(profile.strict_contract)

    def test_claim_profile_relaxes_simple_exact_number_contracts(self) -> None:
        classification = QueryClassification(
            query="At standard atmospheric pressure, what is the boiling point of water in Celsius?",
            normalized_query="At standard atmospheric pressure, what is the boiling point of water in Celsius?",
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
        )

        profile = PydanticAIQueryIntelligence._claim_profile_from_output(
            _ClaimProfileOutput(
                answer_shape="exact_number",
                primary_source_required=False,
                min_independent_sources=2,
                routing_bias="iterative_loop",
                required_dimensions=["number", "source"],
                strict_contract=True,
                focus_terms=["boiling point", "Celsius"],
            ),
            classification,
        )

        self.assertEqual(profile.answer_shape, "exact_number")
        self.assertFalse(profile.primary_source_required)
        self.assertEqual(profile.min_independent_sources, 1)
        self.assertIsNone(profile.routing_bias)
        self.assertFalse(profile.strict_contract)
        self.assertNotIn("event_context", [value.casefold() for value in profile.required_dimensions])

    def test_claim_profile_strips_spurious_event_context_from_generic_exact_number_facts(self) -> None:
        classification = QueryClassification(
            query="At standard atmospheric pressure, what is the boiling point of water in Celsius?",
            normalized_query="At standard atmospheric pressure, what is the boiling point of water in Celsius?",
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
        )

        profile = PydanticAIQueryIntelligence._claim_profile_from_output(
            _ClaimProfileOutput(
                answer_shape="exact_number",
                primary_source_required=True,
                min_independent_sources=2,
                routing_bias="iterative_loop",
                required_dimensions=["number", "event_context"],
                strict_contract=True,
                focus_terms=["boiling point", "Celsius"],
            ),
            classification,
        )

        self.assertFalse(profile.primary_source_required)
        self.assertEqual(profile.min_independent_sources, 1)
        self.assertIsNone(profile.routing_bias)
        self.assertFalse(profile.strict_contract)
        self.assertEqual(profile.required_dimensions, ["number"])

    def test_claim_profile_keeps_official_fact_contracts_strict(self) -> None:
        classification = QueryClassification(
            query="Is the IRS a U.S. government agency?",
            normalized_query="Is the IRS a U.S. government agency?",
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
        )

        profile = PydanticAIQueryIntelligence._claim_profile_from_output(
            _ClaimProfileOutput(
                answer_shape="fact",
                primary_source_required=False,
                min_independent_sources=1,
                preferred_domain_types=["official", "major_media"],
                routing_bias=None,
                required_dimensions=["agency_classification"],
                strict_contract=False,
                focus_terms=["IRS", "government agency"],
            ),
            classification,
        )

        self.assertTrue(profile.primary_source_required)
        self.assertEqual(profile.min_independent_sources, 2)
        self.assertEqual(profile.routing_bias, "short_path")
        self.assertTrue(profile.strict_contract)

    def test_claim_profile_relaxes_simple_exact_date_contracts(self) -> None:
        classification = QueryClassification(
            query="When was Python 3.13.0 released?",
            normalized_query="When was Python 3.13.0 released?",
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
        )

        profile = PydanticAIQueryIntelligence._claim_profile_from_output(
            _ClaimProfileOutput(
                answer_shape="exact_date",
                primary_source_required=True,
                min_independent_sources=2,
                routing_bias="iterative_loop",
                required_dimensions=["release_date"],
                strict_contract=True,
                focus_terms=["release date"],
            ),
            classification,
        )

        self.assertEqual(profile.answer_shape, "exact_date")
        self.assertEqual(profile.min_independent_sources, 1)
        self.assertIsNone(profile.routing_bias)
        self.assertFalse(profile.strict_contract)

    def test_verify_claim_maps_structured_quotes_to_spans(self):
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        claim = Claim(
            claim_id="claim-1",
            claim_text="Satya Nadella is the CEO of Microsoft.",
            priority=1,
            needs_freshness=False,
            entity_set=["Satya Nadella", "Microsoft"],
        )
        passage = Passage(
            passage_id="p1",
            url="https://news.microsoft.com/source/exec/satya-nadella/",
            canonical_url="https://news.microsoft.com/source/exec/satya-nadella/",
            host="news.microsoft.com",
            title="Satya Nadella",
            section="Leadership",
            published_at="2026-03-24T00:00:00+00:00",
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p1",
            text="Satya Nadella is Chairman and Chief Executive Officer of Microsoft.",
            source_score=0.9,
            utility_score=0.9,
        )
        result = type(
            "Result",
            (),
            {
                "output": _VerificationOutput(
                    verdict="supported",
                    confidence=0.95,
                    supporting_passages=[_EvidenceQuote(passage_id="p1", quote="Chief Executive Officer of Microsoft")],
                    rationale="Direct official leadership page.",
                )
            },
        )()

        with patch.object(service._verifier_agent, "run_sync", return_value=result):
            verification = service.verify_claim(claim, [passage])

        self.assertEqual(verification.verdict, "supported")
        self.assertEqual(len(verification.supporting_spans), 1)
        self.assertIn("Chief Executive Officer", verification.supporting_spans[0].text)

    def test_verify_claim_floors_supported_confidence_when_near_zero(self) -> None:
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        claim = Claim(
            claim_id="claim-1",
            claim_text="Satya Nadella leads Microsoft.",
            priority=1,
            needs_freshness=False,
            entity_set=["Microsoft"],
        )
        passage = Passage(
            passage_id="p1",
            url="https://news.microsoft.com/source/exec/satya-nadella/",
            canonical_url="https://news.microsoft.com/source/exec/satya-nadella/",
            host="news.microsoft.com",
            title="Satya Nadella",
            section="Leadership",
            published_at=None,
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p1",
            text="Satya Nadella is Chairman and CEO of Microsoft.",
            source_score=0.9,
            utility_score=0.9,
        )
        result = type(
            "Result",
            (),
            {
                "output": _VerificationOutput(
                    verdict="supported",
                    confidence=0.0,
                    supporting_passages=[_EvidenceQuote(passage_id="p1", quote="CEO")],
                    rationale="",
                )
            },
        )()

        with patch.object(service._verifier_agent, "run_sync", return_value=result):
            verification = service.verify_claim(claim, [passage])

        self.assertEqual(verification.verdict, "supported")
        self.assertGreaterEqual(verification.confidence, 0.38)

    def test_verify_claim_preserves_llm_insufficient_without_template_boost(self) -> None:
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        claim = Claim(
            claim_id="claim-1",
            claim_text="Python 3.8 introduced assignment expressions (walrus operator).",
            priority=1,
            needs_freshness=False,
            entity_set=["Python"],
        )
        long_body = "Introductory prose about assignment expressions. " * 20
        passage = Passage(
            passage_id="p-doc",
            url="https://docs.python.org/3/whatsnew/3.8.html",
            canonical_url="https://docs.python.org/3/whatsnew/3.8.html",
            host="docs.python.org",
            title="What is new in Python 3.8",
            section="",
            published_at=None,
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p-doc",
            text=long_body,
            source_score=0.9,
            utility_score=0.3,
        )
        result = type(
            "Result",
            (),
            {
                "output": _VerificationOutput(
                    verdict="insufficient_evidence",
                    confidence=0.15,
                    supporting_passages=[],
                    rationale="Need clearer quote.",
                )
            },
        )()

        with patch.object(service._verifier_agent, "run_sync", return_value=result):
            verification = service.verify_claim(claim, [passage])

        self.assertEqual(verification.verdict, "insufficient_evidence")
        self.assertEqual(verification.supporting_spans, [])

    def test_verify_claim_uses_rescue_llm_pass_when_primary_fails(self) -> None:
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        claim = Claim(
            claim_id="claim-1",
            claim_text="Who is the CEO of Microsoft?",
            priority=1,
            needs_freshness=False,
            entity_set=["Microsoft"],
        )
        passage = Passage(
            passage_id="p-role",
            url="https://news.microsoft.com/source/exec/satya-nadella/",
            canonical_url="https://news.microsoft.com/source/exec/satya-nadella/",
            host="news.microsoft.com",
            title="Satya Nadella",
            section="Leadership",
            published_at=None,
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p-role",
            text="Satya Nadella is Chairman and Chief Executive Officer of Microsoft. Before being named CEO, he held leadership roles across the company.",
            source_score=0.9,
            utility_score=0.4,
        )
        result = type(
            "Result",
            (),
            {
                "output": _VerificationOutput(
                    verdict="supported",
                    confidence=0.72,
                    supporting_passages=[_EvidenceQuote(passage_id="p-role", quote="Chief Executive Officer of Microsoft")],
                    rationale="Official leadership page directly names the CEO.",
                )
            },
        )()

        with patch.object(service._verifier_agent, "run_sync", side_effect=[RuntimeError("provider error"), result]):
            verification = service.verify_claim(claim, [passage])

        self.assertEqual(verification.verdict, "supported")
        self.assertGreaterEqual(verification.confidence, 0.72)
        self.assertTrue(verification.supporting_spans)
        self.assertIn("Chief Executive Officer of Microsoft", verification.supporting_spans[0].text)

    def test_verify_claim_returns_default_insufficient_after_double_failure(self) -> None:
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        claim = Claim(
            claim_id="claim-1",
            claim_text="When was Python 3.13.0 released?",
            priority=1,
            needs_freshness=False,
            entity_set=["Python"],
        )
        passage = Passage(
            passage_id="p-release",
            url="https://www.python.org/downloads/release/python-3130/",
            canonical_url="https://www.python.org/downloads/release/python-3130/",
            host="www.python.org",
            title="Python 3.13.0",
            section="Release",
            published_at=None,
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p-release",
            text="Python 3.13.0, released on October 7, 2024, includes new features and bug fixes. " * 8,
            source_score=0.9,
            utility_score=0.3,
        )
        with patch.object(service._verifier_agent, "run_sync", side_effect=[RuntimeError("provider error"), RuntimeError("provider error")]):
            verification = service.verify_claim(claim, [passage])

        self.assertEqual(verification.verdict, "insufficient_evidence")
        self.assertIn("after retry", verification.rationale)

    def test_synthesize_answer_news_digest_keeps_four_source_footer_entries(self) -> None:
        service = PydanticAIQueryIntelligence(AppSettings(llm_api_key="test-key"))
        passages = [
            Passage(
                passage_id=f"p{i}",
                url=f"https://source{i}.example/story",
                canonical_url=f"https://source{i}.example/story",
                host=f"source{i}.example",
                title=f"Story {i}",
                section="News",
                published_at="2026-03-24T00:00:00+00:00",
                author=None,
                extracted_at="2026-03-24T00:00:00+00:00",
                chunk_id=f"p{i}",
                text=f"Important development {i}.",
                source_score=0.8,
                utility_score=0.8,
            )
            for i in range(1, 5)
        ]
        result = type("Result", (), {"output": "Summary line [1]"})()

        with patch.object(service._synth_agent, "run_sync", return_value=result):
            answer = service.synthesize_answer(
                "What are the latest news on the Iran conflict?",
                passages,
                intent="news_digest",
            )

        self.assertIn("[1] Story 1", answer)
        self.assertIn("[2] Story 2", answer)
        self.assertIn("[3] Story 3", answer)
        self.assertIn("[4] Story 4", answer)


if __name__ == "__main__":
    unittest.main()
