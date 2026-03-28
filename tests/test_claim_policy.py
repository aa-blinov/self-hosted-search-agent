import unittest

from search_agent import tuning
from search_agent.application.claim_policy import (
    claim_contract_gaps,
    post_adjust_verification,
    should_stop_claim_loop,
)
from search_agent.domain.models import Claim, ClaimProfile, EvidenceBundle, Passage, VerificationResult


class ClaimPolicyTests(unittest.TestCase):
    def test_claim_contract_gaps_tracks_primary_sources_and_freshness(self) -> None:
        claim = Claim(
            claim_id="claim-1",
            claim_text="Latest MacBook Neo specifications.",
            priority=1,
            needs_freshness=True,
            claim_profile=ClaimProfile(
                answer_shape="product_specs",
                primary_source_required=True,
                min_independent_sources=2,
            ),
        )

        gaps = claim_contract_gaps(
            claim,
            independent_source_count=1,
            has_primary_source=False,
            freshness_ok=False,
        )

        self.assertEqual(gaps, ["primary_source", "independent_sources", "freshness"])

    def test_post_adjust_verification_demotes_list_like_overview_support(self) -> None:
        claim = Claim(
            claim_id="claim-1",
            claim_text="What features does MacBook Neo have?",
            priority=1,
            needs_freshness=False,
            claim_profile=ClaimProfile(
                answer_shape="overview",
                focus_terms=["display", "processor"],
                strict_contract=False,
            ),
        )
        passage = Passage(
            passage_id="p1",
            url="https://www.apple.com/macbook-neo/specs/",
            canonical_url="https://www.apple.com/macbook-neo/specs/",
            host="www.apple.com",
            title="MacBook Neo specs",
            section="Overview",
            published_at="2026-03-04T00:00:00+00:00",
            author=None,
            extracted_at="2026-03-28T00:00:00+00:00",
            chunk_id="p1",
            text="MacBook Neo has a 13-inch display and an A18 Pro chip.",
            source_score=0.9,
            utility_score=0.9,
        )
        result = VerificationResult(
            verdict="supported",
            confidence=0.91,
            missing_dimensions=[],
            rationale="Matched a specs page.",
        )

        adjusted = post_adjust_verification(claim, [passage], result)

        self.assertEqual(adjusted.verdict, "insufficient_evidence")
        self.assertEqual(adjusted.supporting_spans, [])
        self.assertIn("claim-level insufficient_evidence", adjusted.rationale)

    def test_post_adjust_verification_promotes_explanatory_overview(self) -> None:
        claim = Claim(
            claim_id="claim-1",
            claim_text="How does asyncio work in Python?",
            priority=1,
            needs_freshness=False,
            claim_profile=ClaimProfile(
                answer_shape="overview",
                focus_terms=["asyncio"],
                strict_contract=False,
            ),
        )
        passages = [
            Passage(
                passage_id="p1",
                url="https://docs.python.org/3/library/asyncio.html",
                canonical_url="https://docs.python.org/3/library/asyncio.html",
                host="docs.python.org",
                title="asyncio",
                section="Overview",
                published_at=None,
                author=None,
                extracted_at="2026-03-28T00:00:00+00:00",
                chunk_id="p1",
                text="asyncio provides infrastructure for writing single-threaded concurrent code using coroutines.",
                source_score=0.9,
                utility_score=0.9,
            ),
            Passage(
                passage_id="p2",
                url="https://docs.python.org/3/library/asyncio-task.html",
                canonical_url="https://docs.python.org/3/library/asyncio-task.html",
                host="docs.python.org",
                title="Coroutines and Tasks",
                section="Tasks",
                published_at=None,
                author=None,
                extracted_at="2026-03-28T00:00:00+00:00",
                chunk_id="p2",
                text="The event loop schedules coroutines and manages asynchronous tasks.",
                source_score=0.88,
                utility_score=0.82,
            ),
        ]
        result = VerificationResult(
            verdict="insufficient_evidence",
            confidence=0.12,
            missing_dimensions=["coverage"],
            rationale="Need broader explanation.",
        )

        adjusted = post_adjust_verification(claim, passages, result)

        self.assertEqual(adjusted.verdict, "supported")
        self.assertGreaterEqual(adjusted.confidence, 0.62)
        self.assertEqual(len(adjusted.supporting_spans), 2)

    def test_post_adjust_verification_promotes_simple_fact_with_strong_multi_source_support(self) -> None:
        claim = Claim(
            claim_id="claim-1",
            claim_text="Is the IRS a U.S. government agency?",
            priority=1,
            needs_freshness=False,
            entity_set=["IRS"],
            claim_profile=ClaimProfile(
                answer_shape="fact",
                primary_source_required=True,
                min_independent_sources=2,
                required_dimensions=["agency_classification", "government_affiliation"],
                focus_terms=["IRS", "government agency"],
                strict_contract=True,
            ),
        )
        passages = [
            Passage(
                passage_id="p1",
                url="https://www.usa.gov/agencies/internal-revenue-service",
                canonical_url="https://www.usa.gov/agencies/internal-revenue-service",
                host="www.usa.gov",
                title="Internal Revenue Service (IRS) | USAGov",
                section="Agency",
                published_at=None,
                author=None,
                extracted_at="2026-03-28T00:00:00+00:00",
                chunk_id="p1",
                text="The Internal Revenue Service (IRS) administers and enforces U.S. federal tax laws.",
                source_score=0.71,
                utility_score=0.39,
            ),
            Passage(
                passage_id="p2",
                url="https://usgovernmentmanual.gov/Agency?EntityId=IRS",
                canonical_url="https://usgovernmentmanual.gov/Agency?EntityId=IRS",
                host="usgovernmentmanual.gov",
                title="United States Government Manual",
                section="Commissioner of Internal Revenue",
                published_at=None,
                author=None,
                extracted_at="2026-03-28T00:00:00+00:00",
                chunk_id="p2",
                text="The Internal Revenue Service (IRS) administers and enforces the internal revenue laws.",
                source_score=0.69,
                utility_score=0.37,
            ),
        ]
        result = VerificationResult(
            verdict="insufficient_evidence",
            confidence=0.60,
            missing_dimensions=["source", "coverage"],
            rationale="Need a slightly clearer official corroboration.",
        )

        adjusted = post_adjust_verification(claim, passages, result)

        self.assertEqual(adjusted.verdict, "supported")
        self.assertGreaterEqual(adjusted.confidence, 0.66)
        self.assertEqual(len(adjusted.supporting_spans), 2)

    def test_should_stop_claim_loop_respects_strict_contract(self) -> None:
        claim = Claim(
            claim_id="claim-1",
            claim_text="What was the exact room temperature when Satya Nadella was named CEO?",
            priority=1,
            needs_freshness=False,
            claim_profile=ClaimProfile(
                answer_shape="exact_number",
                required_dimensions=["number"],
                strict_contract=True,
                min_independent_sources=2,
            ),
        )
        bundle = EvidenceBundle(
            claim_id="claim-1",
            claim_text=claim.claim_text,
            independent_source_count=1,
            has_primary_source=False,
            freshness_ok=True,
            verification=VerificationResult(
                verdict="supported",
                confidence=0.8,
            ),
            contract_satisfied=False,
        )

        self.assertFalse(should_stop_claim_loop(claim, bundle, iteration=1))
        self.assertTrue(should_stop_claim_loop(claim, bundle, iteration=tuning.AGENT_MAX_CLAIM_ITERATIONS))
