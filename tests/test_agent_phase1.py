import unittest

from agent import build_evidence_bundle, build_query_variants, compose_answer, gate_serp_results, should_stop_claim_loop
from agent_types import (
    AgentRunResult,
    Claim,
    ClaimRun,
    EvidenceBundle,
    GatedSerpResult,
    Passage,
    QueryClassification,
    SearchSnapshot,
    SerpResult,
    SourceAssessment,
    VerificationResult,
)


class AgentPhase1Tests(unittest.TestCase):
    def test_query_variants_preserve_entities(self):
        classification = QueryClassification(
            query="What did OpenAI announce about GPT-4.1 in Q1 2026?",
            normalized_query="What did OpenAI announce about GPT-4.1 in Q1 2026?",
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
            time_scope="Q1 2026",
        )
        claim = Claim(
            claim_id="claim-1",
            claim_text="What did OpenAI announce about GPT-4.1 in Q1 2026?",
            priority=1,
            needs_freshness=False,
            entity_set=["OpenAI", "GPT-4.1"],
            time_scope="Q1 2026",
        )

        variants = build_query_variants(claim, classification)

        self.assertGreaterEqual(len(variants), 4)
        self.assertEqual(len({variant.query_text for variant in variants}), len(variants))
        self.assertTrue(any('"OpenAI"' in variant.query_text for variant in variants))
        self.assertTrue(any("Q1 2026" in variant.query_text for variant in variants))
        self.assertTrue(any(variant.strategy == "exact_match" for variant in variants))

    def test_serp_gate_dedupes_and_prefers_official(self):
        claim = Claim(
            claim_id="claim-1",
            claim_text="Acme Corp earnings report 2026",
            priority=1,
            needs_freshness=False,
            entity_set=["Acme Corp"],
            time_scope="2026",
        )
        snapshot = SearchSnapshot(
            query="Acme Corp earnings report 2026",
            suggestions=[],
            retrieved_at="2026-03-24T00:00:00+00:00",
            results=[
                SerpResult(
                    result_id="r1",
                    query_variant_id="v1",
                    title="Acme Corp Q1 2026 earnings report",
                    url="https://investor.acme.com/q1-2026?utm_source=test",
                    snippet="Official earnings report from Acme Corp.",
                    canonical_url="https://investor.acme.com/q1-2026",
                    host="investor.acme.com",
                    position=1,
                    published_at="2026-03-20T00:00:00+00:00",
                ),
                SerpResult(
                    result_id="r2",
                    query_variant_id="v2",
                    title="Acme Corp Q1 2026 earnings report",
                    url="https://www.investor.acme.com/q1-2026",
                    snippet="Official earnings report from Acme Corp.",
                    canonical_url="https://investor.acme.com/q1-2026",
                    host="www.investor.acme.com",
                    position=2,
                    published_at="2026-03-20T00:00:00+00:00",
                ),
                SerpResult(
                    result_id="r3",
                    query_variant_id="v3",
                    title="Best Acme earnings review guide",
                    url="https://spam-reviews.best/acme-earnings",
                    snippet="Top review and affiliate breakdown for Acme earnings.",
                    canonical_url="https://spam-reviews.best/acme-earnings",
                    host="spam-reviews.best",
                    position=3,
                    published_at=None,
                ),
            ],
        )

        gated = gate_serp_results(claim, [snapshot], limit=10)

        self.assertEqual(len(gated), 2)
        self.assertEqual(gated[0].serp.canonical_url, "https://investor.acme.com/q1-2026")
        self.assertGreater(gated[0].assessment.source_score, gated[1].assessment.source_score)

    def test_answer_composer_only_uses_supported_claims_as_answer(self):
        supported_passage = Passage(
            passage_id="p1",
            url="https://example.com/openai",
            canonical_url="https://example.com/openai",
            host="example.com",
            title="OpenAI announcement",
            section="Intro",
            published_at="2026-03-20T00:00:00+00:00",
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p1",
            text="OpenAI announced GPT-4.1 in March 2026 with updated API features.",
            source_score=0.9,
            utility_score=0.8,
        )
        supported_bundle = EvidenceBundle(
            claim_id="claim-1",
            claim_text="OpenAI announced GPT-4.1 in March 2026.",
            supporting_passages=[supported_passage],
            considered_passages=[supported_passage],
            independent_source_count=1,
            has_primary_source=False,
            freshness_ok=True,
            verification=VerificationResult(verdict="supported", confidence=0.8),
        )
        unsupported_bundle = EvidenceBundle(
            claim_id="claim-2",
            claim_text="The Moon is made of cheese.",
            considered_passages=[],
            independent_source_count=0,
            has_primary_source=False,
            freshness_ok=False,
            verification=VerificationResult(
                verdict="insufficient_evidence",
                confidence=0.1,
                missing_dimensions=["coverage"],
            ),
        )
        report = AgentRunResult(
            user_query="Tell me about OpenAI and whether the Moon is made of cheese.",
            classification=QueryClassification(
                query="Tell me about OpenAI and whether the Moon is made of cheese.",
                normalized_query="Tell me about OpenAI and whether the Moon is made of cheese.",
                intent="factual",
                complexity="multi_hop",
                needs_freshness=False,
            ),
            claims=[
                ClaimRun(
                    claim=Claim(
                        claim_id="claim-1",
                        claim_text="OpenAI announced GPT-4.1 in March 2026.",
                        priority=1,
                        needs_freshness=False,
                    ),
                    evidence_bundle=supported_bundle,
                ),
                ClaimRun(
                    claim=Claim(
                        claim_id="claim-2",
                        claim_text="The Moon is made of cheese.",
                        priority=2,
                        needs_freshness=False,
                    ),
                    evidence_bundle=unsupported_bundle,
                ),
            ],
            answer="",
        )

        answer = compose_answer(report)

        self.assertIn("OpenAI announced GPT-4.1 in March 2026", answer)
        self.assertIn("[1]", answer)
        self.assertIn("The Moon is made of cheese.: insufficient evidence", answer)
        self.assertIn("Источники", answer)

    def test_serp_gate_prefers_entity_matched_official_host(self):
        claim = Claim(
            claim_id="claim-1",
            claim_text="When was Python 3.13.0 released?",
            priority=1,
            needs_freshness=False,
            entity_set=["Python", "Python 3.13.0"],
        )
        snapshot = SearchSnapshot(
            query="When was Python 3.13.0 released?",
            suggestions=[],
            retrieved_at="2026-03-24T00:00:00+00:00",
            results=[
                SerpResult(
                    result_id="r1",
                    query_variant_id="v1",
                    title="Python 3.13.0 final released",
                    url="https://blog.python.org/2024/10/python-3130-final-released.html",
                    snippet="Python 3.13.0 final released on October 7, 2024.",
                    canonical_url="https://blog.python.org/2024/10/python-3130-final-released.html",
                    host="blog.python.org",
                    position=1,
                    published_at="2024-10-07T00:00:00+00:00",
                ),
                SerpResult(
                    result_id="r2",
                    query_variant_id="v1",
                    title="When should you upgrade to Python 3.13?",
                    url="https://pythonspeed.com/articles/upgrade-python-3.13/",
                    snippet="Python 3.13 article by a third-party blog.",
                    canonical_url="https://pythonspeed.com/articles/upgrade-python-3.13/",
                    host="pythonspeed.com",
                    position=2,
                    published_at="2024-09-12T00:00:00+00:00",
                ),
                SerpResult(
                    result_id="r3",
                    query_variant_id="v1",
                    title="Python 3.13.0 Is Released | Hacker News",
                    url="https://news.ycombinator.com/item?id=41766035",
                    snippet="Discussion about Python 3.13.0 release.",
                    canonical_url="https://news.ycombinator.com/item?id=41766035",
                    host="news.ycombinator.com",
                    position=3,
                    published_at="2024-10-07T00:00:00+00:00",
                ),
            ],
        )

        gated = gate_serp_results(claim, [snapshot], limit=10)

        self.assertEqual(gated[0].serp.host, "blog.python.org")
        self.assertEqual(gated[0].assessment.domain_type, "official")
        self.assertGreater(gated[0].assessment.primary_source_likelihood, gated[1].assessment.primary_source_likelihood)
        self.assertNotEqual(gated[1].assessment.domain_type, "official")

    def test_supported_claim_with_entities_keeps_loop_until_primary_source(self):
        claim = Claim(
            claim_id="claim-1",
            claim_text="When was Python 3.13.0 released?",
            priority=1,
            needs_freshness=False,
            entity_set=["Python", "Python 3.13.0"],
        )
        passage = Passage(
            passage_id="p1",
            url="https://medium.example/python-313",
            canonical_url="https://medium.example/python-313",
            host="medium.example",
            title="Python 3.13.0 released",
            section="Intro",
            published_at="2024-10-07T00:00:00+00:00",
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p1",
            text="Python 3.13.0 was released on October 7, 2024.",
            source_score=0.7,
            utility_score=0.7,
        )
        verification = VerificationResult(verdict="supported", confidence=0.98)
        gated = [
            GatedSerpResult(
                serp=SerpResult(
                    result_id="r1",
                    query_variant_id="v1",
                    title="Python 3.13.0 released",
                    url=passage.url,
                    snippet=passage.text,
                    canonical_url=passage.canonical_url,
                    host=passage.host,
                    position=1,
                ),
                assessment=SourceAssessment(
                    domain_type="unknown",
                    source_prior=0.0,
                    primary_source_likelihood=0.4,
                    freshness_score=0.8,
                    seo_spam_risk=0.1,
                    entity_match_score=1.0,
                    semantic_match_score=1.0,
                    source_score=0.7,
                ),
            )
        ]
        bundle = build_evidence_bundle(claim, [passage], verification, gated)
        bundle.independent_source_count = 2
        bundle.has_primary_source = False

        self.assertFalse(should_stop_claim_loop(claim, bundle, iteration=1))


if __name__ == "__main__":
    unittest.main()
