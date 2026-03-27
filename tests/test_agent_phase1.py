import unittest

from search_agent.application.agent_steps import (
    _answer_type,
    build_evidence_bundle,
    build_query_variants,
    compose_answer,
    gate_serp_results,
    refine_query_variants,
    route_claim_retrieval,
    should_stop_claim_loop,
)
from search_agent.domain.models import (
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
    def test_refine_contradiction_only_when_verdict_contradicted(self):
        """insufficient_evidence must not add refined_contradiction (noisy SERP)."""
        classification = QueryClassification(
            query="test",
            normalized_query="test",
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
        )
        claim = Claim(
            claim_id="c1",
            claim_text="What is new in Python 3.12?",
            priority=1,
            needs_freshness=False,
            entity_set=["Python"],
        )
        insufficient = VerificationResult(
            verdict="insufficient_evidence",
            confidence=0.2,
            missing_dimensions=["coverage"],
        )
        variants_ie = refine_query_variants(
            claim,
            classification,
            insufficient,
            gated_results=[],
            bundle=None,
            iteration=1,
            existing_queries=set(),
        )
        self.assertFalse(
            any(v.strategy == "refined_contradiction" for v in variants_ie),
            "insufficient_evidence should not trigger contradiction query refinement",
        )

        contradicted = VerificationResult(
            verdict="contradicted",
            confidence=0.75,
            missing_dimensions=[],
        )
        variants_ct = refine_query_variants(
            claim,
            classification,
            contradicted,
            gated_results=[],
            bundle=None,
            iteration=1,
            existing_queries=set(),
        )
        self.assertTrue(
            any(v.strategy == "refined_contradiction" for v in variants_ct),
            "contradicted should still offer contradiction-focused refinement",
        )

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
        self.assertIn("Sources", answer)

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

    def test_supported_unit_only_entity_can_stop_without_primary_source(self):
        claim = Claim(
            claim_id="claim-1",
            claim_text="At standard atmospheric pressure, what is the boiling point of water in Celsius?",
            priority=1,
            needs_freshness=False,
            entity_set=["Celsius"],
        )
        passage = Passage(
            passage_id="p1",
            url="https://example.com/boiling-point",
            canonical_url="https://example.com/boiling-point",
            host="example.com",
            title="Boiling point of water",
            section="Answer",
            published_at=None,
            author=None,
            extracted_at="2026-03-24T00:00:00+00:00",
            chunk_id="p1",
            text="Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            source_score=0.8,
            utility_score=0.8,
        )
        verification = VerificationResult(verdict="supported", confidence=0.98)
        bundle = EvidenceBundle(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            supporting_passages=[passage],
            considered_passages=[passage],
            independent_source_count=2,
            has_primary_source=False,
            freshness_ok=True,
            verification=verification,
        )

        self.assertTrue(should_stop_claim_loop(claim, bundle, iteration=1))

    def test_insufficient_exact_numeric_detail_stops_after_strong_guardrail_signal(self):
        claim = Claim(
            claim_id="claim-1",
            claim_text="What was the exact room temperature when Satya Nadella was named CEO of Microsoft?",
            priority=1,
            needs_freshness=False,
            entity_set=["Satya Nadella", "Microsoft"],
        )
        verification = VerificationResult(
            verdict="insufficient_evidence",
            confidence=1.0,
            missing_dimensions=["time", "location"],
        )
        bundle = EvidenceBundle(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            supporting_passages=[],
            considered_passages=[],
            independent_source_count=2,
            has_primary_source=True,
            freshness_ok=True,
            verification=verification,
        )

        self.assertTrue(should_stop_claim_loop(claim, bundle, iteration=1))

    def test_news_digest_query_variants_bias_local_news_and_date(self):
        classification = QueryClassification(
            query="что сегодня было в Астане",
            normalized_query="что 2026-03-25 было в Астане",
            intent="news_digest",
            complexity="single_hop",
            needs_freshness=True,
            time_scope="2026-03-25",
            region_hint="Астане",
        )
        claim = Claim(
            claim_id="claim-1",
            claim_text="что 2026-03-25 было в Астане",
            priority=1,
            needs_freshness=True,
            entity_set=["Астане"],
            time_scope="2026-03-25",
        )

        variants = build_query_variants(claim, classification)

        self.assertGreaterEqual(len(variants), 4)
        self.assertTrue(all(variant.strategy.startswith("news_digest_") for variant in variants))
        self.assertTrue(any("site:.kz" in variant.query_text for variant in variants))
        self.assertTrue(any("2026-03-25" in variant.query_text for variant in variants))
        self.assertTrue(
            any("\u043d\u043e\u0432\u043e\u0441\u0442\u0438" in variant.query_text or "\u0441\u043e\u0431\u044b\u0442\u0438\u044f" in variant.query_text for variant in variants)
        )

    def test_compose_answer_formats_news_digest_from_supported_passages(self):
        first = Passage(
            passage_id="p1",
            url="https://local.example/traffic",
            canonical_url="https://local.example/traffic",
            host="local.example",
            title="Astana traffic restrictions",
            section="City",
            published_at="2026-03-25T10:00:00+05:00",
            author=None,
            extracted_at="2026-03-25T11:00:00+05:00",
            chunk_id="p1",
            text="Authorities reported temporary traffic restrictions in central Astana on March 25, 2026.",
            source_score=0.8,
            utility_score=0.7,
        )
        second = Passage(
            passage_id="p2",
            url="https://city.example/weather",
            canonical_url="https://city.example/weather",
            host="city.example",
            title="Astana weather alert",
            section="Weather",
            published_at="2026-03-25T08:00:00+05:00",
            author=None,
            extracted_at="2026-03-25T11:00:00+05:00",
            chunk_id="p2",
            text="Forecasters issued a wind warning for Astana for March 25, 2026.",
            source_score=0.75,
            utility_score=0.68,
        )
        bundle = EvidenceBundle(
            claim_id="claim-1",
            claim_text="что 2026-03-25 было в Астане",
            supporting_passages=[first, second],
            considered_passages=[first, second],
            independent_source_count=2,
            has_primary_source=False,
            freshness_ok=True,
            verification=VerificationResult(verdict="supported", confidence=0.55),
        )
        report = AgentRunResult(
            user_query="что сегодня было в Астане",
            classification=QueryClassification(
                query="что сегодня было в Астане",
                normalized_query="что 2026-03-25 было в Астане",
                intent="news_digest",
                complexity="single_hop",
                needs_freshness=True,
                time_scope="2026-03-25",
                region_hint="Астане",
            ),
            claims=[
                ClaimRun(
                    claim=Claim(
                        claim_id="claim-1",
                        claim_text="что 2026-03-25 было в Астане",
                        priority=1,
                        needs_freshness=True,
                        entity_set=["Астане"],
                        time_scope="2026-03-25",
                    ),
                    evidence_bundle=bundle,
                )
            ],
            answer="",
        )

        answer = compose_answer(report)

        self.assertIn("Astana traffic restrictions", answer)
        self.assertIn("Astana weather alert", answer)
        self.assertIn("[1]", answer)
        self.assertIn("[2]", answer)

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
        self.assertIn("OpenAI announcement", answer)
        self.assertIn("https://example.com/openai", answer)

    def test_route_exact_numeric_detail_without_dimension_match_uses_iterative_loop(self):
        claim = Claim(
            claim_id="claim-1",
            claim_text="What was the exact room temperature when Satya Nadella was named CEO of Microsoft?",
            priority=1,
            needs_freshness=False,
            entity_set=["Satya Nadella", "Microsoft"],
        )
        gated = [
            GatedSerpResult(
                serp=SerpResult(
                    result_id="r1",
                    query_variant_id="v1",
                    title="Microsoft CEO: Satya Nadella",
                    url="https://news.microsoft.com/source/exec/satya-nadella/",
                    snippet="Satya Nadella is Chairman and Chief Executive Officer of Microsoft. Before being named CEO in February 2014, Nadella held leadership roles across the company.",
                    canonical_url="https://news.microsoft.com/source/exec/satya-nadella/",
                    host="news.microsoft.com",
                    position=1,
                ),
                assessment=SourceAssessment(
                    domain_type="official",
                    source_prior=0.5,
                    primary_source_likelihood=1.0,
                    freshness_score=0.4,
                    seo_spam_risk=0.0,
                    entity_match_score=1.0,
                    semantic_match_score=0.6,
                    source_score=1.0,
                ),
            ),
            GatedSerpResult(
                serp=SerpResult(
                    result_id="r2",
                    query_variant_id="v1",
                    title="Introducing Microsoft's new CEO: Satya Nadella",
                    url="https://blogs.microsoft.com/blog/2014/02/04/introducing-microsofts-new-ceo-satya-nadella/",
                    snippet="Microsoft today named Satya Nadella its next Chief Executive Officer.",
                    canonical_url="https://blogs.microsoft.com/blog/2014/02/04/introducing-microsofts-new-ceo-satya-nadella/",
                    host="blogs.microsoft.com",
                    position=2,
                ),
                assessment=SourceAssessment(
                    domain_type="official",
                    source_prior=0.5,
                    primary_source_likelihood=1.0,
                    freshness_score=0.4,
                    seo_spam_risk=0.0,
                    entity_match_score=1.0,
                    semantic_match_score=0.65,
                    source_score=1.0,
                ),
            ),
        ]

        self.assertEqual(_answer_type(claim), "number")
        decision = route_claim_retrieval(claim, gated)

        self.assertEqual(decision.mode, "iterative_loop")

    def test_route_numeric_dimension_match_can_stay_targeted(self):
        claim = Claim(
            claim_id="claim-1",
            claim_text="At standard atmospheric pressure, what is the boiling point of water in Celsius?",
            priority=1,
            needs_freshness=False,
        )
        gated = [
            GatedSerpResult(
                serp=SerpResult(
                    result_id="r1",
                    query_variant_id="v1",
                    title="What is the boiling point of water in Celsius?",
                    url="https://example.com/boiling-point",
                    snippet="At standard atmospheric pressure, water boils at 100 degrees Celsius (100°C).",
                    canonical_url="https://example.com/boiling-point",
                    host="example.com",
                    position=1,
                ),
                assessment=SourceAssessment(
                    domain_type="unknown",
                    source_prior=0.2,
                    primary_source_likelihood=0.3,
                    freshness_score=0.4,
                    seo_spam_risk=0.0,
                    entity_match_score=0.8,
                    semantic_match_score=0.9,
                    source_score=0.8,
                ),
            ),
            GatedSerpResult(
                serp=SerpResult(
                    result_id="r2",
                    query_variant_id="v1",
                    title="Boiling point of water",
                    url="https://example.org/boiling-water",
                    snippet="The boiling point of water is 100°C at 1 atm pressure.",
                    canonical_url="https://example.org/boiling-water",
                    host="example.org",
                    position=2,
                ),
                assessment=SourceAssessment(
                    domain_type="unknown",
                    source_prior=0.2,
                    primary_source_likelihood=0.2,
                    freshness_score=0.4,
                    seo_spam_risk=0.0,
                    entity_match_score=0.75,
                    semantic_match_score=0.85,
                    source_score=0.75,
                ),
            ),
        ]

        self.assertEqual(_answer_type(claim), "number")
        decision = route_claim_retrieval(claim, gated)

        self.assertIn(decision.mode, {"short_path", "targeted_retrieval"})

    def test_compose_answer_no_supported_lines_uses_heading_not_bullet(self):
        insufficient_bundle = EvidenceBundle(
            claim_id="claim-1",
            claim_text="What was the exact room temperature when Satya Nadella was named CEO of Microsoft?",
            considered_passages=[],
            independent_source_count=0,
            has_primary_source=False,
            freshness_ok=False,
            verification=VerificationResult(
                verdict="insufficient_evidence",
                confidence=0.1,
                missing_dimensions=["time", "location", "source"],
            ),
        )
        report = AgentRunResult(
            user_query="What was the exact room temperature when Satya Nadella was named CEO of Microsoft?",
            classification=QueryClassification(
                query="What was the exact room temperature when Satya Nadella was named CEO of Microsoft?",
                normalized_query="What was the exact room temperature when Satya Nadella was named CEO of Microsoft?",
                intent="factual",
                complexity="single_hop",
                needs_freshness=False,
            ),
            claims=[
                ClaimRun(
                    claim=Claim(
                        claim_id="claim-1",
                        claim_text="What was the exact room temperature when Satya Nadella was named CEO of Microsoft?",
                        priority=1,
                        needs_freshness=False,
                    ),
                    evidence_bundle=insufficient_bundle,
                )
            ],
            answer="",
        )

        answer = compose_answer(report)

        self.assertIn("Answer", answer)
        self.assertIn("Not enough claim-level evidence for a direct answer.", answer)
        self.assertNotIn("- Not enough claim-level evidence for a direct answer.", answer)


if __name__ == "__main__":
    unittest.main()
