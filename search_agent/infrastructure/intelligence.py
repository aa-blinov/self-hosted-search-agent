from __future__ import annotations

from typing import Literal

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from search_agent.domain.models import Claim, EvidenceSpan, Passage, QueryClassification, VerificationResult

from search_agent.application.text_heuristics import (
    COMPARISON_MARKERS,
    clamp,
    extract_entities,
    extract_region_hint,
    extract_time_scope,
    heuristic_verifier,
    is_news_digest_query,
    needs_freshness,
    normalize_relative_time_references,
    normalized_text,
    should_decompose,
)
from search_agent.infrastructure.pydantic_ai_factory import build_model_settings, build_openai_model
from search_agent.infrastructure.telemetry import configure_logfire
from search_agent.settings import AppSettings


class _NormalizedQueryOutput(BaseModel):
    normalized_query: str = Field(min_length=1)


class _ClaimDraft(BaseModel):
    claim_text: str = Field(min_length=1)
    priority: int = 1
    needs_freshness: bool = False
    entity_set: list[str] = Field(default_factory=list)
    time_scope: str | None = None


class _ClaimDecompositionOutput(BaseModel):
    claims: list[_ClaimDraft] = Field(default_factory=list)


class _EvidenceQuote(BaseModel):
    passage_id: str
    quote: str


class _VerificationOutput(BaseModel):
    verdict: Literal["supported", "contradicted", "insufficient_evidence"]
    confidence: float = 0.0
    supporting_passages: list[_EvidenceQuote] = Field(default_factory=list)
    contradicting_passages: list[_EvidenceQuote] = Field(default_factory=list)
    missing_dimensions: list[str] = Field(default_factory=list)
    rationale: str = ""


class PydanticAIQueryIntelligence:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        configure_logfire(settings)
        self._enabled = bool(settings.llm_api_key)
        self._model = build_openai_model(settings)

        self._normalize_agent = Agent(
            self._model,
            output_type=_NormalizedQueryOutput,
            retries=1,
            instrument=True,
            system_prompt=(
                "Rewrite search queries for web search. "
                "Preserve named entities exactly. "
                "Only replace relative time references with explicit dates."
            ),
        )
        self._claim_agent = Agent(
            self._model,
            output_type=_ClaimDecompositionOutput,
            retries=1,
            instrument=True,
            system_prompt=(
                "Break user requests into atomic factual claims or subquestions. "
                "Preserve exact named entities and do not invent facts."
            ),
        )
        self._verifier_agent = Agent(
            self._model,
            output_type=_VerificationOutput,
            retries=1,
            instrument=True,
            system_prompt=(
                "You are a strict claim verifier. "
                "Use only explicit evidence from provided passages."
            ),
        )
    def classify_query(self, query: str, log=None) -> QueryClassification:
        normalized_query = self._normalize_time_references(query, log=log)
        lowered = normalized_query.lower()
        region_hint = extract_region_hint(normalized_query)
        time_scope = extract_time_scope(normalized_query)
        freshness = needs_freshness(query)
        if is_news_digest_query(normalized_query, region_hint=region_hint, freshness=freshness):
            intent = "news_digest"
        elif any(marker in lowered for marker in COMPARISON_MARKERS):
            intent = "comparison"
        else:
            intent = "factual"
        complexity = "multi_hop" if should_decompose(normalized_query) else "single_hop"
        entities = extract_entities(normalized_query)
        entity_disambiguation = any(len(entity) <= 4 for entity in entities)
        return QueryClassification(
            query=query,
            normalized_query=normalized_query,
            intent=intent,
            complexity=complexity,
            needs_freshness=freshness,
            time_scope=time_scope,
            region_hint=region_hint,
            entity_disambiguation=entity_disambiguation,
        )

    def decompose_claims(self, classification: QueryClassification, log=None) -> list[Claim]:
        log = log or (lambda msg: None)
        if classification.intent == "news_digest":
            return self._fallback_claims(classification)
        if not self._enabled or not should_decompose(classification.normalized_query):
            return self._fallback_claims(classification)

        prompt = (
            "Return JSON with one key `claims`.\n"
            "Each claim must have claim_text, priority, needs_freshness, entity_set, time_scope.\n"
            "Keep claims atomic, exact, and capped at 4.\n\n"
            f"User request: {classification.normalized_query}"
        )
        try:
            with logfire.span("query_intelligence.decompose_claims", query=classification.normalized_query):
                result = self._claim_agent.run_sync(
                    prompt,
                    model_settings=self._model_settings(max_tokens=500, temperature=0),
                )
            claims = []
            for idx, item in enumerate(result.output.claims[:4], 1):
                claims.append(
                    Claim(
                        claim_id=f"claim-{idx}",
                        claim_text=normalized_text(item.claim_text),
                        priority=max(1, int(item.priority or idx)),
                        needs_freshness=bool(item.needs_freshness or classification.needs_freshness),
                        entity_set=[
                            normalized_text(entity)
                            for entity in item.entity_set
                            if normalized_text(entity)
                        ] or extract_entities(item.claim_text),
                        time_scope=normalized_text(item.time_scope) if item.time_scope else extract_time_scope(item.claim_text),
                    )
                )
            return claims or self._fallback_claims(classification)
        except Exception as exc:
            log(f"  [dim yellow]⚠ claim decomposition failed: {exc}[/dim yellow]")
            return self._fallback_claims(classification)

    def verify_claim(self, claim: Claim, passages: list[Passage], log=None) -> VerificationResult:
        log = log or (lambda msg: None)
        if is_news_digest_query(
            claim.claim_text,
            region_hint=extract_region_hint(claim.claim_text) or (claim.entity_set[0] if claim.entity_set else None),
            freshness=bool(claim.needs_freshness or claim.time_scope),
        ):
            return heuristic_verifier(claim, passages)
        if not self._enabled:
            return heuristic_verifier(claim, passages)

        prompt_lines = []
        for passage in passages[:8]:
            prompt_lines.append(
                f"[{passage.passage_id}] {passage.title} | {passage.url}\n"
                f"Section: {passage.section}\n"
                f"Text: {passage.text}"
            )
        prompt = (
            "Classify the claim as supported, contradicted, or insufficient_evidence.\n"
            "Return supporting_passages and contradicting_passages with short quotes.\n"
            "Use missing_dimensions from time, entity, number, location, source, coverage.\n\n"
            f"Claim: {claim.claim_text}\n\n" + "\n\n".join(prompt_lines)
        )
        try:
            with logfire.span("query_intelligence.verify_claim", claim_id=claim.claim_id):
                result = self._verifier_agent.run_sync(
                    prompt,
                    model_settings=self._model_settings(max_tokens=700, temperature=0),
                )
            output = result.output
            passage_map = {passage.passage_id: passage for passage in passages}

            def build_spans(items: list[_EvidenceQuote]) -> list[EvidenceSpan]:
                spans = []
                for item in items:
                    passage = passage_map.get(item.passage_id)
                    if passage is None:
                        continue
                    quote = normalized_text(item.quote) or passage.text[:220]
                    spans.append(
                        EvidenceSpan(
                            passage_id=passage.passage_id,
                            url=passage.url,
                            title=passage.title,
                            section=passage.section,
                            text=quote,
                        )
                    )
                return spans

            return VerificationResult(
                verdict=output.verdict,
                confidence=clamp(output.confidence),
                supporting_spans=build_spans(output.supporting_passages),
                contradicting_spans=build_spans(output.contradicting_passages),
                missing_dimensions=[
                    normalized_text(item)
                    for item in output.missing_dimensions
                    if normalized_text(item)
                ],
                rationale=normalized_text(output.rationale),
            )
        except Exception as exc:
            log(f"  [dim yellow]⚠ verifier failed: {exc}[/dim yellow]")
            return heuristic_verifier(claim, passages)

    def _normalize_time_references(self, query: str, log=None) -> str:
        log = log or (lambda msg: None)
        deterministic = normalize_relative_time_references(query)
        if deterministic != query:
            log(f"  [dim]в†і normalized query: [italic]{deterministic}[/italic][/dim]")
            return deterministic

        if not self._enabled or not needs_freshness(query):
            return query

        prompt = (
            "Replace only relative time references with explicit dates.\n"
            "Keep named entities exactly unchanged.\n\n"
            f"Query: {query}"
        )
        try:
            with logfire.span("query_intelligence.normalize_time_references", query=query):
                result = self._normalize_agent.run_sync(
                    prompt,
                    model_settings=self._model_settings(max_tokens=120, temperature=0),
                )
            normalized = normalized_text(result.output.normalized_query)
            if normalized and normalized != query:
                log(f"  [dim]↳ normalized query: [italic]{normalized}[/italic][/dim]")
            return normalized or query
        except Exception as exc:
            log(f"  [dim yellow]⚠ time normalization failed: {exc}[/dim yellow]")
            return query

    @staticmethod
    def _fallback_claims(classification: QueryClassification) -> list[Claim]:
        return [
            Claim(
                claim_id="claim-1",
                claim_text=classification.normalized_query,
                priority=1,
                needs_freshness=classification.needs_freshness,
                entity_set=extract_entities(classification.normalized_query),
                time_scope=classification.time_scope,
            )
        ]

    def _model_settings(self, *, max_tokens: int, temperature: float):
        return build_model_settings(
            self._settings,
            max_tokens=max_tokens,
            temperature=temperature,
        )
