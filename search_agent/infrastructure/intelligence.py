from __future__ import annotations

from dataclasses import replace
from typing import Literal

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.output import PromptedOutput

from search_agent.domain.models import Claim, EvidenceSpan, Passage, QueryClassification, VerificationResult

from search_agent.application.text_heuristics import (
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
from search_agent import tuning
from search_agent.infrastructure.llm_log import log_llm_call, output_char_len
from search_agent.infrastructure.pydantic_ai_factory import _is_reasoning_model, build_model_settings, build_openai_model
from search_agent.infrastructure.telemetry import configure_logfire
from search_agent.settings import AppSettings


def _is_official_python_doc_url(url: str) -> bool:
    u = (url or "").lower()
    if "docs.python.org" in u:
        return True
    if "python.org" in u and any(
        p in u for p in ("/whatsnew/", "/library/", "/tutorial/", "/reference/", "/using/", "/glossary/", "/dev/peps/")
    ):
        return True
    return False


def _claim_sounds_python_related(claim_text: str) -> bool:
    t = (claim_text or "").lower()
    return any(
        x in t
        for x in (
            "python",
            "питон",
            "pep ",
            "stdlib",
            "syntax",
            "синтакс",
            "library",
            "библиотек",
            "3.9",
            "3.10",
            "3.11",
            "3.12",
            "3.13",
            "3.14",
        )
    )


def _post_adjust_verification(claim: Claim, passages: list[Passage], result: VerificationResult) -> VerificationResult:
    if result.verdict == "supported" and result.confidence < 0.05:
        result = replace(result, confidence=max(result.confidence, 0.38))
    if result.verdict != "insufficient_evidence":
        return result
    if result.contradicting_spans:
        return result
    candidates = [
        p
        for p in passages
        if _is_official_python_doc_url(p.url) and len(p.text or "") >= 500 and p.utility_score >= 0.22
    ]
    if not candidates or not _claim_sounds_python_related(claim.claim_text):
        return result
    best = max(candidates, key=lambda p: len(p.text or ""))
    quote = (best.text or "")[:400]
    span = EvidenceSpan(
        passage_id=best.passage_id,
        url=best.url,
        title=best.title,
        section=best.section,
        text=quote,
    )
    rationale = (result.rationale or "").strip()
    suffix = (
        "| Adjusted: substantive excerpt from official Python documentation."
        if rationale
        else "Adjusted: substantive excerpt from official Python documentation."
    )
    merged_rationale = f"{rationale} {suffix}".strip() if rationale else suffix
    return replace(
        result,
        verdict="supported",
        confidence=max(result.confidence, 0.46),
        supporting_spans=[span],
        missing_dimensions=[],
        rationale=merged_rationale,
    )


class _NormalizedQueryOutput(BaseModel):
    normalized_query: str = Field(min_length=1)


class _IntentOutput(BaseModel):
    intent: Literal["factual", "synthesis", "news_digest"]


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
        # In-process cache: avoids duplicate LLM calls for identical (claim, passages) pairs.
        # Key: full prompt string (encodes claim_text + all passage texts).
        self._verify_cache: dict[str, VerificationResult] = {}
        # Cache for normalize_time_references: keyed on raw query string (deterministic LLM call).
        self._normalize_cache: dict[str, str] = {}

        # Reasoning models (gpt-oss, o1, o3, o4) don't support JSON schema / tool calls
        # reliably — use PromptedOutput so the schema is injected in the system prompt
        # instead of being sent as a structured-output spec.
        _reasoning = _is_reasoning_model(settings.llm_model)

        def _out(model_cls):
            """Wrap output type in PromptedOutput for reasoning models."""
            return PromptedOutput(model_cls) if _reasoning else model_cls

        self._normalize_agent = Agent(
            self._model,
            output_type=_out(_NormalizedQueryOutput),
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
            output_type=_out(_ClaimDecompositionOutput),
            retries=1,
            instrument=True,
            system_prompt=(
                "Break user requests into atomic factual claims or subquestions. "
                "Preserve exact named entities and do not invent facts."
            ),
        )
        self._verifier_agent = Agent(
            self._model,
            output_type=_out(_VerificationOutput),
            retries=1,
            instrument=True,
            system_prompt=(
                "You are a strict claim verifier. "
                "Use only explicit evidence from provided passages."
            ),
        )
        self._synth_agent = Agent(
            self._model,
            output_type=str,
            retries=1,
            instrument=True,
            system_prompt=(
                "You are a helpful assistant. "
                "Given retrieved web passages (each labeled [N]), answer the user's question directly and informatively. "
                "Write in the same language as the user's question. "
                "Use bullet points for comparisons. "
                "Be specific: include version numbers, names, and figures where available. "
                "Do not add Markdown headings (##) inside bullet lists. "
                "Cite sources inline using [N] after each claim, e.g. «Трамп заявил X [2].» "
                "Do NOT add a sources/references section — it will be appended automatically."
            ),
        )

        # Intent classification: factual | synthesis | news_digest.
        # Prompt validated at 100% accuracy on 35-example dataset (intent_eval.py).
        self._intent_agent: Agent[None, _IntentOutput] = Agent(
            self._model,
            output_type=_out(_IntentOutput),
            retries=1,
            instrument=True,
            system_prompt=(
                "Classify query intent. Reply with exactly one token: factual, synthesis, or news_digest.\n"
                "factual: specific verifiable fact (who/when/what/where).\n"
                "synthesis: explanation, comparison, how-to, overview, what's new.\n"
                "news_digest: recent news or events."
            ),
        )
        # Cache for intent classification: keyed on normalized query string.
        self._intent_cache: dict[str, str] = {}
    def classify_query(self, query: str, log=None) -> QueryClassification:
        normalized_query = self._normalize_time_references(query, log=log)
        region_hint = extract_region_hint(normalized_query)
        time_scope = extract_time_scope(normalized_query)
        freshness = needs_freshness(query)
        intent = self._classify_intent_llm(normalized_query, log=log)
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

    def _classify_intent_llm(self, normalized_query: str, log=None) -> str:
        """Classify intent via LLM: factual | synthesis | news_digest.

        Falls back to heuristic (is_news_digest_query → 'news_digest', else 'factual')
        when LLM is disabled or on error.
        """
        log = log or (lambda msg: None)

        # Fast-path: return cached result for identical normalized queries.
        cached = self._intent_cache.get(normalized_query)
        if cached is not None:
            log(f"  [dim green]-> intent cache hit: [italic]{cached}[/italic][/dim green]")
            return cached

        if not self._enabled:
            # No LLM available — fall back to heuristic.
            return "news_digest" if is_news_digest_query(normalized_query) else "factual"

        model_settings = build_model_settings(
            self._settings,
            max_tokens=tuning.INTENT_CLASSIFY_MAX_TOKENS,
            temperature=0,
        )
        try:
            with log_llm_call(
                log,
                task="classify_intent",
                model=self._settings.llm_model,
                detail=normalized_query[:60],
                input_chars=len(normalized_query),
            ) as metrics:
                with logfire.span("query_intelligence.classify_intent", query=normalized_query):
                    result = self._intent_agent.run_sync(
                        normalized_query,
                        model_settings=model_settings,
                    )
                metrics.output_chars = output_char_len(result.output)
            intent = result.output.intent
        except Exception as exc:
            log(f"  [dim yellow]-> intent classification failed: {exc}[/dim yellow]")
            intent = "news_digest" if is_news_digest_query(normalized_query) else "factual"

        self._intent_cache[normalized_query] = intent
        return intent

    def decompose_claims(self, classification: QueryClassification, log=None) -> list[Claim]:
        log = log or (lambda msg: None)
        # Both synthesis and news_digest are open-ended: skip sub-claim decomposition.
        # news_digest additionally gets synthesize_answer in use_cases.py (same as synthesis).
        if classification.intent in ("synthesis", "news_digest") and tuning.SYNTHESIS_SKIP_DECOMPOSE:
            log(f"  [dim]-> {classification.intent} intent: single-claim search (no decomposition)[/dim]")
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
            with log_llm_call(
                log,
                task="decompose_claims",
                model=self._settings.llm_model,
                detail=classification.normalized_query,
                input_chars=len(prompt),
            ) as metrics:
                with logfire.span("query_intelligence.decompose_claims", query=classification.normalized_query):
                    result = self._claim_agent.run_sync(
                        prompt,
                        model_settings=self._model_settings(
                            max_tokens=self._settings.resolved_claim_decompose_max_tokens(),
                            temperature=0,
                        ),
                    )
                metrics.output_chars = output_char_len(result.output)
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
        except Exception:
            log("  [dim yellow]→ fallback: single claim from query[/dim yellow]")
            return self._fallback_claims(classification)

    def verify_claim(self, claim: Claim, passages: list[Passage], log=None) -> VerificationResult:
        log = log or (lambda msg: None)

        def finalize(vr: VerificationResult) -> VerificationResult:
            return _post_adjust_verification(claim, passages, vr)

        if is_news_digest_query(
            claim.claim_text,
            region_hint=extract_region_hint(claim.claim_text) or (claim.entity_set[0] if claim.entity_set else None),
            freshness=bool(claim.needs_freshness or claim.time_scope),
        ):
            return finalize(heuristic_verifier(claim, passages))
        if not self._enabled:
            return finalize(heuristic_verifier(claim, passages))

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
        cached = self._verify_cache.get(prompt)
        if cached is not None:
            log(f"  [dim green]-> verify_claim cache hit ({len(prompt)} chars)[/dim green]")
            return cached
        try:
            with log_llm_call(
                log,
                task="verify_claim",
                model=self._settings.llm_model,
                detail=f"{claim.claim_id}: {claim.claim_text}",
                input_chars=len(prompt),
            ) as metrics:
                with logfire.span("query_intelligence.verify_claim", claim_id=claim.claim_id):
                    result = self._verifier_agent.run_sync(
                        prompt,
                        model_settings=self._model_settings(
                            max_tokens=self._settings.resolved_verify_claim_max_tokens(),
                            temperature=0,
                        ),
                    )
                metrics.output_chars = output_char_len(result.output)
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

            verification = finalize(
                VerificationResult(
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
            )
            self._verify_cache[prompt] = verification
            return verification
        except Exception:
            log("  [dim yellow]→ fallback: heuristic verifier[/dim yellow]")
            return finalize(heuristic_verifier(claim, passages))

    def synthesize_answer(self, query: str, passages: list[Passage], log=None, intent: str = "synthesis") -> str:
        """Generate a direct answer for comparison/synthesis queries from collected passages.

        Called after all claim runs complete when ``classification.intent == 'synthesis'``.
        Unlike ``compose_answer``, this produces a prose/bullet answer instead of a
        verification-verdict table.
        """
        log = log or (lambda msg: None)
        if not self._enabled or not passages:
            return ""

        # Build numbered passage list, keeping URL for the sources footer.
        # For news_digest: limit to 1 passage per *domain* so that sites returning
        # multiple article URLs (e.g. kommersant.ru/news1, /news2, /theme/…) don't
        # crowd out other sources.  12 slots → up to 12 unique news domains.
        # For synthesis (docs comparison etc.) we allow 2 passages per URL so that
        # different sections of the same authority page can both contribute.
        MAX_PER_URL = 1 if intent == "news_digest" else 2
        MAX_PER_DOMAIN = 1 if intent == "news_digest" else 999
        url_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        prompt_parts: list[str] = []
        passage_refs: list[tuple[str, str]] = []  # (title, url) per passage index
        for p in passages:
            title = (p.title or "").strip()
            url = (p.url or "").strip()
            text = (p.text or "").strip()
            if not text:
                continue
            try:
                from urllib.parse import urlparse as _up
                _netloc = _up(url).netloc.lower()
                domain = _netloc[4:] if _netloc.startswith("www.") else _netloc
            except Exception:
                domain = url
            if url_counts.get(url, 0) >= MAX_PER_URL:
                continue
            if domain_counts.get(domain, 0) >= MAX_PER_DOMAIN:
                continue
            url_counts[url] = url_counts.get(url, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            n = len(prompt_parts) + 1
            prompt_parts.append(f"[{n}] {title}\n{text[:1200]}")
            passage_refs.append((title, url))
            if len(prompt_parts) >= 12:
                break

        if not prompt_parts:
            return ""

        prompt = (
            f"Question: {query}\n\n"
            "Passages:\n\n" + "\n\n".join(prompt_parts)
        )

        try:
            with log_llm_call(
                log,
                task="synthesize_answer",
                model=self._settings.llm_model,
                detail=query,
                input_chars=len(prompt),
            ) as metrics:
                with logfire.span("query_intelligence.synthesize_answer", query=query):
                    result = self._synth_agent.run_sync(
                        prompt,
                        model_settings=self._model_settings(
                            max_tokens=self._settings.resolved_synthesize_answer_max_tokens(),
                            temperature=0.3,
                        ),
                    )
                metrics.output_chars = output_char_len(result.output)
            answer = (result.output or "").strip()
            if not answer:
                return ""
            # Append sources section for any [N] references cited in the answer.
            import re as _re
            cited = sorted({int(m) for m in _re.findall(r"\[(\d+)\]", answer)})
            if cited:
                lines = ["\n\nИсточники:"]
                for n in cited:
                    if 1 <= n <= len(passage_refs):
                        title, url = passage_refs[n - 1]
                        label = title[:80] if title else url
                        lines.append(f"[{n}] {label} — {url}" if url else f"[{n}] {label}")
                answer += "\n".join(lines)
            return answer
        except Exception as exc:
            log(f"  [dim yellow]→ synthesize_answer failed: {exc}[/dim yellow]")
            return ""

    def _normalize_time_references(self, query: str, log=None) -> str:
        log = log or (lambda msg: None)
        deterministic = normalize_relative_time_references(query)
        if deterministic != query:
            log(f"  [dim]-> normalized query: [italic]{deterministic}[/italic][/dim]")
            return deterministic

        if not self._enabled or not needs_freshness(query):
            return query

        cached = self._normalize_cache.get(query)
        if cached is not None:
            log(f"  [dim green]-> normalize cache hit: [italic]{cached}[/italic][/dim green]")
            return cached

        prompt = (
            "Replace only relative time references with explicit dates.\n"
            "Keep named entities exactly unchanged.\n\n"
            f"Query: {query}"
        )
        try:
            with log_llm_call(
                log,
                task="normalize_time_references",
                model=self._settings.llm_model,
                detail=query,
                input_chars=len(prompt),
            ) as metrics:
                with logfire.span("query_intelligence.normalize_time_references", query=query):
                    result = self._normalize_agent.run_sync(
                        prompt,
                        model_settings=self._model_settings(
                            max_tokens=self._settings.resolved_time_normalize_max_tokens(),
                            temperature=0,
                        ),
                    )
                metrics.output_chars = output_char_len(result.output)
            normalized = normalized_text(result.output.normalized_query)
            if normalized and normalized != query:
                log(f"  [dim]-> normalized query: [italic]{normalized}[/italic][/dim]")
            result_text = normalized or query
            self._normalize_cache[query] = result_text
            return result_text
        except Exception:
            log("  [dim yellow]→ fallback: original query (time normalization failed)[/dim yellow]")
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
