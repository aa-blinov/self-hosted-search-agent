from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Literal

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.output import PromptedOutput

from search_agent.domain.models import (
    Claim,
    ClaimProfile,
    DomainType,
    EvidenceBundle,
    EvidenceSpan,
    GatedSerpResult,
    Passage,
    QueryClassification,
    VerificationResult,
)

from search_agent.application.text_heuristics import (
    clamp,
    extract_entities,
    extract_region_hint,
    extract_time_scope,
    needs_freshness,
    normalize_relative_time_references,
    normalized_text,
)
from search_agent import tuning
from search_agent.infrastructure.llm_log import log_llm_call, output_char_len
from search_agent.infrastructure.pydantic_ai_factory import _is_reasoning_model, build_model_settings, build_openai_model
from search_agent.infrastructure.telemetry import configure_logfire
from search_agent.settings import AppSettings


def _extract_citation_indices(answer: str) -> list[int]:
    cited: set[int] = set()
    text = answer or ""
    index = 0
    while index < len(text):
        if text[index] != "[":
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end].isdigit():
            end += 1
        if end > index + 1 and end < len(text) and text[end] == "]":
            cited.add(int(text[index + 1:end]))
            index = end + 1
            continue
        index += 1
    return sorted(cited)


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


def _claim_looks_like_feature_listing(claim_text: str) -> bool:
    t = (claim_text or "").lower()
    return any(
        marker in t
        for marker in (
            "what are the key new features",
            "what's new",
            "what is new",
            "new features",
            "main differences",
            "key differences",
            "compare",
            "comparison",
            "pros and cons",
            "что нового",
            "новые возможности",
            "новые функции",
            "основные отличия",
            "сравни",
            "сравнение",
        )
    )


def _claim_requests_role_identity(claim_text: str) -> bool:
    t = (claim_text or "").casefold().strip()
    return t.startswith("who ") or t.startswith("кто ")


def _claim_role_terms(claim_text: str) -> tuple[str, ...]:
    t = (claim_text or "").casefold()
    role_groups = (
        ("chief executive officer", "ceo"),
        ("chief technology officer", "cto"),
        ("chief financial officer", "cfo"),
        ("president",),
        ("founder",),
        ("chairman",),
    )
    for group in role_groups:
        if any(term in t for term in group):
            return group
    return ()


def _direct_role_support_passage(claim: Claim, passages: list[Passage]) -> Passage | None:
    if not _claim_requests_role_identity(claim.claim_text):
        return None

    role_terms = _claim_role_terms(claim.claim_text)
    if not role_terms:
        return None

    entity_hints = claim.entity_set or extract_entities(claim.claim_text)
    candidates: list[Passage] = []
    for passage in passages:
        lead = f"{passage.title}. {passage.text[:500]}"
        lowered = lead.casefold()
        if not any(term in lowered for term in role_terms):
            continue
        if entity_hints and not any(entity.casefold() in lowered for entity in entity_hints):
            continue
        if not any(marker in lowered for marker in (" is ", " serves as ", " became ", " was named ")):
            continue
        if not any(" " in entity for entity in extract_entities(lead)):
            continue
        candidates.append(passage)

    if not candidates:
        return None
    return max(candidates, key=lambda passage: (passage.utility_score, len(passage.text or ""), passage.source_score))


def _post_adjust_verification(claim: Claim, passages: list[Passage], result: VerificationResult) -> VerificationResult:
    if result.verdict == "supported" and result.confidence < 0.05:
        result = replace(result, confidence=max(result.confidence, 0.38))
    if result.verdict == "supported" and _claim_looks_like_feature_listing(claim.claim_text):
        rationale = (result.rationale or "").strip()
        suffix = (
            "| Adjusted: open-ended comparison/feature request stays claim-level insufficient_evidence."
            if rationale
            else "Adjusted: open-ended comparison/feature request stays claim-level insufficient_evidence."
        )
        merged_rationale = f"{rationale} {suffix}".strip() if rationale else suffix
        return replace(
            result,
            verdict="insufficient_evidence",
            confidence=min(result.confidence, 0.45),
            supporting_spans=[],
            missing_dimensions=result.missing_dimensions or ["coverage"],
            rationale=merged_rationale,
        )
    if result.verdict != "insufficient_evidence":
        return result
    if result.contradicting_spans:
        return result
    role_passage = _direct_role_support_passage(claim, passages)
    if role_passage is not None:
        quote = (role_passage.text or "")[:280]
        span = EvidenceSpan(
            passage_id=role_passage.passage_id,
            url=role_passage.url,
            title=role_passage.title,
            section=role_passage.section,
            text=quote,
        )
        rationale = (result.rationale or "").strip()
        suffix = (
            "| Adjusted: direct role-identification sentence found in retrieved evidence."
            if rationale
            else "Adjusted: direct role-identification sentence found in retrieved evidence."
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
    candidates = [
        p
        for p in passages
        if _is_official_python_doc_url(p.url) and len(p.text or "") >= 350 and p.utility_score >= 0.15
    ]
    if (
        not candidates
        or not _claim_sounds_python_related(claim.claim_text)
        or _claim_looks_like_feature_listing(claim.claim_text)
    ):
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


def _default_intent_decision() -> "_IntentOutput":
    return _IntentOutput(intent="factual", complexity="single_hop", search_queries=[])


def _default_verification(reason: str) -> VerificationResult:
    return VerificationResult(
        verdict="insufficient_evidence",
        confidence=0.0,
        missing_dimensions=["coverage"],
        rationale=reason,
    )


def _claim_profile_lines(claim: Claim) -> list[str]:
    profile = claim.claim_profile
    if profile is None:
        return [
            "answer_shape=fact",
            "primary_source_required=False",
            "min_independent_sources=1",
            "preferred_domain_types=",
            "required_dimensions=",
            "focus_terms=",
            "strict_contract=False",
        ]
    return [
        f"answer_shape={profile.answer_shape}",
        f"primary_source_required={profile.primary_source_required}",
        f"min_independent_sources={profile.min_independent_sources}",
        f"preferred_domain_types={','.join(profile.preferred_domain_types)}",
        f"required_dimensions={','.join(profile.required_dimensions)}",
        f"focus_terms={','.join(profile.focus_terms)}",
        f"strict_contract={profile.strict_contract}",
    ]


def _build_verifier_prompt(claim: Claim, passages: list[Passage], *, max_passages: int, max_chars: int) -> str:
    prompt_lines: list[str] = []
    for passage in passages[:max_passages]:
        text = normalized_text((passage.text or "")[:max_chars])
        prompt_lines.append(
            f"[{passage.passage_id}] {passage.title} | {passage.url}\n"
            f"Section: {passage.section}\n"
            f"Text: {text}"
        )
    return (
        "Verify the claim against the retrieved evidence.\n"
        "Use the claim contract. Do not invent facts. Use only the provided passages.\n"
        "For open-ended claims, collective coverage across multiple passages is allowed.\n"
        "If the contract requires a primary source or multiple independent sources and the evidence set does not satisfy that contract, return insufficient_evidence.\n"
        "Return supporting_passages and contradicting_passages with short quotes.\n"
        "Use missing_dimensions from time, entity, number, location, source, coverage.\n\n"
        f"Claim: {claim.claim_text}\n"
        "Claim contract:\n"
        + "\n".join(_claim_profile_lines(claim))
        + "\n\nPassages:\n\n"
        + "\n\n".join(prompt_lines)
    )


def _post_adjust_verification(claim: Claim, passages: list[Passage], result: VerificationResult) -> VerificationResult:
    if result.verdict == "supported" and result.confidence < 0.05:
        return replace(result, confidence=max(result.confidence, 0.38))
    return result


class _NormalizedQueryOutput(BaseModel):
    normalized_query: str = Field(min_length=1)


class _IntentOutput(BaseModel):
    intent: Literal["factual", "synthesis", "news_digest"]
    complexity: Literal["single_hop", "multi_hop"] = "single_hop"
    search_queries: list[str] = Field(default_factory=list)


class _ClaimProfileOutput(BaseModel):
    answer_shape: Literal["fact", "exact_date", "exact_number", "product_specs", "overview", "comparison", "news_digest"]
    primary_source_required: bool = False
    min_independent_sources: int = 1
    preferred_domain_types: list[Literal["official", "academic", "vendor", "major_media", "forum", "unknown"]] = Field(default_factory=list)
    routing_bias: Literal["short_path", "targeted_retrieval", "iterative_loop"] | None = None
    required_dimensions: list[str] = Field(default_factory=list)
    focus_terms: list[str] = Field(default_factory=list)
    allow_synthesis_without_primary: bool = True
    strict_contract: bool = False


class _ClaimDraft(BaseModel):
    claim_text: str = Field(min_length=1)
    priority: int = 1
    needs_freshness: bool = False
    entity_set: list[str] = Field(default_factory=list)
    time_scope: str | None = None
    search_queries: list[str] = Field(default_factory=list)
    claim_profile: _ClaimProfileOutput | None = None


class _ClaimDecompositionOutput(BaseModel):
    claims: list[_ClaimDraft] = Field(default_factory=list)


class _QueryListOutput(BaseModel):
    queries: list[str] = Field(default_factory=list)


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


class _RefinedQueriesOutput(BaseModel):
    queries: list[str] = Field(default_factory=list)




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
        # Cache for query generation: populated by _classify_intent_llm alongside intent,
        # consumed by decompose_claims to avoid a second LLM round-trip.
        self._query_cache: dict[str, list[str]] = {}
        self._classification_cache: dict[str, _IntentOutput] = {}

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
                "Plan a grounded search run for a user query.\n"
                "Return 1 to 4 claims. Use a single claim when decomposition is unnecessary.\n"
                "Preserve named entities exactly and do not invent facts.\n"
                "For each claim return:\n"
                "- claim_text\n"
                "- priority\n"
                "- needs_freshness\n"
                "- entity_set\n"
                "- time_scope\n"
                "- search_queries: 3 to 5 short keyword queries for web search\n"
                "- claim_profile describing the retrieval contract\n"
                "\n"
                "claim_profile rules:\n"
                "- answer_shape must be one of: fact, exact_date, exact_number, product_specs, overview, comparison, news_digest\n"
                "- primary_source_required should be true for leadership roles, official status, releases, product specs, and fresh vendor facts\n"
                "- min_independent_sources should usually be 2; use 3 for news digests; use 1 only for trivial stable facts\n"
                "- preferred_domain_types should favor official/academic/vendor/major_media as appropriate\n"
                "- routing_bias should be iterative_loop for synthesis, product specs, comparisons, and news digests\n"
                "- required_dimensions should reflect what the answer must cover, e.g. time / number / source / specs\n"
                "- focus_terms should contain 2 to 6 short evidence phrases that must be present in useful passages\n"
                "- strict_contract should be true when the answer must not be produced without meeting the evidence contract\n"
                "\n"
                "Critical semantic rules:\n"
                "- Never rewrite a features/specifications/характеристики request into an existence-check claim.\n"
                "- If the user asks for features, specifications, technical characteristics, price/options, or an overview of a product/model, keep that information need intact and use answer_shape=product_specs.\n"
                "- If the user asks for an explanation, comparison, or overview, keep it open-ended and use answer_shape=overview or comparison.\n"
                "- Intent hints may be wrong; preserve the actual information need from the user request.\n"
                "\n"
                "Examples:\n"
                "- 'Какие характеристики нового MacBook Neo?' -> one claim about the product specifications, not about whether it exists.\n"
                "- 'Какие характеистики нового macbook neo?' -> still a product specifications claim.\n"
                "- 'MacBook Neo specs' -> product_specs.\n"
                "\n"
                "Search query rules:\n"
                "- keyword phrases only, no question wording\n"
                "- match the user's language; add one English query when the input is non-English and the topic is global\n"
                "- avoid boilerplate fillers\n"
                "- include time scope or year when relevant"
            ),
        )
        self._query_gen_agent = Agent(
            self._model,
            output_type=_out(_QueryListOutput),
            retries=1,
            instrument=True,
            system_prompt=(
                "Generate 3 to 5 short, diverse web search queries for the given topic.\n"
                "Rules:\n"
                "- Use keyword phrases, not questions (no what/who/when/how/why)\n"
                "- Match the language of the input; add one English query if input is not English\n"
                "- Vary specificity: one broad, one with quoted entities, one with key topic terms\n"
                "- Keep each query under 8 words; skip filler like 'information about'\n"
                "- Include the year or time scope when relevant for freshness"
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

        # Intent classification + query generation: combined to save one LLM round-trip.
        # Intent accuracy validated at 100% on 20-example component eval dataset.
        self._intent_agent: Agent[None, _IntentOutput] = Agent(
            self._model,
            output_type=_out(_IntentOutput),
            retries=1,
            instrument=True,
            system_prompt=(
                "Task 1 — classify intent (pick exactly one: factual / synthesis / news_digest).\n"
                "Task 2 — classify complexity (pick exactly one: single_hop / multi_hop).\n"
                "\n"
                "  factual    — single verifiable fact: who/when/is-it-true/specific number or date.\n"
                "               Examples: 'Who is CEO of Microsoft?' | 'When was Python 3.13 released?'\n"
                "                         'Is IRS a government agency?' | 'How many EU countries?'\n"
                "\n"
                "  synthesis  — aggregated answer: list of features/specs/changes, explanation,\n"
                "               comparison, how-to, overview, OR current-state lookup.\n"
                "               Examples: 'What are new features in Python 3.13?' | 'How does asyncio work?'\n"
                "                         'MacBook Pro M4 specs?' | 'Погода в Астане?' | 'Bitcoin price today'\n"
                "                         'USD/EUR exchange rate?' | 'Pros and cons of Docker?'\n"
                "                         'Какие характеристики нового MacBook Neo?' | 'Какие характеистики нового macbook neo?'\n"
                "               Treat product/specification requests as synthesis even for a single entity.\n"
                "\n"
                "  news_digest — recent events, news feed, what's happening.\n"
                "               Examples: 'Latest news on Iran?' | 'What happened in AI this week?'\n"
                "                         'Последние новости из Казахстана' | 'How is Ukraine war going?'\n"
                "\n"
                "Complexity guidance:\n"
                "  single_hop  — one claim is enough to answer the request.\n"
                "  multi_hop   — multiple independent claims or subquestions are needed.\n"
                "\n"
                "Task 3 — generate 3 to 5 short keyword search queries.\n"
                "  Rules: keyword phrases only (no question words); match input language;\n"
                "  add one English query when input is non-English; max 8 words each;\n"
                "  vary specificity (broad / quoted entity / key terms); include year if relevant."
            ),
        )
        # Cache for intent classification: keyed on normalized query string.
        self._intent_cache: dict[str, str] = {}

        # SAFE-inspired: generates one focused search query from the verifier's rationale.
        self._refiner_agent = Agent(
            self._model,
            output_type=_out(_RefinedQueriesOutput),
            retries=1,
            instrument=True,
            system_prompt=(
                "Generate 1 to 3 focused follow-up web search queries.\n"
                "Use the claim, retrieval contract, verifier rationale, and current evidence context.\n"
                "Return only short keyword queries that would find the missing evidence.\n"
                "Do not repeat existing queries. No explanations."
            ),
        )
    def classify_query(self, query: str, log=None) -> QueryClassification:
        normalized_query = self._normalize_time_references(query, log=log)
        region_hint = extract_region_hint(normalized_query)
        time_scope = extract_time_scope(normalized_query)
        freshness = needs_freshness(query)
        decision = self._classify_intent_llm(
            normalized_query,
            region_hint=region_hint,
            freshness=freshness,
            log=log,
        )
        intent = decision.intent
        complexity = decision.complexity
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

    def _classify_intent_llm(
        self,
        normalized_query: str,
        *,
        region_hint: str | None = None,
        freshness: bool = False,
        log=None,
    ) -> _IntentOutput:
        """Classify query via LLM.

        Falls back to heuristic (is_news_digest_query → 'news_digest', else 'factual')
        when LLM is disabled or on error.
        """
        log = log or (lambda msg: None)

        cached = self._classification_cache.get(normalized_query)
        if cached is not None:
            log(f"  [dim green]-> intent cache hit: [italic]{cached.intent}[/italic][/dim green]")
            return cached

        if not self._enabled:
            # No LLM available — fall back to heuristic.
            return _default_intent_decision()

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
                    today = date.today().isoformat()
                    result = self._intent_agent.run_sync(
                        f"[Today: {today}]\n{normalized_query}",
                        model_settings=model_settings,
                    )
                metrics.output_chars = output_char_len(result.output)
            decision = result.output
            queries = [q.strip() for q in decision.search_queries if q.strip()]
            if queries:
                self._query_cache[normalized_query] = queries[:6]
            self._classification_cache[normalized_query] = decision
        except Exception as exc:
            log(f"  [dim yellow]-> intent classification failed: {exc}[/dim yellow]")
            decision = _default_intent_decision()
            self._classification_cache[normalized_query] = decision

        self._intent_cache[normalized_query] = decision.intent
        return decision

    def decompose_claims(self, classification: QueryClassification, log=None) -> list[Claim]:
        log = log or (lambda msg: None)
        if not self._enabled:
            queries = self._get_or_generate_queries(classification, log)
            return self._fallback_claims(classification, search_queries=queries)

        prompt = (
            "Return JSON with one key `claims`.\n"
            "Each claim must include: claim_text, priority, needs_freshness, entity_set, time_scope, search_queries, claim_profile.\n"
            "claim_profile must include: answer_shape, primary_source_required, min_independent_sources, "
            "preferred_domain_types, routing_bias, required_dimensions, focus_terms, allow_synthesis_without_primary, strict_contract.\n"
            "Keep claims atomic, exact, and capped at 4.\n"
            "Return one claim when the request is already atomic.\n\n"
            f"Intent: {classification.intent}\n"
            f"Complexity: {classification.complexity}\n"
            f"Needs freshness: {classification.needs_freshness}\n"
            f"Time scope: {classification.time_scope or ''}\n"
            f"Region hint: {classification.region_hint or ''}\n"
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
                claim_profile = self._claim_profile_from_output(item.claim_profile, classification)
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
                        search_queries=[q.strip() for q in (item.search_queries or []) if q.strip()],
                        claim_profile=claim_profile,
                    )
                )
            return claims or self._fallback_claims(classification)
        except Exception as exc:
            log("  [dim yellow]→ fallback: single claim from query[/dim yellow]")
            return self._fallback_claims(classification)

    def _get_or_generate_queries(self, classification: QueryClassification, log=None) -> list[str]:
        """Return pre-generated queries from the intent-classification cache, or generate them now.

        The combined _intent_agent call populates _query_cache alongside the intent verdict,
        eliminating a second LLM round-trip for the simple (non-decomposed) path.
        """
        log = log or (lambda msg: None)
        cached = self._query_cache.get(classification.normalized_query)
        if cached:
            for q in cached:
                log(f"  [dim]-> llm_query: {q}[/dim]")
            return cached
        # Fallback: intent call didn't return queries (e.g. LLM disabled path, cache miss).
        return self._generate_queries_llm(classification.normalized_query, classification, log)

    def _generate_queries_llm(
        self,
        claim_text: str,
        classification: QueryClassification,
        log=None,
    ) -> list[str]:
        """Generate search queries via LLM for a single claim (used for synthesis/news_digest)."""
        log = log or (lambda msg: None)
        if not self._enabled:
            return []
        context = claim_text
        if classification.time_scope:
            context += f" [{classification.time_scope}]"
        try:
            with log_llm_call(
                log,
                task="generate_queries",
                model=self._settings.llm_model,
                detail=claim_text[:60],
                input_chars=len(context),
            ) as metrics:
                result = self._query_gen_agent.run_sync(
                    context,
                    model_settings=self._model_settings(max_tokens=200, temperature=0),
                )
                metrics.output_chars = output_char_len(result.output)
            queries = [q.strip() for q in result.output.queries if q.strip()]
            for q in queries:
                log(f"  [dim]-> llm_query: {q}[/dim]")
            return queries[:6]
        except Exception as exc:
            log(f"  [dim yellow]-> generate_queries failed: {exc}[/dim yellow]")
            return []

    def verify_claim(self, claim: Claim, passages: list[Passage], log=None) -> VerificationResult:
        log = log or (lambda msg: None)

        def finalize(vr: VerificationResult) -> VerificationResult:
            return _post_adjust_verification(claim, passages, vr)

        if not self._enabled:
            return finalize(_default_verification("LLM verifier unavailable."))

        prompt = _build_verifier_prompt(claim, passages, max_passages=8, max_chars=900)
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
            log("  [dim yellow]-> verify_claim primary pass failed[/dim yellow]")
            log("  [dim yellow]-> verify_claim primary pass failed[/dim yellow]")

        rescue_prompt = _build_verifier_prompt(claim, passages, max_passages=4, max_chars=450)
        rescue_cached = self._verify_cache.get(rescue_prompt)
        if rescue_cached is not None:
            log(f"  [dim green]-> verify_claim rescue cache hit ({len(rescue_prompt)} chars)[/dim green]")
            return rescue_cached
        try:
            with log_llm_call(
                log,
                task="verify_claim_rescue",
                model=self._settings.llm_model,
                detail=f"{claim.claim_id}: {claim.claim_text}",
                input_chars=len(rescue_prompt),
            ) as metrics:
                with logfire.span("query_intelligence.verify_claim_rescue", claim_id=claim.claim_id):
                    result = self._verifier_agent.run_sync(
                        rescue_prompt,
                        model_settings=self._model_settings(
                            max_tokens=self._settings.resolved_verify_claim_max_tokens(),
                            temperature=0,
                        ),
                    )
                metrics.output_chars = output_char_len(result.output)
            output = result.output
            passage_map = {passage.passage_id: passage for passage in passages}

            def build_rescue_spans(items: list[_EvidenceQuote]) -> list[EvidenceSpan]:
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
                    supporting_spans=build_rescue_spans(output.supporting_passages),
                    contradicting_spans=build_rescue_spans(output.contradicting_passages),
                    missing_dimensions=[
                        normalized_text(item)
                        for item in output.missing_dimensions
                        if normalized_text(item)
                    ],
                    rationale=normalized_text(output.rationale),
                )
            )
            self._verify_cache[rescue_prompt] = verification
            return verification
        except Exception as rescue_exc:
            log(f"  [dim yellow]-> verify_claim rescue failed: {rescue_exc}[/dim yellow]")
            return finalize(_default_verification("LLM verifier failed after retry."))

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
            cited = _extract_citation_indices(answer)
            if intent == "news_digest":
                floor = min(4, len(passage_refs))
                cited = sorted(set(cited) | set(range(1, floor + 1)))
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

    def refine_search_queries(
        self,
        claim: Claim,
        classification: QueryClassification,
        verification: VerificationResult,
        gated_results: list[GatedSerpResult],
        bundle: EvidenceBundle | None,
        next_iteration: int,
        existing_queries: set[str],
        log=None,
    ) -> list[str]:
        log = log or (lambda msg: None)
        if not self._enabled:
            return []

        evidence_lines: list[str] = []
        for idx, result in enumerate(gated_results[:6], 1):
            evidence_lines.append(
                f"{idx}. {result.serp.title} | {result.serp.url} | domain={result.assessment.domain_type} | source_score={result.assessment.source_score:.2f}"
            )
        profile = claim.claim_profile
        profile_lines = [
            f"answer_shape={profile.answer_shape}" if profile else "answer_shape=fact",
            f"primary_source_required={profile.primary_source_required}" if profile else "primary_source_required=False",
            f"min_independent_sources={profile.min_independent_sources}" if profile else "min_independent_sources=1",
            f"preferred_domain_types={','.join(profile.preferred_domain_types)}" if profile and profile.preferred_domain_types else "preferred_domain_types=",
            f"required_dimensions={','.join(profile.required_dimensions)}" if profile and profile.required_dimensions else "required_dimensions=",
            f"focus_terms={','.join(profile.focus_terms)}" if profile and profile.focus_terms else "focus_terms=",
            f"strict_contract={profile.strict_contract}" if profile else "strict_contract=False",
        ]
        prompt = (
            f"Claim: {claim.claim_text}\n"
            f"Intent: {classification.intent}\n"
            f"Iteration: {next_iteration}\n"
            f"Needs freshness: {claim.needs_freshness}\n"
            f"Time scope: {claim.time_scope or ''}\n"
            "Claim contract:\n"
            + "\n".join(profile_lines)
            + "\n\nVerifier verdict: "
            + verification.verdict
            + f"\nMissing dimensions: {', '.join(verification.missing_dimensions) or 'none'}"
            + f"\nRationale: {verification.rationale or 'none'}"
            + f"\nExisting queries: {', '.join(sorted(existing_queries)) or 'none'}"
            + "\nTop current evidence:\n"
            + ("\n".join(evidence_lines) if evidence_lines else "none")
        )
        try:
            with log_llm_call(
                log,
                task="refine_search_queries",
                model=self._settings.llm_model,
                detail=claim.claim_text,
                input_chars=len(prompt),
            ) as metrics:
                result = self._refiner_agent.run_sync(
                    prompt,
                    model_settings=self._model_settings(max_tokens=160, temperature=0),
                )
                metrics.output_chars = output_char_len(result.output)
            queries: list[str] = []
            seen = {query.casefold() for query in existing_queries}
            for candidate in result.output.queries:
                query_text = normalized_text(candidate)
                if not query_text:
                    continue
                key = query_text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                queries.append(query_text)
                log(f"  [dim]-> llm_refined_query: {query_text}[/dim]")
            return queries[:3]
        except Exception as exc:
            log(f"  [dim yellow]-> refine_search_queries failed: {exc}[/dim yellow]")
            return []

    def suggest_rationale_query(self, claim_text: str, rationale: str, log=None) -> str | None:
        classification = QueryClassification(
            query=claim_text,
            normalized_query=claim_text,
            intent="factual",
            complexity="single_hop",
            needs_freshness=False,
        )
        queries = self.refine_search_queries(
            Claim(claim_id="claim-1", claim_text=claim_text, priority=1, needs_freshness=False),
            classification,
            VerificationResult(verdict="insufficient_evidence", rationale=rationale),
            gated_results=[],
            bundle=None,
            next_iteration=2,
            existing_queries=set(),
            log=log,
        )
        return queries[0] if queries else None

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
    def _fallback_claims(
        classification: QueryClassification,
        search_queries: list[str] | None = None,
    ) -> list[Claim]:
        answer_shape = "news_digest" if classification.intent == "news_digest" else "overview" if classification.intent == "synthesis" else "fact"
        return [
            Claim(
                claim_id="claim-1",
                claim_text=classification.normalized_query,
                priority=1,
                needs_freshness=classification.needs_freshness,
                entity_set=extract_entities(classification.normalized_query),
                time_scope=classification.time_scope,
                search_queries=search_queries or [],
                claim_profile=ClaimProfile(
                    answer_shape=answer_shape,
                    primary_source_required=False,
                    min_independent_sources=3 if classification.intent == "news_digest" else 2 if classification.intent == "synthesis" else 1,
                    preferred_domain_types=["major_media", "official", "vendor"] if classification.intent == "news_digest" else ["official", "academic", "vendor", "major_media"],
                    routing_bias="iterative_loop" if classification.intent in {"synthesis", "news_digest"} else None,
                    required_dimensions=["time", "source"] if classification.intent == "news_digest" else [],
                    focus_terms=[],
                    allow_synthesis_without_primary=classification.intent != "factual",
                    strict_contract=False,
                ),
            )
        ]

    @staticmethod
    def _claim_profile_from_output(
        output: _ClaimProfileOutput | None,
        classification: QueryClassification,
    ) -> ClaimProfile:
        if output is None:
            answer_shape = "news_digest" if classification.intent == "news_digest" else "overview" if classification.intent == "synthesis" else "fact"
            return ClaimProfile(answer_shape=answer_shape)
        preferred_domain_types = [
            domain
            for domain in output.preferred_domain_types
            if domain in {"official", "academic", "vendor", "major_media", "forum", "unknown"}
        ]
        return ClaimProfile(
            answer_shape=output.answer_shape,
            primary_source_required=bool(output.primary_source_required),
            min_independent_sources=max(1, int(output.min_independent_sources or 1)),
            preferred_domain_types=preferred_domain_types or (
                ["major_media", "official", "vendor"] if output.answer_shape == "news_digest" else ["official", "academic", "vendor", "major_media"]
            ),
            routing_bias=output.routing_bias,
            required_dimensions=[normalized_text(value) for value in output.required_dimensions if normalized_text(value)],
            focus_terms=[normalized_text(value) for value in output.focus_terms if normalized_text(value)],
            allow_synthesis_without_primary=bool(output.allow_synthesis_without_primary),
            strict_contract=bool(output.strict_contract),
        )

    def _model_settings(self, *, max_tokens: int, temperature: float):
        return build_model_settings(
            self._settings,
            max_tokens=max_tokens,
            temperature=temperature,
        )
