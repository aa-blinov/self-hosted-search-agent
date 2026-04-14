from __future__ import annotations

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
from search_agent.application.claim_policy import (
    claim_profile_lines as _claim_profile_lines,
    post_adjust_verification as _post_adjust_verification,
)
from search_agent.application.text_heuristics import (
    clamp,
    extract_entities,
    extract_numbers,
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


def _normalize_citation_groups(answer: str) -> str:
    text = answer or ""
    normalized: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "[":
            normalized.append(text[index])
            index += 1
            continue
        end = text.find("]", index + 1)
        if end == -1:
            normalized.append(text[index])
            index += 1
            continue
        content = text[index + 1:end]
        values: list[str] = []
        current = ""
        valid = True
        for ch in content:
            if ch.isdigit():
                current += ch
                continue
            if ch in {",", ";", " "}:
                if current:
                    values.append(current)
                    current = ""
                continue
            valid = False
            break
        if current:
            values.append(current)
        if valid and len(values) > 1:
            normalized.extend(f"[{value}]" for value in values)
        else:
            normalized.append(text[index : end + 1])
        index = end + 1
    return "".join(normalized)


def _default_intent_decision() -> "_IntentOutput":
    return _IntentOutput(intent="factual", complexity="single_hop", search_queries=[])


def _default_verification(reason: str) -> VerificationResult:
    return VerificationResult(
        verdict="insufficient_evidence",
        confidence=0.0,
        missing_dimensions=["coverage"],
        rationale=reason,
    )


def _digit_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    current: list[str] = []
    for ch in text or "":
        if ch.isalnum() or ch in {".", "-", "/"}:
            current.append(ch)
            continue
        if current:
            token = "".join(current).strip(".,-/")
            if token and any(char.isdigit() for char in token):
                tokens.add(token.casefold())
            current = []
    if current:
        token = "".join(current).strip(".,-/")
        if token and any(char.isdigit() for char in token):
            tokens.add(token.casefold())
    return tokens


def _preserve_original_factual_claim(
    classification: QueryClassification,
    claim_text: str,
    claim_profile: ClaimProfile | None = None,
) -> str:
    if classification.intent != "factual":
        return claim_text
    single_hop = classification.complexity != "multi_hop"
    original = normalized_text(classification.normalized_query)
    if not original:
        return claim_text
    if single_hop and classification.time_scope and normalized_text(classification.time_scope).casefold() not in claim_text.casefold():
        return original
    query_numbers = set(extract_numbers(original)) | _digit_tokens(original)
    claim_numbers = set(extract_numbers(claim_text)) | _digit_tokens(claim_text)
    if single_hop and claim_profile is not None and claim_profile.answer_shape in {"exact_date", "exact_number"}:
        extra_numbers = {value for value in claim_numbers if value not in query_numbers}
        if extra_numbers:
            return original
    if single_hop and query_numbers and not query_numbers.issubset(claim_numbers):
        return original
    return claim_text


def _verdict_guidance_for_shape(claim: Claim) -> str:
    shape = claim.claim_profile.answer_shape if claim.claim_profile else "fact"
    if shape in ("overview", "comparison", "news_digest", "product_specs"):
        return (
            "This is an open-ended claim. Collective coverage across multiple passages is allowed.\n"
            "You may return supported when the passages collectively address the main aspects of the claim, "
            "even if not every detail is covered.\n"
            "Return insufficient_evidence only when the passages fail to address the core of the claim.\n"
        )
    return (
        "For open-ended claims, collective coverage across multiple passages is allowed.\n"
        "For explanatory overviews, you may return supported when the passages directly explain the concept or mechanism.\n"
        "For list-like overviews or comparisons that require multiple features, differences, highlights, or options, keep the claim-level verdict as insufficient_evidence and let synthesis compose the final answer later.\n"
    )


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
        "Before producing the JSON verdict, briefly summarize the key evidence from the passages "
        "in the rationale field — what supports, what contradicts, and what is missing.\n"
        + _verdict_guidance_for_shape(claim)
        + "For simple identity, classification, membership, or agency-status claims, you may return supported when an official or otherwise strong passage directly states the relationship and a second strong passage corroborates it.\n"
        "For agency-status or institutional-affiliation claims, an official passage that identifies the entity as a bureau, office, component, or service within a government department counts as explicit support for government-agency classification.\n"
        "If the contract requires a primary source or multiple independent sources and the evidence set does not satisfy that contract, return insufficient_evidence.\n"
        "Return supporting_passages and contradicting_passages with short quotes.\n"
        "Use missing_dimensions from time, entity, number, location, source, coverage.\n\n"
        f"Claim: {claim.claim_text}\n"
        "Claim contract:\n"
        + "\n".join(_claim_profile_lines(claim))
        + "\n\nPassages:\n\n"
        + "\n\n".join(prompt_lines)
    )


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
    needs_broad_retrieval: bool = False
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
                "Use only explicit evidence from provided passages. "
                "Explanatory overviews may be supported, but list-like overviews and comparisons should usually remain insufficient_evidence at claim level. "
                "Simple identity or classification claims may be supported when one strong passage states the relationship and another strong passage corroborates it. "
                "For agency-status claims, official bureau or departmental affiliation statements count as explicit classification support."
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
                "When multiple sources are available for synthesis or comparisons, ground the answer in at least two distinct sources. "
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
                "               Use this for identity, role, institutional status, exact date,\n"
                "               and exact-number questions that can be answered with one grounded claim.\n"
                "\n"
                "  synthesis  — aggregated answer: list of features/specs/changes, explanation,\n"
                "               comparison, how-to, overview, OR current-state lookup.\n"
                "               Use this for open-ended explanations, comparisons, feature lists,\n"
                "               product specifications, pros/cons, how-to guidance, and live state lookups\n"
                "               such as weather, prices, or exchange rates.\n"
                "               Treat product/specification requests as synthesis even for a single entity.\n"
                "\n"
                "  news_digest — recent events, news feed, what's happening.\n"
                "               Use this for recent developments, timelines of events, and requests for\n"
                "               a roundup of what happened in a place, topic, or industry.\n"
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
        self._claim_agent = Agent(
            self._model,
            output_type=_out(_ClaimDecompositionOutput),
            retries=1,
            instrument=True,
            system_prompt=(
                "Plan a grounded search run for a user query.\n"
                "Return 1 to 4 claims. Use a single claim when decomposition is unnecessary.\n"
                "Preserve named entities exactly and do not invent facts.\n"
                "Each claim_text must be fully self-contained: resolve all pronouns, references, and implicit entities "
                "to explicit names from the original query. A claim must be understandable without seeing the query.\n"
                "For each claim return: claim_text, priority, needs_freshness, entity_set, time_scope, search_queries, claim_profile.\n"
                "\n"
                "claim_profile rules:\n"
                "- answer_shape must be one of: fact, exact_date, exact_number, product_specs, overview, comparison, news_digest\n"
                "- overview/comparison/news_digest are open-ended contracts: primary_source_required=false and strict_contract=false\n"
                "- news_digest must use min_independent_sources>=3 and needs_broad_retrieval=true\n"
                "- product_specs must use primary_source_required=true, min_independent_sources>=2, allow_synthesis_without_primary=false, needs_broad_retrieval=true, and strict_contract=true\n"
                "- exact event-anchored number lookups should prefer needs_broad_retrieval=true and make the requested measurement the focus_terms\n"
                "- for list-like overviews, use explicit required_dimensions such as feature_list, improvements, changes, or highlights\n"
                "- focus_terms must describe the requested evidence itself, not generic context words\n"
                "- do not use placeholder strings such as exact_date, exact_number, event details, or announcement details inside claim_text or focus_terms\n"
                "\n"
                "Semantic rules:\n"
                "- Never rewrite a features or specifications request into an existence-check claim.\n"
                "- Do not rewrite explanation or mechanism questions into a feature-list request.\n"
                "- Keep explanation, comparison, and overview requests open-ended.\n"
                "- Never answer the claim inside claim_text. Do not inject candidate dates, numbers, or facts that were not already present in the user request.\n"
                "- Intent hints may be wrong; preserve the actual information need from the user request.\n"
                "- Simple factual classification or affiliation claims should remain fact-shaped and prefer official evidence when the contract depends on institutional status.\n"
                "- Mechanism or explanation requests must remain explanatory rather than being rewritten into feature lists.\n"
                "- Generic stable numeric facts should keep required_dimensions focused on number and should not gain event_context unless the number depends on a specific event or dated situation.\n"
                "- Event-anchored exact-number requests may use event_context and iterative retrieval when the requested value depends on a particular event, announcement, or circumstance.\n"
                "\n"
                "Search query rules:\n"
                "- keyword phrases only, no question wording\n"
                "- match the user's language; add one English query when the input is non-English and the topic is global\n"
                "- avoid boilerplate fillers\n"
                "- for official classification or affiliation claims, target organizational structure, statutory authority, bureau, or parent-department pages instead of vague status wording\n"
                "- include time scope or year when relevant"
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
        """Classify query via LLM and return a structured intent decision."""
        log = log or (lambda msg: None)

        cached = self._classification_cache.get(normalized_query)
        if cached is not None:
            log(f"  [dim green]-> intent cache hit: [italic]{cached.intent}[/italic][/dim green]")
            return cached

        if not self._enabled:
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
            "preferred_domain_types, needs_broad_retrieval, required_dimensions, focus_terms, allow_synthesis_without_primary, strict_contract.\n"
            "Keep claims atomic, exact, and capped at 4.\n"
            "Return one claim when the request is already atomic.\n"
            "For overview, comparison, and news_digest claims, keep the contract open-ended and set strict_contract=false.\n"
            "For product_specs claims, keep the specs request intact and require primary-source evidence.\n"
            "For explanation or mechanism questions, keep the mechanism/explanation need intact and do not rewrite into a feature-list request.\n"
            "For simple factual classification or membership claims such as agency status, official role, or institutional affiliation, prefer official sources, require primary-source evidence, and require at least two independent sources.\n"
            "For generic exact-number facts such as scientific constants, measurements under standard conditions, or stable reference values, keep required_dimensions centered on number, avoid event_context, and do not require a primary source by default.\n"
            "For exact event-anchored number lookups, use focus_terms for the requested measurement and prefer needs_broad_retrieval=true.\n"
            "For exact event-anchored number lookups, include event_context in required_dimensions when the number depends on a specific event or situation.\n"
            "Never insert a candidate answer into claim_text. Keep exact_date and exact_number claims unresolved unless that date or number was already stated in the user request.\n"
            "Never emit placeholder phrases like exact_date, exact_number, event details, or announcement details.\n\n"
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
                claim_text = _preserve_original_factual_claim(
                    classification,
                    normalized_text(item.claim_text),
                    claim_profile,
                )
                claims.append(
                    Claim(
                        claim_id=f"claim-{idx}",
                        claim_text=claim_text,
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
            answer = _normalize_citation_groups((result.output or "").strip())
            if not answer:
                return ""
            # Append sources section for any [N] references cited in the answer.
            cited = _extract_citation_indices(answer)
            floor_indices: list[int] = []
            seen_urls: set[str] = set()
            required_unique = 4 if intent == "news_digest" else 2 if intent == "synthesis" else 1
            total_unique_urls = len({url for _, url in passage_refs if url})
            for idx, (_, url) in enumerate(passage_refs, 1):
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                floor_indices.append(idx)
                if len(seen_urls) >= min(required_unique, total_unique_urls):
                    break
            if intent in {"news_digest", "synthesis"}:
                cited = sorted(set(cited) | set(floor_indices))
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
        passage_lines: list[str] = []
        if bundle and bundle.considered_passages:
            for p in bundle.considered_passages[:3]:
                snippet = (p.text or "")[:100].replace("\n", " ")
                passage_lines.append(f"- {p.title}: {snippet}")

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
            + "\nTop retrieved passages:\n"
            + ("\n".join(passage_lines) if passage_lines else "none")
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
                    needs_broad_retrieval=classification.intent in {"synthesis", "news_digest"},
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

        answer_shape = output.answer_shape
        preferred_domain_types = [
            domain
            for domain in output.preferred_domain_types
            if domain in {"official", "academic", "vendor", "major_media", "forum", "unknown"}
        ]
        required_dimensions: list[str] = []
        seen_dimensions: set[str] = set()
        for value in output.required_dimensions:
            normalized = normalized_text(value)
            key = normalized.casefold()
            if not normalized or key in seen_dimensions:
                continue
            seen_dimensions.add(key)
            required_dimensions.append(normalized)

        focus_terms: list[str] = []
        seen_focus: set[str] = set()
        for value in output.focus_terms:
            normalized = normalized_text(value)
            key = normalized.casefold()
            if not normalized or key in seen_focus:
                continue
            seen_focus.add(key)
            focus_terms.append(normalized)

        primary_source_required = bool(output.primary_source_required)
        min_independent_sources = max(1, int(output.min_independent_sources or 1))
        needs_broad_retrieval = bool(output.needs_broad_retrieval)
        allow_synthesis_without_primary = bool(output.allow_synthesis_without_primary)
        strict_contract = bool(output.strict_contract)

        if answer_shape in {"overview", "comparison"}:
            primary_source_required = False
            min_independent_sources = max(2, min_independent_sources)
            needs_broad_retrieval = True
            allow_synthesis_without_primary = True
            strict_contract = False
        elif answer_shape == "fact":
            lower_dimensions = [value.casefold() for value in required_dimensions]
            official_fact_markers = ("agency", "affiliation", "membership", "leadership", "title", "office", "role", "classification")
            if "official" in preferred_domain_types and any(
                marker in value
                for value in lower_dimensions
                for marker in official_fact_markers
            ):
                primary_source_required = True
                min_independent_sources = max(2, min_independent_sources)
                allow_synthesis_without_primary = False
                strict_contract = True
        elif answer_shape == "news_digest":
            primary_source_required = False
            min_independent_sources = max(3, min_independent_sources)
            needs_broad_retrieval = True
            allow_synthesis_without_primary = True
            strict_contract = False
            required_dimensions = list(dict.fromkeys(required_dimensions + ["time", "source", "event"]))
        elif answer_shape == "product_specs":
            primary_source_required = True
            min_independent_sources = max(2, min_independent_sources)
            needs_broad_retrieval = True
            allow_synthesis_without_primary = False
            strict_contract = True
            required_dimensions = list(dict.fromkeys(required_dimensions + ["specs", "source"]))
        elif answer_shape == "exact_date":
            if "source" not in required_dimensions and len(required_dimensions) <= 1:
                min_independent_sources = 1
                needs_broad_retrieval = False
                strict_contract = False
        elif answer_shape == "exact_number":
            if "number" not in required_dimensions:
                required_dimensions.append("number")
            lower_dimensions = [value.casefold() for value in required_dimensions]
            query_entities = {
                normalized_text(entity).casefold()
                for entity in extract_entities(classification.normalized_query)
                if normalized_text(entity)
            }
            has_event_anchor_signal = bool(classification.time_scope) or len(query_entities) >= 2
            event_like_number = any(
                marker in value
                for value in lower_dimensions
                for marker in ("event", "context", "time")
            )
            if event_like_number and not has_event_anchor_signal:
                required_dimensions = [
                    value
                    for value in required_dimensions
                    if not any(marker in value.casefold() for marker in ("event", "context"))
                ]
                lower_dimensions = [value.casefold() for value in required_dimensions]
                primary_source_required = False
                min_independent_sources = 1
                needs_broad_retrieval = False
                allow_synthesis_without_primary = True
                strict_contract = False
                event_like_number = False
            if event_like_number:
                min_independent_sources = max(2, min_independent_sources)
                needs_broad_retrieval = True
                strict_contract = True
            elif not primary_source_required and not event_like_number:
                min_independent_sources = 1
                needs_broad_retrieval = False
                strict_contract = False
            elif "source" in lower_dimensions and len(required_dimensions) >= 2:
                needs_broad_retrieval = True
                strict_contract = True
            elif "time" not in lower_dimensions:
                min_independent_sources = 1
                needs_broad_retrieval = False
                strict_contract = False
            else:
                strict_contract = strict_contract or "number" in lower_dimensions

        return ClaimProfile(
            answer_shape=answer_shape,
            primary_source_required=primary_source_required,
            min_independent_sources=min_independent_sources,
            preferred_domain_types=preferred_domain_types or (
                ["major_media", "official", "vendor"]
                if answer_shape == "news_digest"
                else ["official", "academic", "vendor", "major_media"]
            ),
            needs_broad_retrieval=needs_broad_retrieval,
            required_dimensions=required_dimensions,
            focus_terms=focus_terms,
            allow_synthesis_without_primary=allow_synthesis_without_primary,
            strict_contract=strict_contract,
        )

    def _model_settings(self, *, max_tokens: int, temperature: float):
        return build_model_settings(
            self._settings,
            max_tokens=max_tokens,
            temperature=temperature,
        )
