from __future__ import annotations

from datetime import date
from typing import Literal

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.output import PromptedOutput

from search_agent.domain.assessment import Assessment, KeyClaim
from search_agent.domain.models import (
    Claim,
    DomainType,
    EvidenceSpan,
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


# Patterns indicating an explanation/comparison/how-to query — these are
# unambiguously synthesis intent. The LLM classifier occasionally mislabels
# them as factual on cross-language inputs (e.g. Russian "Как работает X"),
# which routes them through the wrong decompose+verify path. We defensively
# override factual -> synthesis when the normalized query starts with one
# of these phrases.
_SYNTHESIS_PATTERNS: tuple[str, ...] = (
    # Russian
    "как работает",
    "как работают",
    "как использовать",
    "как устроен",
    "как устроена",
    "как устроено",
    "что такое",
    "чем отличается",
    "чем отличаются",
    "в чем разница",
    "в чём разница",
    "объясни",
    "опиши",
    "сравни",
    "расскажи про",
    "расскажи о",
    # English
    "how does",
    "how do",
    "how to",
    "what is",
    "what are",
    "explain",
    "describe",
    "compare",
    "difference between",
    "differences between",
    "what's the difference",
    "what is the difference",
)


def _looks_like_synthesis(query: str) -> bool:
    q = (query or "").casefold().strip()
    if not q:
        return False
    return any(q.startswith(pattern) for pattern in _SYNTHESIS_PATTERNS)


def _default_verification(reason: str) -> VerificationResult:
    return VerificationResult(
        verdict="insufficient_evidence",
        confidence=0.0,
        missing_dimensions=["coverage"],
        rationale=reason,
    )


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


class _KeyClaimOutput(BaseModel):
    text: str = Field(min_length=1)
    supporting_citation_numbers: list[int] = Field(default_factory=list)


class _AssessmentOutput(BaseModel):
    answer: str = Field(default="")
    key_claims: list[_KeyClaimOutput] = Field(default_factory=list)
    confidence: float = 0.0
    gaps: list[str] = Field(default_factory=list)
    contradicts_query: bool = False


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

        # Unified pipeline: writes final answer + self-assessment in a single call.
        # Replaces verify_claim + synthesize_answer + compose_answer in the iterative
        # runner.  The output includes key_claims with 1-based citation numbers so
        # the Python-side stop check can count independent domains per assertion.
        self._assess_agent: Agent[None, _AssessmentOutput] = Agent(
            self._model,
            output_type=_out(_AssessmentOutput),
            retries=1,
            instrument=True,
            system_prompt=(
                "You answer a user query using numbered web passages as evidence.\n"
                "Return JSON with: answer, key_claims, confidence, gaps, contradicts_query.\n"
                "\n"
                "answer — a grounded markdown answer in the SAME language as the query.\n"
                "  Cite sources inline with [N] where N is a passage number.\n"
                "  Adapt format to the query shape:\n"
                "    - short factual query (who/when/where/exact number) → 1–3 sentences.\n"
                "    - explanation / how-does-X / overview → comprehensive multi-paragraph markdown.\n"
                "    - comparison / what's-the-difference → bulleted or sectioned breakdown per dimension.\n"
                "    - news / latest / what's happening → bullet list of distinct developments, each citing a different source.\n"
                "  Do NOT add a Sources/References section — it will be appended automatically from citations.\n"
                "\n"
                "key_claims — atomic factual assertions the answer makes.\n"
                "  Each key_claim has: text (the assertion) and supporting_citation_numbers\n"
                "  (the list of [N] indices that back it up).\n"
                "  Include 1–6 key claims covering the main points. Omit purely stylistic phrases.\n"
                "\n"
                "confidence — 0.0 to 1.0, your honest estimate that the answer is complete and correct.\n"
                "  Use < 0.5 when passages are thin, off-topic, or contradictory and you cannot\n"
                "  produce a reliable answer.\n"
                "  Use >= 0.75 when the answer is well-grounded and you would be comfortable serving it.\n"
                "\n"
                "gaps — specific missing information the next search iteration should target.\n"
                "  Return an empty list when confidence is high. Keep each gap short and concrete.\n"
                "\n"
                "contradicts_query — set to true ONLY if the user's query assumes a fact that\n"
                "  the passages directly disprove (e.g. wrong year, wrong attribution, wrong value).\n"
                "  When true, the answer MUST explicitly correct the user's assumption.\n"
                "\n"
                "Do not invent facts. Do not cite passage numbers that don't exist.\n"
                "If passages are insufficient, say so in the answer, set confidence low,\n"
                "and list the missing information in gaps."
            ),
        )

        # Follow-up query generation driven by gaps from the previous assessment.
        # Used on iteration 2+ when the stop check fails.
        self._gap_query_agent: Agent[None, _QueryListOutput] = Agent(
            self._model,
            output_type=_out(_QueryListOutput),
            retries=1,
            instrument=True,
            system_prompt=(
                "Generate 1 to 4 focused web search queries that target specific missing information.\n"
                "Input: the original user query, a list of information gaps, and queries already tried.\n"
                "Rules:\n"
                "- keyword phrases only, no question wording\n"
                "- match the user's language; add one English query when input is non-English\n"
                "- each query must target one specific gap, not the whole topic\n"
                "- do not repeat or paraphrase queries from the 'already tried' list\n"
                "- max 8 words per query"
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

        # Heuristic boost: explanation/comparison/how-to patterns are unambiguous
        # synthesis queries. The LLM occasionally classifies these as factual on
        # cross-language inputs (e.g. Russian "Как работает X"), which sends them
        # down the wrong decompose+verify path. Override defensively.
        if decision.intent == "factual" and _looks_like_synthesis(normalized_query):
            log(f"  [dim yellow]-> intent override: factual -> synthesis (explanation pattern)[/dim yellow]")
            decision = _IntentOutput(
                intent="synthesis",
                complexity=decision.complexity,
                search_queries=decision.search_queries,
            )
            self._classification_cache[normalized_query] = decision

        self._intent_cache[normalized_query] = decision.intent
        return decision

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

    # ------------------------------------------------------------------
    # Unified iterative pipeline — replaces classify/decompose/verify/synthesize
    # with two calls: generate_queries_unified + assess_and_answer.
    # ------------------------------------------------------------------

    def generate_queries_unified(
        self,
        *,
        user_query: str,
        normalized_query: str,
        iteration: int,
        prior_assessment: Assessment | None,
        used_queries: set[str],
        log=None,
    ) -> list[str]:
        """Produce search queries for the current iteration of the unified runner.

        Iter 1: reuse the query list generated by the combined intent-classification
        call (``_query_cache``) — one LLM round-trip already fires when ``classify_query``
        runs. If nothing is cached (LLM disabled, cache miss) fall back to
        ``_generate_queries_llm``.

        Iter 2+: feed the prior assessment's ``gaps`` into ``_gap_query_agent`` so the
        next retrieval targets specifically the information the model said was missing.
        """
        log = log or (lambda msg: None)

        def _dedupe(candidates: list[str]) -> list[str]:
            out: list[str] = []
            seen = {q.casefold().strip() for q in used_queries if q}
            for candidate in candidates:
                text = normalized_text(candidate)
                if not text:
                    continue
                key = text.casefold().strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(text)
            return out

        if iteration <= 1 or prior_assessment is None:
            cached = self._query_cache.get(normalized_query)
            if cached:
                for q in cached:
                    log(f"  [dim]-> llm_query: {q}[/dim]")
                return _dedupe(cached)
            fallback_classification = QueryClassification(
                query=user_query,
                normalized_query=normalized_query,
                intent="factual",
                complexity="single_hop",
                needs_freshness=False,
            )
            return _dedupe(
                self._generate_queries_llm(normalized_query, fallback_classification, log=log)
            )

        if not self._enabled:
            return []
        gaps = [gap for gap in (prior_assessment.gaps or []) if (gap or "").strip()]
        if not gaps:
            return []

        prompt = (
            f"User query: {user_query}\n"
            f"Gaps to fill:\n" + "\n".join(f"- {gap}" for gap in gaps) + "\n"
            f"Already tried:\n" + ("\n".join(f"- {q}" for q in sorted(used_queries)) or "- none")
        )
        try:
            with log_llm_call(
                log,
                task="generate_queries_gap",
                model=self._settings.llm_model,
                detail=user_query[:60],
                input_chars=len(prompt),
            ) as metrics:
                result = self._gap_query_agent.run_sync(
                    prompt,
                    model_settings=self._model_settings(max_tokens=200, temperature=0),
                )
                metrics.output_chars = output_char_len(result.output)
            queries = _dedupe(list(result.output.queries))
            for q in queries:
                log(f"  [dim]-> llm_gap_query: {q}[/dim]")
            return queries[:4]
        except Exception as exc:
            log(f"  [dim yellow]-> generate_queries_gap failed: {exc}[/dim yellow]")
            return []

    def assess_and_answer(
        self,
        user_query: str,
        passages: list[Passage],
        log=None,
    ) -> Assessment:
        """Single LLM call that writes the final answer AND self-evaluates sufficiency.

        Returns an ``Assessment`` with the markdown answer, the key_claims the answer
        makes (with [N] citation indices), a confidence score, the gaps that would
        drive the next iteration, and a ``contradicts_query`` flag for counterfactual
        user queries.

        ``passages`` must already be the final ranked and capped list in the exact
        order that citation indices [1..N] will reference.  The caller (unified
        runner) is responsible for URL/domain capping; this method does NOT re-cap
        or re-number, so the indices returned in ``key_claims`` align 1-to-1 with
        ``passages``.
        """
        log = log or (lambda msg: None)
        if not self._enabled or not passages:
            return Assessment(answer="", confidence=0.0, contradicts_query=False)

        prompt_parts: list[str] = []
        passage_refs: list[Passage] = []
        for p in passages:
            text = (p.text or "").strip()
            if not text:
                continue
            n = len(prompt_parts) + 1
            title = (p.title or "").strip()
            url = (p.url or "").strip()
            prompt_parts.append(f"[{n}] {title} | {url}\n{text[:1200]}")
            passage_refs.append(p)

        if not prompt_parts:
            return Assessment(answer="", confidence=0.0, contradicts_query=False)

        prompt = (
            f"Query: {user_query}\n\n"
            f"Passages:\n\n" + "\n\n".join(prompt_parts)
        )

        try:
            with log_llm_call(
                log,
                task="assess_and_answer",
                model=self._settings.llm_model,
                detail=user_query[:60],
                input_chars=len(prompt),
            ) as metrics:
                with logfire.span("query_intelligence.assess_and_answer", query=user_query):
                    result = self._assess_agent.run_sync(
                        prompt,
                        model_settings=self._model_settings(
                            max_tokens=self._settings.resolved_synthesize_answer_max_tokens(),
                            temperature=0.3,
                        ),
                    )
                metrics.output_chars = output_char_len(result.output)
            output = result.output

            answer_text = _normalize_citation_groups((output.answer or "").strip())

            # Append Sources section from citations actually used in the answer,
            # mirroring synthesize_answer's sources footer.  We look up each [N]
            # against the passage_refs we built above.
            cited_indices = _extract_citation_indices(answer_text)
            if cited_indices and answer_text:
                lines = ["\n\nИсточники:"]
                for n in cited_indices:
                    if 1 <= n <= len(passage_refs):
                        passage = passage_refs[n - 1]
                        label = (passage.title or passage.url)[:80]
                        url = passage.url
                        lines.append(f"[{n}] {label} — {url}" if url else f"[{n}] {label}")
                answer_text += "\n".join(lines)

            key_claims = [
                KeyClaim(
                    text=(kc.text or "").strip(),
                    supporting_citation_numbers=[
                        n for n in (kc.supporting_citation_numbers or []) if 1 <= n <= len(passage_refs)
                    ],
                )
                for kc in (output.key_claims or [])
                if (kc.text or "").strip()
            ]

            confidence = clamp(float(output.confidence or 0.0))
            gaps = [normalized_text(g) for g in (output.gaps or []) if normalized_text(g)]

            return Assessment(
                answer=answer_text,
                key_claims=key_claims,
                confidence=confidence,
                gaps=gaps,
                contradicts_query=bool(output.contradicts_query),
            )
        except Exception as exc:
            log(f"  [dim yellow]-> assess_and_answer failed: {exc}[/dim yellow]")
            return Assessment(answer="", confidence=0.0, contradicts_query=False)

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

    def _model_settings(self, *, max_tokens: int, temperature: float):
        return build_model_settings(
            self._settings,
            max_tokens=max_tokens,
            temperature=temperature,
        )
