from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from typing import Iterable

from search_agent.domain.models import (
    AgentRunResult,
    AuditTrail,
    Claim,
    ClaimProfile,
    ClaimRun,
    DomainType,
    EvidenceBundle,
    EvidenceSpan,
    FetchedDocument,
    FetchPlan,
    GatedSerpResult,
    Passage,
    QueryClassification,
    QueryVariant,
    RoutingDecision,
    SearchSnapshot,
    SourceAssessment,
    VerificationResult,
)
from search_agent.domain.source_priors import lookup_source_prior
from search_agent import tuning
from search_agent.application.claim_policy import (
    answer_type as _policy_answer_type,
    claim_answer_shape as _policy_claim_answer_shape,
    claim_contract_gaps as _policy_claim_contract_gaps,
    claim_focus_terms as _policy_claim_focus_terms,
    claim_min_independent_sources as _policy_claim_min_independent_sources,
    claim_requires_primary_source as _policy_claim_requires_primary_source,
    exact_detail_guardrail_claim as _policy_exact_detail_guardrail_claim,
    is_news_digest_claim as _policy_is_news_digest_claim,
    publish_supported_claim as _policy_publish_supported_claim,
    retrieval_contract_can_drive_synthesis as _policy_retrieval_contract_can_drive_synthesis,
    should_stop_claim_loop as _policy_should_stop_claim_loop,
)
from search_agent.application.text_heuristics import (
    compact_text as _shared_compact_text,
    contains_date_like as _shared_contains_date_like,
    extract_entities as _shared_extract_entities,
    extract_numbers as _shared_extract_numbers,
    extract_region_hint as _shared_extract_region_hint,
    extract_time_scope as _shared_extract_time_scope,
    is_cyrillic_text as _shared_is_cyrillic_text,
    normalized_text as _shared_normalized_text,
    tokenize as _shared_tokenize,
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "if",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with", "vs",
    "что", "как", "где", "когда", "кто", "или", "для", "это", "эта", "этот",
    "про", "из", "на", "по", "в", "во", "с", "со", "у", "о", "об", "ли",
    "чем", "какой", "какая", "какие", "каково", "есть", "был", "была", "были",
    "than", "about", "between", "latest", "current",
}

_OFFICIAL_HOST_MARKERS = (
    ".gov", ".mil", ".gouv", "europa.eu", "who.int", "sec.gov", "irs.gov",
)
_ACADEMIC_HOST_MARKERS = (
    ".edu", "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "nature.com", "science.org",
    "springer.com", "acm.org", "ieee.org", "sciencedirect.com", "doi.org",
)
_FORUM_HOST_MARKERS = (
    "reddit.com", "stackoverflow.com", "stackexchange.com", "quora.com",
    "forum.", "community.", "discuss.", "news.ycombinator.com",
)
_MAJOR_MEDIA_HOST_MARKERS = (
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com", "theguardian.com",
    "bloomberg.com", "cnbc.com", "wsj.com", "ft.com", "cnn.com", "npr.org",
    "tass.ru", "interfax.ru", "kommersant.ru", "vedomosti.ru", "rbc.ru",
)
_VENDOR_HOST_MARKERS = (
    "docs.", "developer.", "support.", "help.", "learn.", "cloud.", "github.com",
)
_PRIMARY_SOURCE_CUES = (
    "official", "press release", "announcement", "earnings", "report", "filing",
    "documentation", "docs", "paper", "study", "whitepaper", "transcript",
    "официаль", "пресс-релиз", "отчет", "документация", "исследование",
)
_SPAM_CUES = (
    "best ", "top ", "coupon", "deal", "promo", "discount", "affiliate",
    "alternatives", "vs ", "review", "reviews", "guide", "ultimate",
)
_NON_ENTITY_TOKENS = {
    "who", "what", "when", "where", "why", "how", "which",
    "кто", "что", "когда", "где", "почему", "как", "какой", "какая", "какие",
    "ceo", "cto", "cfo", "founder", "president", "chairman", "capital",
    "release", "released", "latest", "current",
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _build_run_id(query: str, started_at: datetime) -> str:
    digest = hashlib.sha1(f"{query}|{started_at.isoformat()}".encode("utf-8")).hexdigest()[:8]
    return f"{started_at.strftime('%Y%m%dT%H%M%S')}-{digest}"


def _normalized_text(text: str) -> str:
    return _shared_normalized_text(text)


def _compact_text(text: str) -> str:
    return _shared_compact_text(text)


def _tokenize(text: str) -> list[str]:
    return [token for token in _shared_tokenize(text) if token not in _STOPWORDS]


def _contains_date_like(text: str) -> bool:
    return _shared_contains_date_like(text)


def _extract_entities(text: str) -> list[str]:
    return _shared_extract_entities(text)[:tuning.AGENT_MAX_QUERY_VARIANTS]


def _extract_time_scope(text: str) -> str | None:
    return _shared_extract_time_scope(text)

def _variant_keywords(text: str, entities: Iterable[str], excluded_tokens: Iterable[str] = ()) -> str:
    entity_tokens = {token.casefold() for entity in entities for token in _tokenize(entity)}
    excluded = {token.casefold() for token in excluded_tokens}
    tokens: list[str] = []
    for token in _tokenize(text):
        if token.casefold() in entity_tokens:
            continue
        if token.casefold() in excluded:
            continue
        if token not in tokens:
            tokens.append(token)
    return " ".join(tokens[:6])


def _compose_query(*parts: str | None) -> str:
    return _normalized_text(" ".join(part for part in parts if part))


def _news_digest_site_hint(claim: Claim, classification: QueryClassification) -> str | None:
    haystack = " ".join(
        part
        for part in (
            claim.claim_text,
            classification.normalized_query,
            classification.region_hint,
            " ".join(claim.entity_set),
        )
        if part
    ).casefold()
    if any(marker in haystack for marker in ("астан", "алмат", "казах", "қазақ", "kazakh", "kazakhstan", ".kz")):
        return "site:.kz"
    return None


def infer_claim_profile(claim: Claim, classification: QueryClassification) -> ClaimProfile:
    if claim.claim_profile is not None:
        return claim.claim_profile
    if classification.intent == "news_digest":
        return ClaimProfile(answer_shape="news_digest", min_independent_sources=3, routing_bias="iterative_loop")
    if classification.intent == "synthesis":
        return ClaimProfile(answer_shape="overview", min_independent_sources=2, routing_bias="iterative_loop")
    return ClaimProfile(answer_shape="fact", min_independent_sources=1)


def _claim_answer_shape(claim: Claim) -> str:
    return _policy_claim_answer_shape(claim)


def _preferred_domain_bonus(claim: Claim, domain_type: DomainType) -> float:
    profile = claim.claim_profile
    if profile is None or not profile.preferred_domain_types:
        return 0.0
    return 0.08 if domain_type in profile.preferred_domain_types else 0.0


def _product_specs_result_bonus(claim: Claim, title: str, snippet: str, url: str) -> float:
    return 0.0


def build_query_variants(claim: Claim, classification: QueryClassification) -> list[QueryVariant]:
    queries = [_normalized_text(query) for query in (claim.search_queries or []) if _normalized_text(query)]
    if not queries:
        fallback = _normalized_text(claim.claim_text)
        queries = [fallback] if fallback else []

    seen: set[str] = set()
    variants: list[QueryVariant] = []
    for idx, query_text in enumerate(queries, 1):
        key = query_text.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        variants.append(QueryVariant(
            variant_id=f"{claim.claim_id}-q{idx}",
            claim_id=claim.claim_id,
            query_text=query_text,
            strategy=f"llm_{idx}" if claim.search_queries else "claim_text",
            rationale="LLM-planned search query." if claim.search_queries else "Fallback to the raw claim text.",
            source_restriction=None,
            freshness_hint=claim.time_scope,
        ))
        if len(variants) >= tuning.AGENT_MAX_QUERY_VARIANTS:
            break
    return variants


def _is_cyrillic_text(text: str) -> bool:
    return _shared_is_cyrillic_text(text)


def _time_query_terms(time_scope: str | None, *, cyrillic: bool) -> list[str]:
    if not time_scope:
        return []
    terms = [time_scope]
    if (
        len(time_scope) == 10
        and time_scope[4] == "-"
        and time_scope[7] == "-"
        and time_scope[:4].isdigit()
        and time_scope[5:7].isdigit()
        and time_scope[8:10].isdigit()
    ):
        try:
            dt = datetime.fromisoformat(time_scope)
        except ValueError:
            return terms
        if cyrillic:
            ru_months = [
                "\u044f\u043d\u0432\u0430\u0440\u044f",
                "\u0444\u0435\u0432\u0440\u0430\u043b\u044f",
                "\u043c\u0430\u0440\u0442\u0430",
                "\u0430\u043f\u0440\u0435\u043b\u044f",
                "\u043c\u0430\u044f",
                "\u0438\u044e\u043d\u044f",
                "\u0438\u044e\u043b\u044f",
                "\u0430\u0432\u0433\u0443\u0441\u0442\u0430",
                "\u0441\u0435\u043d\u0442\u044f\u0431\u0440\u044f",
                "\u043e\u043a\u0442\u044f\u0431\u0440\u044f",
                "\u043d\u043e\u044f\u0431\u0440\u044f",
                "\u0434\u0435\u043a\u0430\u0431\u0440\u044f",
            ]
            terms.append(f"{dt.day} {ru_months[dt.month - 1]} {dt.year}")
            terms.append(str(dt.year))
        else:
            terms.append(dt.strftime("%B %d %Y"))
            terms.append(str(dt.year))
    return terms


def _is_iso_date_text(text: str | None) -> bool:
    if not text:
        return False
    return (
        len(text) == 10
        and text[4] == "-"
        and text[7] == "-"
        and text[:4].isdigit()
        and text[5:7].isdigit()
        and text[8:10].isdigit()
    )


def _is_year_text(text: str | None) -> bool:
    return bool(text and len(text) == 4 and text.isdigit() and text.startswith("20"))


def _contains_person_span(text: str) -> bool:
    return any(" " in entity for entity in _shared_extract_entities(text))


def _contains_location_span(text: str) -> bool:
    return _shared_extract_region_hint(text) is not None


def _contains_negation_cue(text: str) -> bool:
    lowered = (text or "").casefold()
    tokens = set(_shared_tokenize(text))
    return bool(tokens & {"not", "no", "never", "false", "incorrect"} or "debunked" in lowered or "contradict" in lowered)


def _split_sentences(text: str) -> list[str]:
    source = text or ""
    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(source):
        ch = source[index]
        if ch == "." and 0 < index < len(source) - 1 and source[index - 1].isdigit() and source[index + 1].isdigit():
            index += 1
            continue
        if ch not in ".!?":
            index += 1
            continue
        end = index + 1
        while end < len(source) and source[end] in ".!?":
            end += 1
        sentence = _normalized_text(source[start:end])
        if sentence:
            sentences.append(sentence)
        index = end
        while index < len(source) and source[index].isspace():
            index += 1
        start = index
    remainder = _normalized_text(source[start:])
    if remainder:
        sentences.append(remainder)
    return sentences


def _extract_date_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for sentence in _split_sentences(text):
        scope = _extract_time_scope(sentence)
        if not scope:
            continue
        key = scope.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(scope)
        if len(candidates) >= 3:
            break
    if not candidates:
        scope = _extract_time_scope(text)
        if scope:
            candidates.append(scope)
    return candidates[:3]


def _extract_person_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for entity in _extract_entities(text):
        token_count = len(_tokenize(entity))
        if token_count >= 2:
            candidates.append(entity)
        if len(candidates) >= 3:
            break
    return candidates


def _extract_location_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for sentence in _split_sentences(text):
        region = _shared_extract_region_hint(sentence)
        if not region:
            continue
        key = region.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(region)
        if len(candidates) >= 3:
            break
    if not candidates:
        region = _shared_extract_region_hint(text)
        if region:
            candidates.append(region)
    return candidates[:3]


def _contains_release_date_cue(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(phrase in lowered for phrase in ("released on", "release date", "announced on", "dated"))


def _clean_title_key(title: str) -> str:
    return " ".join(_tokenize(title))


def _extract_author(text: str) -> str | None:
    head = (text or "")[:600]
    for raw_line in head.splitlines():
        line = raw_line.strip()
        lowered = line.casefold()
        for marker in ("by:", "author:", "автор:"):
            if lowered.startswith(marker):
                return _normalized_text(line.split(":", 1)[1])
    return None


def _sanitize_compose_fragment(text: str) -> str:
    if not text:
        return text
    parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.lstrip()
        while line.startswith("#"):
            line = line[1:].lstrip()
        if line:
            parts.append(line)
    if not parts:
        return ""
    return " ".join(" ".join(parts).split())


def _retag_snapshot(snapshot: SearchSnapshot, variant: QueryVariant) -> SearchSnapshot:
    results = [
        replace(result, result_id=f"{variant.variant_id}:{result.position}", query_variant_id=variant.variant_id)
        for result in snapshot.results
    ]
    return SearchSnapshot(
        query=snapshot.query,
        suggestions=snapshot.suggestions,
        results=results,
        retrieved_at=snapshot.retrieved_at,
        profile_name=snapshot.profile_name,
        unresponsive_engines=list(snapshot.unresponsive_engines),
    )


def _domain_type(host: str) -> DomainType:
    if any(marker in host for marker in _OFFICIAL_HOST_MARKERS):
        return "official"
    if any(marker in host for marker in _ACADEMIC_HOST_MARKERS):
        return "academic"
    if any(marker in host for marker in _FORUM_HOST_MARKERS):
        return "forum"
    if any(marker in host for marker in _MAJOR_MEDIA_HOST_MARKERS):
        return "major_media"
    if any(marker in host for marker in _VENDOR_HOST_MARKERS):
        return "vendor"
    return "unknown"


def _host_root(host: str) -> str:
    parts = [part for part in host.split(".") if part]
    if len(parts) <= 2:
        return host
    if parts[-2] in {"co", "com", "org", "net"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _entity_host_match_score(claim: Claim, host: str) -> float:
    labels = [
        _compact_text(part)
        for part in host.split(".")
        if part and part.lower() != "www"
    ]
    if not labels:
        return 0.0

    best = 0.0
    label_set = set(labels)
    for entity in claim.entity_set:
        compact_entity = _compact_text(entity)
        if len(compact_entity) >= 4 and compact_entity in label_set:
            best = max(best, 1.0)
        for token in _tokenize(entity):
            compact_token = _compact_text(token)
            if len(compact_token) < 4:
                continue
            if compact_token in _STOPWORDS or compact_token in _NON_ENTITY_TOKENS:
                continue
            if compact_token in label_set:
                best = max(best, 0.95)
    return best


def _effective_domain_type(claim: Claim, host: str) -> DomainType:
    prior = lookup_source_prior(host)
    if prior.domain_type_override in {"official", "academic", "vendor", "major_media", "forum", "unknown"}:
        return prior.domain_type_override
    base = _domain_type(host)
    if base != "unknown":
        return base
    if _entity_host_match_score(claim, host) >= 0.8:
        return "official"
    return base


def _title_key(title: str) -> str:
    return _clean_title_key(title)


def _semantic_overlap(query_text: str, candidate_text: str) -> float:
    query_tokens = set(_tokenize(query_text))
    if not query_tokens:
        return 0.0
    candidate_tokens = set(_tokenize(candidate_text))
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def _entity_overlap(entities: list[str], candidate_text: str) -> float:
    if not entities:
        return 0.0
    lowered = candidate_text.casefold()
    compact_candidate = _compact_text(candidate_text)
    hits = 0.0
    for entity in entities:
        if entity.casefold() in lowered:
            hits += 1.0
            continue
        compact_entity = _compact_text(entity)
        if compact_entity and compact_entity in compact_candidate:
            hits += 1.0
            continue
        if len(compact_entity) >= 5 and compact_entity[:5] in compact_candidate:
            hits += 0.75
    return _clamp(hits / len(entities))


def _time_scope_alignment(claim: Claim, result) -> float:
    if not claim.time_scope:
        return 0.0
    scope = claim.time_scope.casefold()
    haystack = f"{result.title} {result.snippet} {result.url}".casefold()
    if scope in haystack:
        return 1.0
    if result.published_at and result.published_at.startswith(claim.time_scope):
        return 1.0
    if _is_iso_date_text(claim.time_scope) and result.published_at:
        if result.published_at[:7] == claim.time_scope[:7]:
            return 0.45
    if _is_year_text(claim.time_scope) and result.published_at:
        if result.published_at.startswith(claim.time_scope):
            return 0.8
    return 0.0


def _freshness_score(claim: Claim, result) -> float:
    if not claim.needs_freshness and not result.published_at:
        return 0.5
    if not result.published_at:
        return 0.2 if claim.needs_freshness else 0.35
    try:
        published = datetime.fromisoformat(result.published_at.replace("Z", "+00:00"))
        age_days = max(0, (datetime.now(UTC) - published.astimezone(UTC)).days)
    except ValueError:
        return 0.2 if claim.needs_freshness else 0.35
    window = 45 if claim.needs_freshness else 365
    return _clamp(1 - (age_days / max(window, 1)))


def _spam_risk(result) -> float:
    host = result.host.casefold()
    text = f"{result.title} {result.snippet}".casefold()
    prior = lookup_source_prior(host)
    risk = 0.0
    if host.endswith(".top") or host.endswith(".best"):
        risk += 0.6
    if any(cue in text for cue in _SPAM_CUES):
        risk += 0.25
    if text.count("|") >= 2 or text.count(" - ") >= 3:
        risk += 0.2
    if len(_tokenize(result.title)) > 14:
        risk += 0.1
    if _domain_type(host) in {"official", "academic"}:
        risk -= 0.2
    risk += prior.spam_penalty
    return _clamp(risk)


def _primary_source_likelihood(claim: Claim, result, domain_type: DomainType) -> float:
    prior = lookup_source_prior(result.host)
    base = {
        "official": 0.9,
        "academic": 0.85,
        "vendor": 0.7,
        "major_media": 0.45,
        "forum": 0.15,
        "unknown": 0.4,
    }[domain_type]
    text = f"{result.title} {result.snippet} {result.url}".casefold()
    if any(cue in text for cue in _PRIMARY_SOURCE_CUES):
        base += 0.1
    base += 0.18 * _entity_host_match_score(claim, result.host)
    if any(path_cue in text for path_cue in ("/announcement/", "/press", "/release", "/releases/", "/downloads/release/", "/whatsnew/")):
        base += 0.08
    if domain_type == "forum":
        base -= 0.12
    base += prior.primary_boost
    return _clamp(base)


def gate_serp_results(
    claim: Claim,
    snapshots: list[SearchSnapshot],
    limit: int,
) -> list[GatedSerpResult]:
    merged: dict[str, GatedSerpResult] = {}

    for snapshot in snapshots:
        for result in snapshot.results:
            domain_type = _effective_domain_type(claim, result.host)
            prior = lookup_source_prior(result.host)
            domain_prior = {
                "official": 1.0,
                "academic": 0.95,
                "vendor": 0.75,
                "major_media": 0.7,
                "forum": 0.3,
                "unknown": 0.5,
            }[domain_type]
            semantic_match = _semantic_overlap(claim.claim_text, f"{result.title} {result.snippet}")
            entity_match = _entity_overlap(claim.entity_set, f"{result.title} {result.snippet}")
            host_entity_match = _entity_host_match_score(claim, result.host)
            freshness = _freshness_score(claim, result)
            time_alignment = _time_scope_alignment(claim, result)
            spam = _spam_risk(result)
            primary = _primary_source_likelihood(claim, result, domain_type)
            preferred_domain_bonus = _preferred_domain_bonus(claim, domain_type)
            product_bonus = _product_specs_result_bonus(claim, result.title, result.snippet, result.url)
            source_score = _clamp(
                0.22 * domain_prior
                + 0.22 * primary
                + 0.12 * freshness
                + 0.16 * entity_match
                + 0.12 * semantic_match
                + 0.06 * host_entity_match
                + 0.10 * time_alignment
                + preferred_domain_bonus
                + product_bonus
                + prior.source_prior
                - 0.25 * spam
            )
            reasons = [
                f"domain={domain_type}",
                f"prior={prior.source_prior:.2f}",
                f"primary={primary:.2f}",
                f"host_entity={host_entity_match:.2f}",
                f"freshness={freshness:.2f}",
                f"time_alignment={time_alignment:.2f}",
                f"entity_match={entity_match:.2f}",
                f"spam={spam:.2f}",
                f"preferred_bonus={preferred_domain_bonus:.2f}",
                f"shape_bonus={product_bonus:.2f}",
            ]
            reasons.extend(prior.labels)
            gated = GatedSerpResult(
                serp=result,
                assessment=SourceAssessment(
                    domain_type=domain_type,
                    source_prior=prior.source_prior,
                    primary_source_likelihood=primary,
                    freshness_score=freshness,
                    seo_spam_risk=spam,
                    entity_match_score=entity_match,
                    semantic_match_score=semantic_match,
                    source_score=source_score,
                    reasons=reasons,
                ),
                matched_variant_ids=[result.query_variant_id],
            )

            canonical_key = result.canonical_url
            title_key = f"{_host_root(result.host)}|{_title_key(result.title)}"
            existing = merged.get(canonical_key) or merged.get(title_key)
            if existing is None:
                merged[canonical_key] = gated
                continue

            if result.query_variant_id not in existing.matched_variant_ids:
                existing.matched_variant_ids.append(result.query_variant_id)
            if gated.assessment.source_score > existing.assessment.source_score:
                gated.assessment.duplicate_of = existing.serp.result_id
                merged[canonical_key] = gated
            else:
                existing.assessment.duplicate_of = result.result_id

    gated_results = sorted(
        merged.values(),
        key=lambda item: (
            item.assessment.source_score,
            item.assessment.primary_source_likelihood,
            -item.assessment.seo_spam_risk,
        ),
        reverse=True,
    )
    return gated_results[:limit]


def _answer_type(claim: Claim) -> str:
    return _policy_answer_type(claim)


def _extract_answer_candidates(claim: Claim, text: str) -> list[str]:
    answer_type = _answer_type(claim)
    if answer_type == "time":
        return _extract_date_candidates(text)
    if answer_type == "number":
        return _shared_extract_numbers(text)[:3]
    if answer_type == "person":
        return _extract_person_candidates(text)
    if answer_type == "location":
        return _extract_location_candidates(text)
    return _extract_entities(text)[:3]


def _extract_author(text: str) -> str | None:
    head = (text or "")[:600]
    for raw_line in head.splitlines():
        line = raw_line.strip()
        lowered = line.casefold()
        for marker in ("by:", "author:", "автор:"):
            if lowered.startswith(marker):
                return _normalized_text(line.split(":", 1)[1])
    return None


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return _normalized_text(stripped.lstrip("#").strip())
    return None


def _make_document(
    gated: GatedSerpResult,
    content: str,
    fetch_depth: str,
    *,
    title: str | None = None,
    author: str | None = None,
    published_at: str | None = None,
    meta_description: str | None = None,
    headings: list[str] | None = None,
    first_paragraphs: list[str] | None = None,
    schema_org: dict | None = None,
) -> FetchedDocument:
    extracted_at = datetime.now(UTC).isoformat()
    title = title or _markdown_title(content) or gated.serp.title or gated.serp.url
    normalized = _normalized_text(content)
    content_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return FetchedDocument(
        doc_id=f"doc-{content_hash[:12]}",
        url=gated.serp.url,
        canonical_url=gated.serp.canonical_url,
        host=gated.serp.host,
        title=title,
        author=author or _extract_author(content),
        published_at=published_at or gated.serp.published_at,
        extracted_at=extracted_at,
        content_hash=content_hash,
        content=normalized,
        fetch_depth=fetch_depth,
        source_score=gated.assessment.source_score,
        meta_description=meta_description,
        headings=headings or [],
        first_paragraphs=first_paragraphs or [],
        schema_org=schema_org or {},
    )


def _select_fetch_candidates(gated_results: list[GatedSerpResult], limit: int) -> list[GatedSerpResult]:
    selected: list[GatedSerpResult] = []
    seen_hosts: set[str] = set()

    for candidate in gated_results:
        root = _host_root(candidate.serp.host)
        if root in seen_hosts:
            continue
        selected.append(candidate)
        seen_hosts.add(root)
        if len(selected) >= limit:
            return selected

    for candidate in gated_results:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            return selected
    return selected


def _routing_limits(profile, decision: RoutingDecision, iteration: int = 1) -> tuple[int, int]:
    if iteration == 1:
        shallow_limit = {
            "short_path": tuning.SHALLOW_FETCH_SHORT_FAST_LIMIT,
            "targeted_retrieval": tuning.SHALLOW_FETCH_TARGETED_FAST_LIMIT,
            "iterative_loop": tuning.SHALLOW_FETCH_ITERATIVE_FAST_LIMIT,
        }[decision.mode]
    else:
        shallow_limit = {
            "short_path": tuning.SHALLOW_FETCH_SHORT_LIMIT,
            "targeted_retrieval": tuning.SHALLOW_FETCH_TARGETED_LIMIT,
            "iterative_loop": tuning.SHALLOW_FETCH_ITERATIVE_LIMIT,
        }[decision.mode]
    deep_limit = {
        "short_path": tuning.DEEP_FETCH_SHORT_LIMIT,
        "targeted_retrieval": tuning.DEEP_FETCH_TARGETED_LIMIT,
        "iterative_loop": tuning.DEEP_FETCH_ITERATIVE_LIMIT,
    }[decision.mode]
    if profile.fetch_top_n == 0:
        deep_limit = 0
    else:
        deep_limit = min(deep_limit, max(profile.fetch_top_n, tuning.AGENT_FETCH_TOP_N))
    return shallow_limit, deep_limit


def _verification_source_bonus(claim: Claim, *, host: str, title: str, url: str) -> float:
    domain_type = _effective_domain_type(claim, host)
    prior = lookup_source_prior(host)
    bonus = {
        "official": 0.38,
        "academic": 0.34,
        "vendor": 0.18,
        "major_media": 0.08,
        "forum": -0.20,
        "unknown": 0.0,
    }[domain_type]
    lowered = f"{title} {url}".casefold()
    if any(cue in lowered for cue in ("announcement", "press", "release", "released", "downloads/release", "whatsnew", "/blog/")):
        bonus += 0.14
    if any(cue in lowered for cue in ("hacker news", "reddit", "forum", "comment")):
        bonus -= 0.12
    bonus += prior.verification_bonus
    return bonus


def score_shallow_document_for_claim(claim: Claim, document: FetchedDocument) -> float:
    overview = " ".join(
        [
            document.title,
            document.meta_description or "",
            " ".join(document.headings[:3]),
            " ".join(document.first_paragraphs[:2]),
        ]
    )
    source_bonus = _verification_source_bonus(
        claim,
        host=document.host,
        title=document.title,
        url=document.url,
    )
    return _clamp(
        0.30 * _semantic_overlap(claim.claim_text, overview)
        + 0.22 * _entity_overlap(claim.entity_set, overview)
        + 0.18 * _dimension_coverage_score(claim, overview)
        + 0.18 * document.source_score
        + 0.12 * max(source_bonus, 0.0)
    )


def _make_shallow_document(candidate: GatedSerpResult, payload: dict) -> FetchedDocument:
    summary = payload.get("content") or candidate.serp.snippet or candidate.serp.title
    return _make_document(
        candidate,
        summary,
        "shallow",
        title=payload.get("title") or candidate.serp.title,
        author=payload.get("author"),
        published_at=payload.get("published_at"),
        meta_description=payload.get("meta_description"),
        headings=payload.get("headings") or [],
        first_paragraphs=payload.get("first_paragraphs") or [],
        schema_org=payload.get("schema_org") or {},
    )


def build_snippet_passages(gated_results: list[GatedSerpResult]) -> list[Passage]:
    """Build lightweight Passage objects directly from SERP snippets (no HTTP fetch)."""
    now = datetime.now(UTC).isoformat()
    passages = []
    for i, gated in enumerate(gated_results):
        snippet = (gated.serp.snippet or "").strip()
        if len(snippet) < 20:
            continue
        passages.append(
            Passage(
                passage_id=f"snip-{i}",
                url=gated.serp.url,
                canonical_url=gated.serp.canonical_url,
                host=gated.serp.host,
                title=gated.serp.title or "",
                section="snippet",
                published_at=gated.serp.published_at,
                author=None,
                extracted_at=now,
                chunk_id=f"snip-{i}-0",
                text=snippet,
                source_score=gated.assessment.source_score,
                utility_score=0.0,
            )
        )
    return passages


def fetch_claim_documents(
    claim: Claim,
    gated_results: list[GatedSerpResult],
    profile,
    routing_decision: RoutingDecision,
    seen_urls: set[str] | None = None,
    log=None,
    iteration: int = 1,
    page_cache: dict[str, dict] | None = None,
    page_cache_lock=None,
    intent: str = "factual",
) -> tuple[list[FetchPlan], list[FetchedDocument]]:
    log = log or (lambda msg: None)
    from search_agent.infrastructure.extractor import fetch_and_extract_many, shallow_fetch_many

    seen_urls = seen_urls or set()
    shallow_limit, deep_limit = _routing_limits(profile, routing_decision, iteration)
    selected = _select_fetch_candidates(
        [candidate for candidate in gated_results if candidate.serp.url not in seen_urls],
        min(len(gated_results), shallow_limit),
    )

    plans: list[FetchPlan] = [
        FetchPlan(
            depth="shallow",
            url=candidate.serp.url,
            reason=f"Phase 2 shallow fetch ({routing_decision.mode}).",
            source_score=candidate.assessment.source_score,
        )
        for candidate in selected
    ]

    shallow_documents: list[FetchedDocument] = []
    if selected:
        for candidate, payload in zip(
            selected,
            shallow_fetch_many(
                [c.serp.url for c in selected],
                log=log,
                page_cache=page_cache,
                page_cache_lock=page_cache_lock,
                intent=intent,
            ),
        ):
            if payload:
                shallow_documents.append(_make_shallow_document(candidate, payload))
            elif candidate.serp.snippet:
                shallow_documents.append(
                    _make_document(
                        candidate,
                        candidate.serp.snippet,
                        "snippet_only",
                        title=candidate.serp.title,
                    )
                )

    shallow_ranked = sorted(
        shallow_documents,
        key=lambda document: score_shallow_document_for_claim(claim, document),
        reverse=True,
    )

    deep_candidates: list[tuple[GatedSerpResult, FetchedDocument]] = []
    gated_by_url = {candidate.serp.url: candidate for candidate in selected}
    for document in shallow_ranked:
        candidate = gated_by_url.get(document.url)
        if candidate is None:
            continue
        deep_candidates.append((candidate, document))
        if len(deep_candidates) >= deep_limit:
            break

    if deep_limit > 0 and shallow_ranked:
        source_priority_ranked = sorted(
            shallow_ranked,
            key=lambda document: (
                _verification_source_bonus(
                    claim,
                    host=document.host,
                    title=document.title,
                    url=document.url,
                ),
                document.source_score,
                score_shallow_document_for_claim(claim, document),
            ),
            reverse=True,
        )
        preferred_document = source_priority_ranked[0]
        preferred_candidate = gated_by_url.get(preferred_document.url)
        selected_urls = {document.url for _, document in deep_candidates}
        preferred_bonus = _verification_source_bonus(
            claim,
            host=preferred_document.host,
            title=preferred_document.title,
            url=preferred_document.url,
        )
        if (
            preferred_candidate is not None
            and preferred_document.url not in selected_urls
            and preferred_bonus > 0.18
        ):
            if len(deep_candidates) < deep_limit:
                deep_candidates.append((preferred_candidate, preferred_document))
            elif deep_candidates:
                deep_candidates[-1] = (preferred_candidate, preferred_document)

    deep_documents: list[FetchedDocument] = []
    for candidate, shallow_document in deep_candidates:
        plans.append(
            FetchPlan(
                depth="deep",
                url=candidate.serp.url,
                reason="Selective deep fetch after shallow rerank.",
                source_score=candidate.assessment.source_score,
            )
        )
    if deep_candidates:
        deep_urls = [c.serp.url for c, _ in deep_candidates]
        deep_contents = fetch_and_extract_many(deep_urls, log=log)
        for (candidate, shallow_document), content in zip(deep_candidates, deep_contents):
            if content:
                deep_documents.append(
                    _make_document(
                        candidate,
                        content,
                        "deep",
                        title=shallow_document.title,
                        author=shallow_document.author,
                        published_at=shallow_document.published_at,
                        meta_description=shallow_document.meta_description,
                        headings=shallow_document.headings,
                        first_paragraphs=shallow_document.first_paragraphs,
                        schema_org=shallow_document.schema_org,
                    )
                )

    documents = shallow_ranked + deep_documents
    if not deep_documents:
        documents.extend(
            _make_document(
                gated_by_url.get(document.url, selected[0]),
                document.content,
                "snippet_only",
                title=document.title,
                author=document.author,
                published_at=document.published_at,
                meta_description=document.meta_description,
                headings=document.headings,
                first_paragraphs=document.first_paragraphs,
                schema_org=document.schema_org,
            )
            for document in shallow_ranked[:tuning.AGENT_SNIPPET_FALLBACK_DOCS]
            if document.fetch_depth == "shallow"
        )

    return plans, documents


def _split_into_passages(document: FetchedDocument) -> list[Passage]:
    if not document.content:
        return []

    passages: list[Passage] = []
    current_section = "Intro"
    section_index = 0
    chunk_index = 0
    buffer: list[str] = []

    def flush() -> None:
        nonlocal chunk_index
        text = _normalized_text(" ".join(buffer))
        buffer.clear()
        if len(text) < 60:
            return
        chunks = _split_sentences(text)
        running = ""
        for piece in chunks:
            piece = _normalized_text(piece)
            if not piece:
                continue
            candidate = f"{running} {piece}".strip()
            if len(candidate) <= 420:
                running = candidate
                continue
            if running:
                passage_id = f"{document.doc_id}:{section_index}:{chunk_index}"
                passages.append(
                    Passage(
                        passage_id=passage_id,
                        url=document.url,
                        canonical_url=document.canonical_url,
                        host=document.host,
                        title=document.title,
                        section=current_section,
                        published_at=document.published_at,
                        author=document.author,
                        extracted_at=document.extracted_at,
                        chunk_id=passage_id,
                        text=running,
                        source_score=document.source_score,
                    )
                )
                chunk_index += 1
            running = piece
        if running:
            passage_id = f"{document.doc_id}:{section_index}:{chunk_index}"
            passages.append(
                Passage(
                    passage_id=passage_id,
                    url=document.url,
                    canonical_url=document.canonical_url,
                    host=document.host,
                    title=document.title,
                    section=current_section,
                    published_at=document.published_at,
                    author=document.author,
                    extracted_at=document.extracted_at,
                    chunk_id=passage_id,
                    text=running,
                    source_score=document.source_score,
                )
            )
            chunk_index += 1

    for raw_line in document.content.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            flush()
            current_section = _normalized_text(line.lstrip("#").strip()) or current_section
            section_index += 1
            continue
        buffer.append(line)
    flush()

    if passages:
        return passages

    passage_id = f"{document.doc_id}:0:0"
    return [
        Passage(
            passage_id=passage_id,
            url=document.url,
            canonical_url=document.canonical_url,
            host=document.host,
            title=document.title,
            section=current_section,
            published_at=document.published_at,
            author=document.author,
            extracted_at=document.extracted_at,
            chunk_id=passage_id,
            text=document.content[:420],
            source_score=document.source_score,
        )
    ]


def _documents_for_passage_extraction(documents: list[FetchedDocument]) -> list[FetchedDocument]:
    deep_documents = [document for document in documents if document.fetch_depth == "deep"]
    if deep_documents:
        selected = list(deep_documents)
        deep_hosts = {_host_root(document.host) for document in deep_documents}
        for document in sorted(
            [document for document in documents if document.fetch_depth in {"shallow", "snippet_only"}],
            key=lambda item: item.source_score,
            reverse=True,
        ):
            root = _host_root(document.host)
            if root in deep_hosts:
                continue
            selected.append(document)
            deep_hosts.add(root)
            if len(selected) >= len(deep_documents) + 2:
                break
        return selected
    snippet_documents = [document for document in documents if document.fetch_depth == "snippet_only"]
    if snippet_documents:
        return snippet_documents
    return [document for document in documents if document.fetch_depth == "shallow"]


def cheap_passage_filter(claim: Claim, passages: list[Passage], limit: int = tuning.CHEAP_PASSAGE_LIMIT) -> list[Passage]:
    scored: list[tuple[float, Passage]] = []
    for passage in passages:
        score = cheap_passage_score(claim, passage)
        if score >= 0.18:
            scored.append((score, passage))
    if not scored:
        scored = [(cheap_passage_score(claim, passage), passage) for passage in passages]
    scored.sort(key=lambda item: (item[0], item[1].source_score), reverse=True)
    return [passage for _, passage in scored[:limit]]


def utility_rerank_passages(claim: Claim, passages: list[Passage], limit: int = tuning.AGENT_PASSAGE_TOP_K) -> list[Passage]:
    if _claim_answer_shape(claim) == "product_specs":
        limit = max(limit, 4)
    reranked = [
        replace(passage, utility_score=utility_score_for_claim(claim, passage))
        for passage in passages
    ]
    reranked.sort(key=lambda item: (item.utility_score, item.source_score), reverse=True)

    selected: list[Passage] = []
    seen_hosts: set[str] = set()
    for passage in reranked:
        root = _host_root(passage.host)
        if root in seen_hosts and len(selected) < max(2, limit // 2):
            continue
        selected.append(passage)
        seen_hosts.add(root)
        if len(selected) >= limit:
            break

    # Diversity floor: if all selected passages share the same host root,
    # include the highest-scoring passage from a different host (if one exists).
    if len(seen_hosts) < 2:
        for passage in reranked:
            if _host_root(passage.host) not in seen_hosts:
                selected.append(passage)
                break

    return selected


def _news_digest_region_hint_from_claim(claim: Claim) -> str | None:
    if claim.entity_set:
        return claim.entity_set[0]
    return _shared_extract_region_hint(claim.claim_text)


def _news_digest_time_match(claim: Claim, passage: Passage) -> float:
    if not claim.time_scope:
        return 0.5
    haystack = f"{passage.title} {passage.text}".casefold()
    if claim.time_scope.casefold() in haystack:
        return 1.0
    if passage.published_at and passage.published_at.startswith(claim.time_scope):
        return 1.0
    if _is_iso_date_text(claim.time_scope):
        return 0.0
    return 0.0


def _local_news_host_bonus(host: str) -> float:
    lowered = (host or "").casefold()
    if lowered.endswith(".kz") or ".kz" in lowered:
        return 1.0
    if any(marker in lowered for marker in ("astana", "kaz", "tengri", "zakon", "inform", "kt.kz")):
        return 0.7
    return 0.0


def _select_passages_from_spans(passages: list[Passage], spans: list[EvidenceSpan]) -> list[Passage]:
    wanted = {span.passage_id for span in spans}
    return [passage for passage in passages if passage.passage_id in wanted]


def _claim_contract_gaps(
    claim: Claim,
    verification: VerificationResult,
    *,
    independent_source_count: int,
    has_primary_source: bool,
    freshness_ok: bool,
) -> list[str]:
    return _policy_claim_contract_gaps(
        claim,
        verification,
        independent_source_count=independent_source_count,
        has_primary_source=has_primary_source,
        freshness_ok=freshness_ok,
    )


def _retrieval_contract_can_drive_synthesis(claim: Claim) -> bool:
    return _policy_retrieval_contract_can_drive_synthesis(claim)


def _primary_domain_types_for_claim(claim: Claim) -> set[DomainType]:
    domain_types: set[DomainType] = {"official", "academic"}
    if _claim_answer_shape(claim) == "product_specs":
        domain_types.add("vendor")
    return domain_types


def _is_primary_support_passage(
    claim: Claim,
    passage: Passage,
    gated_by_url: dict[str, GatedSerpResult] | None = None,
) -> bool:
    if gated_by_url is not None:
        gated = gated_by_url.get(passage.canonical_url) or gated_by_url.get(passage.url)
        if gated and gated.assessment.primary_source_likelihood >= 0.7:
            return True
    return _effective_domain_type(claim, passage.host) in _primary_domain_types_for_claim(claim)


def build_evidence_bundle(
    claim: Claim,
    passages: list[Passage],
    verification: VerificationResult,
    gated_results: list[GatedSerpResult],
) -> EvidenceBundle:
    gated_by_url = {result.serp.canonical_url: result for result in gated_results}
    supporting_passages = _select_passages_from_spans(passages, verification.supporting_spans)
    contradicting_passages = _select_passages_from_spans(passages, verification.contradicting_spans)
    support_limit = max(3, _claim_min_independent_sources(claim) + 1)
    all_supporting = (
        _bundle_support_passages(
            claim,
            supporting_passages,
            passages,
            max_count=support_limit,
            gated_by_url=gated_by_url,
        )
        or passages[: max(2, _claim_min_independent_sources(claim))]
    )
    independent_sources = {_host_root(passage.host) for passage in all_supporting}
    has_primary = any(_is_primary_support_passage(claim, passage, gated_by_url) for passage in all_supporting)
    freshness_ok = True
    if claim.needs_freshness:
        freshness_ok = any(passage.published_at for passage in all_supporting)
    contract_gaps = _claim_contract_gaps(
        claim,
        verification,
        independent_source_count=len(independent_sources),
        has_primary_source=has_primary,
        freshness_ok=freshness_ok,
    )
    contract_satisfied = not contract_gaps and (
        verification.verdict == "supported"
        or (_retrieval_contract_can_drive_synthesis(claim) and bool(all_supporting))
    )

    return EvidenceBundle(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        supporting_passages=all_supporting,
        contradicting_passages=contradicting_passages,
        considered_passages=passages,
        independent_source_count=len(independent_sources),
        has_primary_source=has_primary,
        freshness_ok=freshness_ok,
        verification=verification,
        contract_satisfied=contract_satisfied,
        contract_gaps=contract_gaps,
    )


def _preferred_source_host(gated_results: list[GatedSerpResult]) -> str | None:
    ranked = sorted(
        gated_results,
        key=lambda result: (
            result.assessment.primary_source_likelihood,
            result.assessment.source_score,
        ),
        reverse=True,
    )
    for result in ranked:
        root = _host_root(result.serp.host)
        if result.assessment.primary_source_likelihood >= 0.65 and "." in root:
            return root
    return None


def _candidate_aliases(claim: Claim, gated_results: list[GatedSerpResult]) -> list[str]:
    aliases: list[str] = []
    existing = {entity.casefold() for entity in claim.entity_set}
    for result in gated_results[:6]:
        for entity in _extract_entities(f"{result.serp.title} {result.serp.snippet}"):
            key = entity.casefold()
            if key in existing or key in aliases:
                continue
            aliases.append(entity)
            if len(aliases) >= 3:
                return aliases
    return aliases


def refine_query_variants(
    claim: Claim,
    classification: QueryClassification,
    verification: VerificationResult,
    gated_results: list[GatedSerpResult],
    bundle: EvidenceBundle | None,
    iteration: int,
    existing_queries: set[str],
) -> list[QueryVariant]:
    return []


def _claim_focus_terms(claim: Claim) -> tuple[str, ...]:
    return _policy_claim_focus_terms(claim)


def _focus_term_overlap(claim: Claim, text: str) -> float:
    focus_terms = _claim_focus_terms(claim)
    if not focus_terms:
        return 0.0
    lowered = (text or "").casefold()
    compact = _compact_text(text)
    text_tokens = set(_tokenize(text))
    hits = 0.0
    for term in focus_terms:
        term_text = (term or "").casefold()
        term_key = _compact_text(term)
        if term_text and term_text in lowered:
            hits += 1.0
            continue
        if term_key and term_key in compact:
            hits += 1.0
            continue
        term_tokens = set(_tokenize(term))
        if term_tokens and term_tokens <= text_tokens:
            hits += 1.0
    return _clamp(hits / len(focus_terms))


def _claim_requires_primary_source(claim: Claim) -> bool:
    return _policy_claim_requires_primary_source(claim)


def _claim_min_independent_sources(claim: Claim) -> int:
    return _policy_claim_min_independent_sources(claim)


def _exact_detail_guardrail_claim(claim: Claim) -> bool:
    return _policy_exact_detail_guardrail_claim(claim)


def _is_news_digest_claim(claim: Claim) -> bool:
    return _policy_is_news_digest_claim(claim)


def _dimension_coverage_score(claim: Claim, text: str) -> float:
    lowered = text.casefold()
    score = 0.0
    answer_type = _answer_type(claim)
    focus_overlap = _focus_term_overlap(claim, text)
    if answer_type == "time" and _contains_date_like(text):
        score += 0.35
        score += 0.20 * focus_overlap
    elif answer_type == "number":
        if _shared_extract_numbers(text):
            score += 0.35 if focus_overlap > 0.25 or not _claim_focus_terms(claim) else 0.05
        score += 0.25 * focus_overlap
    elif focus_overlap > 0.0:
        score += 0.35 * focus_overlap
    if claim.time_scope and claim.time_scope.casefold() in lowered:
        score += 0.15
    if answer_type == "person" and _contains_person_span(text):
        score += 0.2
    if answer_type == "location" and _contains_location_span(text):
        score += 0.2
    return _clamp(score)


def route_claim_retrieval(
    claim: Claim,
    gated_results: list[GatedSerpResult],
) -> RoutingDecision:
    top_results = gated_results[:5]
    if not top_results:
        return RoutingDecision(
            mode="iterative_loop",
            certainty=0.0,
            consistency=0.0,
            evidence_sufficiency=0.0,
            rationale="No gated results available.",
        )

    certainty = sum(result.assessment.source_score for result in top_results[:3]) / min(len(top_results[:3]), 3)
    detail_coverages = [
        _dimension_coverage_score(claim, f"{result.serp.title} {result.serp.snippet}")
        for result in top_results
    ]
    dimension_alignment = sum(detail_coverages) / len(detail_coverages)
    max_detail_coverage = max(detail_coverages) if detail_coverages else 0.0

    candidates: list[str] = []
    for result in top_results:
        text = f"{result.serp.title} {result.serp.snippet}"
        candidates.extend([candidate.casefold() for candidate in _extract_answer_candidates(claim, text)])
    if candidates:
        counts: dict[str, int] = {}
        for candidate in candidates:
            counts[candidate] = counts.get(candidate, 0) + 1
        consistency = max(counts.values()) / len(candidates)
    else:
        semantic_scores = [result.assessment.semantic_match_score for result in top_results]
        consistency = sum(semantic_scores) / len(semantic_scores)

    evidence_sufficiency = 0.0
    combined_top_text = " ".join(f"{result.serp.title} {result.serp.snippet}" for result in top_results)
    evidence_sufficiency += min(0.45, 0.11 * sum(result.assessment.source_score >= 0.6 for result in gated_results[:8]))
    evidence_sufficiency += 0.2 if any(result.assessment.primary_source_likelihood >= 0.7 for result in top_results) else 0.0
    evidence_sufficiency += 0.15 if any(result.assessment.entity_match_score >= 0.7 for result in top_results) else 0.0
    evidence_sufficiency += 0.15 if any(result.assessment.semantic_match_score >= 0.7 for result in top_results) else 0.0
    evidence_sufficiency += 0.15 * max_detail_coverage
    evidence_sufficiency += 0.12 * _focus_term_overlap(claim, combined_top_text)
    evidence_sufficiency = _clamp(evidence_sufficiency)
    certainty = _clamp(certainty)
    consistency = _clamp(max(consistency, dimension_alignment * 0.6))

    profile = claim.claim_profile
    answer_shape = profile.answer_shape if profile is not None else "fact"
    routing_bias = profile.routing_bias if profile is not None else None
    open_ended = answer_shape in {"product_specs", "overview", "comparison", "news_digest"}
    exact_detail_request = _exact_detail_guardrail_claim(claim)
    answer_type = _answer_type(claim)
    number_targeted_threshold = 0.4 if profile is not None and profile.answer_shape == "exact_number" and not profile.strict_contract else 0.45

    if exact_detail_request and (max_detail_coverage < 0.45 or _focus_term_overlap(claim, combined_top_text) < 0.6):
        mode = "iterative_loop"
    elif routing_bias == "iterative_loop":
        mode = "iterative_loop"
    elif not open_ended and certainty >= 0.8 and consistency >= 0.35 and evidence_sufficiency >= 0.6:
        mode = "short_path"
    elif answer_type == "number" and certainty >= number_targeted_threshold and max_detail_coverage >= 0.25 and evidence_sufficiency >= 0.35:
        mode = "targeted_retrieval"
    elif answer_type == "time" and certainty >= 0.45 and max_detail_coverage >= 0.25 and evidence_sufficiency >= 0.35:
        mode = "targeted_retrieval"
    elif not open_ended and certainty >= 0.5 and consistency >= 0.5 and evidence_sufficiency >= 0.5:
        mode = "targeted_retrieval"
    elif certainty >= 0.55 and evidence_sufficiency >= 0.45:
        mode = "targeted_retrieval"
    else:
        mode = "iterative_loop"

    return RoutingDecision(
        mode=mode,
        certainty=certainty,
        consistency=consistency,
        evidence_sufficiency=evidence_sufficiency,
        rationale=(
            f"certainty={certainty:.2f}, consistency={consistency:.2f}, "
            f"evidence_sufficiency={evidence_sufficiency:.2f}, "
            f"dimension_alignment={dimension_alignment:.2f}"
        ),
    )


def cheap_passage_score(claim: Claim, passage: Passage) -> float:
    overlap = _semantic_overlap(claim.claim_text, passage.text)
    entity_overlap = _entity_overlap(claim.entity_set, passage.text)
    dimension_overlap = _dimension_coverage_score(claim, passage.text)
    focus_overlap = _focus_term_overlap(claim, passage.text)
    claim_numbers = set(_shared_extract_numbers(claim.claim_text))
    passage_numbers = set(_shared_extract_numbers(passage.text))
    number_overlap = 1.0 if claim_numbers and claim_numbers & passage_numbers else 0.0

    if _is_news_digest_claim(claim):
        region = _news_digest_region_hint_from_claim(claim)
        haystack = f"{passage.title} {passage.text[:220]} {passage.url}"
        region_match = _entity_overlap([region], haystack) if region else entity_overlap
        time_match = _news_digest_time_match(claim, passage)
        return _clamp(
            0.18 * overlap
            + 0.28 * region_match
            + 0.22 * time_match
            + 0.16 * focus_overlap
            + 0.16 * passage.source_score
        )

    if _claim_answer_shape(claim) == "product_specs":
        preferred_bonus = _preferred_domain_bonus(claim, _effective_domain_type(claim, passage.host))
        return _clamp(
            0.28 * overlap
            + 0.20 * entity_overlap
            + 0.16 * dimension_overlap
            + 0.16 * focus_overlap
            + 0.10 * passage.source_score
            + 0.10 * preferred_bonus
        )

    return _clamp(
        0.30 * overlap
        + 0.22 * entity_overlap
        + 0.18 * dimension_overlap
        + 0.15 * focus_overlap
        + 0.07 * number_overlap
        + 0.08 * passage.source_score
    )


def utility_score_for_claim(claim: Claim, passage: Passage) -> float:
    source_bonus = _verification_source_bonus(
        claim,
        host=passage.host,
        title=passage.title,
        url=passage.url,
    )
    focus_overlap = _focus_term_overlap(claim, passage.text)

    if _is_news_digest_claim(claim):
        region = _news_digest_region_hint_from_claim(claim)
        haystack = f"{passage.title} {passage.text[:220]} {passage.url}"
        region_match = _entity_overlap([region], haystack) if region else _entity_overlap(claim.entity_set, haystack)
        time_match = _news_digest_time_match(claim, passage)
        return _clamp(
            0.24 * cheap_passage_score(claim, passage)
            + 0.24 * region_match
            + 0.18 * time_match
            + 0.16 * focus_overlap
            + 0.10 * max(source_bonus, 0.0)
            + 0.08 * passage.source_score
        )

    if _claim_answer_shape(claim) == "product_specs":
        preferred_bonus = _preferred_domain_bonus(claim, _effective_domain_type(claim, passage.host))
        return _clamp(
            0.36 * cheap_passage_score(claim, passage)
            + 0.18 * focus_overlap
            + 0.16 * preferred_bonus
            + 0.18 * max(source_bonus, 0.0)
            + 0.12 * passage.source_score
        )

    directness = 0.0
    answer_type = _answer_type(claim)
    if answer_type == "time" and _contains_date_like(passage.text):
        directness += 0.3
    elif answer_type == "number" and _shared_extract_numbers(passage.text):
        directness += 0.3
    elif answer_type == "person" and _contains_person_span(passage.text):
        directness += 0.2
    elif answer_type == "location" and _contains_location_span(passage.text):
        directness += 0.2

    contradiction_signal = 0.15 if _contains_negation_cue(passage.text.casefold()) else 0.0
    return _clamp(
        0.30 * cheap_passage_score(claim, passage)
        + 0.20 * _dimension_coverage_score(claim, passage.text)
        + 0.15 * focus_overlap
        + 0.15 * directness
        + 0.12 * max(source_bonus, 0.0)
        + 0.08 * passage.source_score
        + 0.05 * contradiction_signal
    )


def should_stop_claim_loop(claim: Claim, bundle: EvidenceBundle, iteration: int) -> bool:
    return _policy_should_stop_claim_loop(claim, bundle, iteration)


def _truncate_compose_line(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _sanitize_compose_fragment(text: str) -> str:
    """Remove Markdown heading markers that leak from source pages into one line."""
    if not text:
        return text
    parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.lstrip()
        while line.startswith("#"):
            line = line[1:].lstrip()
        if line:
            parts.append(line)
    if not parts:
        return ""
    return " ".join(" ".join(parts).split())


def _compose_ui_labels_legacy_mojibake(user_query: str) -> dict[str, str]:
    """Section titles: Russian if the user query contains Cyrillic, else English."""
    rq = user_query or ""
    ru = _is_cyrillic_text(rq)
    if ru:
        return {
            "answer": "Ответ",
            "sources": "Источники",
            "insufficient": "Недостаточно данных",
            "caveats": "Оговорки",
            "no_evidence": "Недостаточно подтверждённых утверждений для прямого ответа.",
            "digest_header": "События из источников:",
        }
    return {
        "answer": "Answer",
        "sources": "Sources",
        "insufficient": "Insufficient data",
        "caveats": "Caveats",
        "no_evidence": "Not enough claim-level evidence for a direct answer.",
        "digest_header": "Events from retrieved sources:",
    }


def _best_sentence_for_claim(claim: Claim, passage: Passage) -> str:
    cap = tuning.COMPOSE_ANSWER_MAX_SPAN_CHARS
    head = passage.text[: max(cap, 4000)]
    sentences = _split_sentences(head)
    if not sentences:
        return _truncate_compose_line(passage.text, cap)
    scored = sorted(
        sentences,
        key=lambda sentence: _answer_sentence_score(claim, sentence),
        reverse=True,
    )
    best = _normalized_text(scored[0]) if scored else _truncate_compose_line(passage.text, cap)
    return _truncate_compose_line(best, cap)


def _best_span_text(verification: VerificationResult, passages: list[Passage], claim: Claim, contradicted: bool = False) -> str | None:
    cap = tuning.COMPOSE_ANSWER_MAX_SPAN_CHARS
    spans = verification.contradicting_spans if contradicted else verification.supporting_spans
    if spans:
        text = _normalized_text(spans[0].text)
        return _truncate_compose_line(text, cap)
    if passages:
        return _best_sentence_for_claim(claim, passages[0])
    return None


def _format_citations(url_to_index: dict[str, int], passages: list[Passage]) -> str:
    seen: list[int] = []
    for passage in passages:
        idx = url_to_index.get(passage.url)
        if idx is not None and idx not in seen:
            seen.append(idx)
    return "".join(f"[{idx}]" for idx in seen)


def _extract_citation_indices(text: str) -> set[int]:
    cited: set[int] = set()
    content = text or ""
    index = 0
    while index < len(content):
        if content[index] != "[":
            index += 1
            continue
        end = index + 1
        while end < len(content) and content[end].isdigit():
            end += 1
        if end > index + 1 and end < len(content) and content[end] == "]":
            cited.add(int(content[index + 1:end]))
            index = end + 1
            continue
        index += 1
    return cited


def _digest_sentence(passage: Passage) -> str:
    sentence = _best_sentence_for_claim(
        Claim(
            claim_id="digest",
            claim_text=passage.title,
            priority=1,
            needs_freshness=False,
        ),
        passage,
    )
    title = _normalized_text(passage.title)
    if title and title.casefold() not in sentence.casefold():
        combined = f"{title}. {sentence}"
    else:
        combined = sentence
    return _truncate_compose_line(combined, tuning.COMPOSE_ANSWER_DIGEST_LINE_CHARS)


def _aligned_news_digest_passages(run: ClaimRun) -> list[Passage]:
    bundle = run.evidence_bundle
    if bundle is None:
        return []

    region = _news_digest_region_hint_from_claim(run.claim)
    scored: list[tuple[float, Passage]] = []
    for passage in bundle.considered_passages or bundle.supporting_passages:
        lead = f"{passage.title} {passage.text[:220]}"
        region_match = _entity_overlap([region], lead) if region else 0.0
        time_match = _news_digest_time_match(run.claim, passage)
        local_bonus = _local_news_host_bonus(passage.host)
        if region and region_match < 0.25:
            continue
        if run.claim.time_scope and time_match <= 0.0:
            continue
        score = (
            0.42 * region_match
            + 0.25 * time_match
            + 0.15 * local_bonus
            + 0.18 * max(passage.utility_score, passage.source_score)
        )
        scored.append((score, passage))

    if not scored:
        for passage in bundle.supporting_passages:
            scored.append((max(passage.utility_score, passage.source_score), passage))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[Passage] = []
    seen_urls: set[str] = set()
    for _, passage in scored:
        if passage.url in seen_urls:
            continue
        seen_urls.add(passage.url)
        selected.append(passage)
        if len(selected) >= 4:
            break
    return selected


def _compose_ui_labels_mojibake(user_query: str) -> dict[str, str]:
    rq = user_query or ""
    ru = _is_cyrillic_text(rq)
    if ru:
        return {
            "answer": "РћС‚РІРµС‚",
            "sources": "РСЃС‚РѕС‡РЅРёРєРё",
            "insufficient": "РќРµРґРѕСЃС‚Р°С‚РѕС‡РЅРѕ РґР°РЅРЅС‹С…",
            "caveats": "РћРіРѕРІРѕСЂРєРё",
            "no_evidence": "РќРµРґРѕСЃС‚Р°С‚РѕС‡РЅРѕ РїРѕРґС‚РІРµСЂР¶РґС‘РЅРЅС‹С… СѓС‚РІРµСЂР¶РґРµРЅРёР№ РґР»СЏ РїСЂСЏРјРѕРіРѕ РѕС‚РІРµС‚Р°.",
            "digest_header": "РЎРѕР±С‹С‚РёСЏ РёР· РёСЃС‚РѕС‡РЅРёРєРѕРІ:",
        }
    return {
        "answer": "Answer",
        "sources": "Sources",
        "insufficient": "Insufficient data",
        "caveats": "Caveats",
        "no_evidence": "Not enough claim-level evidence for a direct answer.",
        "digest_header": "Events from retrieved sources:",
    }


def _compose_ui_labels(user_query: str) -> dict[str, str]:
    rq = user_query or ""
    ru = _is_cyrillic_text(rq)
    if ru:
        return {
            "answer": "\u041e\u0442\u0432\u0435\u0442",
            "sources": "\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0438",
            "insufficient": "\u041d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0434\u0430\u043d\u043d\u044b\u0445",
            "caveats": "\u041e\u0433\u043e\u0432\u043e\u0440\u043a\u0438",
            "no_evidence": "\u041d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0451\u043d\u043d\u044b\u0445 \u0443\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u0438\u0439 \u0434\u043b\u044f \u043f\u0440\u044f\u043c\u043e\u0433\u043e \u043e\u0442\u0432\u0435\u0442\u0430.",
            "digest_header": "\u0421\u043e\u0431\u044b\u0442\u0438\u044f \u0438\u0437 \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u043e\u0432:",
        }
    return {
        "answer": "Answer",
        "sources": "Sources",
        "insufficient": "Insufficient data",
        "caveats": "Caveats",
        "no_evidence": "Not enough claim-level evidence for a direct answer.",
        "digest_header": "Events from retrieved sources:",
    }


def _time_specificity_score(text: str) -> float:
    scope = _extract_time_scope(text)
    if not scope:
        return 0.0
    if _is_iso_date_text(scope):
        return 1.0
    if _is_year_text(scope):
        return 0.2
    if any(ch.isalpha() for ch in scope) and any(ch.isdigit() for ch in scope):
        return 0.95
    if scope.casefold().startswith("q"):
        return 0.7
    return 0.5


def _answer_sentence_score(claim: Claim, sentence: str) -> float:
    score = _semantic_overlap(claim.claim_text, sentence) + 0.15 * _entity_overlap(claim.entity_set, sentence)
    answer_type = _answer_type(claim)
    if answer_type == "time":
        score += 0.65 * _time_specificity_score(sentence)
    elif answer_type == "number" and _shared_extract_numbers(sentence):
        score += 0.35
    elif answer_type == "person" and _contains_person_span(sentence):
        score += 0.2
    elif answer_type == "location" and _contains_location_span(sentence):
        score += 0.2
    if claim.time_scope and claim.time_scope.casefold() in sentence.casefold():
        score += 0.1
    return score


def _best_sentence_for_claim(claim: Claim, passage: Passage) -> str:
    cap = tuning.COMPOSE_ANSWER_MAX_SPAN_CHARS
    head = passage.text[: max(cap, 4000)]
    sentences = _split_sentences(head)
    if not sentences:
        return _truncate_compose_line(passage.text, cap)
    scored = sorted(sentences, key=lambda sentence: _answer_sentence_score(claim, sentence), reverse=True)
    best = _normalized_text(scored[0]) if scored else _truncate_compose_line(passage.text, cap)
    return _truncate_compose_line(best, cap)


def _best_sentence_from_passages(claim: Claim, passages: list[Passage]) -> str | None:
    best_sentence: str | None = None
    best_score = -1.0
    for passage in passages[:3]:
        sentence = _best_sentence_for_claim(claim, passage)
        score = _answer_sentence_score(claim, sentence)
        if score > best_score:
            best_sentence = sentence
            best_score = score
    return best_sentence


def _best_span_text(verification: VerificationResult, passages: list[Passage], claim: Claim, contradicted: bool = False) -> str | None:
    cap = tuning.COMPOSE_ANSWER_MAX_SPAN_CHARS
    spans = verification.contradicting_spans if contradicted else verification.supporting_spans
    if spans:
        text = _normalized_text(spans[0].text)
        if not contradicted:
            answer_type = _answer_type(claim)
            fallback = _best_sentence_from_passages(claim, passages)
            if answer_type == "time":
                if fallback and _time_specificity_score(fallback) > _time_specificity_score(text):
                    return _truncate_compose_line(fallback, cap)
            elif answer_type == "number":
                if fallback and not _shared_extract_numbers(text):
                    return _truncate_compose_line(fallback, cap)
        return _truncate_compose_line(text, cap)
    fallback = _best_sentence_from_passages(claim, passages)
    if fallback:
        return _truncate_compose_line(fallback, cap)
    return None


def _bundle_support_passages(
    claim: Claim,
    supporting_passages: list[Passage],
    passages: list[Passage],
    *,
    max_count: int,
    gated_by_url: dict[str, GatedSerpResult] | None = None,
) -> list[Passage]:
    selected: list[Passage] = []
    seen_urls: set[str] = set()
    seen_hosts: set[str] = set()

    def add(passage: Passage) -> None:
        url = passage.url or passage.canonical_url
        host = _host_root(passage.host)
        if url in seen_urls:
            return
        seen_urls.add(url)
        if host:
            seen_hosts.add(host)
        selected.append(passage)

    def support_score(passage: Passage) -> float:
        lead = f"{passage.title} {passage.text[:320]} {passage.url}"
        score = (
            0.36 * _semantic_overlap(claim.claim_text, lead)
            + 0.16 * _entity_overlap(claim.entity_set, lead)
            + 0.16 * _dimension_coverage_score(claim, lead)
            + 0.16 * max(passage.utility_score, passage.source_score)
            + 0.10 * max(
                _verification_source_bonus(
                    claim,
                    host=passage.host,
                    title=passage.title,
                    url=passage.url,
                ),
                0.0,
            )
            + 0.06 * _preferred_domain_bonus(claim, _effective_domain_type(claim, passage.host))
        )
        if _claim_answer_shape(claim) == "product_specs":
            score += 0.14 * _preferred_domain_bonus(claim, _effective_domain_type(claim, passage.host))
        return score

    for passage in sorted(
        supporting_passages,
        key=lambda item: (support_score(item), item.source_score),
        reverse=True,
    ):
        add(passage)
        if len(selected) >= max_count:
            return selected[:max_count]

    if _claim_requires_primary_source(claim) and not any(
        _is_primary_support_passage(claim, passage, gated_by_url) for passage in selected
    ):
        for passage in sorted(
            passages,
            key=lambda item: (support_score(item), item.source_score),
            reverse=True,
        ):
            if not _is_primary_support_passage(claim, passage, gated_by_url):
                continue
            add(passage)
            if len(selected) >= max_count:
                return selected[:max_count]
            break

    ranked = sorted(
        passages,
        key=lambda passage: (
            support_score(passage),
            passage.source_score,
        ),
        reverse=True,
    )
    for passage in ranked:
        host = _host_root(passage.host)
        if host in seen_hosts and seen_hosts:
            continue
        add(passage)
        if len(selected) >= max_count or len(seen_hosts) >= 2:
            break
    return selected[:max_count]


def _supporting_answer_passages(claim: Claim, bundle: EvidenceBundle, *, max_count: int) -> list[Passage]:
    selected: list[Passage] = []
    seen_urls: set[str] = set()
    seen_hosts: set[str] = set()

    def add(passage: Passage) -> None:
        url = passage.url or passage.canonical_url
        host = _host_root(passage.host)
        if url in seen_urls:
            return
        seen_urls.add(url)
        if host:
            seen_hosts.add(host)
        selected.append(passage)

    def support_score(passage: Passage) -> float:
        lead = f"{passage.title} {passage.text[:320]} {passage.url}"
        score = (
            0.45 * _semantic_overlap(claim.claim_text, lead)
            + 0.20 * _entity_overlap(claim.entity_set, lead)
            + 0.20 * _dimension_coverage_score(claim, lead)
            + 0.08 * max(
                _verification_source_bonus(
                    claim,
                    host=passage.host,
                    title=passage.title,
                    url=passage.url,
                ),
                0.0,
            )
            + 0.07 * max(passage.utility_score, passage.source_score)
        )
        if _claim_answer_shape(claim) == "product_specs":
            score += (
                0.18 * _preferred_domain_bonus(claim, _effective_domain_type(claim, passage.host))
            )
        return score

    for passage in sorted(
        bundle.supporting_passages,
        key=lambda item: (support_score(item), item.source_score),
        reverse=True,
    ):
        add(passage)
        if len(selected) >= max_count:
            return selected[:max_count]

    for passage in sorted(
        bundle.considered_passages,
        key=lambda item: (support_score(item), item.source_score),
        reverse=True,
    ):
        host = _host_root(passage.host)
        if host in seen_hosts and seen_hosts:
            continue
        add(passage)
        if len(selected) >= max_count or len(seen_hosts) >= 2:
            break

    if not selected:
        for passage in bundle.considered_passages[:max_count]:
            add(passage)
    return selected[:max_count]


def _publish_supported_claim(claim: Claim, bundle: EvidenceBundle) -> bool:
    return _policy_publish_supported_claim(claim, bundle)


def _contract_gap_text(claim: Claim, bundle: EvidenceBundle, *, cyrillic: bool = False) -> str:
    if not bundle.contract_gaps:
        return "покрытие" if cyrillic else "coverage"

    min_sources = _claim_min_independent_sources(claim)
    labels: list[str] = []
    for gap in bundle.contract_gaps:
        if gap == "primary_source":
            labels.append(
                "нужно подтверждение из первичного источника"
                if cyrillic
                else "needs primary-source evidence"
            )
        elif gap == "independent_sources":
            labels.append(
                f"нужно минимум {min_sources} независимых источника"
                if cyrillic
                else f"needs {min_sources} independent sources"
            )
        elif gap == "freshness":
            labels.append(
                "нужно свежее датированное подтверждение"
                if cyrillic
                else "needs fresh dated evidence"
            )
        else:
            labels.append(gap.replace("_", " "))
    return ", ".join(labels)


def compose_answer(report: AgentRunResult) -> str:
    lb = _compose_ui_labels(report.user_query)
    ru = _is_cyrillic_text(report.user_query)
    if report.classification.intent == "news_digest":
        selected_passages: list[Passage] = []
        for run in sorted(report.claims, key=lambda item: item.claim.priority):
            bundle = run.evidence_bundle
            if bundle is None or bundle.verification is None or bundle.verification.verdict != "supported":
                continue
            selected_passages.extend(_aligned_news_digest_passages(run))

        url_to_index: dict[str, int] = {}
        indexed_sources: list[tuple[int, str, str]] = []
        digest_lines: list[str] = []
        for passage in selected_passages:
            if passage.url not in url_to_index:
                idx = len(url_to_index) + 1
                url_to_index[passage.url] = idx
                indexed_sources.append((idx, passage.title, passage.url))
            digest_lines.append(f"- {_sanitize_compose_fragment(_digest_sentence(passage))} {_format_citations(url_to_index, [passage])}".rstrip())

        if digest_lines:
            lines = [f"- {lb['digest_header']}"]
            lines.extend(digest_lines[:4])
            if indexed_sources:
                lines.append("")
                lines.append(lb["sources"])
                for idx, title, url in indexed_sources:
                    lines.append(f"[{idx}] {title} - {url}")
            return "\n".join(lines)

    cited_passages: list[Passage] = []
    for run in report.claims:
        bundle = run.evidence_bundle
        if bundle is None or bundle.verification is None:
            continue
        if bundle.verification.verdict == "supported":
            if not _publish_supported_claim(run.claim, bundle):
                continue
            cited_passages.extend(_supporting_answer_passages(run.claim, bundle, max_count=3))
        elif bundle.verification.verdict == "contradicted":
            cited_passages.extend((bundle.contradicting_passages[:2] or bundle.considered_passages[:2]))

    url_to_index: dict[str, int] = {}
    indexed_sources: list[tuple[int, str, str]] = []
    for passage in cited_passages:
        if passage.url in url_to_index:
            continue
        idx = len(url_to_index) + 1
        url_to_index[passage.url] = idx
        indexed_sources.append((idx, passage.title, passage.url))

    supported_lines: list[str] = []
    caveat_lines: list[str] = []
    gap_lines: list[str] = []

    for run in sorted(report.claims, key=lambda item: item.claim.priority):
        bundle = run.evidence_bundle
        if bundle is None or bundle.verification is None:
            continue
        verification = bundle.verification
        if verification.verdict == "supported":
            if not _publish_supported_claim(run.claim, bundle):
                verdict_text = "недостаточно данных" if ru else "insufficient evidence"
                gap_lines.append(
                    f"- {run.claim.claim_text}: {verdict_text} ({_contract_gap_text(run.claim, bundle, cyrillic=ru)})."
                )
                continue
            passages = _supporting_answer_passages(run.claim, bundle, max_count=2)
            if not passages:
                continue
            sentence = _best_span_text(verification, passages, run.claim)
            if not sentence:
                continue
            sentence = _sanitize_compose_fragment(sentence)
            citations = _format_citations(url_to_index, passages)
            qualifier = ""
            min_sources = _claim_min_independent_sources(run.claim)
            if bundle.independent_source_count < min_sources:
                qualifier = (
                    f" (подтверждено менее чем {min_sources} независимыми источниками)"
                    if ru
                    else f" (supported by fewer than {min_sources} independent sources)"
                )
            elif _claim_requires_primary_source(run.claim) and not bundle.has_primary_source:
                qualifier = (
                    " (без обнаруженного первичного источника)"
                    if ru
                    else " (without a detected primary source)"
                )
            elif run.claim.needs_freshness and not bundle.freshness_ok:
                qualifier = (
                    " (без свежего датированного подтверждения)"
                    if ru
                    else " (without fresh dated evidence)"
                )
            supported_lines.append(f"- {sentence}{qualifier} {citations}".rstrip())
        elif verification.verdict == "contradicted":
            passages = bundle.contradicting_passages[:2] or bundle.considered_passages[:1]
            if not passages:
                continue
            sentence = _best_span_text(verification, passages, run.claim, contradicted=True)
            if not sentence:
                continue
            sentence = _sanitize_compose_fragment(sentence)
            citations = _format_citations(url_to_index, passages)
            verdict_text = "противоречит данным" if ru else "contradicted"
            caveat_lines.append(f"- {run.claim.claim_text}: {verdict_text}. {sentence} {citations}".rstrip())
        else:
            missing = ", ".join(verification.missing_dimensions) if verification.missing_dimensions else (
                "покрытие" if ru else "coverage"
            )
            verdict_text = "недостаточно данных" if ru else "insufficient evidence"
            gap_lines.append(f"- {run.claim.claim_text}: {verdict_text} ({missing}).")

    lines: list[str] = []
    if supported_lines:
        lines.extend(supported_lines)
    else:
        lines.append(lb["answer"])
        lines.append(lb["no_evidence"])

    if caveat_lines:
        lines.append("")
        lines.append(lb["caveats"])
        lines.extend(caveat_lines)

    if gap_lines:
        lines.append("")
        lines.append(lb["insufficient"])
        lines.extend(gap_lines)

    used_source_indices = set()
    for line in supported_lines + caveat_lines:
        used_source_indices.update(_extract_citation_indices(line))

    if indexed_sources:
        lines.append("")
        lines.append(lb["sources"])
        for idx, title, url in indexed_sources:
            if used_source_indices and idx not in used_source_indices:
                continue
            lines.append(f"[{idx}] {title} - {url}")

    return "\n".join(lines)


def _estimate_search_cost(claim_runs: list[ClaimRun]) -> float:
    shallow = 0
    deep = 0
    snippet_only = 0
    search_queries = 0
    for run in claim_runs:
        search_queries += len(run.query_variants)
        for plan in run.fetch_plans:
            if plan.depth == "shallow":
                shallow += 1
            elif plan.depth == "deep":
                deep += 1
            else:
                snippet_only += 1
    return round(
        search_queries
        + 0.25 * shallow
        + 1.0 * deep
        + 0.10 * snippet_only
        + 0.50 * len(claim_runs),
        3,
    )


def run_search_agent(
    query: str,
    profile,
    client=None,
    receipts_dir: str | None = None,
    log=None,
) -> AgentRunResult:
    from search_agent import build_search_agent_use_case

    return build_search_agent_use_case().run(
        query,
        profile,
        receipts_dir=receipts_dir,
        log=log,
    )
