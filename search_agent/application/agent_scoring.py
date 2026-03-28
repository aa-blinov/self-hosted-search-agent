from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime

from search_agent import tuning
from search_agent.application import policy_tuning
from search_agent.application.claim_policy import (
    answer_type as _policy_answer_type,
    claim_answer_shape as _policy_claim_answer_shape,
    claim_contract_gaps as _policy_claim_contract_gaps,
    claim_focus_terms as _policy_claim_focus_terms,
    claim_min_independent_sources as _policy_claim_min_independent_sources,
    claim_requires_primary_source as _policy_claim_requires_primary_source,
    exact_detail_guardrail_claim as _policy_exact_detail_guardrail_claim,
    is_news_digest_claim as _policy_is_news_digest_claim,
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
    normalized_text as _shared_normalized_text,
    tokenize as _shared_tokenize,
)
from search_agent.domain.models import (
    Claim,
    DomainType,
    EvidenceBundle,
    EvidenceSpan,
    FetchedDocument,
    FetchPlan,
    GatedSerpResult,
    Passage,
    RoutingDecision,
    SourceAssessment,
    VerificationResult,
)
from search_agent.domain.source_priors import lookup_source_prior

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
    "официал", "пресс-релиз", "отчет", "документация", "исследование",
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
        if len(_tokenize(entity)) >= 2:
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


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return _normalized_text(stripped.lstrip("#").strip())
    return None


def _claim_answer_shape(claim: Claim) -> str:
    return _policy_claim_answer_shape(claim)


def _claim_focus_terms(claim: Claim) -> tuple[str, ...]:
    return _policy_claim_focus_terms(claim)


def _claim_requires_primary_source(claim: Claim) -> bool:
    return _policy_claim_requires_primary_source(claim)


def _claim_min_independent_sources(claim: Claim) -> int:
    return _policy_claim_min_independent_sources(claim)


def _exact_detail_guardrail_claim(claim: Claim) -> bool:
    return _policy_exact_detail_guardrail_claim(claim)


def _is_news_digest_claim(claim: Claim) -> bool:
    return _policy_is_news_digest_claim(claim)


def _preferred_domain_bonus(claim: Claim, domain_type: DomainType) -> float:
    profile = claim.claim_profile
    if profile is None or not profile.preferred_domain_types:
        return 0.0
    return policy_tuning.PREFERRED_DOMAIN_TYPE_BONUS if domain_type in profile.preferred_domain_types else 0.0


def _product_specs_result_bonus(claim: Claim, title: str, snippet: str, url: str) -> float:
    return 0.0


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
                best = max(best, policy_tuning.ENTITY_HOST_MATCH_TOKEN_SCORE)
    return best


def _effective_domain_type(claim: Claim, host: str) -> DomainType:
    prior = lookup_source_prior(host)
    if prior.domain_type_override in {"official", "academic", "vendor", "major_media", "forum", "unknown"}:
        return prior.domain_type_override
    base = _domain_type(host)
    if base != "unknown":
        return base
    if _entity_host_match_score(claim, host) >= policy_tuning.EFFECTIVE_DOMAIN_ENTITY_MATCH_THRESHOLD:
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
            hits += policy_tuning.ENTITY_OVERLAP_PARTIAL_MATCH_SCORE
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
    if _is_iso_date_text(claim.time_scope) and result.published_at and result.published_at[:7] == claim.time_scope[:7]:
        return policy_tuning.TIME_SCOPE_MONTH_MATCH_SCORE
    if _is_year_text(claim.time_scope) and result.published_at and result.published_at.startswith(claim.time_scope):
        return policy_tuning.TIME_SCOPE_YEAR_MATCH_SCORE
    return 0.0


def _freshness_score(claim: Claim, result) -> float:
    if not claim.needs_freshness and not result.published_at:
        return policy_tuning.FRESHNESS_NEUTRAL_SCORE
    if not result.published_at:
        return (
            policy_tuning.FRESHNESS_MISSING_REQUIRED_SCORE
            if claim.needs_freshness
            else policy_tuning.FRESHNESS_MISSING_OPTIONAL_SCORE
        )
    try:
        published = datetime.fromisoformat(result.published_at.replace("Z", "+00:00"))
        age_days = max(0, (datetime.now(UTC) - published.astimezone(UTC)).days)
    except ValueError:
        return (
            policy_tuning.FRESHNESS_MISSING_REQUIRED_SCORE
            if claim.needs_freshness
            else policy_tuning.FRESHNESS_MISSING_OPTIONAL_SCORE
        )
    window = (
        policy_tuning.FRESHNESS_REQUIRED_WINDOW_DAYS
        if claim.needs_freshness
        else policy_tuning.FRESHNESS_OPTIONAL_WINDOW_DAYS
    )
    return _clamp(1 - (age_days / max(window, 1)))


def _spam_risk(result) -> float:
    host = result.host.casefold()
    text = f"{result.title} {result.snippet}".casefold()
    prior = lookup_source_prior(host)
    risk = 0.0
    if host.endswith(".top") or host.endswith(".best"):
        risk += policy_tuning.SPAM_SUFFIX_PENALTY
    if any(cue in text for cue in _SPAM_CUES):
        risk += policy_tuning.SPAM_CUE_PENALTY
    if text.count("|") >= 2 or text.count(" - ") >= 3:
        risk += policy_tuning.SPAM_TITLE_SEPARATOR_PENALTY
    if len(_tokenize(result.title)) > 14:
        risk += policy_tuning.SPAM_LONG_TITLE_PENALTY
    if _domain_type(host) in {"official", "academic"}:
        risk -= policy_tuning.SPAM_AUTHORITATIVE_DOMAIN_DISCOUNT
    risk += prior.spam_penalty
    return _clamp(risk)


def _primary_source_likelihood(claim: Claim, result, domain_type: DomainType) -> float:
    prior = lookup_source_prior(result.host)
    base = policy_tuning.PRIMARY_SOURCE_BASE_BY_DOMAIN_TYPE[domain_type]
    text = f"{result.title} {result.snippet} {result.url}".casefold()
    if any(cue in text for cue in _PRIMARY_SOURCE_CUES):
        base += policy_tuning.PRIMARY_SOURCE_CUE_BOOST
    base += policy_tuning.PRIMARY_SOURCE_ENTITY_HOST_MATCH_WEIGHT * _entity_host_match_score(claim, result.host)
    if any(path_cue in text for path_cue in ("/announcement/", "/press", "/release", "/releases/", "/downloads/release/", "/whatsnew/")):
        base += policy_tuning.PRIMARY_SOURCE_PATH_CUE_BOOST
    if domain_type == "forum":
        base -= policy_tuning.PRIMARY_SOURCE_FORUM_PENALTY
    base += prior.primary_boost
    return _clamp(base)


def gate_serp_results(
    claim: Claim,
    snapshots: list,
    limit: int,
) -> list[GatedSerpResult]:
    merged: dict[str, GatedSerpResult] = {}

    for snapshot in snapshots:
        for result in snapshot.results:
            domain_type = _effective_domain_type(claim, result.host)
            prior = lookup_source_prior(result.host)
            domain_prior = policy_tuning.SERP_DOMAIN_PRIOR_BY_TYPE[domain_type]
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
                policy_tuning.SERP_DOMAIN_PRIOR_WEIGHT * domain_prior
                + policy_tuning.SERP_PRIMARY_WEIGHT * primary
                + policy_tuning.SERP_FRESHNESS_WEIGHT * freshness
                + policy_tuning.SERP_ENTITY_MATCH_WEIGHT * entity_match
                + policy_tuning.SERP_SEMANTIC_MATCH_WEIGHT * semantic_match
                + policy_tuning.SERP_HOST_ENTITY_WEIGHT * host_entity_match
                + policy_tuning.SERP_TIME_ALIGNMENT_WEIGHT * time_alignment
                + preferred_domain_bonus
                + product_bonus
                + prior.source_prior
                - policy_tuning.SERP_SPAM_PENALTY_WEIGHT * spam
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
    bonus = policy_tuning.VERIFICATION_BONUS_BY_DOMAIN_TYPE[domain_type]
    lowered = f"{title} {url}".casefold()
    if any(cue in lowered for cue in ("announcement", "press", "release", "released", "downloads/release", "whatsnew", "/blog/")):
        bonus += policy_tuning.VERIFICATION_RELEASE_CUE_BOOST
    if any(cue in lowered for cue in ("hacker news", "reddit", "forum", "comment")):
        bonus -= policy_tuning.VERIFICATION_FORUM_CUE_PENALTY
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
    weights = policy_tuning.SHALLOW_DOCUMENT_SCORE_WEIGHTS
    return _clamp(
        weights["semantic_overlap"] * _semantic_overlap(claim.claim_text, overview)
        + weights["entity_overlap"] * _entity_overlap(claim.entity_set, overview)
        + weights["dimension_coverage"] * _dimension_coverage_score(claim, overview)
        + weights["source_score"] * document.source_score
        + weights["source_bonus"] * max(source_bonus, 0.0)
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
    now = datetime.now(UTC).isoformat()
    passages: list[Passage] = []
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

    plans = [
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
            and preferred_bonus > policy_tuning.DEEP_FETCH_PREFERRED_BONUS_THRESHOLD
        ):
            if len(deep_candidates) < deep_limit:
                deep_candidates.append((preferred_candidate, preferred_document))
            elif deep_candidates:
                deep_candidates[-1] = (preferred_candidate, preferred_document)

    deep_documents: list[FetchedDocument] = []
    for candidate, _ in deep_candidates:
        plans.append(
            FetchPlan(
                depth="deep",
                url=candidate.serp.url,
                reason="Selective deep fetch after shallow rerank.",
                source_score=candidate.assessment.source_score,
            )
        )
    if deep_candidates:
        deep_urls = [candidate.serp.url for candidate, _ in deep_candidates]
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
    if not deep_documents and selected:
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


def _dimension_coverage_score(claim: Claim, text: str) -> float:
    lowered = text.casefold()
    score = 0.0
    answer_type = _answer_type(claim)
    focus_overlap = _focus_term_overlap(claim, text)
    weights = policy_tuning.DIMENSION_COVERAGE
    if answer_type == "time" and _contains_date_like(text):
        score += weights["time_base"]
        score += weights["time_focus_weight"] * focus_overlap
    elif answer_type == "number":
        if _shared_extract_numbers(text):
            score += (
                weights["number_base_with_focus"]
                if focus_overlap > weights["number_focus_overlap_threshold"] or not _claim_focus_terms(claim)
                else weights["number_base_without_focus"]
            )
        score += weights["number_focus_weight"] * focus_overlap
    elif focus_overlap > 0.0:
        score += weights["generic_focus_weight"] * focus_overlap
    if claim.time_scope and claim.time_scope.casefold() in lowered:
        score += weights["time_scope_boost"]
    if answer_type == "person" and _contains_person_span(text):
        score += weights["person_boost"]
    if answer_type == "location" and _contains_location_span(text):
        score += weights["location_boost"]
    return _clamp(score)


def route_claim_retrieval(claim: Claim, gated_results: list[GatedSerpResult]) -> RoutingDecision:
    routing = policy_tuning.ROUTING
    top_results = gated_results[:routing["top_results_limit"]]
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
    evidence_sufficiency += min(
        routing["evidence_score_cap"],
        routing["evidence_source_weight"]
        * sum(
            result.assessment.source_score >= routing["evidence_source_threshold"]
            for result in gated_results[:routing["gated_results_support_window"]]
        ),
    )
    evidence_sufficiency += (
        routing["evidence_primary_boost"]
        if any(result.assessment.primary_source_likelihood >= routing["evidence_primary_threshold"] for result in top_results)
        else 0.0
    )
    evidence_sufficiency += (
        routing["evidence_entity_boost"]
        if any(result.assessment.entity_match_score >= routing["evidence_entity_threshold"] for result in top_results)
        else 0.0
    )
    evidence_sufficiency += (
        routing["evidence_semantic_boost"]
        if any(result.assessment.semantic_match_score >= routing["evidence_semantic_threshold"] for result in top_results)
        else 0.0
    )
    evidence_sufficiency += routing["evidence_detail_weight"] * max_detail_coverage
    evidence_sufficiency += routing["evidence_focus_weight"] * _focus_term_overlap(claim, combined_top_text)
    evidence_sufficiency = _clamp(evidence_sufficiency)
    certainty = _clamp(certainty)
    consistency = _clamp(max(consistency, dimension_alignment * routing["consistency_dimension_weight"]))

    profile = claim.claim_profile
    answer_shape = profile.answer_shape if profile is not None else "fact"
    routing_bias = profile.routing_bias if profile is not None else None
    open_ended = answer_shape in {"product_specs", "overview", "comparison", "news_digest"}
    exact_detail_request = _exact_detail_guardrail_claim(claim)
    answer_type = _answer_type(claim)
    number_targeted_threshold = (
        routing["exact_number_targeted_threshold"]
        if profile is not None and profile.answer_shape == "exact_number" and not profile.strict_contract
        else routing["default_number_targeted_threshold"]
    )

    if (
        exact_detail_request
        and (
            max_detail_coverage < routing["exact_detail_min_coverage"]
            or _focus_term_overlap(claim, combined_top_text) < routing["exact_detail_min_focus_overlap"]
        )
    ):
        mode = "iterative_loop"
    elif routing_bias == "iterative_loop":
        mode = "iterative_loop"
    elif (
        not open_ended
        and certainty >= routing["short_path_certainty"]
        and consistency >= routing["short_path_consistency"]
        and evidence_sufficiency >= routing["short_path_sufficiency"]
    ):
        mode = "short_path"
    elif (
        answer_type == "number"
        and certainty >= number_targeted_threshold
        and max_detail_coverage >= routing["targeted_number_min_coverage"]
        and evidence_sufficiency >= routing["targeted_number_min_sufficiency"]
    ):
        mode = "targeted_retrieval"
    elif (
        answer_type == "time"
        and certainty >= routing["targeted_time_certainty"]
        and max_detail_coverage >= routing["targeted_number_min_coverage"]
        and evidence_sufficiency >= routing["targeted_number_min_sufficiency"]
    ):
        mode = "targeted_retrieval"
    elif (
        not open_ended
        and certainty >= routing["targeted_default_certainty"]
        and consistency >= routing["targeted_default_consistency"]
        and evidence_sufficiency >= routing["targeted_default_sufficiency"]
    ):
        mode = "targeted_retrieval"
    elif certainty >= routing["targeted_fallback_certainty"] and evidence_sufficiency >= routing["targeted_fallback_sufficiency"]:
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
        weights = policy_tuning.CHEAP_PASSAGE_NEWS_WEIGHTS
        return _clamp(
            weights["overlap"] * overlap
            + weights["region_match"] * region_match
            + weights["time_match"] * time_match
            + weights["focus_overlap"] * focus_overlap
            + weights["source_score"] * passage.source_score
        )

    if _claim_answer_shape(claim) == "product_specs":
        preferred_bonus = _preferred_domain_bonus(claim, _effective_domain_type(claim, passage.host))
        weights = policy_tuning.CHEAP_PASSAGE_PRODUCT_WEIGHTS
        return _clamp(
            weights["overlap"] * overlap
            + weights["entity_overlap"] * entity_overlap
            + weights["dimension_overlap"] * dimension_overlap
            + weights["focus_overlap"] * focus_overlap
            + weights["source_score"] * passage.source_score
            + weights["preferred_bonus"] * preferred_bonus
        )

    weights = policy_tuning.CHEAP_PASSAGE_DEFAULT_WEIGHTS
    return _clamp(
        weights["overlap"] * overlap
        + weights["entity_overlap"] * entity_overlap
        + weights["dimension_overlap"] * dimension_overlap
        + weights["focus_overlap"] * focus_overlap
        + weights["number_overlap"] * number_overlap
        + weights["source_score"] * passage.source_score
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
        weights = policy_tuning.UTILITY_NEWS_WEIGHTS
        return _clamp(
            weights["cheap_score"] * cheap_passage_score(claim, passage)
            + weights["region_match"] * region_match
            + weights["time_match"] * time_match
            + weights["focus_overlap"] * focus_overlap
            + weights["source_bonus"] * max(source_bonus, 0.0)
            + weights["source_score"] * passage.source_score
        )

    if _claim_answer_shape(claim) == "product_specs":
        preferred_bonus = _preferred_domain_bonus(claim, _effective_domain_type(claim, passage.host))
        weights = policy_tuning.UTILITY_PRODUCT_WEIGHTS
        return _clamp(
            weights["cheap_score"] * cheap_passage_score(claim, passage)
            + weights["focus_overlap"] * focus_overlap
            + weights["preferred_bonus"] * preferred_bonus
            + weights["source_bonus"] * max(source_bonus, 0.0)
            + weights["source_score"] * passage.source_score
        )

    directness = 0.0
    answer_type = _answer_type(claim)
    if answer_type == "time" and _contains_date_like(passage.text):
        directness += policy_tuning.DIRECTNESS_TIME_BOOST
    elif answer_type == "number" and _shared_extract_numbers(passage.text):
        directness += policy_tuning.DIRECTNESS_NUMBER_BOOST
    elif answer_type == "person" and _contains_person_span(passage.text):
        directness += policy_tuning.DIRECTNESS_PERSON_BOOST
    elif answer_type == "location" and _contains_location_span(passage.text):
        directness += policy_tuning.DIRECTNESS_LOCATION_BOOST

    contradiction_signal = policy_tuning.CONTRADICTION_SIGNAL_BOOST if _contains_negation_cue(passage.text.casefold()) else 0.0
    weights = policy_tuning.UTILITY_DEFAULT_WEIGHTS
    return _clamp(
        weights["cheap_score"] * cheap_passage_score(claim, passage)
        + weights["dimension_coverage"] * _dimension_coverage_score(claim, passage.text)
        + weights["focus_overlap"] * focus_overlap
        + weights["directness"] * directness
        + weights["source_bonus"] * max(source_bonus, 0.0)
        + weights["source_score"] * passage.source_score
        + weights["contradiction_signal"] * contradiction_signal
    )


def cheap_passage_filter(
    claim: Claim,
    passages: list[Passage],
    limit: int = tuning.CHEAP_PASSAGE_LIMIT,
) -> list[Passage]:
    scored: list[tuple[float, Passage]] = []
    for passage in passages:
        score = cheap_passage_score(claim, passage)
        if score >= 0.18:
            scored.append((score, passage))
    if not scored:
        scored = [(cheap_passage_score(claim, passage), passage) for passage in passages]
    scored.sort(key=lambda item: (item[0], item[1].source_score), reverse=True)
    return [passage for _, passage in scored[:limit]]


def utility_rerank_passages(
    claim: Claim,
    passages: list[Passage],
    limit: int = tuning.AGENT_PASSAGE_TOP_K,
) -> list[Passage]:
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
        return policy_tuning.LOCAL_NEWS_HOST_EXACT_BONUS
    if any(marker in lowered for marker in ("astana", "kaz", "tengri", "zakon", "inform", "kt.kz")):
        return policy_tuning.LOCAL_NEWS_HOST_HINT_BONUS
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
        if gated and gated.assessment.primary_source_likelihood >= policy_tuning.PRIMARY_SUPPORT_PASSAGE_THRESHOLD:
            return True
    return _effective_domain_type(claim, passage.host) in _primary_domain_types_for_claim(claim)


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
        weights = policy_tuning.SUPPORT_SELECTION_WEIGHTS
        score = (
            weights["semantic_overlap"] * _semantic_overlap(claim.claim_text, lead)
            + weights["entity_overlap"] * _entity_overlap(claim.entity_set, lead)
            + weights["dimension_coverage"] * _dimension_coverage_score(claim, lead)
            + weights["utility_or_source"] * max(passage.utility_score, passage.source_score)
            + weights["verification_bonus"] * max(
                _verification_source_bonus(
                    claim,
                    host=passage.host,
                    title=passage.title,
                    url=passage.url,
                ),
                0.0,
            )
            + weights["preferred_bonus"] * _preferred_domain_bonus(claim, _effective_domain_type(claim, passage.host))
        )
        if _claim_answer_shape(claim) == "product_specs":
            score += (
                policy_tuning.SUPPORT_SELECTION_PRODUCT_PREFERRED_BONUS
                * _preferred_domain_bonus(claim, _effective_domain_type(claim, passage.host))
            )
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


def should_stop_claim_loop(claim: Claim, bundle: EvidenceBundle, iteration: int) -> bool:
    return _policy_should_stop_claim_loop(claim, bundle, iteration)
