from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from datetime import UTC, datetime
from typing import Iterable

from search_agent.domain.models import (
    AgentRunResult,
    AuditTrail,
    Claim,
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

_DATE_LIKE_RE = re.compile(
    r"(?:"
    r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b|"
    r"\b\d{1,2}\s+[А-Яа-яЁё]+\s+\d{4}\b|"
    r"\b[A-Z][a-z]{2,9}\.?\s+\d{1,2},\s+\d{4}\b|"
    r"\b[А-ЯЁ][а-яё]{2,12}\s+\d{1,2},\s+\d{4}\b|"
    r"\b20\d{2}\b"
    r")"
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _build_run_id(query: str, started_at: datetime) -> str:
    digest = hashlib.sha1(f"{query}|{started_at.isoformat()}".encode("utf-8")).hexdigest()[:8]
    return f"{started_at.strftime('%Y%m%dT%H%M%S')}-{digest}"


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9а-яё]+", "", (text or "").casefold())


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9.+/-]*", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    ] 


def _contains_date_like(text: str) -> bool:
    return bool(_DATE_LIKE_RE.search(text or ""))


def _extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    for quoted in re.findall(r'"([^"]{2,80})"', text):
        entities.append(_normalized_text(quoted))

    patterns = [
        r"(?:[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9.+/-]*)(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9.+/-]*){0,3}",
        r"(?:[A-Z]{2,}[A-Za-z0-9.+/-]*)(?:\s+[A-Z]{2,}[A-Za-z0-9.+/-]*){0,2}",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            cleaned = _normalized_text(match)
            cleaned_key = cleaned.casefold()
            token_keys = [token.casefold() for token in _tokenize(cleaned)]
            if (
                len(cleaned) > 1
                and cleaned_key not in _NON_ENTITY_TOKENS
                and token_keys
                and not all(token in _NON_ENTITY_TOKENS for token in token_keys)
            ):
                entities.append(cleaned)

    deduped: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        key = entity.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(entity)
    return deduped[:tuning.AGENT_MAX_QUERY_VARIANTS]


def _extract_time_scope(text: str) -> str | None:
    patterns = [
        r"\b(20\d{2})\b",
        r"\bQ[1-4]\s+20\d{2}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}\s+[A-Z][a-z]+\s+20\d{2}\b",
        r"\b\d{1,2}\s+[А-Яа-яЁё]+\s+20\d{2}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None

def _variant_keywords(text: str, entities: Iterable[str]) -> str:
    entity_tokens = {token.casefold() for entity in entities for token in _tokenize(entity)}
    tokens: list[str] = []
    for token in _tokenize(text):
        if token.casefold() in entity_tokens:
            continue
        if token not in tokens:
            tokens.append(token)
    return " ".join(tokens[:6])


def _source_restricted_query(claim: Claim) -> tuple[str, str] | None:
    lowered = claim.claim_text.lower()
    if any(term in lowered for term in ("paper", "study", "research", "исследование", "статья")):
        return (f'{claim.claim_text} site:arxiv.org', "Target likely academic primary sources.")
    if any(term in lowered for term in ("law", "regulation", "filing", "policy", "закон", "регулятор")):
        return (f'{claim.claim_text} site:.gov', "Prefer official or regulatory sources.")
    if any(term in lowered for term in ("official", "announcement", "пресс-релиз", "официаль")) and claim.entity_set:
        return (f'"{claim.entity_set[0]}" official announcement', "Prefer an official source path.")
    return None


def _is_cyrillic_text(text: str) -> bool:
    return bool(re.search(r"[а-яёА-ЯЁ]", text or ""))


def _time_query_terms(time_scope: str | None, *, cyrillic: bool) -> list[str]:
    if not time_scope:
        return []
    terms = [time_scope]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", time_scope):
        try:
            dt = datetime.fromisoformat(time_scope)
        except ValueError:
            return terms
        if cyrillic:
            ru_months = [
                "января", "февраля", "марта", "апреля", "мая", "июня",
                "июля", "августа", "сентября", "октября", "ноября", "декабря",
            ]
            terms.append(f"{dt.day} {ru_months[dt.month - 1]} {dt.year}")
        else:
            terms.append(dt.strftime("%B %d %Y"))
    return terms


def _news_digest_region_terms(claim: Claim, classification: QueryClassification) -> list[str]:
    candidates: list[str] = []
    if classification.region_hint:
        candidates.append(_normalized_text(classification.region_hint))
    for entity in claim.entity_set:
        cleaned = _normalized_text(entity)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
    return candidates[:3]


def _build_news_digest_query_variants(
    claim: Claim,
    classification: QueryClassification,
) -> list[tuple[str, str, str, str | None, str | None]]:
    cyrillic = _is_cyrillic_text(claim.claim_text)
    region_terms = _news_digest_region_terms(claim, classification)
    primary_region = region_terms[0] if region_terms else claim.claim_text
    time_terms = _time_query_terms(claim.time_scope, cyrillic=cyrillic)
    primary_time = time_terms[0] if time_terms else (claim.time_scope or str(datetime.now().year))

    if cyrillic:
        return [
            (
                _normalized_text(f"{primary_region} новости {primary_time}"),
                "news_digest_broad",
                "Target recent local news for the requested place and date.",
                None,
                claim.time_scope,
            ),
            (
                _normalized_text(f'"{primary_region}" события {primary_time}'),
                "news_digest_events",
                "Prefer event-style coverage for the requested location and date.",
                None,
                claim.time_scope,
            ),
            (
                _normalized_text(f"{primary_region} новости {primary_time} site:.kz"),
                "news_digest_local",
                "Bias retrieval toward local Kazakhstan news domains.",
                "site:.kz",
                claim.time_scope,
            ),
            (
                _normalized_text(f"{primary_region} происшествия {primary_time}"),
                "news_digest_incidents",
                "Catch notable incidents that may not use generic news wording.",
                None,
                claim.time_scope,
            ),
        ]

    return [
        (
            _normalized_text(f"{primary_region} news {primary_time}"),
            "news_digest_broad",
            "Target recent local news for the requested place and date.",
            None,
            claim.time_scope,
        ),
        (
            _normalized_text(f'"{primary_region}" events {primary_time}'),
            "news_digest_events",
            "Prefer event coverage for the requested location and date.",
            None,
            claim.time_scope,
        ),
        (
            _normalized_text(f"{primary_region} breaking news {primary_time}"),
            "news_digest_breaking",
            "Catch breaking-news style coverage for the requested place and date.",
            None,
            claim.time_scope,
        ),
    ]


def build_query_variants(claim: Claim, classification: QueryClassification) -> list[QueryVariant]:
    candidates: list[tuple[str, str, str, str | None, str | None]] = []
    keywords = _variant_keywords(claim.claim_text, claim.entity_set)

    candidates.append((
        claim.claim_text,
        "broad",
        "Broad query to preserve recall and capture general evidence.",
        None,
        claim.time_scope,
    ))

    if claim.entity_set:
        locked = " ".join(f'"{entity}"' for entity in claim.entity_set[:3])
        locked = _normalized_text(f"{locked} {keywords}")
        candidates.append((
            locked,
            "entity_locked",
            "Hard-lock named entities to reduce entity drift.",
            None,
            claim.time_scope,
        ))

    exact_target = claim.claim_text if len(claim.claim_text.split()) <= 12 else f"{' '.join(claim.entity_set[:2])} {keywords}"
    candidates.append((
        f'"{_normalized_text(exact_target)}"',
        "exact_match",
        "Exact-match variant for literal phrasing and narrow factual lookups.",
        None,
        claim.time_scope,
    ))

    if claim.needs_freshness or claim.time_scope:
        freshness_suffix = claim.time_scope or str(datetime.now().year)
        candidates.append((
            _normalized_text(f"{claim.claim_text} {freshness_suffix}"),
            "freshness_aware",
            "Adds explicit time scope so retrieval stays aligned to the intended period.",
            None,
            freshness_suffix,
        ))

    restricted = _source_restricted_query(claim)
    if restricted:
        query_text, rationale = restricted
        candidates.append((
            _normalized_text(query_text),
            "source_restricted",
            rationale,
            query_text.split()[-1] if "site:" in query_text else "official",
            claim.time_scope,
        ))

    deduped: list[QueryVariant] = []
    seen: set[str] = set()
    for idx, (query_text, strategy, rationale, source_restriction, freshness_hint) in enumerate(candidates, 1):
        key = query_text.casefold()
        if not query_text or key in seen:
            continue
        seen.add(key)
        deduped.append(
            QueryVariant(
                variant_id=f"{claim.claim_id}-q{idx}",
                claim_id=claim.claim_id,
                query_text=query_text,
                strategy=strategy,
                rationale=rationale,
                source_restriction=source_restriction,
                freshness_hint=freshness_hint,
            )
        )
    return deduped[:tuning.AGENT_MAX_QUERY_VARIANTS]


def _is_cyrillic_text(text: str) -> bool:
    return bool(re.search(r"[\u0400-\u04FF]", text or ""))


def _time_query_terms(time_scope: str | None, *, cyrillic: bool) -> list[str]:
    if not time_scope:
        return []
    terms = [time_scope]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", time_scope):
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


def _news_digest_region_terms(claim: Claim, classification: QueryClassification) -> list[str]:
    candidates: list[str] = []
    if classification.region_hint:
        candidates.append(_normalized_text(classification.region_hint))
    for entity in claim.entity_set:
        cleaned = _normalized_text(entity)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
    return candidates[:3]


def _build_news_digest_query_variants(
    claim: Claim,
    classification: QueryClassification,
) -> list[tuple[str, str, str, str | None, str | None]]:
    cyrillic = _is_cyrillic_text(claim.claim_text)
    region_terms = _news_digest_region_terms(claim, classification)
    primary_region = region_terms[0] if region_terms else claim.claim_text
    time_terms = _time_query_terms(claim.time_scope, cyrillic=cyrillic)
    primary_time = time_terms[0] if time_terms else (claim.time_scope or str(datetime.now().year))
    fallback_time = primary_time

    if cyrillic:
        return [
            (
                _normalized_text(f"{primary_region} \u043d\u043e\u0432\u043e\u0441\u0442\u0438 {primary_time}"),
                "news_digest_broad",
                "Target recent local news for the requested place and date.",
                None,
                claim.time_scope,
            ),
            (
                _normalized_text(f'"{primary_region}" \u0441\u043e\u0431\u044b\u0442\u0438\u044f {primary_time}'),
                "news_digest_events",
                "Prefer event-style coverage for the requested location and date.",
                None,
                claim.time_scope,
            ),
            (
                _normalized_text(f"{primary_region} \u043d\u043e\u0432\u043e\u0441\u0442\u0438 {fallback_time} site:.kz"),
                "news_digest_local",
                "Bias retrieval toward local Kazakhstan news domains.",
                "site:.kz",
                claim.time_scope,
            ),
            (
                _normalized_text(f"{primary_region} \u043f\u0440\u043e\u0438\u0441\u0448\u0435\u0441\u0442\u0432\u0438\u044f {fallback_time}"),
                "news_digest_incidents",
                "Catch notable incidents that may not use generic news wording.",
                None,
                claim.time_scope,
            ),
            (
                _normalized_text(f'"{primary_region}" {fallback_time} site:.kz'),
                "news_digest_exact_local",
                "Force the requested place and time into a local-domain search.",
                "site:.kz",
                claim.time_scope,
            ),
        ]

    return [
        (
            _normalized_text(f"{primary_region} news {primary_time}"),
            "news_digest_broad",
            "Target recent local news for the requested place and date.",
            None,
            claim.time_scope,
        ),
        (
            _normalized_text(f'"{primary_region}" events {primary_time}'),
            "news_digest_events",
            "Prefer event coverage for the requested location and date.",
            None,
            claim.time_scope,
        ),
        (
            _normalized_text(f"{primary_region} breaking news {primary_time}"),
            "news_digest_breaking",
            "Catch breaking-news style coverage for the requested place and date.",
            None,
            claim.time_scope,
        ),
    ]


def build_query_variants(claim: Claim, classification: QueryClassification) -> list[QueryVariant]:
    if classification.intent == "news_digest":
        candidates: list[tuple[str, str, str, str | None, str | None]] = _build_news_digest_query_variants(
            claim,
            classification,
        )
    else:
        candidates = []

    keywords = _variant_keywords(claim.claim_text, claim.entity_set)

    if classification.intent != "news_digest":
        candidates.append((
            claim.claim_text,
            "broad",
            "Broad query to preserve recall and capture general evidence.",
            None,
            claim.time_scope,
        ))

        if claim.entity_set:
            locked = " ".join(f'"{entity}"' for entity in claim.entity_set[:3])
            locked = _normalized_text(f"{locked} {keywords}")
            candidates.append((
                locked,
                "entity_locked",
                "Hard-lock named entities to reduce entity drift.",
                None,
                claim.time_scope,
            ))

        exact_target = claim.claim_text if len(claim.claim_text.split()) <= 12 else f"{' '.join(claim.entity_set[:2])} {keywords}"
        candidates.append((
            f'"{_normalized_text(exact_target)}"',
            "exact_match",
            "Exact-match variant for literal phrasing and narrow factual lookups.",
            None,
            claim.time_scope,
        ))

        if claim.needs_freshness or claim.time_scope:
            freshness_suffix = claim.time_scope or str(datetime.now().year)
            candidates.append((
                _normalized_text(f"{claim.claim_text} {freshness_suffix}"),
                "freshness_aware",
                "Adds explicit time scope so retrieval stays aligned to the intended period.",
                None,
                freshness_suffix,
            ))

        restricted = _source_restricted_query(claim)
        if restricted:
            query_text, rationale = restricted
            candidates.append((
                _normalized_text(query_text),
                "source_restricted",
                rationale,
                query_text.split()[-1] if "site:" in query_text else "official",
                claim.time_scope,
            ))

    deduped: list[QueryVariant] = []
    seen: set[str] = set()
    for idx, (query_text, strategy, rationale, source_restriction, freshness_hint) in enumerate(candidates, 1):
        key = query_text.casefold()
        if not query_text or key in seen:
            continue
        seen.add(key)
        deduped.append(
            QueryVariant(
                variant_id=f"{claim.claim_id}-q{idx}",
                claim_id=claim.claim_id,
                query_text=query_text,
                strategy=strategy,
                rationale=rationale,
                source_restriction=source_restriction,
                freshness_hint=freshness_hint,
            )
        )
    return deduped[:tuning.AGENT_MAX_QUERY_VARIANTS]


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
    return re.sub(r"[^a-zа-яё0-9]+", " ", title.lower()).strip()


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
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", claim.time_scope or "") and result.published_at:
        if result.published_at[:7] == claim.time_scope[:7]:
            return 0.45
    if re.fullmatch(r"20\d{2}", claim.time_scope or "") and result.published_at:
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
            source_score = _clamp(
                0.22 * domain_prior
                + 0.22 * primary
                + 0.12 * freshness
                + 0.16 * entity_match
                + 0.12 * semantic_match
                + 0.06 * host_entity_match
                + 0.10 * time_alignment
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
    lowered = claim.claim_text.lower()
    if lowered.startswith("who ") or " who " in lowered:
        return "person"
    if lowered.startswith("when ") or " when " in lowered or "release" in lowered or "released" in lowered:
        return "time"
    if lowered.startswith("where ") or " where " in lowered:
        return "location"
    if "how many" in lowered or "how much" in lowered or re.search(r"\b\d+(?:\.\d+)?\b", lowered):
        return "number"
    return "fact"


def _extract_answer_candidates(claim: Claim, text: str) -> list[str]:
    answer_type = _answer_type(claim)
    if answer_type == "time":
        return _DATE_LIKE_RE.findall(text)[:3]
    if answer_type == "number":
        return re.findall(r"\b\d+(?:\.\d+)?\b", text)[:3]
    if answer_type == "person":
        return re.findall(
            r"(?:[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё.-]+(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё.-]+){1,2})",
            text,
        )[:3]
    if answer_type == "location":
        return re.findall(
            r"(?:in|at|from)\s+([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё.-]+(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё.-]+){0,2})",
            text,
        )[:3]
    return _extract_entities(text)[:3]


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
    evidence_sufficiency += min(0.5, 0.12 * sum(result.assessment.source_score >= 0.6 for result in gated_results[:8]))
    evidence_sufficiency += 0.2 if any(result.assessment.primary_source_likelihood >= 0.7 for result in top_results) else 0.0
    evidence_sufficiency += 0.15 if any(result.assessment.entity_match_score >= 0.7 for result in top_results) else 0.0
    evidence_sufficiency += 0.15 if any(result.assessment.semantic_match_score >= 0.7 for result in top_results) else 0.0
    evidence_sufficiency = _clamp(evidence_sufficiency)

    if certainty >= 0.8 and consistency >= 0.65 and evidence_sufficiency >= 0.6:
        mode = "short_path"
    elif certainty >= 0.55 and evidence_sufficiency >= 0.45:
        mode = "targeted_retrieval"
    else:
        mode = "iterative_loop"

    return RoutingDecision(
        mode=mode,
        certainty=_clamp(certainty),
        consistency=_clamp(consistency),
        evidence_sufficiency=evidence_sufficiency,
        rationale=(
            f"certainty={certainty:.2f}, consistency={consistency:.2f}, "
            f"evidence_sufficiency={evidence_sufficiency:.2f}"
        ),
    )


def _extract_author(text: str) -> str | None:
    head = text[:600]
    patterns = [
        r"(?:^|\n)(?:By|Author)\s*:\s*([^\n]{2,80})",
        r"(?:^|\n)(?:Автор)\s*:\s*([^\n]{2,80})",
    ]
    for pattern in patterns:
        match = re.search(pattern, head, re.IGNORECASE)
        if match:
            return _normalized_text(match.group(1))
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


def _routing_limits(profile, decision: RoutingDecision) -> tuple[int, int]:
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


def _dimension_coverage_score(claim: Claim, text: str) -> float:
    lowered = text.casefold()
    score = 0.0
    answer_type = _answer_type(claim)

    if answer_type == "time" and _contains_date_like(text):
        score += 0.45
    if answer_type == "number" and re.search(r"\b\d+(?:\.\d+)?\b", text):
        score += 0.45
    if answer_type == "person" and re.search(r"(?:[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё.-]+(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё.-]+){1,2})", text):
        score += 0.35
    if answer_type == "location" and re.search(r"\b(?:in|at|from)\s+[A-ZА-ЯЁ]", text):
        score += 0.35
    if claim.time_scope and claim.time_scope.casefold() in lowered:
        score += 0.2
    return _clamp(score)


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


def fetch_claim_documents(
    claim: Claim,
    gated_results: list[GatedSerpResult],
    profile,
    routing_decision: RoutingDecision,
    seen_urls: set[str] | None = None,
    log=None,
) -> tuple[list[FetchPlan], list[FetchedDocument]]:
    log = log or (lambda msg: None)
    from search_agent.infrastructure.extractor import fetch_and_extract_many, shallow_fetch_many

    seen_urls = seen_urls or set()
    shallow_limit, deep_limit = _routing_limits(profile, routing_decision)
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
            shallow_fetch_many([c.serp.url for c in selected], log=log),
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
        chunks = re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])", text)
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
        return deep_documents
    snippet_documents = [document for document in documents if document.fetch_depth == "snippet_only"]
    if snippet_documents:
        return snippet_documents
    return [document for document in documents if document.fetch_depth == "shallow"]


def cheap_passage_score(claim: Claim, passage: Passage) -> float:
    overlap = _semantic_overlap(claim.claim_text, passage.text)
    entity_overlap = _entity_overlap(claim.entity_set, passage.text)
    dimension_overlap = _dimension_coverage_score(claim, passage.text)
    claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", claim.claim_text))
    passage_numbers = set(re.findall(r"\d+(?:\.\d+)?", passage.text))
    number_overlap = 1.0 if claim_numbers and claim_numbers & passage_numbers else 0.0
    return _clamp(
        0.35 * overlap
        + 0.25 * entity_overlap
        + 0.20 * dimension_overlap
        + 0.10 * number_overlap
        + 0.10 * passage.source_score
    )


def utility_score_for_claim(claim: Claim, passage: Passage) -> float:
    lowered = passage.text.casefold()
    directness = 0.0
    answer_type = _answer_type(claim)
    source_bonus = _verification_source_bonus(
        claim,
        host=passage.host,
        title=passage.title,
        url=passage.url,
    )

    if answer_type == "time":
        if re.search(r"\b(?:released on|release date|announced on|dated)\b", lowered):
            directness += 0.35
        if _contains_date_like(passage.text):
            directness += 0.25
        if any(
            cue in f"{passage.title} {passage.url}".casefold()
            for cue in ("release", "released", "announcement", "downloads/release", "whatsnew")
        ):
            directness += 0.15
        if "release date:" in lowered:
            directness += 0.15
    elif answer_type == "person":
        if any(role in lowered for role in ("ceo", "founder", "president", "chairman")):
            directness += 0.35
        if re.search(r"(?:[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё.-]+(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё.-]+){1,2})", passage.text):
            directness += 0.2
    elif answer_type == "number":
        if re.search(r"\b\d+(?:\.\d+)?\b", passage.text):
            directness += 0.35
    elif answer_type == "location":
        if re.search(r"\b(?:in|at|from)\s+[A-ZА-ЯЁ]", passage.text):
            directness += 0.25

    contradiction_signal = 0.15 if re.search(r"\b(?:not|no|never|false|incorrect|debunked|contradict)\b", lowered) else 0.0
    return _clamp(
        0.28 * cheap_passage_score(claim, passage)
        + 0.24 * _dimension_coverage_score(claim, passage.text)
        + 0.20 * directness
        + 0.18 * max(source_bonus, 0.0)
        + 0.10 * passage.source_score
        + 0.05 * contradiction_signal
    )


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
    return selected


def _news_digest_region_hint_from_claim(claim: Claim) -> str | None:
    if claim.entity_set:
        return claim.entity_set[0]
    match = re.search(
        r"\b(?:in|for|within|at|from|\u0432|\u0434\u043b\u044f|\u043f\u043e)\s+([A-Z\u0410-\u042f\u0401][A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451.-]+(?:\s+[A-Z\u0410-\u042f\u0401][A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451.-]+){0,2})",
        claim.claim_text,
    )
    if match:
        return _normalized_text(match.group(1))
    return None


def _is_news_digest_claim(claim: Claim) -> bool:
    lowered = claim.claim_text.casefold()
    return bool(
        (claim.needs_freshness or claim.time_scope)
        and (
            re.search(r"\bwhat\b.*\bhappened\b", lowered)
            or re.search(r"\b\u0447\u0442\u043e\b.*\b(?:\u043f\u0440\u043e\u0438\u0437\u043e\u0448\u043b\u043e|\u0441\u043b\u0443\u0447\u0438\u043b\u043e\u0441\u044c|\u0431\u044b\u043b\u043e)\b", lowered)
            or "\u043d\u043e\u0432\u043e\u0441\u0442\u0438" in lowered
            or "\u0441\u043e\u0431\u044b\u0442\u0438\u044f" in lowered
            or "\u043f\u0440\u043e\u0438\u0441\u0448\u0435\u0441\u0442\u0432\u0438\u044f" in lowered
        )
    )


def _news_digest_time_match(claim: Claim, passage: Passage) -> float:
    if not claim.time_scope:
        return 0.5
    haystack = f"{passage.title} {passage.text}".casefold()
    if claim.time_scope.casefold() in haystack:
        return 1.0
    if passage.published_at and passage.published_at.startswith(claim.time_scope):
        return 1.0
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", claim.time_scope):
        return 0.0
    return 0.0


def _local_news_host_bonus(host: str) -> float:
    lowered = (host or "").casefold()
    if lowered.endswith(".kz") or ".kz" in lowered:
        return 1.0
    if any(marker in lowered for marker in ("astana", "kaz", "tengri", "zakon", "inform", "kt.kz")):
        return 0.7
    return 0.0


def cheap_passage_score(claim: Claim, passage: Passage) -> float:
    overlap = _semantic_overlap(claim.claim_text, passage.text)
    entity_overlap = _entity_overlap(claim.entity_set, passage.text)
    dimension_overlap = _dimension_coverage_score(claim, passage.text)
    claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", claim.claim_text))
    passage_numbers = set(re.findall(r"\d+(?:\.\d+)?", passage.text))
    number_overlap = 1.0 if claim_numbers and claim_numbers & passage_numbers else 0.0

    if _is_news_digest_claim(claim):
        region = _news_digest_region_hint_from_claim(claim)
        haystack = f"{passage.title} {passage.text[:220]} {passage.url}"
        region_match = _entity_overlap([region], haystack) if region else entity_overlap
        time_match = _news_digest_time_match(claim, passage)
        local_bonus = _local_news_host_bonus(passage.host)
        event_terms = 1.0 if any(
            term in haystack.casefold()
            for term in (
                "\u043d\u043e\u0432\u043e\u0441\u0442",
                "\u0441\u043e\u0431\u044b\u0442",
                "\u043f\u0440\u043e\u0438\u0441\u0448\u0435\u0441\u0442\u0432",
                "news",
                "events",
                "breaking",
            )
        ) else 0.0
        score = (
            0.14 * overlap
            + 0.34 * region_match
            + 0.24 * time_match
            + 0.12 * local_bonus
            + 0.08 * event_terms
            + 0.08 * passage.source_score
        )
        if region and region_match < 0.25:
            score -= 0.30
        if claim.time_scope and time_match <= 0.0:
            score -= 0.20
        return _clamp(score)

    return _clamp(
        0.35 * overlap
        + 0.25 * entity_overlap
        + 0.20 * dimension_overlap
        + 0.10 * number_overlap
        + 0.10 * passage.source_score
    )


def utility_score_for_claim(claim: Claim, passage: Passage) -> float:
    lowered = passage.text.casefold()
    if _is_news_digest_claim(claim):
        region = _news_digest_region_hint_from_claim(claim)
        haystack = f"{passage.title} {passage.text[:220]} {passage.url}"
        region_match = _entity_overlap([region], haystack) if region else _entity_overlap(claim.entity_set, haystack)
        time_match = _news_digest_time_match(claim, passage)
        local_bonus = _local_news_host_bonus(passage.host)
        title_match = 1.0 if region and _entity_overlap([region], passage.title) >= 0.6 else 0.0
        source_bonus = _verification_source_bonus(
            claim,
            host=passage.host,
            title=passage.title,
            url=passage.url,
        )
        score = (
            0.24 * cheap_passage_score(claim, passage)
            + 0.28 * region_match
            + 0.18 * time_match
            + 0.12 * local_bonus
            + 0.08 * title_match
            + 0.05 * max(source_bonus, 0.0)
            + 0.05 * passage.source_score
        )
        if region and region_match < 0.25:
            score -= 0.30
        if claim.time_scope and time_match <= 0.0:
            score -= 0.20
        return _clamp(score)

    directness = 0.0
    answer_type = _answer_type(claim)
    source_bonus = _verification_source_bonus(
        claim,
        host=passage.host,
        title=passage.title,
        url=passage.url,
    )

    if answer_type == "time":
        if re.search(r"\b(?:released on|release date|announced on|dated)\b", lowered):
            directness += 0.35
        if _contains_date_like(passage.text):
            directness += 0.25
        if any(
            cue in f"{passage.title} {passage.url}".casefold()
            for cue in ("release", "released", "announcement", "downloads/release", "whatsnew")
        ):
            directness += 0.15
        if "release date:" in lowered:
            directness += 0.15
    elif answer_type == "person":
        if any(role in lowered for role in ("ceo", "founder", "president", "chairman")):
            directness += 0.35
        if re.search(r"(?:[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё.-]+(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё.-]+){1,2})", passage.text):
            directness += 0.2
    elif answer_type == "number":
        if re.search(r"\b\d+(?:\.\d+)?\b", passage.text):
            directness += 0.35
    elif answer_type == "location":
        if re.search(r"\b(?:in|at|from)\s+[A-ZА-ЯЁ]", passage.text):
            directness += 0.25

    contradiction_signal = 0.15 if re.search(r"\b(?:not|no|never|false|incorrect|debunked|contradict)\b", lowered) else 0.0
    return _clamp(
        0.28 * cheap_passage_score(claim, passage)
        + 0.24 * _dimension_coverage_score(claim, passage.text)
        + 0.20 * directness
        + 0.18 * max(source_bonus, 0.0)
        + 0.10 * passage.source_score
        + 0.05 * contradiction_signal
    )

def _select_passages_from_spans(passages: list[Passage], spans: list[EvidenceSpan]) -> list[Passage]:
    wanted = {span.passage_id for span in spans}
    return [passage for passage in passages if passage.passage_id in wanted]


def build_evidence_bundle(
    claim: Claim,
    passages: list[Passage],
    verification: VerificationResult,
    gated_results: list[GatedSerpResult],
) -> EvidenceBundle:
    gated_by_url = {result.serp.canonical_url: result for result in gated_results}
    supporting_passages = _select_passages_from_spans(passages, verification.supporting_spans)
    contradicting_passages = _select_passages_from_spans(passages, verification.contradicting_spans)
    all_supporting = supporting_passages or passages[:2]
    independent_sources = {_host_root(passage.host) for passage in all_supporting}
    has_primary = any(
        (
            gated_by_url.get(passage.canonical_url)
            and gated_by_url[passage.canonical_url].assessment.primary_source_likelihood >= 0.7
        )
        or _effective_domain_type(claim, passage.host) in {"official", "academic"}
        for passage in all_supporting
    )
    freshness_ok = True
    if claim.needs_freshness:
        freshness_ok = any(passage.published_at for passage in all_supporting)

    return EvidenceBundle(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        supporting_passages=supporting_passages,
        contradicting_passages=contradicting_passages,
        considered_passages=passages,
        independent_source_count=len(independent_sources),
        has_primary_source=has_primary,
        freshness_ok=freshness_ok,
        verification=verification,
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
    if classification.intent == "news_digest":
        region = classification.region_hint or (claim.entity_set[0] if claim.entity_set else claim.claim_text)
        cyrillic = _is_cyrillic_text(claim.claim_text)
        time_terms = _time_query_terms(claim.time_scope, cyrillic=cyrillic)
        explicit_time = time_terms[0] if time_terms else (claim.time_scope or str(datetime.now().year))
        fallback_time = explicit_time
        local_word = "\u043d\u043e\u0432\u043e\u0441\u0442\u0438" if cyrillic else "news"
        event_word = "\u0441\u043e\u0431\u044b\u0442\u0438\u044f" if cyrillic else "events"

        candidates: list[tuple[str, str, str, str | None, str | None]] = [
            (
                _normalized_text(f"{region} {local_word} {explicit_time} site:.kz"),
                "refined_local_news",
                "Tighten the search around local news coverage for the exact place and date.",
                "site:.kz",
                claim.time_scope,
            ),
            (
                _normalized_text(f'"{region}" {event_word} {fallback_time} site:.kz'),
                "refined_local_events",
                "Bias toward local event recaps for the requested place and date.",
                "site:.kz",
                claim.time_scope,
            ),
        ]
        if "source" in verification.missing_dimensions or verification.verdict != "supported":
            candidates.append((
                _normalized_text(f'"{region}" {fallback_time} site:.kz'),
                "refined_local_exact",
                "Force retrieval from local Kazakhstan domains for a precise place/date match.",
                "site:.kz",
                claim.time_scope,
            ))

        variants: list[QueryVariant] = []
        seen = set(existing_queries)
        for idx, (query_text, strategy, rationale, source_restriction, freshness_hint) in enumerate(candidates, 1):
            key = query_text.casefold()
            if not query_text or key in seen:
                continue
            seen.add(key)
            variants.append(
                QueryVariant(
                    variant_id=f"{claim.claim_id}-r{iteration}-{idx}",
                    claim_id=claim.claim_id,
                    query_text=query_text,
                    strategy=strategy,
                    rationale=rationale,
                    source_restriction=source_restriction,
                    freshness_hint=freshness_hint,
                )
            )
        return variants[:tuning.AGENT_MAX_REFINE_VARIANTS]

    candidates: list[tuple[str, str, str, str | None, str | None]] = []
    base_keywords = _variant_keywords(claim.claim_text, claim.entity_set)
    missing_primary_support = bool(
        bundle
        and verification.verdict == "supported"
        and claim.entity_set
        and not bundle.has_primary_source
    )

    if "time" in verification.missing_dimensions:
        explicit_time = claim.time_scope or str(datetime.now().year)
        candidates.append((
            _normalized_text(f"{claim.claim_text} {explicit_time}"),
            "refined_time",
            "Add explicit temporal constraint for verification.",
            None,
            explicit_time,
        ))

    if "entity" in verification.missing_dimensions:
        aliases = _candidate_aliases(claim, gated_results)
        for alias in aliases[:2]:
            candidates.append((
                _normalized_text(f'"{alias}" {base_keywords}'),
                "refined_alias",
                "Add alias discovered from retrieved results.",
                None,
                claim.time_scope,
            ))

    if (
        "source" in verification.missing_dimensions
        or missing_primary_support
        or not any(result.assessment.primary_source_likelihood >= 0.7 for result in gated_results[:5])
    ):
        source_host = _preferred_source_host(gated_results)
        if source_host:
            candidates.append((
                _normalized_text(f"{claim.claim_text} site:{source_host}"),
                "refined_source",
                "Restrict search to the most promising primary source host.",
                source_host,
                claim.time_scope,
            ))
            candidates.append((
                _normalized_text(f'"{base_keywords}" site:{source_host}'),
                "refined_source_exact",
                "Force evidence retrieval from a primary-source host with tighter wording.",
                source_host,
                claim.time_scope,
            ))

    if verification.verdict != "supported" or missing_primary_support:
        candidates.append((
            f'"{claim.claim_text}"',
            "refined_exact",
            "Switch to exact-match wording for narrower evidence retrieval.",
            None,
            claim.time_scope,
        ))

    if verification.verdict in {"contradicted", "insufficient_evidence"} and iteration < tuning.AGENT_MAX_CLAIM_ITERATIONS:
        candidates.append((
            _normalized_text(f"{claim.claim_text} contradiction OR false OR debunked"),
            "refined_contradiction",
            "Search specifically for contradiction or correction evidence.",
            None,
            claim.time_scope,
        ))

    variants: list[QueryVariant] = []
    seen = set(existing_queries)
    for idx, (query_text, strategy, rationale, source_restriction, freshness_hint) in enumerate(candidates, 1):
        key = query_text.casefold()
        if not query_text or key in seen:
            continue
        seen.add(key)
        variants.append(
            QueryVariant(
                variant_id=f"{claim.claim_id}-r{iteration}-{idx}",
                claim_id=claim.claim_id,
                query_text=query_text,
                strategy=strategy,
                rationale=rationale,
                source_restriction=source_restriction,
                freshness_hint=freshness_hint,
            )
        )
    return variants[:tuning.AGENT_MAX_REFINE_VARIANTS]


def should_stop_claim_loop(claim: Claim, bundle: EvidenceBundle, iteration: int) -> bool:
    verification = bundle.verification
    if verification is None:
        return iteration >= tuning.AGENT_MAX_CLAIM_ITERATIONS
    if verification.verdict == "supported":
        if bundle.independent_source_count >= 2 and bundle.has_primary_source:
            return True
        if not claim.entity_set and verification.confidence >= 0.95 and bundle.independent_source_count >= 2:
            return True
    return iteration >= tuning.AGENT_MAX_CLAIM_ITERATIONS


def _best_sentence_for_claim(claim: Claim, passage: Passage) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", passage.text)
    if not sentences:
        return passage.text[:220]
    scored = sorted(
        sentences,
        key=lambda sentence: _semantic_overlap(claim.claim_text, sentence),
        reverse=True,
    )
    best = _normalized_text(scored[0]) if scored else passage.text[:220]
    if len(best) > 240:
        best = best[:237].rstrip() + "..."
    return best


def _best_span_text(verification: VerificationResult, passages: list[Passage], claim: Claim, contradicted: bool = False) -> str | None:
    spans = verification.contradicting_spans if contradicted else verification.supporting_spans
    if spans:
        text = _normalized_text(spans[0].text)
        if len(text) > 240:
            text = text[:237].rstrip() + "..."
        return text
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
    if len(combined) > 260:
        combined = combined[:257].rstrip() + "..."
    return combined


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


def compose_answer(report: AgentRunResult) -> str:
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
            digest_lines.append(f"- {_digest_sentence(passage)} {_format_citations(url_to_index, [passage])}".rstrip())

        if digest_lines:
            lines = ["- Events from retrieved sources:"]
            lines.extend(digest_lines[:4])
            if indexed_sources:
                lines.append("")
                lines.append("Sources")
                for idx, title, url in indexed_sources:
                    lines.append(f"[{idx}] {title} - {url}")
            return "\n".join(lines)

    cited_passages: list[Passage] = []
    for run in report.claims:
        bundle = run.evidence_bundle
        if bundle is None or bundle.verification is None:
            continue
        if bundle.verification.verdict == "supported":
            cited_passages.extend((bundle.supporting_passages[:3] or bundle.considered_passages[:3]))
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
            passages = bundle.supporting_passages[:2] or bundle.considered_passages[:1]
            if not passages:
                continue
            sentence = _best_span_text(verification, passages, run.claim)
            if not sentence:
                continue
            citations = _format_citations(url_to_index, passages)
            qualifier = ""
            if bundle.independent_source_count < 2:
                qualifier = " (supported by fewer than two independent sources)"
            elif not bundle.has_primary_source:
                qualifier = " (without a detected primary source)"
            supported_lines.append(f"- {sentence}{qualifier} {citations}".rstrip())
        elif verification.verdict == "contradicted":
            passages = bundle.contradicting_passages[:2] or bundle.considered_passages[:1]
            if not passages:
                continue
            sentence = _best_span_text(verification, passages, run.claim, contradicted=True)
            if not sentence:
                continue
            citations = _format_citations(url_to_index, passages)
            caveat_lines.append(f"- {run.claim.claim_text}: contradicted. {sentence} {citations}".rstrip())
        else:
            missing = ", ".join(verification.missing_dimensions) if verification.missing_dimensions else "coverage"
            gap_lines.append(f"- {run.claim.claim_text}: insufficient evidence ({missing}).")

    lines: list[str] = []
    if supported_lines:
        lines.extend(supported_lines)
    else:
        lines.append("- Not enough claim-level evidence for a direct answer.")

    if caveat_lines:
        lines.append("")
        lines.append("Caveats")
        lines.extend(caveat_lines)

    if gap_lines:
        lines.append("")
        lines.append("Insufficient data")
        lines.extend(gap_lines)

    if indexed_sources:
        lines.append("")
        lines.append("Sources")
        for idx, title, url in indexed_sources:
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
