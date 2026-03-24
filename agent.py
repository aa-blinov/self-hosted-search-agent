from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from datetime import UTC, datetime
from typing import Iterable

from openai import OpenAI

from agent_types import (
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
from search import search_searxng_with_fallback
from source_priors import lookup_source_prior

MAX_CLAIMS = int(os.getenv("AGENT_MAX_CLAIMS", "4"))
MAX_CLAIM_ITERATIONS = int(os.getenv("AGENT_MAX_CLAIM_ITERATIONS", "3"))
DEFAULT_FETCH_TOP_N = int(os.getenv("AGENT_FETCH_TOP_N", "4"))
PASSAGE_TOP_K = int(os.getenv("AGENT_PASSAGE_TOP_K", "8"))
SERP_GATE_MIN_URLS = int(os.getenv("SERP_GATE_MIN_URLS", "15"))
SERP_GATE_MAX_URLS = int(os.getenv("SERP_GATE_MAX_URLS", "30"))
SNIPPET_FALLBACK_DOCS = int(os.getenv("AGENT_SNIPPET_FALLBACK_DOCS", "2"))
SHALLOW_SHORT_LIMIT = int(os.getenv("SHALLOW_FETCH_SHORT_LIMIT", "8"))
SHALLOW_TARGETED_LIMIT = int(os.getenv("SHALLOW_FETCH_TARGETED_LIMIT", "12"))
SHALLOW_ITERATIVE_LIMIT = int(os.getenv("SHALLOW_FETCH_ITERATIVE_LIMIT", "15"))
DEEP_SHORT_LIMIT = int(os.getenv("DEEP_FETCH_SHORT_LIMIT", "2"))
DEEP_TARGETED_LIMIT = int(os.getenv("DEEP_FETCH_TARGETED_LIMIT", "3"))
DEEP_ITERATIVE_LIMIT = int(os.getenv("DEEP_FETCH_ITERATIVE_LIMIT", "4"))
CHEAP_PASSAGE_LIMIT = int(os.getenv("CHEAP_PASSAGE_LIMIT", "12"))

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "if",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with", "vs",
    "что", "как", "где", "когда", "кто", "или", "для", "это", "эта", "этот",
    "про", "из", "на", "по", "в", "во", "с", "со", "у", "о", "об", "ли",
    "чем", "какой", "какая", "какие", "каково", "есть", "был", "была", "были",
    "than", "about", "between", "latest", "current",
}

_RELATIVE_TIME_MARKERS = {
    "today", "tomorrow", "yesterday", "this week", "next week", "last week",
    "this month", "next month", "last month", "this quarter", "next quarter",
    "last quarter", "current", "latest",
    "сегодня", "завтра", "вчера", "эта неделя", "на этой неделе",
    "следующая неделя", "прошлая неделя", "этот месяц", "следующий месяц",
    "прошлый месяц", "этот квартал", "следующий квартал", "прошлый квартал",
    "текущий", "последний",
}

_COMPARISON_MARKERS = {
    "compare", "comparison", "difference", "versus", "vs", "contrast",
    "сравни", "сравнение", "разница", "отличается", "лучше", "хуже",
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


def _parse_json_object(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _parse_json_array(text: str):
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


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
    return deduped[:6]


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


def _extract_region_hint(text: str) -> str | None:
    patterns = [
        r"\b(?:in|for|within)\s+([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+){0,2})",
        r"\b(?:в|для|по)\s+([А-ЯЁ][А-Яа-яЁё.-]+(?:\s+[А-ЯЁ][А-Яа-яЁё.-]+){0,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _needs_freshness(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _RELATIVE_TIME_MARKERS)


def _should_decompose(text: str) -> bool:
    lowered = text.lower()
    return (
        any(marker in lowered for marker in _COMPARISON_MARKERS)
        or " and " in lowered
        or " и " in lowered
        or len(text) > 90
        or text.count("?") > 1
    )


def _normalize_time_references(query: str, client: OpenAI | None, log=None) -> str:
    log = log or (lambda msg: None)
    if client is None or not _needs_freshness(query):
        return query

    today = datetime.now().strftime("%d %B %Y, %A")
    prompt = (
        f"Today is {today}.\n"
        "Rewrite the search query by replacing only relative time references with explicit dates.\n"
        "Keep all named entities exactly as written.\n"
        "Return only the rewritten query.\n\n"
        f"Query: {query}"
    )
    try:
        from llm import MODEL, _extra

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
            extra_body=_extra(),
        )
        normalized = _normalized_text(response.choices[0].message.content or "")
        if normalized:
            if normalized != query:
                log(f"  [dim]↳ normalized query: [italic]{normalized}[/italic][/dim]")
            return normalized
    except Exception as exc:
        log(f"  [dim yellow]⚠ time normalization failed: {exc}[/dim yellow]")
    return query


def classify_query(query: str, client: OpenAI | None = None, log=None) -> QueryClassification:
    normalized_query = _normalize_time_references(query, client, log=log)
    lowered = normalized_query.lower()
    intent = "comparison" if any(marker in lowered for marker in _COMPARISON_MARKERS) else "factual"
    complexity = "multi_hop" if _should_decompose(normalized_query) else "single_hop"
    entities = _extract_entities(normalized_query)
    entity_disambiguation = any(len(entity) <= 4 for entity in entities)
    return QueryClassification(
        query=query,
        normalized_query=normalized_query,
        intent=intent,
        complexity=complexity,
        needs_freshness=_needs_freshness(query),
        time_scope=_extract_time_scope(normalized_query),
        region_hint=_extract_region_hint(normalized_query),
        entity_disambiguation=entity_disambiguation,
    )


def _fallback_claims(classification: QueryClassification) -> list[Claim]:
    return [
        Claim(
            claim_id="claim-1",
            claim_text=classification.normalized_query,
            priority=1,
            needs_freshness=classification.needs_freshness,
            entity_set=_extract_entities(classification.normalized_query),
            time_scope=classification.time_scope,
        )
    ]


def decompose_claims(
    classification: QueryClassification,
    client: OpenAI | None = None,
    log=None,
) -> list[Claim]:
    log = log or (lambda msg: None)
    if client is None or not _should_decompose(classification.normalized_query):
        return _fallback_claims(classification)

    prompt = (
        "Break the user request into atomic factual claims or subquestions that can be verified on the web.\n"
        "Return a JSON array. Each item must have:\n"
        "- claim_text\n"
        "- priority (1 = highest)\n"
        "- needs_freshness (true/false)\n"
        "- entity_set (array of exact entities)\n"
        "- time_scope (string or null)\n"
        "Rules:\n"
        "- Preserve exact named entities.\n"
        "- Keep claims atomic.\n"
        "- Do not invent missing facts.\n"
        "- Use at most 4 claims.\n\n"
        f"User request: {classification.normalized_query}"
    )
    try:
        from llm import MODEL, _extra

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
            extra_body=_extra(),
        )
        parsed = _parse_json_array(response.choices[0].message.content or "")
        if isinstance(parsed, list):
            claims: list[Claim] = []
            for idx, item in enumerate(parsed[:MAX_CLAIMS], 1):
                if not isinstance(item, dict) or not item.get("claim_text"):
                    continue
                claims.append(
                    Claim(
                        claim_id=f"claim-{idx}",
                        claim_text=_normalized_text(str(item["claim_text"])),
                        priority=max(1, int(item.get("priority", idx))),
                        needs_freshness=bool(item.get("needs_freshness", classification.needs_freshness)),
                        entity_set=[
                            _normalized_text(str(entity))
                            for entity in item.get("entity_set", [])
                            if _normalized_text(str(entity))
                        ] or _extract_entities(str(item["claim_text"])),
                        time_scope=(
                            _normalized_text(str(item["time_scope"]))
                            if item.get("time_scope") not in (None, "", "null")
                            else _extract_time_scope(str(item["claim_text"]))
                        ),
                    )
                )
            if claims:
                return claims
    except Exception as exc:
        log(f"  [dim yellow]⚠ claim decomposition failed: {exc}[/dim yellow]")
    return _fallback_claims(classification)


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
    return deduped[:6]


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
    hits = sum(1 for entity in entities if entity.casefold() in lowered)
    return hits / len(entities)


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
            spam = _spam_risk(result)
            primary = _primary_source_likelihood(claim, result, domain_type)
            source_score = _clamp(
                0.24 * domain_prior
                + 0.24 * primary
                + 0.14 * freshness
                + 0.16 * entity_match
                + 0.14 * semantic_match
                + 0.08 * host_entity_match
                + prior.source_prior
                - 0.25 * spam
            )
            reasons = [
                f"domain={domain_type}",
                f"prior={prior.source_prior:.2f}",
                f"primary={primary:.2f}",
                f"host_entity={host_entity_match:.2f}",
                f"freshness={freshness:.2f}",
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
        "short_path": SHALLOW_SHORT_LIMIT,
        "targeted_retrieval": SHALLOW_TARGETED_LIMIT,
        "iterative_loop": SHALLOW_ITERATIVE_LIMIT,
    }[decision.mode]
    deep_limit = {
        "short_path": DEEP_SHORT_LIMIT,
        "targeted_retrieval": DEEP_TARGETED_LIMIT,
        "iterative_loop": DEEP_ITERATIVE_LIMIT,
    }[decision.mode]
    if profile.fetch_top_n == 0:
        deep_limit = 0
    else:
        deep_limit = min(deep_limit, max(profile.fetch_top_n, DEFAULT_FETCH_TOP_N))
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
    from extractor import fetch_and_extract, shallow_fetch

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
    for candidate in selected:
        payload = shallow_fetch(candidate.serp.url, log=log)
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
        content = fetch_and_extract(candidate.serp.url, log=log)
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
            for document in shallow_ranked[:SNIPPET_FALLBACK_DOCS]
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


def cheap_passage_filter(claim: Claim, passages: list[Passage], limit: int = CHEAP_PASSAGE_LIMIT) -> list[Passage]:
    scored: list[tuple[float, Passage]] = []
    for passage in passages:
        score = cheap_passage_score(claim, passage)
        if score >= 0.18:
            scored.append((score, passage))
    if not scored:
        scored = [(cheap_passage_score(claim, passage), passage) for passage in passages]
    scored.sort(key=lambda item: (item[0], item[1].source_score), reverse=True)
    return [passage for _, passage in scored[:limit]]


def utility_rerank_passages(claim: Claim, passages: list[Passage], limit: int = PASSAGE_TOP_K) -> list[Passage]:
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


def _heuristic_verifier(claim: Claim, passages: list[Passage]) -> VerificationResult:
    if not passages:
        return VerificationResult(
            verdict="insufficient_evidence",
            confidence=0.1,
            missing_dimensions=["source"],
            rationale="No passages were available for verification.",
        )

    strong = [passage for passage in passages if passage.utility_score >= 0.35]
    if len(strong) >= 2 and len({_host_root(p.host) for p in strong[:3]}) >= 2:
        supporting = [
            EvidenceSpan(
                passage_id=passage.passage_id,
                url=passage.url,
                title=passage.title,
                section=passage.section,
                text=passage.text[:220],
            )
            for passage in strong[:3]
        ]
        return VerificationResult(
            verdict="supported",
            confidence=0.6,
            supporting_spans=supporting,
            rationale="Multiple passages from independent sources align with the claim.",
        )

    missing: list[str] = []
    if claim.time_scope and not any(claim.time_scope.casefold() in passage.text.casefold() for passage in passages):
        missing.append("time")
    if claim.entity_set and not any(
        entity.casefold() in passage.text.casefold()
        for entity in claim.entity_set
        for passage in passages
    ):
        missing.append("entity")
    if re.search(r"\d", claim.claim_text) and not any(re.search(r"\d", passage.text) for passage in passages):
        missing.append("number")

    return VerificationResult(
        verdict="insufficient_evidence",
        confidence=0.35,
        supporting_spans=[
            EvidenceSpan(
                passage_id=passage.passage_id,
                url=passage.url,
                title=passage.title,
                section=passage.section,
                text=passage.text[:220],
            )
            for passage in strong[:2]
        ],
        missing_dimensions=missing or ["coverage"],
        rationale="Evidence is partial or does not fully cover the claim scope.",
    )


def verify_claim(claim: Claim, passages: list[Passage], client: OpenAI | None = None, log=None) -> VerificationResult:
    log = log or (lambda msg: None)
    if client is None:
        return _heuristic_verifier(claim, passages)

    prompt_lines = []
    for passage in passages[:PASSAGE_TOP_K]:
        prompt_lines.append(
            f"[{passage.passage_id}] {passage.title} | {passage.url}\n"
            f"Section: {passage.section}\n"
            f"Text: {passage.text}"
        )
    prompt = (
        "You are a claim verifier. Decide whether the evidence supports, contradicts, or is insufficient.\n"
        "Return one JSON object with keys:\n"
        "- verdict: supported | contradicted | insufficient_evidence\n"
        "- confidence: number between 0 and 1\n"
        "- supporting_passages: array of {passage_id, quote}\n"
        "- contradicting_passages: array of {passage_id, quote}\n"
        "- missing_dimensions: array from [time, entity, number, location, source, coverage]\n"
        "- rationale: short string\n"
        "Rules:\n"
        "- Use only explicit evidence.\n"
        "- If entity, time, number, or location scope is missing, prefer insufficient_evidence.\n"
        "- Quotes must be short excerpts copied from the passage.\n\n"
        f"Claim: {claim.claim_text}\n\n"
        + "\n\n".join(prompt_lines)
    )
    try:
        from llm import MODEL, _extra

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
            extra_body=_extra(),
        )
        parsed = _parse_json_object(response.choices[0].message.content or "")
        if not isinstance(parsed, dict):
            return _heuristic_verifier(claim, passages)

        passage_map = {passage.passage_id: passage for passage in passages}

        def build_spans(items: list[dict]) -> list[EvidenceSpan]:
            spans: list[EvidenceSpan] = []
            for item in items:
                passage = passage_map.get(str(item.get("passage_id", "")))
                if passage is None:
                    continue
                quote = _normalized_text(str(item.get("quote", ""))) or passage.text[:220]
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

        verdict = str(parsed.get("verdict", "insufficient_evidence"))
        if verdict not in {"supported", "contradicted", "insufficient_evidence"}:
            verdict = "insufficient_evidence"
        return VerificationResult(
            verdict=verdict,
            confidence=_clamp(float(parsed.get("confidence", 0.0))),
            supporting_spans=build_spans(parsed.get("supporting_passages", [])),
            contradicting_spans=build_spans(parsed.get("contradicting_passages", [])),
            missing_dimensions=[
                _normalized_text(str(item))
                for item in parsed.get("missing_dimensions", [])
                if _normalized_text(str(item))
            ],
            rationale=_normalized_text(str(parsed.get("rationale", ""))),
        )
    except Exception as exc:
        log(f"  [dim yellow]⚠ verifier failed: {exc}[/dim yellow]")
        return _heuristic_verifier(claim, passages)


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

    if verification.verdict in {"contradicted", "insufficient_evidence"} and iteration < MAX_CLAIM_ITERATIONS:
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
    return variants


def should_stop_claim_loop(claim: Claim, bundle: EvidenceBundle, iteration: int) -> bool:
    verification = bundle.verification
    if verification is None:
        return iteration >= MAX_CLAIM_ITERATIONS
    if verification.verdict == "supported":
        if bundle.independent_source_count >= 2 and bundle.has_primary_source:
            return True
        if not claim.entity_set and verification.confidence >= 0.95 and bundle.independent_source_count >= 2:
            return True
    return iteration >= MAX_CLAIM_ITERATIONS


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


def compose_answer(report: AgentRunResult) -> str:
    cited_passages: list[Passage] = []
    for run in report.claims:
        bundle = run.evidence_bundle
        if bundle is None or bundle.verification is None:
            continue
        if bundle.verification.verdict == "supported":
            cited_passages.extend((bundle.supporting_passages[:2] or bundle.considered_passages[:2]))
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
                qualifier = " (пока подтверждено менее чем двумя независимыми источниками)"
            elif not bundle.has_primary_source:
                qualifier = " (без найденного первичного источника)"
            supported_lines.append(f"- {sentence}{qualifier} {citations}".rstrip())
        elif verification.verdict == "contradicted":
            passages = bundle.contradicting_passages[:2] or bundle.considered_passages[:1]
            if not passages:
                continue
            sentence = _best_span_text(verification, passages, run.claim, contradicted=True)
            if not sentence:
                continue
            citations = _format_citations(url_to_index, passages)
            caveat_lines.append(f"- {run.claim.claim_text}: найдено опровержение. {sentence} {citations}".rstrip())
        else:
            missing = ", ".join(verification.missing_dimensions) if verification.missing_dimensions else "coverage"
            gap_lines.append(f"- {run.claim.claim_text}: insufficient evidence ({missing}).")

    lines: list[str] = []
    lines.append("Ответ")
    if supported_lines:
        lines.extend(supported_lines)
    else:
        lines.append("- Недостаточно подтверждённых claim-level доказательств для прямого ответа.")

    if caveat_lines:
        lines.append("")
        lines.append("Оговорки")
        lines.extend(caveat_lines)

    if gap_lines:
        lines.append("")
        lines.append("Недостаточно данных")
        lines.extend(gap_lines)

    if indexed_sources:
        lines.append("")
        lines.append("Источники")
        for idx, title, url in indexed_sources:
            lines.append(f"[{idx}] {title} — {url}")

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
    client: OpenAI | None = None,
    receipts_dir: str | None = None,
    log=None,
) -> AgentRunResult:
    log = log or (lambda msg: None)
    started_at = datetime.now(UTC)

    classification = classify_query(query, client=client, log=log)
    claims = decompose_claims(classification, client=client, log=log)
    log(f"\n[bold]Agent Search[/bold] [dim]{len(claims)} claim(s)[/dim]")

    claim_runs: list[ClaimRun] = []
    audit = AuditTrail(
        run_id=_build_run_id(query, started_at),
        profile_name=getattr(profile, "name", None),
        started_at=started_at.isoformat(),
    )

    for claim in claims:
        log(f"\n[bold]  Claim[/bold] [italic]{claim.claim_text}[/italic]")
        all_variants: list[QueryVariant] = []
        snapshots: list[SearchSnapshot] = []
        gated_results: list[GatedSerpResult] = []
        fetch_plans: list[FetchPlan] = []
        documents: list[FetchedDocument] = []
        final_passages: list[Passage] = []
        routing_decision = RoutingDecision(
            mode="iterative_loop",
            certainty=0.0,
            consistency=0.0,
            evidence_sufficiency=0.0,
            rationale="Not evaluated yet.",
        )
        bundle: EvidenceBundle | None = None

        existing_queries: set[str] = set()
        seen_urls: set[str] = set()
        seen_documents: set[tuple[str, str, str]] = set()
        next_variants = build_query_variants(claim, classification)
        iterations_used = 0

        for iteration in range(1, MAX_CLAIM_ITERATIONS + 1):
            if not next_variants:
                break
            iterations_used = iteration

            log(f"  [bold]Iteration {iteration}/{MAX_CLAIM_ITERATIONS}[/bold]")
            for variant in next_variants:
                log(f"  [dim]↳ {variant.strategy}: {variant.query_text}[/dim]")
                existing_queries.add(variant.query_text.casefold())
            all_variants.extend(next_variants)

            new_snapshots: list[SearchSnapshot] = []
            for variant in next_variants:
                variant_snapshots = search_searxng_with_fallback(variant.query_text, profile, log=log)
                new_snapshots.extend(_retag_snapshot(snapshot, variant) for snapshot in variant_snapshots)
            snapshots.extend(new_snapshots)

            gated_limit = min(SERP_GATE_MAX_URLS, max(SERP_GATE_MIN_URLS, profile.max_results))
            gated_results = gate_serp_results(claim, snapshots, gated_limit)
            routing_decision = route_claim_retrieval(claim, gated_results)
            if bundle and bundle.verification and bundle.verification.verdict != "supported":
                if routing_decision.mode == "short_path":
                    routing_decision = replace(
                        routing_decision,
                        mode="targeted_retrieval",
                        rationale=routing_decision.rationale + " | escalated after weak verification",
                    )
                elif iteration > 1:
                    routing_decision = replace(
                        routing_decision,
                        mode="iterative_loop",
                        rationale=routing_decision.rationale + " | iterative escalation",
                    )

            log(
                f"  [dim]route={routing_decision.mode} · "
                f"certainty={routing_decision.certainty:.2f} · "
                f"consistency={routing_decision.consistency:.2f} · "
                f"sufficiency={routing_decision.evidence_sufficiency:.2f}[/dim]"
            )
            log(f"  [dim]SERP gate kept {len(gated_results)} URLs[/dim]")

            new_fetch_plans, new_documents = fetch_claim_documents(
                claim,
                gated_results,
                profile,
                routing_decision,
                seen_urls=seen_urls,
                log=log,
            )
            fetch_plans.extend(new_fetch_plans)
            for document in new_documents:
                key = (document.url, document.fetch_depth, document.content_hash)
                if key in seen_documents:
                    continue
                seen_documents.add(key)
                seen_urls.add(document.url)
                documents.append(document)

            passage_documents = _documents_for_passage_extraction(documents)
            passages: list[Passage] = []
            for document in passage_documents:
                passages.extend(_split_into_passages(document))

            cheap_filtered = cheap_passage_filter(claim, passages, CHEAP_PASSAGE_LIMIT)
            final_passages = utility_rerank_passages(claim, cheap_filtered, PASSAGE_TOP_K)
            log(
                f"  [dim]passages: {len(passages)} total · "
                f"{len(cheap_filtered)} after cheap filter · "
                f"{len(final_passages)} after utility rerank[/dim]"
            )

            verification = verify_claim(claim, final_passages, client=client, log=log)
            bundle = build_evidence_bundle(claim, final_passages, verification, gated_results)
            log(
                f"  [dim]verdict={verification.verdict} · "
                f"confidence={verification.confidence:.2f} · "
                f"independent_sources={bundle.independent_source_count}[/dim]"
            )

            if should_stop_claim_loop(claim, bundle, iteration):
                break

            next_variants = refine_query_variants(
                claim,
                classification,
                verification,
                gated_results,
                bundle,
                iteration + 1,
                existing_queries,
            )
        else:
            next_variants = []

        claim_runs.append(
            ClaimRun(
                claim=claim,
                query_variants=all_variants,
                search_snapshots=snapshots,
                gated_results=gated_results,
                fetch_plans=fetch_plans,
                fetched_documents=documents,
                passages=final_passages,
                evidence_bundle=bundle,
                routing_decision=routing_decision,
            )
        )

        audit.query_variants.extend(all_variants)
        audit.serp_snapshots.extend(snapshots)
        audit.selected_urls.extend([result.serp.url for result in gated_results])
        audit.crawl_events.extend(
            {
                "claim_id": claim.claim_id,
                "url": document.url,
                "fetched_at": document.extracted_at,
                "content_hash": document.content_hash,
                "fetch_depth": document.fetch_depth,
            }
            for document in documents
        )
        audit.passage_ids.extend([passage.passage_id for passage in final_passages])
        audit.claim_to_passages[claim.claim_id] = [passage.passage_id for passage in final_passages]
        audit.claim_iterations[claim.claim_id] = iterations_used
        if bundle and bundle.verification:
            audit.verification_results[claim.claim_id] = bundle.verification
            audit.final_verdicts[claim.claim_id] = bundle.verification.verdict

    report = AgentRunResult(
        user_query=query,
        classification=classification,
        claims=claim_runs,
        answer="",
        audit_trail=audit,
    )
    report.answer = compose_answer(report)
    completed_at = datetime.now(UTC)
    report.audit_trail.completed_at = completed_at.isoformat()
    report.audit_trail.latency_ms = int((completed_at - started_at).total_seconds() * 1000)
    report.audit_trail.estimated_search_cost = _estimate_search_cost(claim_runs)

    resolved_receipts_dir = receipts_dir or os.getenv("AGENT_RECEIPTS_DIR", "").strip() or None
    if resolved_receipts_dir:
        from receipts import write_receipt

        report.audit_trail.receipt_path = write_receipt(report, output_dir=resolved_receipts_dir)
    return report
