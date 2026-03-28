from __future__ import annotations

from search_agent import tuning
from search_agent.application import policy_tuning
from search_agent.application.claim_policy import (
    answer_type as _policy_answer_type,
    claim_answer_shape as _policy_claim_answer_shape,
    claim_focus_terms as _policy_claim_focus_terms,
    claim_min_independent_sources as _policy_claim_min_independent_sources,
    claim_requires_primary_source as _policy_claim_requires_primary_source,
    exact_detail_guardrail_claim as _policy_exact_detail_guardrail_claim,
    is_news_digest_claim as _policy_is_news_digest_claim,
)
from search_agent.application.text_heuristics import (
    compact_text as _shared_compact_text,
    contains_date_like as _shared_contains_date_like,
    extract_entities as _shared_extract_entities,
    extract_region_hint as _shared_extract_region_hint,
    extract_time_scope as _shared_extract_time_scope,
    normalized_text as _shared_normalized_text,
    tokenize as _shared_tokenize,
)
from search_agent.domain.models import Claim, DomainType

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


def _answer_type(claim: Claim) -> str:
    return _policy_answer_type(claim)


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
