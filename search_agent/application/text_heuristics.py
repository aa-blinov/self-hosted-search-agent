from __future__ import annotations

import re
from datetime import datetime, timedelta

from search_agent.domain.models import Claim, EvidenceSpan, Passage, VerificationResult

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "if",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with", "vs",
    "than", "about", "between", "latest", "current",
}

RELATIVE_TIME_MARKERS = {
    "today", "tomorrow", "yesterday", "this week", "next week", "last week",
    "this month", "next month", "last month", "this quarter", "next quarter",
    "last quarter", "current", "latest",
    "сегодня", "завтра", "вчера", "текущий", "последний",
}

COMPARISON_MARKERS = {
    "compare", "comparison", "difference", "versus", "vs", "contrast",
    "сравни", "сравнение", "разница", "отличается", "лучше", "хуже",
}

NEWS_DIGEST_PATTERNS = (
    r"\bwhat happened\b",
    r"\bwhat happened in\b",
    r"\bwhat is happening in\b",
    r"\bwhat happened today\b",
    r"\bwhat's happening in\b",
    r"\bnews in\b",
    r"\bupdates in\b",
    r"\bчто произошло\b",
    r"\bчто случилось\b",
    r"\bчто было\b",
    r"\bкакие новости\b",
    r"\bновости\b",
    r"\bсобытия\b",
    r"\bпроисшествия\b",
)

NON_ENTITY_TOKENS = {
    "who", "what", "when", "where", "why", "how", "which",
    "кто", "что", "когда", "где", "почему", "как", "какой", "какая", "какие",
    "ceo", "cto", "cfo", "founder", "president", "chairman", "capital",
    "release", "released", "latest", "current",
}

DATE_LIKE_RE = re.compile(
    r"(?:"
    r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b|"
    r"\b\d{1,2}\s+[А-Яа-яЁё]+\s+\d{4}\b|"
    r"\b[A-Z][a-z]{2,9}\.?\s+\d{1,2},\s+\d{4}\b|"
    r"\b[А-ЯЁ][а-яё]{2,12}\s+\d{1,2},\s+\d{4}\b|"
    r"\b20\d{2}\b"
    r")"
)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9.+/-]*", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    for quoted in re.findall(r'"([^"]{2,80})"', text):
        entities.append(normalized_text(quoted))

    patterns = [
        r"(?:[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9.+/-]*)(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9.+/-]*){0,3}",
        r"(?:[A-Z]{2,}[A-Za-z0-9.+/-]*)(?:\s+[A-Z]{2,}[A-Za-z0-9.+/-]*){0,2}",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            cleaned = normalized_text(match)
            cleaned_key = cleaned.casefold()
            token_keys = [token.casefold() for token in tokenize(cleaned)]
            if (
                len(cleaned) > 1
                and cleaned_key not in NON_ENTITY_TOKENS
                and token_keys
                and not all(token in NON_ENTITY_TOKENS for token in token_keys)
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


def extract_time_scope(text: str) -> str | None:
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


def extract_region_hint(text: str) -> str | None:
    patterns = [
        r"\b(?:in|for|within)\s+([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+){0,2})",
        r"\b(?:в|для|по)\s+([А-ЯЁ][А-Яа-яЁё.-]+(?:\s+[А-ЯЁ][А-Яа-яЁё.-]+){0,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def needs_freshness(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in RELATIVE_TIME_MARKERS)


def normalize_relative_time_references(query: str, *, now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    today = now.date()
    replacements = {
        "today": today.isoformat(),
        "сегодня": today.isoformat(),
        "tomorrow": (today + timedelta(days=1)).isoformat(),
        "завтра": (today + timedelta(days=1)).isoformat(),
        "yesterday": (today - timedelta(days=1)).isoformat(),
        "вчера": (today - timedelta(days=1)).isoformat(),
    }
    normalized = query
    for marker, value in replacements.items():
        normalized = re.sub(rf"\b{re.escape(marker)}\b", value, normalized, flags=re.IGNORECASE)
    return normalized


def is_news_digest_query(
    text: str,
    *,
    region_hint: str | None = None,
    freshness: bool = False,
) -> bool:
    lowered = text.casefold()
    if not any(re.search(pattern, lowered) for pattern in NEWS_DIGEST_PATTERNS):
        return False
    return freshness or bool(region_hint)


def extract_region_hint(text: str) -> str | None:
    patterns = [
        r"\b(?:in|for|within)\s+([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+){0,2})",
        r"\b(?:\u0432|\u0434\u043b\u044f|\u043f\u043e)\s+([\u0410-\u042f\u0401][\u0410-\u042f\u0430-\u044f\u0401\u0451.-]+(?:\s+[\u0410-\u042f\u0401][\u0410-\u042f\u0430-\u044f\u0401\u0451.-]+){0,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def extract_time_scope(text: str) -> str | None:
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\bQ[1-4]\s+20\d{2}\b",
        r"\b\d{1,2}\s+[A-Z][a-z]+\s+20\d{2}\b",
        r"\b\d{1,2}\s+[\u0410-\u042f\u0430-\u044f\u0401\u0451]+\s+20\d{2}\b",
        r"\b20\d{2}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def needs_freshness(text: str) -> bool:
    lowered = text.casefold()
    markers = {
        "today",
        "tomorrow",
        "yesterday",
        "this week",
        "next week",
        "last week",
        "this month",
        "next month",
        "last month",
        "this quarter",
        "next quarter",
        "last quarter",
        "current",
        "latest",
        "\u0441\u0435\u0433\u043e\u0434\u043d\u044f",
        "\u0437\u0430\u0432\u0442\u0440\u0430",
        "\u0432\u0447\u0435\u0440\u0430",
        "\u0442\u0435\u043a\u0443\u0449\u0438\u0439",
        "\u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0439",
    }
    return any(marker in lowered for marker in markers)


def normalize_relative_time_references(query: str, *, now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    today = now.date()
    replacements = {
        "today": today.isoformat(),
        "\u0441\u0435\u0433\u043e\u0434\u043d\u044f": today.isoformat(),
        "tomorrow": (today + timedelta(days=1)).isoformat(),
        "\u0437\u0430\u0432\u0442\u0440\u0430": (today + timedelta(days=1)).isoformat(),
        "yesterday": (today - timedelta(days=1)).isoformat(),
        "\u0432\u0447\u0435\u0440\u0430": (today - timedelta(days=1)).isoformat(),
    }
    normalized = query
    for marker, value in replacements.items():
        normalized = re.sub(rf"\b{re.escape(marker)}\b", value, normalized, flags=re.IGNORECASE)
    return normalized


def is_news_digest_query(
    text: str,
    *,
    region_hint: str | None = None,
    freshness: bool = False,
) -> bool:
    lowered = text.casefold()
    has_digest_cue = any(
        re.search(pattern, lowered)
        for pattern in (
            r"\bwhat\b.*\bhappened\b",
            r"\bwhat\b.*\bhappening\b",
            r"\bnews in\b",
            r"\bupdates in\b",
            r"\b\u0447\u0442\u043e\b.*\b(?:\u043f\u0440\u043e\u0438\u0437\u043e\u0448\u043b\u043e|\u0441\u043b\u0443\u0447\u0438\u043b\u043e\u0441\u044c|\u0431\u044b\u043b\u043e)\b",
            r"\b\u043a\u0430\u043a\u0438\u0435\b.*\b\u043d\u043e\u0432\u043e\u0441\u0442\u0438\b",
            r"\b\u043d\u043e\u0432\u043e\u0441\u0442\u0438\b",
            r"\b\u0441\u043e\u0431\u044b\u0442\u0438\u044f\b",
            r"\b\u043f\u0440\u043e\u0438\u0441\u0448\u0435\u0441\u0442\u0432\u0438\u044f\b",
        )
    )
    if not has_digest_cue:
        return False
    return freshness or bool(region_hint)


def should_decompose(text: str) -> bool:
    lowered = text.lower()
    return (
        any(marker in lowered for marker in COMPARISON_MARKERS)
        or " and " in lowered
        or " и " in lowered
        or len(text) > 90
        or text.count("?") > 1
    )


def host_root(host: str) -> str:
    parts = (host or "").split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", "", (text or "").casefold())


def _soft_text_match(needle: str, haystack: str) -> bool:
    compact_needle = _compact_text(needle)
    compact_haystack = _compact_text(haystack)
    if not compact_needle or not compact_haystack:
        return False
    if compact_needle in compact_haystack:
        return True
    return len(compact_needle) >= 5 and compact_needle[:5] in compact_haystack


def _time_scope_matches_passage(time_scope: str | None, passage: Passage) -> bool:
    if not time_scope:
        return True
    haystack = f"{passage.title} {passage.text}"
    if time_scope.casefold() in haystack.casefold():
        return True
    return bool(passage.published_at and passage.published_at.startswith(time_scope))


def heuristic_verifier(claim: Claim, passages: list[Passage]) -> VerificationResult:
    if not passages:
        return VerificationResult(
            verdict="insufficient_evidence",
            confidence=0.1,
            missing_dimensions=["source"],
            rationale="No passages were available for verification.",
        )

    strong = [passage for passage in passages if passage.utility_score >= 0.35]
    region_hint = extract_region_hint(claim.claim_text) or (claim.entity_set[0] if claim.entity_set else None)
    if is_news_digest_query(
        claim.claim_text,
        region_hint=region_hint,
        freshness=bool(claim.needs_freshness or claim.time_scope),
    ):
        digest_support = [
            passage
            for passage in passages
            if passage.utility_score >= 0.25
            and (not region_hint or _soft_text_match(region_hint, f"{passage.title} {passage.text}"))
            and _time_scope_matches_passage(claim.time_scope, passage)
        ]
        if len(digest_support) >= 2 and len({host_root(p.host) for p in digest_support[:4]}) >= 2:
            supporting = [
                EvidenceSpan(
                    passage_id=passage.passage_id,
                    url=passage.url,
                    title=passage.title,
                    section=passage.section,
                    text=passage.text[:220],
                )
                for passage in digest_support[:4]
            ]
            return VerificationResult(
                verdict="supported",
                confidence=0.55,
                supporting_spans=supporting,
                rationale="Multiple independent reports align on the requested place/time scope.",
            )

    if len(strong) >= 2 and len({host_root(p.host) for p in strong[:3]}) >= 2:
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
