from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from search_agent.domain.models import Passage

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "if",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with", "vs",
    "than", "about", "between", "latest", "current",
    "что", "как", "где", "когда", "кто", "или", "для", "это", "эта", "этот",
    "про", "из", "на", "по", "в", "во", "с", "со", "у", "о", "об", "ли",
    "чем", "какой", "какая", "какие", "каково", "есть", "был", "была", "были",
}

RELATIVE_TIME_MARKERS = {
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
    "сегодня",
    "завтра",
    "вчера",
    "текущий",
    "последний",
}

COMPARISON_MARKERS = {
    "compare",
    "comparison",
    "difference",
    "versus",
    "vs",
    "contrast",
    "сравни",
    "сравнение",
    "разница",
    "отличается",
    "лучше",
    "хуже",
}

NEWS_DIGEST_PHRASES = (
    "what happened",
    "what happened in",
    "what is happening",
    "what's happening",
    "latest news",
    "news on",
    "news in",
    "updates on",
    "updates in",
    "latest developments",
    "последние новости",
    "какие новости",
    "что произошло",
    "что случилось",
    "что было",
    "события",
    "происшествия",
    "обновления",
)

NON_ENTITY_TOKENS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "кто",
    "что",
    "когда",
    "где",
    "почему",
    "как",
    "какой",
    "какая",
    "какие",
    "ceo",
    "cto",
    "cfo",
    "founder",
    "president",
    "chairman",
    "capital",
    "release",
    "released",
    "latest",
    "current",
}

_ENTITY_JOINERS = {"of", "the", "and", "for", "de", "la", "и"}
_REGION_PREPOSITIONS = {"in", "for", "within", "в", "для", "по"}
_RELATIVE_DATE_REPLACEMENTS = {
    "today": 0,
    "tomorrow": 1,
    "yesterday": -1,
    "сегодня": 0,
    "завтра": 1,
    "вчера": -1,
}
_EN_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_RU_MONTHS = {
    "янв": 1,
    "января": 1,
    "фев": 2,
    "февраля": 2,
    "мар": 3,
    "марта": 3,
    "апр": 4,
    "апреля": 4,
    "мая": 5,
    "июн": 6,
    "июня": 6,
    "июл": 7,
    "июля": 7,
    "авг": 8,
    "августа": 8,
    "сен": 9,
    "сентября": 9,
    "окт": 10,
    "октября": 10,
    "ноя": 11,
    "ноября": 11,
    "дек": 12,
    "декабря": 12,
}


@dataclass(slots=True)
class _WordSpan:
    text: str
    start: int
    end: int


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def normalized_text(text: str) -> str:
    return " ".join((text or "").split())


def compact_text(text: str) -> str:
    return "".join(ch for ch in (text or "").casefold() if ch.isalnum())


def has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text or "")


def is_cyrillic_text(text: str) -> bool:
    return any("а" <= ch.casefold() <= "я" or ch in {"ё", "Ё"} for ch in text or "")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for span in _iter_word_spans(text):
        token = _clean_token(span.text).casefold()
        if len(token) <= 1:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def extract_numbers(text: str) -> list[str]:
    numbers: list[str] = []
    for span in _iter_word_spans(text):
        token = _clean_token(span.text)
        if token and _looks_like_number_token(token):
            numbers.append(token)
    return numbers


def contains_date_like(text: str) -> bool:
    return extract_time_scope(text) is not None


def extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    for quoted in _quoted_segments(text):
        cleaned = normalized_text(quoted)
        if 2 <= len(cleaned) <= 80:
            entities.append(cleaned)

    words = _iter_word_spans(text)
    index = 0
    while index < len(words):
        word = words[index]
        if not _looks_like_entity_token(word.text):
            index += 1
            continue
        collected = [word.text]
        next_index = index + 1
        while next_index < len(words):
            token = _clean_token(words[next_index].text).casefold()
            separator = text[words[next_index - 1].end:words[next_index].start]
            if separator and not separator.isspace():
                break
            if token in _ENTITY_JOINERS:
                if next_index + 1 >= len(words):
                    break
                after = words[next_index + 1]
                after_sep = text[words[next_index].end:after.start]
                if after_sep and not after_sep.isspace():
                    break
                if not _looks_like_entity_token(after.text):
                    break
                collected.append(words[next_index].text)
                next_index += 1
                continue
            if not _looks_like_entity_token(words[next_index].text):
                break
            collected.append(words[next_index].text)
            if len(collected) >= 4:
                break
            next_index += 1
        candidate = normalized_text(" ".join(collected))
        if _entity_candidate_ok(candidate):
            entities.append(candidate)
        index += 1

    deduped: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        key = entity.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped[:6]


def extract_time_scope(text: str) -> str | None:
    words = _iter_word_spans(text)
    clean_words = [_clean_token(word.text) for word in words]
    for token in clean_words:
        if _is_iso_date(token):
            return token
    for index, token in enumerate(clean_words):
        lowered = token.casefold()
        if lowered in {"q1", "q2", "q3", "q4"} and index + 1 < len(clean_words):
            year = clean_words[index + 1]
            if _is_year(year):
                return f"{token} {year}"
    for index in range(len(clean_words) - 2):
        day = clean_words[index]
        month = clean_words[index + 1]
        year = clean_words[index + 2]
        if _is_day(day) and _is_month(month) and _is_year(year):
            return f"{day} {month} {year}"
        if _is_month(day) and _is_day(month) and _is_year(year):
            return f"{day} {month} {year}"
    for token in clean_words:
        if _is_year(token):
            return token
    return None


def extract_region_hint(text: str) -> str | None:
    words = _iter_word_spans(text)
    for index, word in enumerate(words):
        token = _clean_token(word.text).casefold()
        if token not in _REGION_PREPOSITIONS:
            continue
        collected: list[str] = []
        next_index = index + 1
        while next_index < len(words) and len(collected) < 3:
            candidate = _clean_token(words[next_index].text)
            if not candidate or not _looks_like_region_token(candidate):
                break
            collected.append(candidate)
            next_index += 1
        if collected:
            return normalized_text(" ".join(collected))
    return None


def needs_freshness(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(marker in lowered for marker in RELATIVE_TIME_MARKERS)


def normalize_relative_time_references(query: str, *, now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    today = now.date()
    replacements = {
        marker: (today + timedelta(days=offset)).isoformat()
        for marker, offset in _RELATIVE_DATE_REPLACEMENTS.items()
    }
    spans = _iter_word_spans(query)
    if not spans:
        return query
    parts: list[str] = []
    cursor = 0
    for span in spans:
        token = _clean_token(span.text).casefold()
        replacement = replacements.get(token)
        if replacement is None:
            continue
        parts.append(query[cursor:span.start])
        parts.append(replacement)
        cursor = span.end
    if cursor == 0:
        return query
    parts.append(query[cursor:])
    return "".join(parts)


def is_news_digest_query(
    text: str,
    *,
    region_hint: str | None = None,
    freshness: bool = False,
) -> bool:
    lowered = (text or "").casefold()
    if not any(phrase in lowered for phrase in NEWS_DIGEST_PHRASES):
        return False
    return freshness or bool(region_hint)


def should_decompose(text: str) -> bool:
    lowered = (text or "").casefold()
    return (
        any(marker in lowered for marker in COMPARISON_MARKERS)
        or " and " in lowered
        or " и " in lowered
        or len(text or "") > 90
        or (text or "").count("?") > 1
    )


def host_root(host: str) -> str:
    parts = (host or "").split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def _iter_word_spans(text: str) -> list[_WordSpan]:
    spans: list[_WordSpan] = []
    source = text or ""
    index = 0
    length = len(source)
    while index < length:
        if not _is_word_start_char(source[index]):
            index += 1
            continue
        start = index
        index += 1
        while index < length and _is_word_char(source[index]):
            index += 1
        spans.append(_WordSpan(text=source[start:index], start=start, end=index))
    return spans


def _quoted_segments(text: str) -> list[str]:
    pairs = {'"': '"', "“": "”", "«": "»"}
    source = text or ""
    segments: list[str] = []
    index = 0
    while index < len(source):
        opener = source[index]
        closer = pairs.get(opener)
        if closer is None:
            index += 1
            continue
        end = source.find(closer, index + 1)
        if end == -1:
            break
        segment = source[index + 1:end]
        if segment:
            segments.append(segment)
        index = end + 1
    return segments


def _clean_token(token: str) -> str:
    start = 0
    end = len(token)
    while start < end and not token[start].isalnum():
        start += 1
    while end > start and not token[end - 1].isalnum():
        end -= 1
    return token[start:end]


def _is_word_start_char(ch: str) -> bool:
    return ch.isalnum()


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch in {".", "+", "/", "-", "_", "'"}


def _looks_like_entity_token(token: str) -> bool:
    cleaned = _clean_token(token)
    if not cleaned:
        return False
    lowered = cleaned.casefold()
    if lowered in NON_ENTITY_TOKENS:
        return False
    if cleaned.isupper() and any(ch.isalpha() for ch in cleaned):
        return True
    if cleaned[0].isupper():
        return True
    if cleaned[0].isdigit():
        return any(ch.isupper() for ch in cleaned if ch.isalpha())
    return False


def _entity_candidate_ok(candidate: str) -> bool:
    if len(candidate) <= 1:
        return False
    lowered = candidate.casefold()
    if lowered in NON_ENTITY_TOKENS:
        return False
    token_keys = tokenize(candidate)
    if not token_keys:
        return False
    return not all(token in NON_ENTITY_TOKENS for token in token_keys)


def _looks_like_region_token(token: str) -> bool:
    if not token:
        return False
    if token.casefold() in STOPWORDS:
        return False
    return token[0].isupper()


def _soft_text_match(needle: str, haystack: str) -> bool:
    compact_needle = compact_text(needle)
    compact_haystack = compact_text(haystack)
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


def _is_year(token: str) -> bool:
    return len(token) == 4 and token.isdigit() and token.startswith("20")


def _is_iso_date(token: str) -> bool:
    return (
        len(token) == 10
        and token[4] == "-"
        and token[7] == "-"
        and token[:4].isdigit()
        and token[5:7].isdigit()
        and token[8:10].isdigit()
    )


def _is_day(token: str) -> bool:
    if not token.isdigit():
        return False
    day = int(token)
    return 1 <= day <= 31


def _is_month(token: str) -> bool:
    lowered = token.casefold().rstrip(".")
    return lowered in _EN_MONTHS or lowered in _RU_MONTHS


def _looks_like_number_token(token: str) -> bool:
    if not token:
        return False
    dot_seen = False
    for ch in token:
        if ch.isdigit():
            continue
        if ch in {".", ","} and not dot_seen:
            dot_seen = True
            continue
        return False
    return any(ch.isdigit() for ch in token)
