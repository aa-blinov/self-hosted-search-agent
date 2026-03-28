from __future__ import annotations

from search_agent import tuning
from search_agent.application import policy_tuning
from search_agent.application.agent_scoring import (
    _answer_type,
    _claim_answer_shape,
    _contains_location_span,
    _contains_person_span,
    _dimension_coverage_score,
    _effective_domain_type,
    _entity_overlap,
    _host_root,
    _is_iso_date_text,
    _is_year_text,
    _local_news_host_bonus,
    _news_digest_region_hint_from_claim,
    _news_digest_time_match,
    _normalized_text,
    _semantic_overlap,
    _split_sentences,
    _verification_source_bonus,
    _preferred_domain_bonus,
)
from search_agent.application.claim_policy import (
    claim_min_independent_sources as _policy_claim_min_independent_sources,
    claim_requires_primary_source as _policy_claim_requires_primary_source,
    publish_supported_claim as _policy_publish_supported_claim,
)
from search_agent.application.text_heuristics import (
    extract_numbers as _shared_extract_numbers,
    extract_time_scope as _shared_extract_time_scope,
    is_cyrillic_text as _shared_is_cyrillic_text,
)
from search_agent.domain.models import AgentRunResult, Claim, ClaimRun, EvidenceBundle, Passage, VerificationResult


def _truncate_compose_line(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


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


def _compose_ui_labels(user_query: str) -> dict[str, str]:
    ru = _shared_is_cyrillic_text(user_query or "")
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


def _time_specificity_score(text: str) -> float:
    scope = _shared_extract_time_scope(text)
    if not scope:
        return 0.0
    scores = policy_tuning.TIME_SPECIFICITY_SCORES
    if _is_iso_date_text(scope):
        return scores["iso_date"]
    if _is_year_text(scope):
        return scores["year"]
    if any(ch.isalpha() for ch in scope) and any(ch.isdigit() for ch in scope):
        return scores["named_period"]
    if scope.casefold().startswith("q"):
        return scores["quarter"]
    return scores["fallback"]


def _answer_sentence_score(claim: Claim, sentence: str) -> float:
    weights = policy_tuning.ANSWER_SENTENCE
    score = _semantic_overlap(claim.claim_text, sentence) + weights["entity_weight"] * _entity_overlap(claim.entity_set, sentence)
    answer_type = _answer_type(claim)
    if answer_type == "time":
        score += weights["time_specificity_weight"] * _time_specificity_score(sentence)
    elif answer_type == "number" and _shared_extract_numbers(sentence):
        score += weights["number_boost"]
    elif answer_type == "person" and _contains_person_span(sentence):
        score += weights["person_boost"]
    elif answer_type == "location" and _contains_location_span(sentence):
        score += weights["location_boost"]
    if claim.time_scope and claim.time_scope.casefold() in sentence.casefold():
        score += weights["time_scope_boost"]
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


def _best_span_text(
    verification: VerificationResult,
    passages: list[Passage],
    claim: Claim,
    contradicted: bool = False,
) -> str | None:
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
        if region and region_match < policy_tuning.NEWS_SELECTION["min_region_match"]:
            continue
        if run.claim.time_scope and time_match <= 0.0:
            continue
        weights = policy_tuning.NEWS_SELECTION
        score = (
            weights["region_weight"] * region_match
            + weights["time_weight"] * time_match
            + weights["local_bonus_weight"] * local_bonus
            + weights["utility_or_source_weight"] * max(passage.utility_score, passage.source_score)
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
        weights = policy_tuning.SUPPORT_SCORE_WEIGHTS
        score = (
            weights["semantic_overlap"] * _semantic_overlap(claim.claim_text, lead)
            + weights["entity_overlap"] * _entity_overlap(claim.entity_set, lead)
            + weights["dimension_coverage"] * _dimension_coverage_score(claim, lead)
            + weights["verification_bonus"] * max(
                _verification_source_bonus(
                    claim,
                    host=passage.host,
                    title=passage.title,
                    url=passage.url,
                ),
                0.0,
            )
            + weights["utility_or_source"] * max(passage.utility_score, passage.source_score)
        )
        if _claim_answer_shape(claim) == "product_specs":
            score += (
                policy_tuning.SUPPORT_SCORE_PRODUCT_PREFERRED_BONUS
                * _preferred_domain_bonus(claim, _effective_domain_type(claim, passage.host))
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

    min_sources = _policy_claim_min_independent_sources(claim)
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
    labels = _compose_ui_labels(report.user_query)
    ru = _shared_is_cyrillic_text(report.user_query)
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
            digest_lines.append(
                f"- {_sanitize_compose_fragment(_digest_sentence(passage))} {_format_citations(url_to_index, [passage])}".rstrip()
            )

        if digest_lines:
            lines = [f"- {labels['digest_header']}"]
            lines.extend(digest_lines[:4])
            if indexed_sources:
                lines.append("")
                lines.append(labels["sources"])
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
            cited_passages.extend(bundle.contradicting_passages[:2] or bundle.considered_passages[:2])

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
            min_sources = _policy_claim_min_independent_sources(run.claim)
            if bundle.independent_source_count < min_sources:
                qualifier = (
                    f" (подтверждено менее чем {min_sources} независимыми источниками)"
                    if ru
                    else f" (supported by fewer than {min_sources} independent sources)"
                )
            elif _policy_claim_requires_primary_source(run.claim) and not bundle.has_primary_source:
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
            missing = ", ".join(verification.missing_dimensions) if verification.missing_dimensions else ("покрытие" if ru else "coverage")
            verdict_text = "недостаточно данных" if ru else "insufficient evidence"
            gap_lines.append(f"- {run.claim.claim_text}: {verdict_text} ({missing}).")

    lines: list[str] = []
    if supported_lines:
        lines.extend(supported_lines)
    else:
        lines.append(labels["answer"])
        lines.append(labels["no_evidence"])

    if caveat_lines:
        lines.append("")
        lines.append(labels["caveats"])
        lines.extend(caveat_lines)

    if gap_lines:
        lines.append("")
        lines.append(labels["insufficient"])
        lines.extend(gap_lines)

    used_source_indices = set()
    for line in supported_lines + caveat_lines:
        used_source_indices.update(_extract_citation_indices(line))

    if indexed_sources:
        lines.append("")
        lines.append(labels["sources"])
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
        + policy_tuning.SEARCH_COST_WEIGHTS["shallow"] * shallow
        + policy_tuning.SEARCH_COST_WEIGHTS["deep"] * deep
        + policy_tuning.SEARCH_COST_WEIGHTS["snippet_only"] * snippet_only
        + policy_tuning.SEARCH_COST_WEIGHTS["claim_run"] * len(claim_runs),
        3,
    )
