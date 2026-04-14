from __future__ import annotations

from dataclasses import replace

from search_agent import tuning
from search_agent.application import policy_tuning
from search_agent.application.agent_scoring_shared import (
    _answer_type,
    _claim_answer_shape,
    _claim_focus_terms,
    _clamp,
    _compact_text,
    _contains_date_like,
    _contains_location_span,
    _contains_negation_cue,
    _contains_person_span,
    _exact_detail_guardrail_claim,
    _extract_date_candidates,
    _extract_entities,
    _extract_location_candidates,
    _extract_person_candidates,
    _is_iso_date_text,
    _is_news_digest_claim,
    _preferred_domain_bonus,
    _tokenize,
)
from search_agent.application.agent_sources import (
    _effective_domain_type,
    _entity_overlap,
    _host_root,
    _semantic_overlap,
    _verification_source_bonus,
)
from search_agent.application.text_heuristics import (
    extract_numbers as _shared_extract_numbers,
    extract_region_hint as _shared_extract_region_hint,
    is_cyrillic_text as _is_cyrillic_text,
)
from search_agent.domain.models import Claim, GatedSerpResult, Passage, RoutingDecision


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
    """Iteration-based routing: fast on iter1, full on iter2+.

    The mode is always 'fast' here (called on first encounter).
    Escalation to 'full' happens in use_cases._route_iteration based on
    verification results — the real adaptive decision is should_stop_claim_loop.
    """
    if not gated_results:
        return RoutingDecision(
            mode="full",
            certainty=0.0,
            consistency=0.0,
            evidence_sufficiency=0.0,
            rationale="No gated results available.",
        )

    top = gated_results[:5]
    certainty = _clamp(sum(r.assessment.source_score for r in top[:3]) / min(len(top[:3]), 3))
    consistency = _clamp(
        sum(r.assessment.semantic_match_score for r in top) / len(top)
    )
    evidence_sufficiency = _clamp(
        sum(1 for r in gated_results[:8] if r.assessment.source_score >= 0.6) * 0.11
        + (0.2 if any(r.assessment.primary_source_likelihood >= 0.7 for r in top) else 0.0)
    )

    return RoutingDecision(
        mode="fast",
        certainty=certainty,
        consistency=consistency,
        evidence_sufficiency=evidence_sufficiency,
        rationale=(
            f"certainty={certainty:.2f}, consistency={consistency:.2f}, "
            f"evidence_sufficiency={evidence_sufficiency:.2f}"
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


def _is_cross_language(claim: Claim, passages: list[Passage]) -> bool:
    """Check if claim and passages are in different scripts (Cyrillic vs Latin)."""
    claim_cyrillic = _is_cyrillic_text(claim.claim_text)
    if not passages:
        return False
    sample = passages[:5]
    passage_cyrillic_count = sum(1 for p in sample if _is_cyrillic_text(p.text))
    passage_majority_cyrillic = passage_cyrillic_count > len(sample) // 2
    return claim_cyrillic != passage_majority_cyrillic


def cheap_passage_filter(
    claim: Claim,
    passages: list[Passage],
    limit: int = tuning.CHEAP_PASSAGE_LIMIT,
) -> list[Passage]:
    cross_lang = _is_cross_language(claim, passages)
    threshold = 0.0 if cross_lang else 0.18
    scored: list[tuple[float, Passage]] = []
    for passage in passages:
        score = cheap_passage_score(claim, passage)
        if score >= threshold:
            scored.append((score, passage))
    if not scored:
        scored = [(cheap_passage_score(claim, passage), passage) for passage in passages]
    scored.sort(key=lambda item: (item[0], item[1].source_score), reverse=True)
    return [passage for _, passage in scored[:limit]]


_PRIOR_ITERATION_BOOST = 0.12


def utility_rerank_passages(
    claim: Claim,
    passages: list[Passage],
    limit: int = tuning.AGENT_PASSAGE_TOP_K,
    prior_passage_ids: set[str] | None = None,
) -> list[Passage]:
    if _claim_answer_shape(claim) == "product_specs":
        limit = max(limit, 4)
    reranked = []
    for passage in passages:
        score = utility_score_for_claim(claim, passage)
        if prior_passage_ids and passage.passage_id in prior_passage_ids:
            score = _clamp(score + _PRIOR_ITERATION_BOOST)
        reranked.append(replace(passage, utility_score=score))
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
