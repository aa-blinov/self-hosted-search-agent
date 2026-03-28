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
