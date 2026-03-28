from __future__ import annotations

from search_agent.application import policy_tuning
from search_agent.application.agent_passage_scoring import _dimension_coverage_score
from search_agent.application.agent_scoring_shared import (
    _claim_answer_shape,
    _claim_min_independent_sources,
    _claim_requires_primary_source,
    _preferred_domain_bonus,
)
from search_agent.application.agent_sources import (
    _effective_domain_type,
    _entity_overlap,
    _host_root,
    _semantic_overlap,
    _verification_source_bonus,
)
from search_agent.application.claim_policy import (
    claim_contract_gaps as _policy_claim_contract_gaps,
    retrieval_contract_can_drive_synthesis as _policy_retrieval_contract_can_drive_synthesis,
    should_stop_claim_loop as _policy_should_stop_claim_loop,
)
from search_agent.domain.models import (
    Claim,
    DomainType,
    EvidenceBundle,
    EvidenceSpan,
    GatedSerpResult,
    Passage,
    VerificationResult,
)


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
