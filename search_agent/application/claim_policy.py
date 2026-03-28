from __future__ import annotations

from dataclasses import replace

from search_agent import tuning
from search_agent.application.text_heuristics import compact_text, normalized_text
from search_agent.domain.models import Claim, ClaimProfile, EvidenceBundle, EvidenceSpan, Passage, VerificationResult


_LIST_LIKE_DIMENSIONS = {"feature_list", "improvements", "changes", "highlights", "options"}
_RETRIEVAL_DRIVEN_SHAPES = {"product_specs", "overview", "comparison", "news_digest"}
_OPEN_ENDED_STOP_SHAPES = {"overview", "comparison", "news_digest"}
_SIMPLE_FACT_BLOCKED_DIMENSIONS = {"time", "date", "number", "location", "price", "source"}


def claim_profile_lines(claim: Claim) -> list[str]:
    profile = claim.claim_profile
    if profile is None:
        return [
            "answer_shape=fact",
            "primary_source_required=False",
            "min_independent_sources=1",
            "preferred_domain_types=",
            "required_dimensions=",
            "focus_terms=",
            "strict_contract=False",
        ]
    return [
        f"answer_shape={profile.answer_shape}",
        f"primary_source_required={profile.primary_source_required}",
        f"min_independent_sources={profile.min_independent_sources}",
        f"preferred_domain_types={','.join(profile.preferred_domain_types)}",
        f"required_dimensions={','.join(profile.required_dimensions)}",
        f"focus_terms={','.join(profile.focus_terms)}",
        f"strict_contract={profile.strict_contract}",
    ]


def is_list_like_contract(profile: ClaimProfile | None) -> bool:
    if profile is None:
        return False
    if profile.answer_shape == "comparison":
        return True
    if profile.answer_shape != "overview":
        return False
    dimension_keys = {value.casefold() for value in profile.required_dimensions}
    if not dimension_keys and len(profile.focus_terms) >= 2:
        return True
    return bool(dimension_keys & _LIST_LIKE_DIMENSIONS)


def claim_answer_shape(claim: Claim) -> str:
    if claim.claim_profile is not None:
        return claim.claim_profile.answer_shape
    return "fact"


def claim_requires_primary_source(claim: Claim) -> bool:
    return bool(claim.claim_profile and claim.claim_profile.primary_source_required)


def claim_min_independent_sources(claim: Claim) -> int:
    if claim.claim_profile is not None:
        return max(1, claim.claim_profile.min_independent_sources)
    return 1


def exact_detail_guardrail_claim(claim: Claim) -> bool:
    profile = claim.claim_profile
    if profile is None:
        return False
    return profile.answer_shape == "exact_number" and profile.strict_contract and "number" in profile.required_dimensions


def is_news_digest_claim(claim: Claim) -> bool:
    return bool(claim.claim_profile and claim.claim_profile.answer_shape == "news_digest")


def answer_type(claim: Claim) -> str:
    profile = claim.claim_profile
    if profile is not None:
        if profile.answer_shape == "exact_date":
            return "time"
        if profile.answer_shape == "exact_number":
            return "number"
    return "fact"


def claim_focus_terms(claim: Claim) -> tuple[str, ...]:
    profile = claim.claim_profile
    if profile is None:
        return ()
    seen: set[str] = set()
    ordered: list[str] = []
    for term in profile.focus_terms:
        key = compact_text(term)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(term)
    return tuple(ordered)


def _host_root(host: str) -> str:
    parts = [part for part in (host or "").casefold().split(".") if part and part != "www"]
    if len(parts) <= 2:
        return ".".join(parts)
    if parts[-2] in {"co", "com", "org", "net"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _simple_fact_contract(profile: ClaimProfile | None) -> bool:
    if profile is None or profile.answer_shape != "fact":
        return False
    return not any(
        blocked in dimension.casefold()
        for dimension in profile.required_dimensions
        for blocked in _SIMPLE_FACT_BLOCKED_DIMENSIONS
    )


def _passage_support_floor(passage: Passage) -> float:
    return max(passage.utility_score, passage.source_score)


def _supporting_text(passage: Passage) -> str:
    title = normalized_text(passage.title)
    text = normalized_text((passage.text or "")[:260])
    return f"{title} {text}".strip()


def claim_contract_gaps(
    claim: Claim,
    verification: VerificationResult | None = None,
    *,
    independent_source_count: int,
    has_primary_source: bool,
    freshness_ok: bool,
) -> list[str]:
    del verification
    gaps: list[str] = []
    if claim_requires_primary_source(claim) and not has_primary_source:
        gaps.append("primary_source")

    min_sources = claim_min_independent_sources(claim)
    if independent_source_count < min_sources:
        gaps.append("independent_sources")

    if claim.needs_freshness and not freshness_ok:
        gaps.append("freshness")

    return gaps


def retrieval_contract_can_drive_synthesis(claim: Claim) -> bool:
    profile = claim.claim_profile
    if profile is None:
        return False
    return profile.answer_shape in _RETRIEVAL_DRIVEN_SHAPES


def publish_supported_claim(claim: Claim, bundle: EvidenceBundle) -> bool:
    profile = claim.claim_profile
    if profile is None:
        return True
    if profile.strict_contract and not bundle.contract_satisfied:
        return False
    return True


def should_stop_claim_loop(claim: Claim, bundle: EvidenceBundle, iteration: int) -> bool:
    verification = bundle.verification
    if verification is None:
        return iteration >= tuning.AGENT_MAX_CLAIM_ITERATIONS
    primary_sensitive = claim_requires_primary_source(claim)
    min_sources = claim_min_independent_sources(claim)
    profile = claim.claim_profile
    answer_shape = profile.answer_shape if profile is not None else "fact"
    open_ended = answer_shape in _OPEN_ENDED_STOP_SHAPES

    if verification.verdict == "supported":
        if bundle.contract_satisfied:
            return True
        if claim.needs_freshness and not bundle.freshness_ok:
            return iteration >= tuning.AGENT_MAX_CLAIM_ITERATIONS
        if profile is not None and profile.strict_contract:
            return iteration >= tuning.AGENT_MAX_CLAIM_ITERATIONS
        if primary_sensitive and not bundle.has_primary_source:
            return iteration >= tuning.AGENT_MAX_CLAIM_ITERATIONS
        if bundle.independent_source_count >= min_sources and (not primary_sensitive or bundle.has_primary_source):
            return True
        if verification.confidence >= 0.75 and bundle.independent_source_count >= min_sources and not primary_sensitive:
            return True

    if verification.verdict == "insufficient_evidence" and profile is not None and profile.strict_contract:
        if bundle.contract_satisfied and bundle.considered_passages:
            return True
        if exact_detail_guardrail_claim(claim) and verification.confidence >= 0.9 and bundle.independent_source_count >= 2:
            return True

    if open_ended and bundle.considered_passages:
        return True

    return iteration >= tuning.AGENT_MAX_CLAIM_ITERATIONS


def post_adjust_verification(claim: Claim, passages: list[Passage], result: VerificationResult) -> VerificationResult:
    if result.verdict == "supported" and result.confidence < 0.05:
        return replace(result, confidence=max(result.confidence, 0.38))
    profile = claim.claim_profile
    if result.verdict == "supported" and profile is not None and not profile.strict_contract and is_list_like_contract(profile):
        rationale = (result.rationale or "").strip()
        suffix = "Adjusted: open-ended contract stays claim-level insufficient_evidence until synthesis."
        merged_rationale = f"{rationale} {suffix}".strip() if rationale else suffix
        return replace(
            result,
            verdict="insufficient_evidence",
            confidence=min(result.confidence, 0.62),
            supporting_spans=[],
            missing_dimensions=result.missing_dimensions or ["coverage"],
            rationale=merged_rationale,
        )
    if (
        result.verdict == "insufficient_evidence"
        and profile is not None
        and not profile.strict_contract
        and profile.answer_shape == "overview"
    ):
        if is_list_like_contract(profile):
            return result
        robust = [
            passage
            for passage in passages
            if passage.utility_score >= 0.2 or passage.source_score >= 0.7
        ]
        unique_hosts = {
            (passage.host or "").casefold().removeprefix("www.")
            for passage in robust
            if passage.host
        }
        if len(robust) >= 2 and len(unique_hosts) >= 1:
            supporting = [
                EvidenceSpan(
                    passage_id=passage.passage_id,
                    url=passage.url,
                    title=passage.title,
                    section=passage.section,
                    text=normalized_text(passage.text[:220]),
                )
                for passage in robust[:2]
            ]
            return replace(
                result,
                verdict="supported",
                confidence=max(result.confidence, 0.62),
                supporting_spans=supporting,
                missing_dimensions=[],
            )
    if (
        result.verdict == "insufficient_evidence"
        and _simple_fact_contract(profile)
        and not claim.needs_freshness
        and set(result.missing_dimensions or ()) <= {"source", "coverage"}
        and not result.contradicting_spans
        and result.confidence >= 0.55
    ):
        robust = sorted(
            [
                passage
                for passage in passages
                if _passage_support_floor(passage) >= 0.35
            ],
            key=lambda passage: (_passage_support_floor(passage), passage.source_score),
            reverse=True,
        )
        selected: list[Passage] = []
        seen_hosts: set[str] = set()
        for passage in robust:
            root = _host_root(passage.host)
            if root and root in seen_hosts:
                continue
            selected.append(passage)
            if root:
                seen_hosts.add(root)
            if len(selected) >= 2:
                break
        has_primary_like_source = any(passage.source_score >= 0.66 for passage in selected)
        if len(selected) >= 2 and len(seen_hosts) >= 2 and has_primary_like_source:
            supporting = [
                EvidenceSpan(
                    passage_id=passage.passage_id,
                    url=passage.url,
                    title=passage.title,
                    section=passage.section,
                    text=_supporting_text(passage),
                )
                for passage in selected[:2]
            ]
            rationale = (result.rationale or "").strip()
            suffix = "Adjusted: simple factual classification is sufficiently supported by multiple robust passages."
            merged_rationale = f"{rationale} {suffix}".strip() if rationale else suffix
            return replace(
                result,
                verdict="supported",
                confidence=max(result.confidence, 0.66),
                supporting_spans=supporting,
                missing_dimensions=[],
                rationale=merged_rationale,
            )
    return result
