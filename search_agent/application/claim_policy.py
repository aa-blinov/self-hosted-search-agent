from __future__ import annotations

from dataclasses import replace

from search_agent import tuning
from search_agent.application import policy_tuning
from search_agent.application.text_heuristics import compact_text, normalized_text
from search_agent.domain.models import Claim, ClaimProfile, EvidenceBundle, EvidenceSpan, Passage, VerificationResult


_LIST_LIKE_DIMENSIONS = {"feature_list", "improvements", "changes", "highlights", "options"}
_RETRIEVAL_DRIVEN_SHAPES = {"product_specs", "overview", "comparison", "news_digest"}
_OPEN_ENDED_STOP_SHAPES = {"overview", "comparison", "news_digest"}
_RELATIONSHIP_CONTRACT_MARKERS = {
    "affiliation",
    "agency",
    "bureau",
    "classification",
    "component",
    "department",
    "entity_status",
    "government",
    "institutional",
    "membership",
    "organizational",
    "parent",
    "status",
}
_RELATIONSHIP_TEXT_CUES = (
    " agency ",
    " bureau ",
    " bureaus ",
    " department ",
    " component ",
    " administration ",
    " ministry ",
    " treasury ",
    " member of ",
    " part of ",
    " within ",
)
_OFFICIAL_HOST_SUFFIXES = (".gov", ".mil", ".gouv")
_OFFICIAL_HOST_EXACT = {"europa.eu", "who.int"}


def claim_profile_lines(claim: Claim) -> list[str]:
    profile = claim.claim_profile
    if profile is None:
        return [
            "answer_shape=fact",
            "primary_source_required=False",
            f"min_independent_sources={policy_tuning.DEFAULT_FACT_MIN_INDEPENDENT_SOURCES}",
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
        return max(policy_tuning.DEFAULT_FACT_MIN_INDEPENDENT_SOURCES, claim.claim_profile.min_independent_sources)
    return policy_tuning.DEFAULT_FACT_MIN_INDEPENDENT_SOURCES


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
    lowered = (host or "").casefold()
    if lowered.startswith("www."):
        lowered = lowered[4:]
    parts = lowered.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return lowered


def _is_official_host(host: str) -> bool:
    lowered = (host or "").casefold()
    if lowered.startswith("www."):
        lowered = lowered[4:]
    if lowered in _OFFICIAL_HOST_EXACT:
        return True
    return any(lowered.endswith(suffix) for suffix in _OFFICIAL_HOST_SUFFIXES)


def _is_relationship_fact_contract(profile: ClaimProfile | None) -> bool:
    if profile is None:
        return False
    if profile.answer_shape != "fact" or not profile.strict_contract or not profile.primary_source_required:
        return False
    contract_text = " ".join(profile.required_dimensions + profile.focus_terms).casefold()
    return any(marker in contract_text for marker in _RELATIONSHIP_CONTRACT_MARKERS)


def _relationship_cue_score(claim: Claim, passage: Passage) -> float:
    haystack = f" {normalized_text(f'{passage.title} {passage.text[:320]}').casefold()} "
    score = 0.0
    if any(cue in haystack for cue in _RELATIONSHIP_TEXT_CUES):
        score += 1.0
    if any(entity.casefold() in haystack for entity in claim.entity_set if entity):
        score += 0.5
    if _is_official_host(passage.host):
        score += 0.5
    return score


def _official_relationship_rescue(
    claim: Claim,
    passages: list[Passage],
    result: VerificationResult,
) -> VerificationResult | None:
    profile = claim.claim_profile
    if result.verdict != "insufficient_evidence" or not _is_relationship_fact_contract(profile):
        return None

    robust = [
        passage
        for passage in passages
        if passage.source_score >= policy_tuning.OFFICIAL_RELATIONSHIP_SOURCE_THRESHOLD
    ]
    if len({_host_root(passage.host) for passage in robust if passage.host}) < policy_tuning.OFFICIAL_RELATIONSHIP_MIN_UNIQUE_HOSTS:
        return None

    official = [passage for passage in robust if _is_official_host(passage.host)]
    if not official:
        return None

    explicit = [
        passage
        for passage in robust
        if _relationship_cue_score(claim, passage) >= policy_tuning.OFFICIAL_RELATIONSHIP_CUE_SCORE_THRESHOLD
    ]
    if not explicit:
        return None

    selected = explicit[: policy_tuning.OFFICIAL_RELATIONSHIP_SUPPORTING_PASSAGE_COUNT]
    supporting = [
        EvidenceSpan(
            passage_id=passage.passage_id,
            url=passage.url,
            title=passage.title,
            section=passage.section,
            text=normalized_text(passage.text[:policy_tuning.OFFICIAL_RELATIONSHIP_SUPPORTING_SPAN_CHARS]),
        )
        for passage in selected
    ]
    rationale = (result.rationale or "").strip()
    suffix = "Adjusted: multiple official passages explicitly support the institutional relationship."
    return replace(
        result,
        verdict="supported",
        confidence=max(result.confidence, policy_tuning.OFFICIAL_RELATIONSHIP_PROMOTED_CONFIDENCE_FLOOR),
        supporting_spans=supporting,
        missing_dimensions=[],
        rationale=f"{rationale} {suffix}".strip() if rationale else suffix,
    )


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
        if (
            verification.confidence >= policy_tuning.SUPPORTED_STOP_CONFIDENCE
            and bundle.independent_source_count >= min_sources
            and not primary_sensitive
        ):
            return True

    if verification.verdict == "insufficient_evidence" and profile is not None and profile.strict_contract:
        if bundle.contract_satisfied and bundle.considered_passages:
            return True
        if (
            exact_detail_guardrail_claim(claim)
            and verification.confidence >= policy_tuning.STRICT_EXACT_DETAIL_STOP_CONFIDENCE
            and bundle.independent_source_count >= policy_tuning.STRICT_EXACT_DETAIL_STOP_MIN_SOURCES
        ):
            return True

    if open_ended and bundle.considered_passages:
        return True

    return iteration >= tuning.AGENT_MAX_CLAIM_ITERATIONS


def post_adjust_verification(claim: Claim, passages: list[Passage], result: VerificationResult) -> VerificationResult:
    rescued = _official_relationship_rescue(claim, passages, result)
    if rescued is not None:
        return rescued
    if result.verdict == "supported" and result.confidence < policy_tuning.SUPPORTED_CONFIDENCE_FLOOR_THRESHOLD:
        return replace(result, confidence=max(result.confidence, policy_tuning.SUPPORTED_CONFIDENCE_FLOOR_VALUE))
    profile = claim.claim_profile
    if result.verdict == "supported" and profile is not None and not profile.strict_contract and is_list_like_contract(profile):
        rationale = (result.rationale or "").strip()
        suffix = "Adjusted: open-ended contract stays claim-level insufficient_evidence until synthesis."
        merged_rationale = f"{rationale} {suffix}".strip() if rationale else suffix
        return replace(
            result,
            verdict="insufficient_evidence",
            confidence=min(result.confidence, policy_tuning.OPEN_ENDED_DEMOTION_CONFIDENCE_CAP),
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
            if (
                passage.utility_score >= policy_tuning.OVERVIEW_ROBUST_PASSAGE_UTILITY_THRESHOLD
                or passage.source_score >= policy_tuning.OVERVIEW_ROBUST_PASSAGE_SOURCE_THRESHOLD
            )
        ]
        unique_hosts = {
            (passage.host or "").casefold().removeprefix("www.")
            for passage in robust
            if passage.host
        }
        if (
            len(robust) >= policy_tuning.OVERVIEW_SUPPORTING_PASSAGE_COUNT
            and len(unique_hosts) >= policy_tuning.OVERVIEW_SUPPORTING_MIN_UNIQUE_HOSTS
        ):
            supporting = [
                EvidenceSpan(
                    passage_id=passage.passage_id,
                    url=passage.url,
                    title=passage.title,
                    section=passage.section,
                    text=normalized_text(passage.text[:policy_tuning.OVERVIEW_SUPPORTING_SPAN_CHARS]),
                )
                for passage in robust[:policy_tuning.OVERVIEW_SUPPORTING_PASSAGE_COUNT]
            ]
            return replace(
                result,
                verdict="supported",
                confidence=max(result.confidence, policy_tuning.OVERVIEW_PROMOTED_CONFIDENCE_FLOOR),
                supporting_spans=supporting,
                missing_dimensions=[],
            )
    return result
