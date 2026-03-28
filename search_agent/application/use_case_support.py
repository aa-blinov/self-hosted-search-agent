from __future__ import annotations

from dataclasses import replace
from urllib.parse import urlparse

from search_agent.application import policy_tuning
from search_agent.domain.models import AuditTrail, ClaimRun, GatedSerpResult

_OPEN_ENDED_ANSWER_SHAPES = {"product_specs", "overview", "comparison", "news_digest"}


def _passage_domain_key(passage) -> str:
    host = (getattr(passage, "host", None) or "").casefold()
    if not host:
        host = urlparse(getattr(passage, "url", "") or "").netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    return host


def select_synthesis_passages(passages: list, *, intent: str, limit: int) -> list:
    if limit <= 0:
        return []
    max_per_url = (
        policy_tuning.NEWS_DIGEST_REPEAT_PASSAGES_PER_URL
        if intent == "news_digest"
        else policy_tuning.DEFAULT_REPEAT_PASSAGES_PER_URL
    )
    max_per_domain = (
        policy_tuning.NEWS_DIGEST_REPEAT_PASSAGES_PER_DOMAIN
        if intent == "news_digest"
        else policy_tuning.DEFAULT_REPEAT_PASSAGES_PER_DOMAIN
    )
    primary_url_limit = (
        policy_tuning.NEWS_DIGEST_PRIMARY_PASSAGES_PER_URL
        if intent == "news_digest"
        else policy_tuning.DEFAULT_PRIMARY_PASSAGES_PER_URL
    )
    primary_domain_limit = (
        policy_tuning.NEWS_DIGEST_PRIMARY_PASSAGES_PER_DOMAIN
        if intent == "news_digest"
        else policy_tuning.DEFAULT_PRIMARY_PASSAGES_PER_DOMAIN
    )

    selected: list = []
    url_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}

    for passage in passages:
        url = getattr(passage, "url", "") or getattr(passage, "canonical_url", "")
        domain = _passage_domain_key(passage) or url
        if url and url_counts.get(url, 0) >= primary_url_limit:
            continue
        if domain and domain_counts.get(domain, 0) >= primary_domain_limit:
            continue
        if url:
            url_counts[url] = primary_url_limit
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        selected.append(passage)
        if len(selected) >= limit:
            return selected

    for passage in passages:
        url = getattr(passage, "url", "") or getattr(passage, "canonical_url", "")
        domain = _passage_domain_key(passage) or url
        if url and url_counts.get(url, 0) >= max_per_url:
            continue
        if domain and domain_counts.get(domain, 0) >= max_per_domain:
            continue
        if url:
            url_counts[url] = url_counts.get(url, 0) + 1
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        selected.append(passage)
        if len(selected) >= limit:
            break
    return selected


def claim_run_allows_synthesis(claim_run: ClaimRun) -> bool:
    bundle = claim_run.evidence_bundle
    if bundle is None or bundle.verification is None:
        return False
    if bundle.verification.verdict != "supported":
        profile = claim_run.claim.claim_profile
        if profile is None:
            return False
        if profile.answer_shape == "news_digest":
            fetched_urls = {
                getattr(document, "canonical_url", None) or getattr(document, "url", None)
                for document in claim_run.fetched_documents
                if getattr(document, "canonical_url", None) or getattr(document, "url", None)
            }
            return bool(
                bundle.considered_passages
                and (
                    bundle.independent_source_count >= policy_tuning.NEWS_DIGEST_MIN_SYNTHESIS_SOURCES
                    or len(fetched_urls) >= policy_tuning.NEWS_DIGEST_MIN_FETCHED_URLS
                )
            )
        if profile.strict_contract:
            return bundle.contract_satisfied and bool(bundle.considered_passages)
        return bool(profile.allow_synthesis_without_primary and bundle.considered_passages)
    profile = claim_run.claim.claim_profile
    if profile is None:
        return True
    if profile.strict_contract:
        return bundle.contract_satisfied
    if profile.primary_source_required and not profile.allow_synthesis_without_primary and not bundle.has_primary_source:
        return False
    return True


def reconcile_classification_with_claims(classification, claims: list) -> object:
    if not claims:
        return classification
    answer_shapes = {
        claim.claim_profile.answer_shape
        for claim in claims
        if getattr(claim, "claim_profile", None) is not None
    }
    intent = classification.intent
    if "news_digest" in answer_shapes:
        intent = "news_digest"
    elif answer_shapes & _OPEN_ENDED_ANSWER_SHAPES:
        intent = "synthesis"
    complexity = "multi_hop" if len(claims) > 1 else classification.complexity
    if intent == classification.intent and complexity == classification.complexity:
        return classification
    return replace(classification, intent=intent, complexity=complexity)


def merge_gated_results(existing: list[GatedSerpResult], new_results: list[GatedSerpResult]) -> list[GatedSerpResult]:
    by_url: dict[str, GatedSerpResult] = {}
    for result in existing + new_results:
        key = result.serp.canonical_url or result.serp.url
        current = by_url.get(key)
        if current is None:
            by_url[key] = result
            continue
        current_rank = (
            current.assessment.primary_source_likelihood,
            current.assessment.source_score,
        )
        next_rank = (
            result.assessment.primary_source_likelihood,
            result.assessment.source_score,
        )
        if next_rank > current_rank:
            by_url[key] = result
    merged = list(by_url.values())
    merged.sort(
        key=lambda result: (
            result.assessment.primary_source_likelihood,
            result.assessment.source_score,
        ),
        reverse=True,
    )
    return merged


def extend_audit(audit: AuditTrail, claim_run: ClaimRun, iterations_used: int) -> None:
    claim = claim_run.claim
    bundle = claim_run.evidence_bundle
    audit.query_variants.extend(claim_run.query_variants)
    audit.serp_snapshots.extend(claim_run.search_snapshots)
    audit.selected_urls.extend([result.serp.url for result in claim_run.gated_results])
    audit.crawl_events.extend(
        {
            "claim_id": claim.claim_id,
            "url": document.url,
            "fetched_at": document.extracted_at,
            "content_hash": document.content_hash,
            "fetch_depth": document.fetch_depth,
        }
        for document in claim_run.fetched_documents
    )
    audit.passage_ids.extend([passage.passage_id for passage in claim_run.passages])
    audit.claim_to_passages[claim.claim_id] = [passage.passage_id for passage in claim_run.passages]
    audit.claim_iterations[claim.claim_id] = iterations_used
    if bundle and bundle.verification:
        audit.verification_results[claim.claim_id] = bundle.verification
        audit.final_verdicts[claim.claim_id] = bundle.verification.verdict
