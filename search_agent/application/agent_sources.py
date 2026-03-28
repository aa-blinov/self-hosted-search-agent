from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from search_agent import tuning
from search_agent.application import policy_tuning
from search_agent.application.agent_scoring_shared import (
    _ACADEMIC_HOST_MARKERS,
    _FORUM_HOST_MARKERS,
    _MAJOR_MEDIA_HOST_MARKERS,
    _NON_ENTITY_TOKENS,
    _OFFICIAL_HOST_MARKERS,
    _PRIMARY_SOURCE_CUES,
    _SPAM_CUES,
    _STOPWORDS,
    _VENDOR_HOST_MARKERS,
    _clean_title_key,
    _clamp,
    _compact_text,
    _extract_author,
    _is_iso_date_text,
    _is_year_text,
    _markdown_title,
    _normalized_text,
    _preferred_domain_bonus,
    _product_specs_result_bonus,
    _split_sentences,
    _tokenize,
)
from search_agent.domain.models import (
    Claim,
    DomainType,
    FetchedDocument,
    FetchPlan,
    GatedSerpResult,
    Passage,
    RoutingDecision,
    SourceAssessment,
)
from search_agent.domain.source_priors import lookup_source_prior


def _domain_type(host: str) -> DomainType:
    if any(marker in host for marker in _OFFICIAL_HOST_MARKERS):
        return "official"
    if any(marker in host for marker in _ACADEMIC_HOST_MARKERS):
        return "academic"
    if any(marker in host for marker in _FORUM_HOST_MARKERS):
        return "forum"
    if any(marker in host for marker in _MAJOR_MEDIA_HOST_MARKERS):
        return "major_media"
    if any(marker in host for marker in _VENDOR_HOST_MARKERS):
        return "vendor"
    return "unknown"


def _host_root(host: str) -> str:
    parts = [part for part in host.split(".") if part]
    if len(parts) <= 2:
        return host
    if parts[-2] in {"co", "com", "org", "net"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _entity_host_match_score(claim: Claim, host: str) -> float:
    labels = [
        _compact_text(part)
        for part in host.split(".")
        if part and part.lower() != "www"
    ]
    if not labels:
        return 0.0

    best = 0.0
    label_set = set(labels)
    for entity in claim.entity_set:
        compact_entity = _compact_text(entity)
        if len(compact_entity) >= 4 and compact_entity in label_set:
            best = max(best, 1.0)
        for token in _tokenize(entity):
            compact_token = _compact_text(token)
            if len(compact_token) < 4:
                continue
            if compact_token in _STOPWORDS or compact_token in _NON_ENTITY_TOKENS:
                continue
            if compact_token in label_set:
                best = max(best, policy_tuning.ENTITY_HOST_MATCH_TOKEN_SCORE)
    return best


def _effective_domain_type(claim: Claim, host: str) -> DomainType:
    prior = lookup_source_prior(host)
    if prior.domain_type_override in {"official", "academic", "vendor", "major_media", "forum", "unknown"}:
        return prior.domain_type_override
    base = _domain_type(host)
    if base != "unknown":
        return base
    if _entity_host_match_score(claim, host) >= policy_tuning.EFFECTIVE_DOMAIN_ENTITY_MATCH_THRESHOLD:
        return "official"
    return base


def _title_key(title: str) -> str:
    return _clean_title_key(title)


def _semantic_overlap(query_text: str, candidate_text: str) -> float:
    query_tokens = set(_tokenize(query_text))
    if not query_tokens:
        return 0.0
    candidate_tokens = set(_tokenize(candidate_text))
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def _entity_overlap(entities: list[str], candidate_text: str) -> float:
    if not entities:
        return 0.0
    lowered = candidate_text.casefold()
    compact_candidate = _compact_text(candidate_text)
    hits = 0.0
    for entity in entities:
        if entity.casefold() in lowered:
            hits += 1.0
            continue
        compact_entity = _compact_text(entity)
        if compact_entity and compact_entity in compact_candidate:
            hits += 1.0
            continue
        if len(compact_entity) >= 5 and compact_entity[:5] in compact_candidate:
            hits += policy_tuning.ENTITY_OVERLAP_PARTIAL_MATCH_SCORE
    return _clamp(hits / len(entities))


def _time_scope_alignment(claim: Claim, result) -> float:
    if not claim.time_scope:
        return 0.0
    scope = claim.time_scope.casefold()
    haystack = f"{result.title} {result.snippet} {result.url}".casefold()
    if scope in haystack:
        return 1.0
    if result.published_at and result.published_at.startswith(claim.time_scope):
        return 1.0
    if _is_iso_date_text(claim.time_scope) and result.published_at and result.published_at[:7] == claim.time_scope[:7]:
        return policy_tuning.TIME_SCOPE_MONTH_MATCH_SCORE
    if _is_year_text(claim.time_scope) and result.published_at and result.published_at.startswith(claim.time_scope):
        return policy_tuning.TIME_SCOPE_YEAR_MATCH_SCORE
    return 0.0


def _freshness_score(claim: Claim, result) -> float:
    if not claim.needs_freshness and not result.published_at:
        return policy_tuning.FRESHNESS_NEUTRAL_SCORE
    if not result.published_at:
        return (
            policy_tuning.FRESHNESS_MISSING_REQUIRED_SCORE
            if claim.needs_freshness
            else policy_tuning.FRESHNESS_MISSING_OPTIONAL_SCORE
        )
    try:
        published = datetime.fromisoformat(result.published_at.replace("Z", "+00:00"))
        age_days = max(0, (datetime.now(UTC) - published.astimezone(UTC)).days)
    except ValueError:
        return (
            policy_tuning.FRESHNESS_MISSING_REQUIRED_SCORE
            if claim.needs_freshness
            else policy_tuning.FRESHNESS_MISSING_OPTIONAL_SCORE
        )
    window = (
        policy_tuning.FRESHNESS_REQUIRED_WINDOW_DAYS
        if claim.needs_freshness
        else policy_tuning.FRESHNESS_OPTIONAL_WINDOW_DAYS
    )
    return _clamp(1 - (age_days / max(window, 1)))


def _spam_risk(result) -> float:
    host = result.host.casefold()
    text = f"{result.title} {result.snippet}".casefold()
    prior = lookup_source_prior(host)
    risk = 0.0
    if host.endswith(".top") or host.endswith(".best"):
        risk += policy_tuning.SPAM_SUFFIX_PENALTY
    if any(cue in text for cue in _SPAM_CUES):
        risk += policy_tuning.SPAM_CUE_PENALTY
    if text.count("|") >= 2 or text.count(" - ") >= 3:
        risk += policy_tuning.SPAM_TITLE_SEPARATOR_PENALTY
    if len(_tokenize(result.title)) > 14:
        risk += policy_tuning.SPAM_LONG_TITLE_PENALTY
    if _domain_type(host) in {"official", "academic"}:
        risk -= policy_tuning.SPAM_AUTHORITATIVE_DOMAIN_DISCOUNT
    risk += prior.spam_penalty
    return _clamp(risk)


def _primary_source_likelihood(claim: Claim, result, domain_type: DomainType) -> float:
    prior = lookup_source_prior(result.host)
    base = policy_tuning.PRIMARY_SOURCE_BASE_BY_DOMAIN_TYPE[domain_type]
    text = f"{result.title} {result.snippet} {result.url}".casefold()
    if any(cue in text for cue in _PRIMARY_SOURCE_CUES):
        base += policy_tuning.PRIMARY_SOURCE_CUE_BOOST
    base += policy_tuning.PRIMARY_SOURCE_ENTITY_HOST_MATCH_WEIGHT * _entity_host_match_score(claim, result.host)
    if any(path_cue in text for path_cue in ("/announcement/", "/press", "/release", "/releases/", "/downloads/release/", "/whatsnew/")):
        base += policy_tuning.PRIMARY_SOURCE_PATH_CUE_BOOST
    if domain_type == "forum":
        base -= policy_tuning.PRIMARY_SOURCE_FORUM_PENALTY
    base += prior.primary_boost
    return _clamp(base)


def gate_serp_results(
    claim: Claim,
    snapshots: list,
    limit: int,
) -> list[GatedSerpResult]:
    merged: dict[str, GatedSerpResult] = {}

    for snapshot in snapshots:
        for result in snapshot.results:
            domain_type = _effective_domain_type(claim, result.host)
            prior = lookup_source_prior(result.host)
            domain_prior = policy_tuning.SERP_DOMAIN_PRIOR_BY_TYPE[domain_type]
            semantic_match = _semantic_overlap(claim.claim_text, f"{result.title} {result.snippet}")
            entity_match = _entity_overlap(claim.entity_set, f"{result.title} {result.snippet}")
            host_entity_match = _entity_host_match_score(claim, result.host)
            freshness = _freshness_score(claim, result)
            time_alignment = _time_scope_alignment(claim, result)
            spam = _spam_risk(result)
            primary = _primary_source_likelihood(claim, result, domain_type)
            preferred_domain_bonus = _preferred_domain_bonus(claim, domain_type)
            product_bonus = _product_specs_result_bonus(claim, result.title, result.snippet, result.url)
            source_score = _clamp(
                policy_tuning.SERP_DOMAIN_PRIOR_WEIGHT * domain_prior
                + policy_tuning.SERP_PRIMARY_WEIGHT * primary
                + policy_tuning.SERP_FRESHNESS_WEIGHT * freshness
                + policy_tuning.SERP_ENTITY_MATCH_WEIGHT * entity_match
                + policy_tuning.SERP_SEMANTIC_MATCH_WEIGHT * semantic_match
                + policy_tuning.SERP_HOST_ENTITY_WEIGHT * host_entity_match
                + policy_tuning.SERP_TIME_ALIGNMENT_WEIGHT * time_alignment
                + preferred_domain_bonus
                + product_bonus
                + prior.source_prior
                - policy_tuning.SERP_SPAM_PENALTY_WEIGHT * spam
            )
            reasons = [
                f"domain={domain_type}",
                f"prior={prior.source_prior:.2f}",
                f"primary={primary:.2f}",
                f"host_entity={host_entity_match:.2f}",
                f"freshness={freshness:.2f}",
                f"time_alignment={time_alignment:.2f}",
                f"entity_match={entity_match:.2f}",
                f"spam={spam:.2f}",
                f"preferred_bonus={preferred_domain_bonus:.2f}",
                f"shape_bonus={product_bonus:.2f}",
            ]
            reasons.extend(prior.labels)
            gated = GatedSerpResult(
                serp=result,
                assessment=SourceAssessment(
                    domain_type=domain_type,
                    source_prior=prior.source_prior,
                    primary_source_likelihood=primary,
                    freshness_score=freshness,
                    seo_spam_risk=spam,
                    entity_match_score=entity_match,
                    semantic_match_score=semantic_match,
                    source_score=source_score,
                    reasons=reasons,
                ),
                matched_variant_ids=[result.query_variant_id],
            )

            canonical_key = result.canonical_url
            title_key = f"{_host_root(result.host)}|{_title_key(result.title)}"
            existing = merged.get(canonical_key) or merged.get(title_key)
            if existing is None:
                merged[canonical_key] = gated
                continue

            if result.query_variant_id not in existing.matched_variant_ids:
                existing.matched_variant_ids.append(result.query_variant_id)
            if gated.assessment.source_score > existing.assessment.source_score:
                gated.assessment.duplicate_of = existing.serp.result_id
                merged[canonical_key] = gated
            else:
                existing.assessment.duplicate_of = result.result_id

    gated_results = sorted(
        merged.values(),
        key=lambda item: (
            item.assessment.source_score,
            item.assessment.primary_source_likelihood,
            -item.assessment.seo_spam_risk,
        ),
        reverse=True,
    )
    return gated_results[:limit]


def _make_document(
    gated: GatedSerpResult,
    content: str,
    fetch_depth: str,
    *,
    title: str | None = None,
    author: str | None = None,
    published_at: str | None = None,
    meta_description: str | None = None,
    headings: list[str] | None = None,
    first_paragraphs: list[str] | None = None,
    schema_org: dict | None = None,
) -> FetchedDocument:
    extracted_at = datetime.now(UTC).isoformat()
    title = title or _markdown_title(content) or gated.serp.title or gated.serp.url
    normalized = _normalized_text(content)
    content_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return FetchedDocument(
        doc_id=f"doc-{content_hash[:12]}",
        url=gated.serp.url,
        canonical_url=gated.serp.canonical_url,
        host=gated.serp.host,
        title=title,
        author=author or _extract_author(content),
        published_at=published_at or gated.serp.published_at,
        extracted_at=extracted_at,
        content_hash=content_hash,
        content=normalized,
        fetch_depth=fetch_depth,
        source_score=gated.assessment.source_score,
        meta_description=meta_description,
        headings=headings or [],
        first_paragraphs=first_paragraphs or [],
        schema_org=schema_org or {},
    )


def _select_fetch_candidates(gated_results: list[GatedSerpResult], limit: int) -> list[GatedSerpResult]:
    selected: list[GatedSerpResult] = []
    seen_hosts: set[str] = set()

    for candidate in gated_results:
        root = _host_root(candidate.serp.host)
        if root in seen_hosts:
            continue
        selected.append(candidate)
        seen_hosts.add(root)
        if len(selected) >= limit:
            return selected

    for candidate in gated_results:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            return selected
    return selected


def _routing_limits(profile, decision: RoutingDecision, iteration: int = 1) -> tuple[int, int]:
    if iteration == 1:
        shallow_limit = {
            "short_path": tuning.SHALLOW_FETCH_SHORT_FAST_LIMIT,
            "targeted_retrieval": tuning.SHALLOW_FETCH_TARGETED_FAST_LIMIT,
            "iterative_loop": tuning.SHALLOW_FETCH_ITERATIVE_FAST_LIMIT,
        }[decision.mode]
    else:
        shallow_limit = {
            "short_path": tuning.SHALLOW_FETCH_SHORT_LIMIT,
            "targeted_retrieval": tuning.SHALLOW_FETCH_TARGETED_LIMIT,
            "iterative_loop": tuning.SHALLOW_FETCH_ITERATIVE_LIMIT,
        }[decision.mode]
    deep_limit = {
        "short_path": tuning.DEEP_FETCH_SHORT_LIMIT,
        "targeted_retrieval": tuning.DEEP_FETCH_TARGETED_LIMIT,
        "iterative_loop": tuning.DEEP_FETCH_ITERATIVE_LIMIT,
    }[decision.mode]
    if profile.fetch_top_n == 0:
        deep_limit = 0
    else:
        deep_limit = min(deep_limit, max(profile.fetch_top_n, tuning.AGENT_FETCH_TOP_N))
    return shallow_limit, deep_limit


def _verification_source_bonus(claim: Claim, *, host: str, title: str, url: str) -> float:
    domain_type = _effective_domain_type(claim, host)
    prior = lookup_source_prior(host)
    bonus = policy_tuning.VERIFICATION_BONUS_BY_DOMAIN_TYPE[domain_type]
    lowered = f"{title} {url}".casefold()
    if any(cue in lowered for cue in ("announcement", "press", "release", "released", "downloads/release", "whatsnew", "/blog/")):
        bonus += policy_tuning.VERIFICATION_RELEASE_CUE_BOOST
    if any(cue in lowered for cue in ("hacker news", "reddit", "forum", "comment")):
        bonus -= policy_tuning.VERIFICATION_FORUM_CUE_PENALTY
    bonus += prior.verification_bonus
    return bonus


def score_shallow_document_for_claim(claim: Claim, document: FetchedDocument) -> float:
    from search_agent.application.agent_passage_scoring import _dimension_coverage_score

    overview = " ".join(
        [
            document.title,
            document.meta_description or "",
            " ".join(document.headings[:3]),
            " ".join(document.first_paragraphs[:2]),
        ]
    )
    source_bonus = _verification_source_bonus(
        claim,
        host=document.host,
        title=document.title,
        url=document.url,
    )
    weights = policy_tuning.SHALLOW_DOCUMENT_SCORE_WEIGHTS
    return _clamp(
        weights["semantic_overlap"] * _semantic_overlap(claim.claim_text, overview)
        + weights["entity_overlap"] * _entity_overlap(claim.entity_set, overview)
        + weights["dimension_coverage"] * _dimension_coverage_score(claim, overview)
        + weights["source_score"] * document.source_score
        + weights["source_bonus"] * max(source_bonus, 0.0)
    )


def _make_shallow_document(candidate: GatedSerpResult, payload: dict) -> FetchedDocument:
    summary = payload.get("content") or candidate.serp.snippet or candidate.serp.title
    return _make_document(
        candidate,
        summary,
        "shallow",
        title=payload.get("title") or candidate.serp.title,
        author=payload.get("author"),
        published_at=payload.get("published_at"),
        meta_description=payload.get("meta_description"),
        headings=payload.get("headings") or [],
        first_paragraphs=payload.get("first_paragraphs") or [],
        schema_org=payload.get("schema_org") or {},
    )


def build_snippet_passages(gated_results: list[GatedSerpResult]) -> list[Passage]:
    now = datetime.now(UTC).isoformat()
    passages: list[Passage] = []
    for i, gated in enumerate(gated_results):
        snippet = (gated.serp.snippet or "").strip()
        if len(snippet) < 20:
            continue
        passages.append(
            Passage(
                passage_id=f"snip-{i}",
                url=gated.serp.url,
                canonical_url=gated.serp.canonical_url,
                host=gated.serp.host,
                title=gated.serp.title or "",
                section="snippet",
                published_at=gated.serp.published_at,
                author=None,
                extracted_at=now,
                chunk_id=f"snip-{i}-0",
                text=snippet,
                source_score=gated.assessment.source_score,
                utility_score=0.0,
            )
        )
    return passages


def fetch_claim_documents(
    claim: Claim,
    gated_results: list[GatedSerpResult],
    profile,
    routing_decision: RoutingDecision,
    seen_urls: set[str] | None = None,
    log=None,
    iteration: int = 1,
    page_cache: dict[str, dict] | None = None,
    page_cache_lock=None,
    intent: str = "factual",
) -> tuple[list[FetchPlan], list[FetchedDocument]]:
    log = log or (lambda msg: None)
    from search_agent.infrastructure.extractor import fetch_and_extract_many, shallow_fetch_many

    seen_urls = seen_urls or set()
    shallow_limit, deep_limit = _routing_limits(profile, routing_decision, iteration)
    selected = _select_fetch_candidates(
        [candidate for candidate in gated_results if candidate.serp.url not in seen_urls],
        min(len(gated_results), shallow_limit),
    )

    plans = [
        FetchPlan(
            depth="shallow",
            url=candidate.serp.url,
            reason=f"Phase 2 shallow fetch ({routing_decision.mode}).",
            source_score=candidate.assessment.source_score,
        )
        for candidate in selected
    ]

    shallow_documents: list[FetchedDocument] = []
    if selected:
        for candidate, payload in zip(
            selected,
            shallow_fetch_many(
                [c.serp.url for c in selected],
                log=log,
                page_cache=page_cache,
                page_cache_lock=page_cache_lock,
                intent=intent,
            ),
        ):
            if payload:
                shallow_documents.append(_make_shallow_document(candidate, payload))
            elif candidate.serp.snippet:
                shallow_documents.append(
                    _make_document(
                        candidate,
                        candidate.serp.snippet,
                        "snippet_only",
                        title=candidate.serp.title,
                    )
                )

    shallow_ranked = sorted(
        shallow_documents,
        key=lambda document: score_shallow_document_for_claim(claim, document),
        reverse=True,
    )

    deep_candidates: list[tuple[GatedSerpResult, FetchedDocument]] = []
    gated_by_url = {candidate.serp.url: candidate for candidate in selected}
    for document in shallow_ranked:
        candidate = gated_by_url.get(document.url)
        if candidate is None:
            continue
        deep_candidates.append((candidate, document))
        if len(deep_candidates) >= deep_limit:
            break

    if deep_limit > 0 and shallow_ranked:
        source_priority_ranked = sorted(
            shallow_ranked,
            key=lambda document: (
                _verification_source_bonus(
                    claim,
                    host=document.host,
                    title=document.title,
                    url=document.url,
                ),
                document.source_score,
                score_shallow_document_for_claim(claim, document),
            ),
            reverse=True,
        )
        preferred_document = source_priority_ranked[0]
        preferred_candidate = gated_by_url.get(preferred_document.url)
        selected_urls = {document.url for _, document in deep_candidates}
        preferred_bonus = _verification_source_bonus(
            claim,
            host=preferred_document.host,
            title=preferred_document.title,
            url=preferred_document.url,
        )
        if (
            preferred_candidate is not None
            and preferred_document.url not in selected_urls
            and preferred_bonus > policy_tuning.DEEP_FETCH_PREFERRED_BONUS_THRESHOLD
        ):
            if len(deep_candidates) < deep_limit:
                deep_candidates.append((preferred_candidate, preferred_document))
            elif deep_candidates:
                deep_candidates[-1] = (preferred_candidate, preferred_document)

    deep_documents: list[FetchedDocument] = []
    for candidate, _ in deep_candidates:
        plans.append(
            FetchPlan(
                depth="deep",
                url=candidate.serp.url,
                reason="Selective deep fetch after shallow rerank.",
                source_score=candidate.assessment.source_score,
            )
        )
    if deep_candidates:
        deep_urls = [candidate.serp.url for candidate, _ in deep_candidates]
        deep_contents = fetch_and_extract_many(deep_urls, log=log)
        for (candidate, shallow_document), content in zip(deep_candidates, deep_contents):
            if content:
                deep_documents.append(
                    _make_document(
                        candidate,
                        content,
                        "deep",
                        title=shallow_document.title,
                        author=shallow_document.author,
                        published_at=shallow_document.published_at,
                        meta_description=shallow_document.meta_description,
                        headings=shallow_document.headings,
                        first_paragraphs=shallow_document.first_paragraphs,
                        schema_org=shallow_document.schema_org,
                    )
                )

    documents = shallow_ranked + deep_documents
    if not deep_documents and selected:
        documents.extend(
            _make_document(
                gated_by_url.get(document.url, selected[0]),
                document.content,
                "snippet_only",
                title=document.title,
                author=document.author,
                published_at=document.published_at,
                meta_description=document.meta_description,
                headings=document.headings,
                first_paragraphs=document.first_paragraphs,
                schema_org=document.schema_org,
            )
            for document in shallow_ranked[:tuning.AGENT_SNIPPET_FALLBACK_DOCS]
            if document.fetch_depth == "shallow"
        )

    return plans, documents


def _split_into_passages(document: FetchedDocument) -> list[Passage]:
    if not document.content:
        return []

    passages: list[Passage] = []
    current_section = "Intro"
    section_index = 0
    chunk_index = 0
    buffer: list[str] = []

    def flush() -> None:
        nonlocal chunk_index
        text = _normalized_text(" ".join(buffer))
        buffer.clear()
        if len(text) < 60:
            return
        chunks = _split_sentences(text)
        running = ""
        for piece in chunks:
            piece = _normalized_text(piece)
            if not piece:
                continue
            candidate = f"{running} {piece}".strip()
            if len(candidate) <= 420:
                running = candidate
                continue
            if running:
                passage_id = f"{document.doc_id}:{section_index}:{chunk_index}"
                passages.append(
                    Passage(
                        passage_id=passage_id,
                        url=document.url,
                        canonical_url=document.canonical_url,
                        host=document.host,
                        title=document.title,
                        section=current_section,
                        published_at=document.published_at,
                        author=document.author,
                        extracted_at=document.extracted_at,
                        chunk_id=passage_id,
                        text=running,
                        source_score=document.source_score,
                    )
                )
                chunk_index += 1
            running = piece
        if running:
            passage_id = f"{document.doc_id}:{section_index}:{chunk_index}"
            passages.append(
                Passage(
                    passage_id=passage_id,
                    url=document.url,
                    canonical_url=document.canonical_url,
                    host=document.host,
                    title=document.title,
                    section=current_section,
                    published_at=document.published_at,
                    author=document.author,
                    extracted_at=document.extracted_at,
                    chunk_id=passage_id,
                    text=running,
                    source_score=document.source_score,
                )
            )
            chunk_index += 1

    for raw_line in document.content.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            flush()
            current_section = _normalized_text(line.lstrip("#").strip()) or current_section
            section_index += 1
            continue
        buffer.append(line)
    flush()

    if passages:
        return passages

    passage_id = f"{document.doc_id}:0:0"
    return [
        Passage(
            passage_id=passage_id,
            url=document.url,
            canonical_url=document.canonical_url,
            host=document.host,
            title=document.title,
            section=current_section,
            published_at=document.published_at,
            author=document.author,
            extracted_at=document.extracted_at,
            chunk_id=passage_id,
            text=document.content[:420],
            source_score=document.source_score,
        )
    ]


def _documents_for_passage_extraction(documents: list[FetchedDocument]) -> list[FetchedDocument]:
    deep_documents = [document for document in documents if document.fetch_depth == "deep"]
    if deep_documents:
        selected = list(deep_documents)
        deep_hosts = {_host_root(document.host) for document in deep_documents}
        for document in sorted(
            [document for document in documents if document.fetch_depth in {"shallow", "snippet_only"}],
            key=lambda item: item.source_score,
            reverse=True,
        ):
            root = _host_root(document.host)
            if root in deep_hosts:
                continue
            selected.append(document)
            deep_hosts.add(root)
            if len(selected) >= len(deep_documents) + 2:
                break
        return selected
    snippet_documents = [document for document in documents if document.fetch_depth == "snippet_only"]
    if snippet_documents:
        return snippet_documents
    return [document for document in documents if document.fetch_depth == "shallow"]
