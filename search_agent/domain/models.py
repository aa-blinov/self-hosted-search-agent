from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


DomainType = Literal["official", "academic", "vendor", "major_media", "forum", "unknown"]
Verdict = Literal["supported", "contradicted", "insufficient_evidence"]
RouteMode = Literal["short_path", "targeted_retrieval", "iterative_loop"]
FetchDepth = Literal["shallow", "deep", "snippet_only"]


@dataclass(slots=True)
class QueryClassification:
    query: str
    normalized_query: str
    intent: str
    complexity: str
    needs_freshness: bool
    time_scope: str | None = None
    region_hint: str | None = None
    entity_disambiguation: bool = False


@dataclass(slots=True)
class Claim:
    claim_id: str
    claim_text: str
    priority: int
    needs_freshness: bool
    entity_set: list[str] = field(default_factory=list)
    time_scope: str | None = None
    search_queries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryVariant:
    variant_id: str
    claim_id: str
    query_text: str
    strategy: str
    rationale: str
    source_restriction: str | None = None
    freshness_hint: str | None = None


@dataclass(slots=True)
class SearchSnapshot:
    query: str
    suggestions: list[str]
    results: list["SerpResult"]
    retrieved_at: str
    profile_name: str | None = None
    unresponsive_engines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SerpResult:
    result_id: str
    query_variant_id: str
    title: str
    url: str
    snippet: str
    canonical_url: str
    host: str
    position: int
    score: float | None = None
    engine: str | None = None
    published_at: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass(slots=True)
class SourceAssessment:
    domain_type: DomainType
    source_prior: float
    primary_source_likelihood: float
    freshness_score: float
    seo_spam_risk: float
    entity_match_score: float
    semantic_match_score: float
    source_score: float
    duplicate_of: str | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GatedSerpResult:
    serp: SerpResult
    assessment: SourceAssessment
    matched_variant_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RoutingDecision:
    mode: RouteMode
    certainty: float
    consistency: float
    evidence_sufficiency: float
    rationale: str


@dataclass(slots=True)
class FetchPlan:
    depth: FetchDepth
    url: str
    reason: str
    source_score: float


@dataclass(slots=True)
class FetchedDocument:
    doc_id: str
    url: str
    canonical_url: str
    host: str
    title: str
    author: str | None
    published_at: str | None
    extracted_at: str
    content_hash: str
    content: str
    fetch_depth: FetchDepth
    source_score: float
    meta_description: str | None = None
    headings: list[str] = field(default_factory=list)
    first_paragraphs: list[str] = field(default_factory=list)
    schema_org: dict = field(default_factory=dict)


@dataclass(slots=True)
class Passage:
    passage_id: str
    url: str
    canonical_url: str
    host: str
    title: str
    section: str
    published_at: str | None
    author: str | None
    extracted_at: str
    chunk_id: str
    text: str
    source_score: float = 0.0
    utility_score: float = 0.0


@dataclass(slots=True)
class EvidenceSpan:
    passage_id: str
    url: str
    title: str
    section: str
    text: str


@dataclass(slots=True)
class VerificationResult:
    verdict: Verdict
    confidence: float
    supporting_spans: list[EvidenceSpan] = field(default_factory=list)
    contradicting_spans: list[EvidenceSpan] = field(default_factory=list)
    missing_dimensions: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class EvidenceBundle:
    claim_id: str
    claim_text: str
    supporting_passages: list[Passage] = field(default_factory=list)
    contradicting_passages: list[Passage] = field(default_factory=list)
    considered_passages: list[Passage] = field(default_factory=list)
    independent_source_count: int = 0
    has_primary_source: bool = False
    freshness_ok: bool = True
    verification: VerificationResult | None = None


@dataclass(slots=True)
class ClaimRun:
    claim: Claim
    query_variants: list[QueryVariant] = field(default_factory=list)
    search_snapshots: list[SearchSnapshot] = field(default_factory=list)
    gated_results: list[GatedSerpResult] = field(default_factory=list)
    fetch_plans: list[FetchPlan] = field(default_factory=list)
    fetched_documents: list[FetchedDocument] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    evidence_bundle: EvidenceBundle | None = None
    routing_decision: RoutingDecision | None = None


@dataclass(slots=True)
class AuditTrail:
    run_id: str | None = None
    profile_name: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    latency_ms: int = 0
    estimated_search_cost: float = 0.0
    receipt_path: str | None = None
    query_variants: list[QueryVariant] = field(default_factory=list)
    serp_snapshots: list[SearchSnapshot] = field(default_factory=list)
    selected_urls: list[str] = field(default_factory=list)
    crawl_events: list[dict] = field(default_factory=list)
    passage_ids: list[str] = field(default_factory=list)
    claim_to_passages: dict[str, list[str]] = field(default_factory=dict)
    claim_iterations: dict[str, int] = field(default_factory=dict)
    verification_results: dict[str, VerificationResult] = field(default_factory=dict)
    final_verdicts: dict[str, Verdict] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRunResult:
    user_query: str
    classification: QueryClassification
    claims: list[ClaimRun]
    answer: str
    audit_trail: AuditTrail = field(default_factory=AuditTrail)
