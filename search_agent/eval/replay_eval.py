from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

from search_agent.application.step_library import AgentStepLibrary
from search_agent.application.use_cases import SearchAgentUseCase
from search_agent.config.profiles import get_profile
from search_agent.domain.models import (
    AgentRunResult,
    Claim,
    ClaimProfile,
    EvidenceSpan,
    FetchedDocument,
    FetchPlan,
    GatedSerpResult,
    Passage,
    QueryClassification,
    SearchSnapshot,
    SerpResult,
    VerificationResult,
)
from search_agent.evaluation import EvaluationCase, ExpectedClaim, score_reports


@dataclass(slots=True)
class ReplayFixture:
    classification: QueryClassification
    claims: list[Claim]
    search: dict[str, list[SearchSnapshot]]
    fetch: dict[str, dict[int, tuple[list[FetchPlan], list[FetchedDocument]]]]
    verify: dict[str, dict[int, VerificationResult]]
    refine: dict[str, dict[int, list[str]]]
    synthesize: dict[str, str]


@dataclass(slots=True)
class ReplayEvaluationCase:
    evaluation: EvaluationCase
    fixture: ReplayFixture


def _normalize_query_key(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _load_expected_claims(items: list[dict]) -> list[ExpectedClaim]:
    return [
        ExpectedClaim(
            match=item["match"],
            expected_verdict=item["expected_verdict"],
            requires_primary_source=bool(item.get("requires_primary_source", False)),
            expected_routes=(
                [item["expected_route"]]
                if isinstance(item.get("expected_route"), str)
                else [str(route) for route in item.get("expected_route", [])]
            ),
            min_independent_sources=item.get("min_independent_sources"),
        )
        for item in items
    ]


def _load_claim_profile(raw: dict | None) -> ClaimProfile | None:
    if raw is None:
        return None
    return ClaimProfile(
        answer_shape=raw["answer_shape"],
        primary_source_required=bool(raw.get("primary_source_required", False)),
        min_independent_sources=int(raw.get("min_independent_sources", 1)),
        preferred_domain_types=list(raw.get("preferred_domain_types") or []),
        required_dimensions=list(raw.get("required_dimensions") or []),
        focus_terms=list(raw.get("focus_terms") or []),
        allow_synthesis_without_primary=bool(raw.get("allow_synthesis_without_primary", True)),
        strict_contract=bool(raw.get("strict_contract", False)),
    )


def _load_claim(raw: dict) -> Claim:
    return Claim(
        claim_id=raw["claim_id"],
        claim_text=raw["claim_text"],
        priority=int(raw.get("priority", 1)),
        needs_freshness=bool(raw.get("needs_freshness", False)),
        entity_set=list(raw.get("entity_set") or []),
        time_scope=raw.get("time_scope"),
        search_queries=list(raw.get("search_queries") or []),
        claim_profile=_load_claim_profile(raw.get("claim_profile")),
    )


def _load_query_classification(raw: dict) -> QueryClassification:
    return QueryClassification(
        query=raw["query"],
        normalized_query=raw["normalized_query"],
        intent=raw["intent"],
        complexity=raw["complexity"],
        needs_freshness=bool(raw.get("needs_freshness", False)),
        time_scope=raw.get("time_scope"),
        region_hint=raw.get("region_hint"),
        entity_disambiguation=bool(raw.get("entity_disambiguation", False)),
    )


def _load_evidence_span(raw: dict) -> EvidenceSpan:
    return EvidenceSpan(
        passage_id=raw["passage_id"],
        url=raw["url"],
        title=raw.get("title", ""),
        section=raw.get("section", "Intro"),
        text=raw.get("text", ""),
    )


def _load_verification_result(raw: dict) -> VerificationResult:
    return VerificationResult(
        verdict=raw["verdict"],
        confidence=float(raw["confidence"]),
        supporting_spans=[_load_evidence_span(item) for item in (raw.get("supporting_spans") or [])],
        contradicting_spans=[_load_evidence_span(item) for item in (raw.get("contradicting_spans") or [])],
        missing_dimensions=list(raw.get("missing_dimensions") or []),
        rationale=raw.get("rationale", ""),
    )


def _load_serp_result(raw: dict) -> SerpResult:
    return SerpResult(
        result_id=raw["result_id"],
        query_variant_id=raw.get("query_variant_id", ""),
        title=raw.get("title", ""),
        url=raw["url"],
        snippet=raw.get("snippet", ""),
        canonical_url=raw.get("canonical_url", raw["url"]),
        host=raw.get("host", ""),
        position=int(raw.get("position", 0)),
        score=raw.get("score"),
        engine=raw.get("engine"),
        published_at=raw.get("published_at"),
        raw=dict(raw.get("raw") or {}),
    )


def _load_search_snapshot(raw: dict) -> SearchSnapshot:
    return SearchSnapshot(
        query=raw["query"],
        suggestions=list(raw.get("suggestions") or []),
        results=[_load_serp_result(item) for item in (raw.get("results") or [])],
        retrieved_at=raw.get("retrieved_at", ""),
        profile_name=raw.get("profile_name"),
        unresponsive_engines=list(raw.get("unresponsive_engines") or []),
    )


def _load_fetch_plan(raw: dict) -> FetchPlan:
    return FetchPlan(
        depth=raw["depth"],
        url=raw["url"],
        reason=raw.get("reason", ""),
        source_score=float(raw.get("source_score", 0.0)),
    )


def _load_fetched_document(raw: dict) -> FetchedDocument:
    return FetchedDocument(
        doc_id=raw["doc_id"],
        url=raw["url"],
        canonical_url=raw.get("canonical_url", raw["url"]),
        host=raw.get("host", ""),
        title=raw.get("title", ""),
        author=raw.get("author"),
        published_at=raw.get("published_at"),
        extracted_at=raw["extracted_at"],
        content_hash=raw["content_hash"],
        content=raw.get("content", ""),
        fetch_depth=raw["fetch_depth"],
        source_score=float(raw.get("source_score", 0.0)),
        meta_description=raw.get("meta_description"),
        headings=list(raw.get("headings") or []),
        first_paragraphs=list(raw.get("first_paragraphs") or []),
        schema_org=dict(raw.get("schema_org") or {}),
    )


def _load_replay_fixture(raw: dict) -> ReplayFixture:
    search = {
        _normalize_query_key(query): [_load_search_snapshot(item) for item in snapshots]
        for query, snapshots in (raw.get("search") or {}).items()
    }
    fetch: dict[str, dict[int, tuple[list[FetchPlan], list[FetchedDocument]]]] = {}
    for claim_id, iterations in (raw.get("fetch") or {}).items():
        fetch[claim_id] = {}
        for iteration, payload in iterations.items():
            fetch[claim_id][int(iteration)] = (
                [_load_fetch_plan(item) for item in (payload.get("plans") or [])],
                [_load_fetched_document(item) for item in (payload.get("documents") or [])],
            )
    verify: dict[str, dict[int, VerificationResult]] = {}
    for claim_id, iterations in (raw.get("verify") or {}).items():
        verify[claim_id] = {
            int(iteration): _load_verification_result(payload)
            for iteration, payload in iterations.items()
        }
    refine: dict[str, dict[int, list[str]]] = {}
    for claim_id, iterations in (raw.get("refine") or {}).items():
        refine[claim_id] = {
            int(iteration): list(payload or [])
            for iteration, payload in iterations.items()
        }
    synthesize_raw = raw.get("synthesize")
    if isinstance(synthesize_raw, str):
        synthesize = {"__default__": synthesize_raw}
    else:
        synthesize = {
            str(key): str(value)
            for key, value in (synthesize_raw or {}).items()
        }
    return ReplayFixture(
        classification=_load_query_classification(raw["classification"]),
        claims=[_load_claim(item) for item in (raw.get("claims") or [])],
        search=search,
        fetch=fetch,
        verify=verify,
        refine=refine,
        synthesize=synthesize,
    )


def load_replay_cases(path: str | Path) -> list[ReplayEvaluationCase]:
    source = Path(path)
    content = source.read_text(encoding="utf-8-sig").strip()
    if not content:
        return []
    if content.startswith("["):
        items = json.loads(content)
    else:
        items = [json.loads(line) for line in content.splitlines() if line.strip()]

    cases: list[ReplayEvaluationCase] = []
    for item in items:
        replay_raw = item.get("replay")
        fixture_path = item.get("fixture_path")
        if replay_raw is None and fixture_path:
            fixture_file = (source.parent / fixture_path).resolve()
            replay_raw = json.loads(fixture_file.read_text(encoding="utf-8"))
        if replay_raw is None:
            raise KeyError(f"Replay case {item.get('case_id')} is missing replay or fixture_path")
        evaluation = EvaluationCase(
            case_id=item["case_id"],
            split=item["split"],
            query=item["query"],
            profile=item.get("profile", "web"),
            expected_claims=_load_expected_claims(item.get("expected_claims", [])),
            min_answer_chars=item.get("min_answer_chars"),
            min_unique_sources=item.get("min_unique_sources"),
        )
        cases.append(
            ReplayEvaluationCase(
                evaluation=evaluation,
                fixture=_load_replay_fixture(replay_raw),
            )
        )
    return cases


class _ReplayIntelligence:
    def __init__(self, fixture: ReplayFixture) -> None:
        self._fixture = fixture
        self._verify_calls: dict[str, int] = {}

    def classify_query(self, query: str, log=None) -> QueryClassification:
        return copy.deepcopy(self._fixture.classification)

    def decompose_claims(self, classification: QueryClassification, log=None) -> list[Claim]:
        return copy.deepcopy(self._fixture.claims)

    def verify_claim(self, claim: Claim, passages: list[Passage], log=None) -> VerificationResult:
        iteration = self._verify_calls.get(claim.claim_id, 0) + 1
        self._verify_calls[claim.claim_id] = iteration
        payload = self._fixture.verify.get(claim.claim_id, {}).get(iteration)
        if payload is None:
            raise KeyError(f"Missing replay verify payload for claim={claim.claim_id} iteration={iteration}")
        return copy.deepcopy(payload)

    def synthesize_answer(self, query: str, passages: list[Passage], log=None, intent: str = "synthesis") -> str:
        payload = self._fixture.synthesize
        return payload.get(query) or payload.get(intent) or payload.get("__default__", "")

    def suggest_rationale_query(self, claim_text: str, rationale: str, log=None) -> str | None:
        return None

    def refine_search_queries(
        self,
        claim: Claim,
        classification: QueryClassification,
        verification: VerificationResult,
        gated_results: list[GatedSerpResult],
        bundle,
        next_iteration: int,
        existing_queries: set[str],
        log=None,
    ) -> list[str]:
        return copy.deepcopy(self._fixture.refine.get(claim.claim_id, {}).get(next_iteration, []))


class _ReplaySearchGateway:
    def __init__(self, fixture: ReplayFixture) -> None:
        self._fixture = fixture

    def search_variant(self, query: str, profile, log=None) -> list[SearchSnapshot]:
        payload = self._fixture.search.get(_normalize_query_key(query))
        if payload is None:
            raise KeyError(f"Missing replay search payload for query={query!r}")
        return copy.deepcopy(payload)


class _ReplayFetchGateway:
    def __init__(self, fixture: ReplayFixture) -> None:
        self._fixture = fixture

    def fetch_claim_documents(
        self,
        claim: Claim,
        gated_results: list[GatedSerpResult],
        profile,
        routing_decision,
        *,
        seen_urls: set[str],
        log=None,
        iteration: int = 1,
        page_cache=None,
        page_cache_lock=None,
        intent: str = "factual",
    ) -> tuple[list[FetchPlan], list[FetchedDocument]]:
        payload = self._fixture.fetch.get(claim.claim_id, {}).get(iteration)
        if payload is None:
            return [], []
        return copy.deepcopy(payload)


class _ReplayReceiptWriter:
    def write(self, report: AgentRunResult, output_dir: str) -> str:
        return str(Path(output_dir) / "replay-receipt.json")


def build_replay_use_case(fixture: ReplayFixture) -> SearchAgentUseCase:
    return SearchAgentUseCase(
        intelligence=_ReplayIntelligence(fixture),
        search_gateway=_ReplaySearchGateway(fixture),
        fetch_gateway=_ReplayFetchGateway(fixture),
        receipt_writer=_ReplayReceiptWriter(),
        steps=AgentStepLibrary(),
    )


def evaluate_replay_dataset(
    dataset_path: str | Path,
    *,
    log=None,
) -> dict:
    log = log or (lambda msg: None)
    replay_cases = load_replay_cases(dataset_path)
    evaluation_cases = [case.evaluation for case in replay_cases]
    reports: dict[str, AgentRunResult] = {}
    latencies_ms: dict[str, int] = {}

    for idx, case in enumerate(replay_cases, 1):
        log(f"[replay-eval] case {idx}/{len(replay_cases)}: {case.evaluation.case_id}")
        started = time.perf_counter()
        report = build_replay_use_case(case.fixture).run(
            case.evaluation.query,
            get_profile(case.evaluation.profile),
            receipts_dir=None,
            log=log,
        )
        latencies_ms[case.evaluation.case_id] = int((time.perf_counter() - started) * 1000)
        reports[case.evaluation.case_id] = report

    summary = score_reports(evaluation_cases, reports, latencies_ms)
    summary["dataset_path"] = str(Path(dataset_path))
    summary["mode"] = "replay"
    return summary
