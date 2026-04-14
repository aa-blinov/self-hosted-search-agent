from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from search_agent.config.profiles import get_profile
from search_agent.domain.models import AgentRunResult, ClaimRun
from search_agent import tuning


@dataclass(slots=True)
class ExpectedClaim:
    match: str
    expected_verdict: str
    requires_primary_source: bool = False
    expected_routes: list[str] = field(default_factory=list)
    min_independent_sources: int | None = None


@dataclass(slots=True)
class EvaluationCase:
    case_id: str
    split: str
    query: str
    profile: str = "web"
    expected_claims: list[ExpectedClaim] = field(default_factory=list)
    min_answer_chars: int | None = None   # synthesis/news_digest answer depth check
    min_unique_sources: int | None = None  # news_digest source diversity check


def load_evaluation_cases(path: str) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        raw = line.strip()
        if not raw:
            continue
        item = json.loads(raw)
        cases.append(
            EvaluationCase(
                case_id=item["case_id"],
                split=item["split"],
                query=item["query"],
                profile=item.get("profile", "web"),
                min_answer_chars=item.get("min_answer_chars"),
                min_unique_sources=item.get("min_unique_sources"),
                expected_claims=[
                    ExpectedClaim(
                        match=claim["match"],
                        expected_verdict=claim["expected_verdict"],
                        requires_primary_source=bool(claim.get("requires_primary_source", False)),
                        expected_routes=(
                            [claim["expected_route"]]
                            if isinstance(claim.get("expected_route"), str)
                            else [str(route) for route in claim.get("expected_route", [])]
                        ),
                        min_independent_sources=claim.get("min_independent_sources"),
                    )
                    for claim in item.get("expected_claims", [])
                ],
            )
        )
    return cases


def _match_claim(run_list: list[ClaimRun], expected: ExpectedClaim) -> ClaimRun | None:
    needle = expected.match.casefold()
    needle_tokens = set(_match_tokens(expected.match))
    best_run: ClaimRun | None = None
    best_score = 0.0
    for run in run_list:
        haystack = run.claim.claim_text.casefold()
        if needle in haystack or haystack in needle:
            return run
        haystack_tokens = set(_match_tokens(run.claim.claim_text))
        if not needle_tokens or not haystack_tokens:
            continue
        overlap = len(needle_tokens & haystack_tokens) / len(needle_tokens)
        if overlap > best_score:
            best_score = overlap
            best_run = run
    return best_run if best_score >= 0.6 else None


def _match_tokens(text: str) -> list[str]:
    stopwords = {"a", "an", "the", "of", "is", "was", "were", "on", "in", "at", "for", "and"}
    tokens: list[str] = []
    current: list[str] = []
    for ch in (text or "").casefold():
        if ch.isalnum():
            current.append(ch)
            continue
        if current:
            token = "".join(current)
            if token not in stopwords:
                tokens.append(token)
            current = []
    if current:
        token = "".join(current)
        if token not in stopwords:
            tokens.append(token)
    return tokens


def _answer_unique_source_count(answer: str) -> int:
    """Count unique domains cited in the answer sources section."""
    from urllib.parse import urlparse
    domains: set[str] = set()
    for url in _answer_source_urls(answer):
        try:
            netloc = urlparse(url if url.startswith("http") else "https://" + url).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            if netloc:
                domains.add(netloc)
        except Exception:
            pass
    return len(domains)


def _answer_source_urls(answer: str) -> list[str]:
    urls: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped.startswith("["):
            continue
        if "http://" in stripped:
            urls.append("http://" + stripped.rsplit("http://", 1)[1].strip())
        elif "https://" in stripped:
            urls.append("https://" + stripped.rsplit("https://", 1)[1].strip())
    return urls


def _answer_bullets(answer: str) -> list[str]:
    """Bullets in the main answer block (before first blank line). Supports legacy 'Ответ' and plain '- ...' opens."""
    lines = answer.splitlines()
    bullets: list[str] = []
    start_idx: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "Ответ":
            start_idx = i + 1
            break
        if stripped == "Answer":
            start_idx = i + 1
            break
        if stripped.startswith("- "):
            start_idx = i
            break
    if start_idx is None:
        return []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("- "):
            bullets.append(stripped)
    return bullets


def _is_guardrail_bullet(text: str) -> bool:
    lowered = text.casefold()
    return (
        "недостаточно подтвержд" in lowered
        or "insufficient evidence" in lowered
        or "not enough supported" in lowered
    )


def compute_search_cost(report: AgentRunResult) -> float:
    if report.audit_trail.estimated_search_cost:
        return report.audit_trail.estimated_search_cost

    shallow = 0
    deep = 0
    snippet_only = 0
    for run in report.claims:
        for plan in run.fetch_plans:
            if plan.depth == "shallow":
                shallow += 1
            elif plan.depth == "deep":
                deep += 1
            else:
                snippet_only += 1
    return round(
        len(report.audit_trail.query_variants)
        + 0.25 * shallow
        + 1.0 * deep
        + 0.1 * snippet_only
        + 0.5 * len(report.claims),
        3,
    )


def _detect_backend_issue(report: AgentRunResult) -> bool:
    snapshots = report.audit_trail.serp_snapshots
    if not snapshots:
        return False
    saw_unresponsive = False
    for snapshot in snapshots:
        if snapshot.unresponsive_engines:
            saw_unresponsive = True
        if snapshot.results:
            return False
    return saw_unresponsive


def score_reports(
    cases: list[EvaluationCase],
    reports: dict[str, AgentRunResult],
    latencies_ms: dict[str, int] | None = None,
) -> dict:
    latencies_ms = latencies_ms or {}

    supported_expected = 0
    supported_hits = 0
    contradicted_expected = 0
    contradicted_hits = 0
    insufficient_expected = 0
    insufficient_hits = 0
    supported_with_primary = 0
    actual_supported = 0
    primary_requirements = 0
    primary_requirement_hits = 0
    route_expectations = 0
    route_hits = 0
    source_requirements = 0
    source_requirement_hits = 0
    backend_issue_cases = 0

    valid_citations = 0
    total_citations = 0
    unsupported_bullets = 0
    total_answer_bullets = 0
    answer_depth_requirements = 0
    answer_depth_hits = 0
    source_diversity_requirements = 0
    source_diversity_hits = 0
    answer_chars_all: list[int] = []
    costs: list[float] = []
    latency_values: list[int] = []
    case_details: list[dict] = []
    all_iterations: list[int] = []
    route_counts: dict[str, int] = {"fast": 0, "full": 0}

    by_split: dict[str, dict[str, list[float] | int]] = {}

    for case in cases:
        report = reports[case.case_id]
        case_cost = compute_search_cost(report)
        case_latency = latencies_ms.get(case.case_id, report.audit_trail.latency_ms)
        backend_issue = _detect_backend_issue(report)
        if backend_issue:
            backend_issue_cases += 1
        costs.append(case_cost)
        latency_values.append(case_latency)

        for claim_id, iters in report.audit_trail.claim_iterations.items():
            all_iterations.append(iters)
        for run in report.claims:
            if run.routing_decision:
                route_counts[run.routing_decision.mode] = route_counts.get(run.routing_decision.mode, 0) + 1

        split_bucket = by_split.setdefault(
            case.split,
            {
                "supported_expected": 0,
                "supported_hits": 0,
                "contradicted_expected": 0,
                "contradicted_hits": 0,
                "insufficient_expected": 0,
                "insufficient_hits": 0,
                "route_expectations": 0,
                "route_hits": 0,
                "primary_requirements": 0,
                "primary_requirement_hits": 0,
                "source_requirements": 0,
                "source_requirement_hits": 0,
                "backend_issue_cases": 0,
                "costs": [],
                "latencies": [],
            },
        )
        split_bucket["costs"].append(case_cost)
        split_bucket["latencies"].append(case_latency)
        if backend_issue:
            split_bucket["backend_issue_cases"] += 1

        # citation_validity and unsupported_statement are meaningful only for factual
        # queries where the answer is grounded in claim-level passages.  Synthesis and
        # news_digest answers use a separate passage path (synthesize_answer), so their
        # cited URLs won't appear in run.passages and all bullets look "unsupported".
        is_factual_split = not case.split.startswith("synthesis") and case.split != "news_digest"

        supported_claim_count = 0
        for run in report.claims:
            bundle = run.evidence_bundle
            if bundle and bundle.verification and bundle.verification.verdict == "supported":
                supported_claim_count += 1

        if is_factual_split:
            answer_bullets = [bullet for bullet in _answer_bullets(report.answer) if not _is_guardrail_bullet(bullet)]
            total_answer_bullets += len(answer_bullets)
            unsupported_bullets += max(0, len(answer_bullets) - supported_claim_count)

            cited_urls = _answer_source_urls(report.answer)
            valid_urls = {
                passage.url
                for run in report.claims
                for passage in run.passages
            }
            total_citations += len(cited_urls)
            valid_citations += sum(1 for url in cited_urls if url in valid_urls)

        # Answer depth and source diversity checks (synthesis / news_digest)
        answer_chars = len(report.answer)
        answer_chars_all.append(answer_chars)
        unique_sources = _answer_unique_source_count(report.answer)
        if case.min_answer_chars is not None:
            answer_depth_requirements += 1
            if answer_chars >= case.min_answer_chars:
                answer_depth_hits += 1
        if case.min_unique_sources is not None:
            source_diversity_requirements += 1
            if unique_sources >= case.min_unique_sources:
                source_diversity_hits += 1

        detail = {
            "case_id": case.case_id,
            "split": case.split,
            "query": case.query,
            "profile": case.profile,
            "latency_ms": case_latency,
            "search_cost": case_cost,
            "answer": report.answer,
            "answer_chars": answer_chars,
            "unique_sources_in_answer": unique_sources,
            "backend_issue": backend_issue,
            "claims": [],
        }

        for expected in case.expected_claims:
            run = _match_claim(report.claims, expected)
            bundle = run.evidence_bundle if run else None
            actual_verdict = (
                bundle.verification.verdict
                if bundle and bundle.verification
                else None
            )
            actual_route = run.routing_decision.mode if run and run.routing_decision else None
            if expected.expected_verdict == "supported":
                supported_expected += 1
                split_bucket["supported_expected"] += 1
                if actual_verdict == "supported":
                    supported_hits += 1
                    split_bucket["supported_hits"] += 1
            if expected.expected_verdict == "contradicted":
                contradicted_expected += 1
                split_bucket["contradicted_expected"] += 1
                if actual_verdict == "contradicted":
                    contradicted_hits += 1
                    split_bucket["contradicted_hits"] += 1
            if expected.expected_verdict == "insufficient_evidence":
                insufficient_expected += 1
                split_bucket["insufficient_expected"] += 1
                if actual_verdict == "insufficient_evidence":
                    insufficient_hits += 1
                    split_bucket["insufficient_hits"] += 1
            if expected.expected_routes:
                route_expectations += 1
                split_bucket["route_expectations"] += 1
                if actual_route in expected.expected_routes:
                    route_hits += 1
                    split_bucket["route_hits"] += 1
            if expected.requires_primary_source:
                primary_requirements += 1
                split_bucket["primary_requirements"] += 1
                if actual_verdict == "supported" and bundle and bundle.has_primary_source:
                    primary_requirement_hits += 1
                    split_bucket["primary_requirement_hits"] += 1
            if expected.min_independent_sources is not None:
                source_requirements += 1
                split_bucket["source_requirements"] += 1
                if (
                    actual_verdict == "supported"
                    and bundle
                    and bundle.independent_source_count >= expected.min_independent_sources
                ):
                    source_requirement_hits += 1
                    split_bucket["source_requirement_hits"] += 1
            if actual_verdict == "supported":
                actual_supported += 1
                if bundle and bundle.has_primary_source:
                    supported_with_primary += 1

            detail["claims"].append({
                "match": expected.match,
                "expected_verdict": expected.expected_verdict,
                "actual_verdict": actual_verdict,
                "expected_routes": expected.expected_routes,
                "actual_route": actual_route,
                "route_ok": (not expected.expected_routes) or actual_route in expected.expected_routes,
                "requires_primary_source": expected.requires_primary_source,
                "has_primary_source": bundle.has_primary_source if bundle else False,
                "primary_ok": (not expected.requires_primary_source) or bool(bundle and bundle.has_primary_source and actual_verdict == "supported"),
                "min_independent_sources": expected.min_independent_sources,
                "independent_source_count": bundle.independent_source_count if bundle else 0,
                "source_count_ok": (
                    expected.min_independent_sources is None
                    or bool(bundle and actual_verdict == "supported" and bundle.independent_source_count >= expected.min_independent_sources)
                ),
            })

        case_details.append(detail)

    def _rate(hits: int, expectations: int) -> float | None:
        if not expectations:
            return None
        return round(hits / expectations, 4)

    metrics = {
        "claim_support_rate": round(supported_hits / supported_expected, 4) if supported_expected else 0.0,
        "citation_validity_rate": round(valid_citations / total_citations, 4) if total_citations else 1.0,
        "unsupported_statement_rate": round(unsupported_bullets / total_answer_bullets, 4) if total_answer_bullets else 0.0,
        "primary_source_coverage": round(supported_with_primary / actual_supported, 4) if actual_supported else 0.0,
        "contradiction_detection_rate": round(contradicted_hits / contradicted_expected, 4) if contradicted_expected else 0.0,
        "insufficient_detection_rate": round(insufficient_hits / insufficient_expected, 4) if insufficient_expected else 0.0,
        "route_match_rate": _rate(route_hits, route_expectations),
        "primary_requirement_rate": _rate(primary_requirement_hits, primary_requirements),
        "source_requirement_rate": _rate(source_requirement_hits, source_requirements),
        "answer_depth_rate": _rate(answer_depth_hits, answer_depth_requirements),
        "source_diversity_rate": _rate(source_diversity_hits, source_diversity_requirements),
        "median_answer_chars": round(statistics.median(answer_chars_all), 0) if answer_chars_all else 0.0,
        "backend_issue_rate": round(backend_issue_cases / len(cases), 4) if cases else 0.0,
        "median_search_cost": round(statistics.median(costs), 3) if costs else 0.0,
        "median_answer_latency": round(statistics.median(latency_values), 1) if latency_values else 0.0,
        "avg_iterations_per_claim": round(statistics.mean(all_iterations), 2) if all_iterations else 0.0,
        "route_fast_rate": round(route_counts["fast"] / sum(route_counts.values()), 4) if sum(route_counts.values()) else 0.0,
        "route_full_rate": round(route_counts["full"] / sum(route_counts.values()), 4) if sum(route_counts.values()) else 0.0,
    }

    split_metrics: dict[str, dict] = {}
    for split, bucket in by_split.items():
        split_metrics[split] = {
            "claim_support_rate": round(
                bucket["supported_hits"] / bucket["supported_expected"], 4
            ) if bucket["supported_expected"] else 0.0,
            "contradiction_detection_rate": round(
                bucket["contradicted_hits"] / bucket["contradicted_expected"], 4
            ) if bucket["contradicted_expected"] else 0.0,
            "insufficient_detection_rate": round(
                bucket["insufficient_hits"] / bucket["insufficient_expected"], 4
            ) if bucket["insufficient_expected"] else 0.0,
            "route_match_rate": _rate(bucket["route_hits"], bucket["route_expectations"]),
            "primary_requirement_rate": _rate(
                bucket["primary_requirement_hits"], bucket["primary_requirements"]
            ),
            "source_requirement_rate": _rate(
                bucket["source_requirement_hits"], bucket["source_requirements"]
            ),
            "backend_issue_rate": round(
                bucket["backend_issue_cases"] / max(1, len(bucket["costs"])), 4
            ),
            "median_search_cost": round(statistics.median(bucket["costs"]), 3) if bucket["costs"] else 0.0,
            "median_answer_latency": round(statistics.median(bucket["latencies"]), 1) if bucket["latencies"] else 0.0,
        }

    return {
        "case_count": len(cases),
        "metrics": metrics,
        "by_split": split_metrics,
        "cases": case_details,
    }


def evaluate_dataset(
    dataset_path: str,
    receipts_dir: str | None = None,
    delay_between_cases: float | None = None,
    log=None,
    # Deprecated: kept for call-site compatibility, ignored.
    client=None,
    unified: bool = True,  # kept as no-op for backward-compat call sites
) -> dict:
    """Run the unified search agent against a JSONL dataset and score it.

    The ``unified`` parameter is retained for call-site compatibility but is
    now a no-op — the classic pipeline has been removed from the eval path.
    """
    log = log or (lambda msg: None)
    cases = load_evaluation_cases(dataset_path)
    reports: dict[str, AgentRunResult] = {}
    latencies_ms: dict[str, int] = {}
    case_delay = (
        delay_between_cases
        if delay_between_cases is not None
        else tuning.EVAL_CASE_DELAY_SEC
    )

    from search_agent import build_unified_search_agent_use_case

    use_case = build_unified_search_agent_use_case()

    for idx, case in enumerate(cases, 1):
        if idx > 1 and case_delay > 0:
            time.sleep(case_delay)
        log(f"[eval] case {idx}/{len(cases)}: {case.case_id}")
        start = time.perf_counter()
        report = use_case.run(
            case.query,
            profile=get_profile(case.profile),
            receipts_dir=receipts_dir,
            log=log,
        )
        latencies_ms[case.case_id] = int((time.perf_counter() - start) * 1000)
        reports[case.case_id] = report

    summary = score_reports(cases, reports, latencies_ms)
    summary["dataset_path"] = str(Path(dataset_path))
    summary["mode"] = "unified"
    return summary
