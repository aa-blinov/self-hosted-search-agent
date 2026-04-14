"""Component eval: route_claim_retrieval — pure Python, no LLM needed."""
from __future__ import annotations

import time
from pathlib import Path

from .metrics import accuracy, per_class_metrics
from .runner import (
    ComponentCaseResult,
    ComponentRunSummary,
    DEFAULT_DATASETS_DIR,
    load_cases,
    load_claim,
    load_gated_serp_result,
)

DEFAULT_DATASET = DEFAULT_DATASETS_DIR / "route_claim.jsonl"
ROUTE_MODES = ["fast", "full"]


def run_component_eval(
    cases: list[dict],
    *,
    dataset_path: str = str(DEFAULT_DATASET),
    **_kwargs,
) -> ComponentRunSummary:
    from search_agent.application.agent_steps import route_claim_retrieval

    results: list[ComponentCaseResult] = []
    correct = 0
    per_class_data: list[dict] = []
    latencies: list[float] = []

    for case in cases:
        claim = load_claim(case["claim"])
        gated = [load_gated_serp_result(g) for g in case["gated_results"]]
        expected_route: str = case["expected_route"]

        t0 = time.perf_counter()
        decision = route_claim_retrieval(claim, gated)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        latencies.append(latency_ms)

        passed = decision.mode == expected_route
        if passed:
            correct += 1

        per_class_data.append({"expected": expected_route, "predicted": decision.mode})

        results.append(ComponentCaseResult(
            case_id=case["case_id"],
            passed=passed,
            latency_ms=latency_ms,
            details={
                "predicted_route": decision.mode,
                "expected_route": expected_route,
                "certainty": round(decision.certainty, 3),
                "consistency": round(decision.consistency, 3),
                "evidence_sufficiency": round(decision.evidence_sufficiency, 3),
                "rationale": decision.rationale,
            },
        ))

    per_mode = per_class_metrics(per_class_data, "expected", "predicted", ROUTE_MODES)
    metrics = {
        "accuracy": accuracy(correct, len(cases)),
        "pass_rate": accuracy(correct, len(cases)),
        "per_mode": per_mode,
        "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else None,
    }

    return ComponentRunSummary(
        component="route_claim",
        dataset_path=dataset_path,
        case_count=len(cases),
        pass_rate=metrics["pass_rate"] or 0.0,
        metrics=metrics,
        cases=results,
    )


def main(dataset_path: Path = DEFAULT_DATASET, **kwargs) -> ComponentRunSummary:
    cases = load_cases(dataset_path)
    return run_component_eval(cases, dataset_path=str(dataset_path), **kwargs)
