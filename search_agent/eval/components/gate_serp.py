"""Component eval: gate_serp_results — pure Python, no LLM needed."""
from __future__ import annotations

import time
from pathlib import Path

from .metrics import accuracy, safe_div
from .runner import (
    ComponentCaseResult,
    ComponentRunSummary,
    DEFAULT_DATASETS_DIR,
    load_cases,
    load_claim,
    load_search_snapshot,
)

DEFAULT_DATASET = DEFAULT_DATASETS_DIR / "gate_serp.jsonl"


def run_component_eval(
    cases: list[dict],
    *,
    dataset_path: str = str(DEFAULT_DATASET),
    **_kwargs,
) -> ComponentRunSummary:
    from search_agent.application.agent_steps import gate_serp_results

    results: list[ComponentCaseResult] = []
    include_hits = include_total = 0
    exclude_hits = exclude_total = 0
    min_count_passes = 0

    for case in cases:
        claim = load_claim(case["claim"])
        snapshots = [load_search_snapshot(s) for s in case["snapshots"]]
        limit = int(case.get("limit", 20))
        expected_include: list[str] = case.get("expected_urls_include", [])
        expected_exclude: list[str] = case.get("expected_urls_exclude", [])
        expected_min: int = int(case.get("expected_min_count", 1))

        t0 = time.perf_counter()
        gated = gate_serp_results(claim, snapshots, limit)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        output_urls = {r.serp.url for r in gated} | {r.serp.canonical_url for r in gated}

        inc_pass = all(u in output_urls for u in expected_include)
        exc_pass = all(u not in output_urls for u in expected_exclude)
        cnt_pass = len(gated) >= expected_min
        passed = inc_pass and exc_pass and cnt_pass

        include_total += len(expected_include)
        include_hits += sum(1 for u in expected_include if u in output_urls)
        exclude_total += len(expected_exclude)
        exclude_hits += sum(1 for u in expected_exclude if u not in output_urls)
        if cnt_pass:
            min_count_passes += 1

        results.append(ComponentCaseResult(
            case_id=case["case_id"],
            passed=passed,
            latency_ms=latency_ms,
            details={
                "output_url_count": len(gated),
                "include_pass": inc_pass,
                "exclude_pass": exc_pass,
                "min_count_pass": cnt_pass,
                "missing_includes": [u for u in expected_include if u not in output_urls],
                "unexpected_excludes": [u for u in expected_exclude if u in output_urls],
            },
        ))

    passed_count = sum(1 for r in results if r.passed)
    metrics = {
        "include_hit_rate": safe_div(include_hits, include_total),
        "exclude_hit_rate": safe_div(exclude_hits, exclude_total, default=1.0),
        "min_count_pass_rate": safe_div(min_count_passes, len(cases)),
        "pass_rate": safe_div(passed_count, len(cases)),
    }

    return ComponentRunSummary(
        component="gate_serp",
        dataset_path=dataset_path,
        case_count=len(cases),
        pass_rate=metrics["pass_rate"],
        metrics=metrics,
        cases=results,
    )


def main(dataset_path: Path = DEFAULT_DATASET, **kwargs) -> ComponentRunSummary:
    cases = load_cases(dataset_path)
    return run_component_eval(cases, dataset_path=str(dataset_path), **kwargs)
