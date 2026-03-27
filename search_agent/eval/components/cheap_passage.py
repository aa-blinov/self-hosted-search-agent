"""Component eval: cheap_passage_filter — pure Python, no LLM needed."""
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
    load_passage,
)

DEFAULT_DATASET = DEFAULT_DATASETS_DIR / "cheap_passage.jsonl"


def run_component_eval(
    cases: list[dict],
    *,
    dataset_path: str = str(DEFAULT_DATASET),
    **_kwargs,
) -> ComponentRunSummary:
    from search_agent.application.agent_steps import cheap_passage_filter

    results: list[ComponentCaseResult] = []
    include_hits = include_total = 0
    exclude_hits = exclude_total = 0
    latencies: list[float] = []

    for case in cases:
        claim = load_claim(case["claim"])
        passages = [load_passage(p) for p in case["passages"]]
        expected_include: list[str] = case.get("expected_passage_ids_include", [])
        expected_exclude: list[str] = case.get("expected_passage_ids_exclude", [])

        t0 = time.perf_counter()
        filtered = cheap_passage_filter(claim, passages)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        latencies.append(latency_ms)

        kept_ids = {p.passage_id for p in filtered}

        inc_pass = all(pid in kept_ids for pid in expected_include)
        exc_pass = all(pid not in kept_ids for pid in expected_exclude)
        passed = inc_pass and exc_pass

        include_total += len(expected_include)
        include_hits += sum(1 for pid in expected_include if pid in kept_ids)
        exclude_total += len(expected_exclude)
        exclude_hits += sum(1 for pid in expected_exclude if pid not in kept_ids)

        results.append(ComponentCaseResult(
            case_id=case["case_id"],
            passed=passed,
            latency_ms=latency_ms,
            details={
                "input_count": len(passages),
                "output_count": len(filtered),
                "kept_ids": sorted(kept_ids),
                "include_pass": inc_pass,
                "exclude_pass": exc_pass,
                "missing_includes": [pid for pid in expected_include if pid not in kept_ids],
                "unexpected_kept": [pid for pid in expected_exclude if pid in kept_ids],
            },
        ))

    passed_count = sum(1 for r in results if r.passed)
    metrics = {
        "include_recall": safe_div(include_hits, include_total),
        "exclude_precision": safe_div(exclude_hits, exclude_total, default=1.0),
        "pass_rate": safe_div(passed_count, len(cases)),
        "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else None,
    }

    return ComponentRunSummary(
        component="cheap_passage",
        dataset_path=dataset_path,
        case_count=len(cases),
        pass_rate=metrics["pass_rate"],
        metrics=metrics,
        cases=results,
    )


def main(dataset_path: Path = DEFAULT_DATASET, **kwargs) -> ComponentRunSummary:
    cases = load_cases(dataset_path)
    return run_component_eval(cases, dataset_path=str(dataset_path), **kwargs)
