"""Component eval: classify_intent — LLM-based."""
from __future__ import annotations

import time
from pathlib import Path

from .metrics import accuracy, per_class_metrics, percentile
from .runner import (
    ComponentCaseResult,
    ComponentRunSummary,
    DEFAULT_DATASETS_DIR,
    load_cases,
)

DEFAULT_DATASET = DEFAULT_DATASETS_DIR / "classify_intent.jsonl"
INTENT_CLASSES = ["factual", "synthesis", "news_digest"]


def _heuristic_classify(query: str) -> str:
    """Simple keyword-based fallback (no LLM)."""
    q = query.lower()
    news_keywords = ("news", "latest", "this week", "новости", "последние", "на этой неделе")
    synthesis_keywords = (
        "how", "why", "what are", "explain", "difference", "compare",
        "features", "overview", "чем отличается", "как работает", "объясни",
    )
    if any(kw in q for kw in news_keywords):
        return "news_digest"
    if any(kw in q for kw in synthesis_keywords):
        return "synthesis"
    return "factual"


def run_component_eval(
    cases: list[dict],
    *,
    dataset_path: str = str(DEFAULT_DATASET),
    settings=None,
    **_kwargs,
) -> ComponentRunSummary:
    intelligence = None
    if settings is not None:
        try:
            from search_agent.infrastructure.intelligence import PydanticAIQueryIntelligence
            intelligence = PydanticAIQueryIntelligence(settings)
        except Exception as exc:
            print(f"  [classify_intent] LLM init failed: {exc}. Using heuristic fallback.")

    results: list[ComponentCaseResult] = []
    llm_correct = llm_total = 0
    heuristic_correct = 0
    latencies: list[float] = []
    per_class_data: list[dict] = []

    for case in cases:
        query: str = case["query"]
        expected: str = case["expected_intent"]

        # Heuristic baseline (always runs)
        heuristic_pred = _heuristic_classify(query)
        if heuristic_pred == expected:
            heuristic_correct += 1

        llm_pred: str | None = None
        latency_ms = 0

        if intelligence is not None:
            t0 = time.perf_counter()
            try:
                classification = intelligence.classify_query(query)
                llm_pred = classification.intent
            except Exception as exc:
                llm_pred = None
                print(f"  [classify_intent] LLM error on {case['case_id']}: {exc}")
            latency_ms = int((time.perf_counter() - t0) * 1000)
            latencies.append(latency_ms)
            llm_total += 1
            if llm_pred == expected:
                llm_correct += 1

        final_pred = llm_pred if llm_pred is not None else heuristic_pred
        passed = final_pred == expected
        per_class_data.append({"expected": expected, "predicted": final_pred})

        results.append(ComponentCaseResult(
            case_id=case["case_id"],
            passed=passed,
            latency_ms=latency_ms,
            details={
                "query": query,
                "expected": expected,
                "llm_prediction": llm_pred,
                "heuristic_prediction": heuristic_pred,
                "final_prediction": final_pred,
            },
        ))

    per_class = per_class_metrics(per_class_data, "expected", "predicted", INTENT_CLASSES)
    passed_count = sum(1 for r in results if r.passed)
    metrics = {
        "accuracy": accuracy(passed_count, len(cases)),
        "pass_rate": accuracy(passed_count, len(cases)),
        "llm_accuracy": accuracy(llm_correct, llm_total),
        "heuristic_accuracy": accuracy(heuristic_correct, len(cases)),
        "per_class": per_class,
        "median_latency_ms": percentile(latencies, 50),
        "llm_cases_run": llm_total,
    }

    return ComponentRunSummary(
        component="classify_intent",
        dataset_path=dataset_path,
        case_count=len(cases),
        pass_rate=metrics["pass_rate"] or 0.0,
        metrics=metrics,
        cases=results,
    )


def main(dataset_path: Path = DEFAULT_DATASET, **kwargs) -> ComponentRunSummary:
    cases = load_cases(dataset_path)
    return run_component_eval(cases, dataset_path=str(dataset_path), **kwargs)
