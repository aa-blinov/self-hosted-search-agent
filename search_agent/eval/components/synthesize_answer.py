"""Component eval: synthesize_answer — LLM-based."""
from __future__ import annotations

import time
from pathlib import Path

from .metrics import percentile, safe_div
from .runner import (
    ComponentCaseResult,
    ComponentRunSummary,
    DEFAULT_DATASETS_DIR,
    load_cases,
    load_passage,
)

DEFAULT_DATASET = DEFAULT_DATASETS_DIR / "synthesize_answer.jsonl"


def run_component_eval(
    cases: list[dict],
    *,
    dataset_path: str = str(DEFAULT_DATASET),
    settings=None,
    **_kwargs,
) -> ComponentRunSummary:
    if settings is None:
        from search_agent.settings import get_settings
        settings = get_settings()

    intelligence = None
    try:
        from search_agent.infrastructure.intelligence import PydanticAIQueryIntelligence
        intelligence = PydanticAIQueryIntelligence(settings)
    except Exception as exc:
        print(f"  [synthesize_answer] LLM init failed: {exc}")
        raise

    results: list[ComponentCaseResult] = []
    keyword_hits = keyword_total = 0
    min_chars_passes = 0
    answer_chars: list[int] = []
    latencies: list[float] = []

    for case in cases:
        query: str = case["query"]
        intent: str = case.get("intent", "synthesis")
        passages = [load_passage(p) for p in case["passages"]]
        expected_keywords: list[str] = case.get("expected_keywords", [])
        min_chars: int = int(case.get("min_answer_chars", 0))

        t0 = time.perf_counter()
        try:
            answer = intelligence.synthesize_answer(query, passages, intent=intent)
            error = None
        except Exception as exc:
            answer = ""
            error = str(exc)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        latencies.append(latency_ms)

        answer_lower = answer.lower()
        found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        missing_keywords = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
        kw_hit = len(found_keywords) == len(expected_keywords)
        chars_pass = len(answer) >= min_chars

        keyword_total += len(expected_keywords)
        keyword_hits += len(found_keywords)
        if chars_pass:
            min_chars_passes += 1
        answer_chars.append(len(answer))

        passed = kw_hit and chars_pass and error is None

        details: dict = {
            "query": query,
            "intent": intent,
            "answer_chars": len(answer),
            "keyword_hit": kw_hit,
            "chars_pass": chars_pass,
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "answer_preview": answer[:200] if answer else "",
        }
        if error:
            details["error"] = error

        results.append(ComponentCaseResult(
            case_id=case["case_id"],
            passed=passed,
            latency_ms=latency_ms,
            details=details,
        ))

    passed_count = sum(1 for r in results if r.passed)
    metrics = {
        "pass_rate": safe_div(passed_count, len(cases)),
        "keyword_hit_rate": safe_div(keyword_hits, keyword_total),
        "min_chars_pass_rate": safe_div(min_chars_passes, len(cases)),
        "median_answer_chars": percentile([float(c) for c in answer_chars], 50),
        "median_latency_ms": percentile(latencies, 50),
    }

    return ComponentRunSummary(
        component="synthesize_answer",
        dataset_path=dataset_path,
        case_count=len(cases),
        pass_rate=metrics["pass_rate"],
        metrics=metrics,
        cases=results,
    )


def main(dataset_path: Path = DEFAULT_DATASET, **kwargs) -> ComponentRunSummary:
    cases = load_cases(dataset_path)
    return run_component_eval(cases, dataset_path=str(dataset_path), **kwargs)
