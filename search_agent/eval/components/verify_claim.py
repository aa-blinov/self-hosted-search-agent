"""Component eval: verify_claim — LLM-based."""
from __future__ import annotations

import time
from pathlib import Path

from .metrics import accuracy, per_class_metrics, percentile, safe_div
from .runner import (
    ComponentCaseResult,
    ComponentRunSummary,
    DEFAULT_DATASETS_DIR,
    load_cases,
    load_claim,
    load_passage,
)

DEFAULT_DATASET = DEFAULT_DATASETS_DIR / "verify_claim.jsonl"
VERDICT_CLASSES = ["supported", "contradicted", "insufficient_evidence"]


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
        print(f"  [verify_claim] LLM init failed: {exc}")
        raise

    results: list[ComponentCaseResult] = []
    correct = 0
    confidence_floor_hits = confidence_floor_total = 0
    false_support_count = 0  # contradicted/insufficient but predicted supported
    latencies: list[float] = []
    per_class_data: list[dict] = []

    for case in cases:
        claim = load_claim({
            "claim_id": case["case_id"],
            "claim_text": case["claim_text"],
            "priority": 1,
            "needs_freshness": False,
        })
        passages = [load_passage(p) for p in case["passages"]]
        expected_verdict: str = case["expected_verdict"]
        confidence_min: float | None = case.get("expected_confidence_min")

        t0 = time.perf_counter()
        try:
            verification = intelligence.verify_claim(claim, passages)
            predicted_verdict = verification.verdict
            confidence = verification.confidence
            error = None
        except Exception as exc:
            predicted_verdict = "insufficient_evidence"
            confidence = 0.0
            error = str(exc)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        latencies.append(latency_ms)

        verdict_correct = predicted_verdict == expected_verdict
        if verdict_correct:
            correct += 1

        conf_pass = True
        if confidence_min is not None:
            confidence_floor_total += 1
            conf_pass = confidence >= confidence_min
            if conf_pass:
                confidence_floor_hits += 1

        # false support: model said supported but truth is not
        if expected_verdict != "supported" and predicted_verdict == "supported":
            false_support_count += 1

        passed = verdict_correct and conf_pass
        per_class_data.append({"expected": expected_verdict, "predicted": predicted_verdict})

        details: dict = {
            "claim_text": case["claim_text"],
            "expected_verdict": expected_verdict,
            "predicted_verdict": predicted_verdict,
            "verdict_correct": verdict_correct,
            "confidence": confidence if error is None else None,
            "confidence_floor_pass": conf_pass,
        }
        if error:
            details["error"] = error

        results.append(ComponentCaseResult(
            case_id=case["case_id"],
            passed=passed,
            latency_ms=latency_ms,
            details=details,
        ))

    per_class = per_class_metrics(per_class_data, "expected", "predicted", VERDICT_CLASSES)
    passed_count = sum(1 for r in results if r.passed)
    metrics = {
        "accuracy": accuracy(correct, len(cases)),
        "pass_rate": accuracy(passed_count, len(cases)),
        "verdict_accuracy": accuracy(correct, len(cases)),
        "confidence_floor_pass_rate": (
            safe_div(confidence_floor_hits, confidence_floor_total)
            if confidence_floor_total > 0 else None
        ),
        "false_support_rate": safe_div(false_support_count, len(cases)),
        "per_class": per_class,
        "median_latency_ms": percentile(latencies, 50),
    }

    return ComponentRunSummary(
        component="verify_claim",
        dataset_path=dataset_path,
        case_count=len(cases),
        pass_rate=metrics["pass_rate"] or 0.0,
        metrics=metrics,
        cases=results,
    )


def main(dataset_path: Path = DEFAULT_DATASET, **kwargs) -> ComponentRunSummary:
    cases = load_cases(dataset_path)
    return run_component_eval(cases, dataset_path=str(dataset_path), **kwargs)
