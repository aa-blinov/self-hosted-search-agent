"""Standalone eval for LLM-based query intent classification.

Run:
    uv run python -m search_agent.eval.intent_eval
    uv run python -m search_agent.eval.intent_eval --dataset eval_data/intent_classification_dataset.jsonl
    uv run python -m search_agent.eval.intent_eval --no-llm   # heuristic baseline only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent

from search_agent.application.text_heuristics import (
    COMPARISON_MARKERS,
    extract_entities,
    extract_region_hint,
    extract_time_scope,
    is_news_digest_query,
    needs_freshness,
    normalize_relative_time_references,
    normalized_text,
    should_decompose,
)
from search_agent.infrastructure.pydantic_ai_factory import build_model_settings, build_openai_model
from search_agent.settings import AppSettings, get_settings

# ---------------------------------------------------------------------------
# Tunable system prompt — edit this constant to iterate on the LLM classifier.
# Keep under 200 chars of instruction to minimise thinking-token burn.
# ---------------------------------------------------------------------------
INTENT_CLASSIFY_PROMPT = (
    "Classify query intent. Reply with exactly one token: factual, synthesis, or news_digest.\n"
    "factual: specific verifiable fact (who/when/what/where).\n"
    "synthesis: explanation, comparison, how-to, overview, what's new.\n"
    "news_digest: recent news or events."
)

Intent = Literal["factual", "synthesis", "news_digest"]

DEFAULT_DATASET = Path(__file__).parents[2] / "eval_data" / "intent_classification_dataset.jsonl"

# The heuristic uses "comparison" for what we call "synthesis" in this eval.
_HEURISTIC_TO_EVAL_INTENT: dict[str, Intent] = {
    "factual": "factual",
    "comparison": "synthesis",
    "news_digest": "news_digest",
}


# ---------------------------------------------------------------------------
# Heuristic baseline  (mirrors PydanticAIQueryIntelligence.classify_query
# but without the LLM time-normalization step and without constructing the
# full class — keeping this file self-contained).
# ---------------------------------------------------------------------------


def heuristic_classify(query: str) -> Intent:
    """Pure-heuristic intent classification, mirrors classify_query() logic."""
    normalized = normalize_relative_time_references(query)
    lowered = normalized.lower()
    region_hint = extract_region_hint(normalized)
    freshness = needs_freshness(query)
    if is_news_digest_query(normalized, region_hint=region_hint, freshness=freshness):
        raw_intent = "news_digest"
    elif any(marker in lowered for marker in COMPARISON_MARKERS):
        raw_intent = "comparison"
    else:
        raw_intent = "factual"
    return _HEURISTIC_TO_EVAL_INTENT[raw_intent]


# ---------------------------------------------------------------------------
# LLM classifier output schema
# ---------------------------------------------------------------------------


class _IntentOutput(BaseModel):
    intent: Intent


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------


class LLMIntentClassifier:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        model = build_openai_model(settings)
        self._agent: Agent[None, _IntentOutput] = Agent(
            model,
            output_type=_IntentOutput,
            retries=1,
            system_prompt=INTENT_CLASSIFY_PROMPT,
        )

    def classify(self, query: str) -> tuple[Intent, float]:
        """Returns (intent, latency_seconds)."""
        # qwen3.5-35b-a3b generates <think>...</think> before the answer,
        # so we need enough tokens for the thinking section + the actual reply.
        model_settings = build_model_settings(
            self._settings,
            max_tokens=300,
            temperature=0,
        )
        t0 = time.perf_counter()
        result = self._agent.run_sync(query, model_settings=model_settings)
        latency = time.perf_counter() - t0
        return result.output.intent, latency


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> list[dict]:
    examples = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] line {lineno}: JSON parse error — {exc}", file=sys.stderr)
                continue
            if "query" not in obj or "expected_intent" not in obj:
                print(f"[WARN] line {lineno}: missing 'query' or 'expected_intent', skipping", file=sys.stderr)
                continue
            examples.append(obj)
    return examples


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _accuracy(correct: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{correct / total:.1%} ({correct}/{total})"


def _percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = max(0, int(len(sorted_data) * pct / 100) - 1)
    return sorted_data[idx]


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main eval logic
# ---------------------------------------------------------------------------


def run_eval(dataset_path: Path, *, run_llm: bool = True) -> None:
    examples = load_dataset(dataset_path)
    if not examples:
        print("No examples loaded — check dataset path.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(examples)} examples from {dataset_path}")

    settings = get_settings()
    if run_llm and not settings.llm_api_key:
        print(
            "[WARN] LLM_API_KEY not set — skipping LLM classifier. "
            "Pass --no-llm to suppress this warning.",
            file=sys.stderr,
        )
        run_llm = False

    llm_classifier: LLMIntentClassifier | None = None
    if run_llm:
        llm_classifier = LLMIntentClassifier(settings)

    # -----------------------------------------------------------------------
    # Per-example results
    # -----------------------------------------------------------------------
    categories: list[Intent] = ["factual", "synthesis", "news_digest"]
    heuristic_correct: dict[Intent, int] = {c: 0 for c in categories}
    heuristic_total: dict[Intent, int] = {c: 0 for c in categories}
    llm_correct: dict[Intent, int] = {c: 0 for c in categories}
    llm_latencies: list[float] = []
    disagreements: list[dict] = []

    _print_section("Per-example results")

    for ex in examples:
        query: str = ex["query"]
        expected: Intent = ex["expected_intent"]
        notes: str = ex.get("notes", "")

        # Heuristic
        h_intent = heuristic_classify(query)
        h_ok = h_intent == expected
        heuristic_total[expected] = heuristic_total.get(expected, 0) + 1
        if h_ok:
            heuristic_correct[expected] = heuristic_correct.get(expected, 0) + 1

        # LLM
        l_intent: Intent | None = None
        l_ok: bool | None = None
        latency: float | None = None
        if llm_classifier is not None:
            l_intent, latency = llm_classifier.classify(query)
            l_ok = l_intent == expected
            llm_latencies.append(latency)
            if l_ok:
                llm_correct[expected] = llm_correct.get(expected, 0) + 1
            if l_intent != h_intent:
                disagreements.append(
                    {
                        "query": query,
                        "expected": expected,
                        "heuristic": h_intent,
                        "llm": l_intent,
                        "llm_correct": l_ok,
                        "heuristic_correct": h_ok,
                        "notes": notes,
                    }
                )

        # Row output
        h_mark = "OK" if h_ok else "FAIL"
        if llm_classifier is not None:
            l_mark = "OK" if l_ok else "FAIL"
            lat_str = f"{latency:.2f}s" if latency is not None else "n/a"
            print(
                f"  [{h_mark}|{l_mark}] heuristic={h_intent:<12} llm={l_intent:<12} "
                f"expected={expected:<12} lat={lat_str}  {query!r}"
            )
        else:
            print(f"  [{h_mark}] heuristic={h_intent:<12} expected={expected:<12}  {query!r}")

    # -----------------------------------------------------------------------
    # Accuracy summary
    # -----------------------------------------------------------------------
    _print_section("Accuracy by category")

    h_total_all = sum(heuristic_total.values())
    h_correct_all = sum(heuristic_correct.values())
    l_total_all = h_total_all  # same denominator
    l_correct_all = sum(llm_correct.values())

    col_w = 14
    print(f"  {'Category':<14}  {'Heuristic':>12}  {'LLM':>12}")
    print(f"  {'-' * 14}  {'-' * 12}  {'-' * 12}")
    for cat in categories:
        n = heuristic_total.get(cat, 0)
        h_acc = _accuracy(heuristic_correct.get(cat, 0), n)
        if llm_classifier is not None:
            l_acc = _accuracy(llm_correct.get(cat, 0), n)
        else:
            l_acc = "--"
        print(f"  {cat:<14}  {h_acc:>12}  {l_acc:>12}")

    print(f"  {'-' * 14}  {'-' * 12}  {'-' * 12}")
    h_overall = _accuracy(h_correct_all, h_total_all)
    l_overall = _accuracy(l_correct_all, l_total_all) if llm_classifier is not None else "--"
    print(f"  {'OVERALL':<14}  {h_overall:>12}  {l_overall:>12}")

    # -----------------------------------------------------------------------
    # Latency stats
    # -----------------------------------------------------------------------
    if llm_latencies:
        _print_section("LLM latency")
        p50 = _percentile(llm_latencies, 50)
        p90 = _percentile(llm_latencies, 90)
        mean = statistics.mean(llm_latencies)
        total_time = sum(llm_latencies)
        print(f"  mean={mean:.2f}s  p50={p50:.2f}s  p90={p90:.2f}s  total={total_time:.1f}s  n={len(llm_latencies)}")

    # -----------------------------------------------------------------------
    # Disagreements between heuristic and LLM
    # -----------------------------------------------------------------------
    if disagreements:
        _print_section(f"Heuristic vs LLM disagreements ({len(disagreements)})")
        for d in disagreements:
            winner = "llm" if d["llm_correct"] else ("heuristic" if d["heuristic_correct"] else "neither")
            print(
                f"  query:     {d['query']!r}\n"
                f"  expected:  {d['expected']}  heuristic={d['heuristic']}  llm={d['llm']}  winner={winner}\n"
                f"  notes:     {d['notes']}\n"
            )
    elif llm_classifier is not None:
        _print_section("Heuristic vs LLM disagreements")
        print("  (none -- perfect agreement)")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Eval LLM-based query intent classification against a labeled dataset.\n"
            "Runs heuristic baseline and (optionally) LLM classifier, reports accuracy + latency."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=f"Path to .jsonl dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM classifier, run heuristic baseline only",
    )
    args = parser.parse_args()

    run_eval(Path(args.dataset), run_llm=not args.no_llm)


if __name__ == "__main__":
    main()
