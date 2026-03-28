from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from search_agent.eval.replay_eval import evaluate_replay_dataset
from search_agent.eval.tracking import (
    DEFAULT_EVAL_RUNS_DIR,
    _git_revision,
    compare_metric_deltas,
    save_eval_run,
)
from search_agent.evaluation import evaluate_dataset
from search_agent.runtime_bootstrap import ensure_utf8_stdio

ensure_utf8_stdio()

DEFAULT_KEY_METRICS = (
    "claim_support_rate",
    "citation_validity_rate",
    "unsupported_statement_rate",
    "contradiction_detection_rate",
    "insufficient_detection_rate",
    "route_match_rate",
    "primary_requirement_rate",
    "source_requirement_rate",
    "answer_depth_rate",
    "source_diversity_rate",
    "median_search_cost",
    "median_answer_latency",
)


def _claim_signature(claim: dict[str, Any]) -> tuple[Any, ...]:
    return (
        claim.get("match"),
        claim.get("actual_verdict"),
        claim.get("actual_route"),
        claim.get("has_primary_source"),
        claim.get("independent_source_count"),
    )


def find_case_differences(replay_summary: dict[str, Any], live_summary: dict[str, Any]) -> list[dict[str, Any]]:
    replay_cases = {
        case["case_id"]: case
        for case in (replay_summary.get("cases") or [])
    }
    live_cases = {
        case["case_id"]: case
        for case in (live_summary.get("cases") or [])
    }
    differences: list[dict[str, Any]] = []
    for case_id in sorted(set(replay_cases) & set(live_cases)):
        replay_case = replay_cases[case_id]
        live_case = live_cases[case_id]
        replay_claims = [_claim_signature(claim) for claim in replay_case.get("claims") or []]
        live_claims = [_claim_signature(claim) for claim in live_case.get("claims") or []]
        answer_chars_delta = int(live_case.get("answer_chars", 0)) - int(replay_case.get("answer_chars", 0))
        unique_sources_delta = int(live_case.get("unique_sources_in_answer", 0)) - int(
            replay_case.get("unique_sources_in_answer", 0)
        )
        backend_issue_changed = bool(live_case.get("backend_issue")) != bool(replay_case.get("backend_issue"))
        if (
            replay_claims != live_claims
            or answer_chars_delta != 0
            or unique_sources_delta != 0
            or backend_issue_changed
        ):
            differences.append(
                {
                    "case_id": case_id,
                    "split": live_case.get("split") or replay_case.get("split"),
                    "replay_claims": replay_case.get("claims") or [],
                    "live_claims": live_case.get("claims") or [],
                    "replay_answer_chars": replay_case.get("answer_chars", 0),
                    "live_answer_chars": live_case.get("answer_chars", 0),
                    "replay_unique_sources": replay_case.get("unique_sources_in_answer", 0),
                    "live_unique_sources": live_case.get("unique_sources_in_answer", 0),
                    "replay_backend_issue": bool(replay_case.get("backend_issue")),
                    "live_backend_issue": bool(live_case.get("backend_issue")),
                }
            )
    return differences


def build_paired_control_report(
    replay_summary: dict[str, Any],
    live_summary: dict[str, Any],
    *,
    replay_artifact: str | None = None,
    live_artifact: str | None = None,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_revision": _git_revision(),
        "git_revision_short": ((_git_revision() or "nogit")[:8]),
        "replay_dataset_path": replay_summary.get("dataset_path"),
        "live_dataset_path": live_summary.get("dataset_path"),
        "replay_artifact_path": replay_artifact,
        "live_artifact_path": live_artifact,
        "replay_metrics": replay_summary.get("metrics") or {},
        "live_metrics": live_summary.get("metrics") or {},
        "metric_deltas": compare_metric_deltas(
            replay_summary,
            live_summary,
            metric_keys=DEFAULT_KEY_METRICS,
        ),
        "case_differences": find_case_differences(replay_summary, live_summary),
    }


def save_paired_control_report(
    report: dict[str, Any],
    output: str | Path,
) -> Path:
    out = Path(output)
    if out.is_dir() or str(output).endswith(("/", "\\")):
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        short = ((_git_revision() or "nogit")[:8])
        out = out / f"paired_control_{ts}_{short}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _print_metric_rows(report: dict[str, Any]) -> None:
    print("Replay vs live control")
    print()
    for key in DEFAULT_KEY_METRICS:
        row = report["metric_deltas"].get(key) or {}
        replay_value = row.get("previous")
        live_value = row.get("current")
        delta = row.get("delta")
        if replay_value is None or live_value is None or delta is None:
            continue
        sign = "+" if delta >= 0 else ""
        print(f"  {key}: replay={replay_value:.4f} live={live_value:.4f} ({sign}{delta:.4f})")
    print()
    if report["case_differences"]:
        print("Case-level differences:")
        for item in report["case_differences"]:
            print(
                "  "
                f"{item['case_id']}: "
                f"chars {item['replay_answer_chars']} -> {item['live_answer_chars']}, "
                f"sources {item['replay_unique_sources']} -> {item['live_unique_sources']}, "
                f"backend_issue={item['live_backend_issue']}"
            )
    else:
        print("Case-level differences: none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run replay_control and live control side by side.")
    parser.add_argument(
        "--replay-dataset",
        default="eval_data/replay_control.jsonl",
        help="Replay dataset path",
    )
    parser.add_argument(
        "--live-dataset",
        default="eval_data/control_dataset.jsonl",
        help="Live evaluation dataset path",
    )
    parser.add_argument(
        "--search-provider",
        choices=("brave", "ddgs"),
        default=None,
        help="Override search provider for the live run",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_EVAL_RUNS_DIR,
        help="Directory or JSON path for the combined paired-control report",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save replay/live artifacts or the combined report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the combined report as JSON",
    )
    args = parser.parse_args()

    if args.search_provider:
        os.environ["SEARCH_PROVIDER_OVERRIDE"] = args.search_provider
        from search_agent.bootstrap import build_search_agent_use_case
        from search_agent.settings import get_settings

        get_settings.cache_clear()
        build_search_agent_use_case.cache_clear()

    replay_summary = evaluate_replay_dataset(args.replay_dataset, log=print)
    live_summary = evaluate_dataset(args.live_dataset, log=print)

    replay_artifact: str | None = None
    live_artifact: str | None = None
    combined_artifact: str | None = None
    if not args.no_save:
        replay_path = save_eval_run(replay_summary, Path(DEFAULT_EVAL_RUNS_DIR), label="paired-replay-control")
        live_path = save_eval_run(live_summary, Path(DEFAULT_EVAL_RUNS_DIR), label="paired-live-control")
        replay_artifact = replay_path.as_posix()
        live_artifact = live_path.as_posix()

    report = build_paired_control_report(
        replay_summary,
        live_summary,
        replay_artifact=replay_artifact,
        live_artifact=live_artifact,
    )

    if not args.no_save:
        combined_path = save_paired_control_report(report, args.out)
        combined_artifact = combined_path.as_posix()
        report["paired_artifact_path"] = combined_artifact

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_metric_rows(report)
        if replay_artifact:
            print()
            print(f"Replay artifact: {replay_artifact}")
            print(f"Live artifact:   {live_artifact}")
            print(f"Paired report:   {combined_artifact}")


if __name__ == "__main__":
    main()
