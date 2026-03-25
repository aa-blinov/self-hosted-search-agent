"""CLI: ``python -m search_agent.eval.compare_cli run_a.json run_b.json``."""

from __future__ import annotations

import argparse
import json

from search_agent.eval.tracking import compare_metric_deltas, load_eval_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two eval JSON artifacts (metrics delta).")
    parser.add_argument("previous", help="Older eval JSON")
    parser.add_argument("current", help="Newer eval JSON")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text",
    )
    args = parser.parse_args()

    a = load_eval_run(args.previous)
    b = load_eval_run(args.current)
    deltas = compare_metric_deltas(a, b)

    if args.json:
        print(json.dumps(deltas, indent=2, ensure_ascii=False))
        return

    ma = a.get("run_metadata") or {}
    mb = b.get("run_metadata") or {}
    print("Previous:", ma.get("created_at_utc"), ma.get("git_revision_short"), ma.get("llm_model"))
    print("Current: ", mb.get("created_at_utc"), mb.get("git_revision_short"), mb.get("llm_model"))
    print()
    for key, row in sorted(deltas.items()):
        prev, cur, d = row["previous"], row["current"], row["delta"]
        if prev is None or cur is None or d is None:
            print(f"  {key}: {prev!r} -> {cur!r}")
        else:
            sign = "+" if d >= 0 else ""
            print(f"  {key}: {prev:.4f} -> {cur:.4f} ({sign}{d:.4f})")


if __name__ == "__main__":
    main()
