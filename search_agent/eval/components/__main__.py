"""CLI: uv run python -m search_agent.eval.components [component] [options]

Examples:
    # All pure-Python components (no API key needed)
    uv run python -m search_agent.eval.components --all --no-llm

    # Single component
    uv run python -m search_agent.eval.components gate_serp
    uv run python -m search_agent.eval.components verify_claim

    # All components (LLM + pure-Python)
    uv run python -m search_agent.eval.components --all

    # Custom dataset path
    uv run python -m search_agent.eval.components route_claim --dataset my_cases.jsonl

    # Compare two component runs (uses existing compare_cli)
    uv run python -m search_agent.eval eval_runs/components/A.json eval_runs/components/B.json
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

# Component module registry
PURE_PYTHON_COMPONENTS = ["gate_serp", "cheap_passage", "route_claim"]
LLM_COMPONENTS = ["classify_intent", "verify_claim", "synthesize_answer"]
ALL_COMPONENTS = PURE_PYTHON_COMPONENTS + LLM_COMPONENTS


def _import_module(name: str):
    if name == "gate_serp":
        from search_agent.eval.components import gate_serp as m
    elif name == "cheap_passage":
        from search_agent.eval.components import cheap_passage as m
    elif name == "route_claim":
        from search_agent.eval.components import route_claim as m
    elif name == "classify_intent":
        from search_agent.eval.components import classify_intent as m
    elif name == "verify_claim":
        from search_agent.eval.components import verify_claim as m
    elif name == "synthesize_answer":
        from search_agent.eval.components import synthesize_answer as m
    else:
        raise ValueError(f"Unknown component: {name!r}")
    return m


def _print_summary(summary, *, verbose: bool = False) -> None:
    icon = "[OK]" if summary.pass_rate >= 1.0 else ("[WARN]" if summary.pass_rate >= 0.5 else "[FAIL]")
    print(f"\n{icon} [{summary.component}]  pass_rate={summary.pass_rate:.2f}  "
          f"cases={summary.case_count}")
    for k, v in summary.metrics.items():
        if k == "per_class" or k == "per_mode":
            continue  # print separately
        if v is None:
            continue
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    # per-class breakdown
    for key in ("per_class", "per_mode"):
        pc = summary.metrics.get(key)
        if pc:
            print(f"   {key}:")
            for cls, vals in pc.items():
                parts = " ".join(f"{m}={vals[m]:.2f}" for m in ("precision", "recall", "f1"))
                print(f"     {cls}: {parts}")

    if verbose:
        print()
        for c in summary.cases:
            icon2 = "PASS" if c.passed else "FAIL"
            print(f"  [{icon2}] {c.case_id}  {c.latency_ms}ms")
            for k, v in c.details.items():
                if k in ("answer_preview",):
                    continue
                print(f"      {k}: {v}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run component-level evals for the search agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            components:
              Pure-Python (no API key): gate_serp, cheap_passage, route_claim
              LLM-based (needs .env):   classify_intent, verify_claim, synthesize_answer
        """),
    )
    parser.add_argument(
        "component",
        nargs="?",
        choices=ALL_COMPONENTS,
        help="Component to evaluate (omit when using --all)",
    )
    parser.add_argument("--all", action="store_true", help="Run all components")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM-dependent components (safe without .env)",
    )
    parser.add_argument("--dataset", type=Path, help="Override dataset path for a single component")
    parser.add_argument("--output-dir", type=Path, help="Override output directory for artifacts")
    parser.add_argument("--label", help="Human label for artifact metadata")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary to stdout")
    parser.add_argument("--no-save", action="store_true", help="Do not write artifact to disk")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-case details")
    args = parser.parse_args()

    if not args.all and not args.component:
        parser.print_help()
        return 2

    components_to_run: list[str]
    if args.all:
        components_to_run = PURE_PYTHON_COMPONENTS if args.no_llm else ALL_COMPONENTS
    else:
        if args.no_llm and args.component in LLM_COMPONENTS:
            print(f"Skipping LLM component {args.component!r} (--no-llm)")
            return 0
        components_to_run = [args.component]

    # Load settings once for LLM components
    settings = None
    if not args.no_llm and any(c in LLM_COMPONENTS for c in components_to_run):
        try:
            from search_agent.settings import get_settings
            settings = get_settings()
        except Exception as exc:
            print(f"Warning: could not load settings ({exc}). LLM components will fail.")

    from search_agent.eval.components.runner import load_cases, save_component_run, DEFAULT_DATASETS_DIR

    all_summaries = []
    any_failed = False
    t_total = time.perf_counter()

    for name in components_to_run:
        mod = _import_module(name)
        dataset_path = args.dataset if (args.dataset and args.component == name) else mod.DEFAULT_DATASET
        print(f"\n>> {name}  [{dataset_path}]")

        try:
            cases = load_cases(dataset_path)
            t0 = time.perf_counter()
            summary = mod.run_component_eval(
                cases,
                dataset_path=str(dataset_path),
                settings=settings,
            )
            elapsed = time.perf_counter() - t0
            print(f"  Completed in {elapsed:.2f}s")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            any_failed = True
            continue

        _print_summary(summary, verbose=args.verbose)
        all_summaries.append(summary)

        if summary.pass_rate < 1.0:
            any_failed = True

        if not args.no_save:
            try:
                artifact = save_component_run(
                    summary,
                    output_dir=args.output_dir,
                    label=args.label,
                )
                print(f"  Saved: {artifact}")
            except Exception as exc:
                print(f"  Warning: could not save artifact ({exc})")

    total_elapsed = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"Total: {len(all_summaries)}/{len(components_to_run)} components  "
          f"in {total_elapsed:.2f}s")

    if args.json:
        payload = [
            {
                "component": s.component,
                "pass_rate": s.pass_rate,
                "case_count": s.case_count,
                "metrics": s.metrics,
            }
            for s in all_summaries
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
