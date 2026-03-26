from __future__ import annotations

"""
Search assistant entrypoint.

Usage:
  uv run search-agent
  uv run search-agent -q "..." [-p web] [-S brave|ddgs]
  uv run search-agent --research
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.status import Status

from search_agent.runtime_bootstrap import ensure_utf8_stdio

load_dotenv()
ensure_utf8_stdio()
console = Console()


def require_api_key() -> str:
    from search_agent.settings import get_settings

    key = (get_settings().llm_api_key or "").strip()
    if not key:
        console.print(
            "[bold red]Error:[/] LLM_API_KEY is not set.\n"
            "Edit .env and add your key."
        )
        sys.exit(1)
    return key


def make_client(api_key: str):
    from openai import OpenAI

    from search_agent.settings import get_settings

    s = get_settings()
    return OpenAI(
        api_key=api_key,
        base_url=s.llm_base_url,
        default_headers={"HTTP-Referer": s.llm_http_referer},
    )


def run_single_query(
    query: str,
    use_case,
    receipts_dir: str | None = None,
    profile_name: str | None = None,
) -> None:
    from search_agent.config.profiles import get_profile

    console.print(Rule(f"[bold]Query:[/] {query}"))

    from search_agent.settings import get_settings

    name = (profile_name or get_settings().default_profile).strip()
    profile = get_profile(name)
    settings = get_settings()
    console.print(
        f"[dim]Profile:[/] [cyan]{profile.name}[/] - {profile.description} "
        f"[dim]· active search:[/] [cyan]{settings.resolved_search_provider()}[/]"
    )
    with Status("[cyan]Searching (claim-level agent)...[/]", console=console):
        report = use_case.run(query, profile=profile, receipts_dir=receipts_dir, log=console.print)

    console.print(f"\n[bold cyan]Claims ({len(report.claims)}):[/]")
    for run in report.claims:
        verdict = (
            run.evidence_bundle.verification.verdict
            if run.evidence_bundle and run.evidence_bundle.verification
            else "unknown"
        )
        source_count = run.evidence_bundle.independent_source_count if run.evidence_bundle else 0
        console.print(f"  - {run.claim.claim_text}")
        console.print(f"    [dim]verdict={verdict} | independent_sources={source_count}[/dim]")

    console.print()
    # Text() avoids Rich interpreting [1] citations as markup.
    console.print(Panel(Text(report.answer), title="[bold green]Answer[/]", border_style="green"))
    if report.audit_trail.receipt_path:
        console.print(f"\n[dim]Receipt: {report.audit_trail.receipt_path}[/dim]")


def run_research(client: OpenAI) -> None:
    from search_agent.infrastructure.llm import analyze_rag_papers
    from search_agent.infrastructure.arxiv_research import fetch_rag_research

    console.print(Rule("[bold magenta]arXiv Pipeline Research[/]"))
    with Status("[magenta]Fetching arXiv papers...[/]", console=console):
        papers = fetch_rag_research(max_per_query=3)
    console.print(f"[dim]Found {len(papers)} papers.[/]\n")
    for i, paper in enumerate(papers, 1):
        console.print(f"  [{i}] {paper['title']}")
        console.print(f"       [dim]{paper['url']}[/dim]")
    console.print()
    with Status("[magenta]Analysing with LLM...[/]", console=console):
        analysis = analyze_rag_papers(papers, client, log=console.print)
    console.print(Panel(
        Markdown(analysis),
        title="[bold magenta]Techniques worth implementing[/]",
        border_style="magenta",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Assistant Prototype")
    parser.add_argument("--query", "-q", help="Single query (non-interactive)")
    parser.add_argument(
        "--profile",
        "-p",
        default=None,
        help="Search profile name (default: DEFAULT_PROFILE or web)",
    )
    parser.add_argument("--research", action="store_true", help="Run arXiv research analysis")
    parser.add_argument("--eval", help="Run evaluation dataset JSONL")
    parser.add_argument(
        "--eval-no-save",
        action="store_true",
        help="Do not write eval JSON artifact (by default each eval run is saved for progress tracking)",
    )
    parser.add_argument(
        "--eval-out",
        metavar="PATH",
        default=None,
        help="Eval JSON path or directory (default: eval_runs/)",
    )
    parser.add_argument("--eval-label", default=None, help="Label stored in eval JSON metadata (e.g. branch or experiment)")
    parser.add_argument("--receipts-dir", help="Persist audit receipts to this directory")
    parser.add_argument(
        "--search-provider",
        "-S",
        choices=("brave", "ddgs"),
        default=None,
        metavar="NAME",
        help="Override search_provider for this run (sets SEARCH_PROVIDER_OVERRIDE; use if .env is fixed)",
    )
    args = parser.parse_args()

    if args.search_provider:
        os.environ["SEARCH_PROVIDER_OVERRIDE"] = args.search_provider
        from search_agent.bootstrap import build_search_agent_use_case
        from search_agent.settings import get_settings

        get_settings.cache_clear()
        build_search_agent_use_case.cache_clear()

    api_key = require_api_key()
    client = make_client(api_key)
    from search_agent import build_search_agent_use_case

    use_case = build_search_agent_use_case()

    if args.eval:
        from pathlib import Path

        from search_agent.evaluation import evaluate_dataset
        from search_agent.eval.tracking import DEFAULT_EVAL_RUNS_DIR, save_eval_run

        summary = evaluate_dataset(
            args.eval,
            receipts_dir=args.receipts_dir,
            log=console.print,
        )
        console.print(Panel(
            Markdown(
                "\n".join(
                    [
                        f"- case_count: {summary['case_count']}",
                        *(f"- {key}: {value}" for key, value in summary["metrics"].items()),
                    ]
                )
            ),
            title="[bold cyan]Evaluation[/]",
            border_style="cyan",
        ))
        if not args.eval_no_save:
            out: str | Path = args.eval_out if args.eval_out is not None else Path(DEFAULT_EVAL_RUNS_DIR)
            path = save_eval_run(summary, out, label=args.eval_label)
            console.print(f"[green]Eval saved:[/] {path}")
        return

    if args.research and not args.query:
        run_research(client)
        return

    if args.query:
        run_single_query(args.query, use_case, receipts_dir=args.receipts_dir, profile_name=args.profile)
        if args.research:
            console.print()
            run_research(client)
        return

    from search_agent.cli import SearchCLI

    SearchCLI(client, use_case).run()


if __name__ == "__main__":
    main()
