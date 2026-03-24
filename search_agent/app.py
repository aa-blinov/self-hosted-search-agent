from __future__ import annotations

"""
Search assistant entrypoint.

Usage:
  uv run search-agent
  uv run search-agent -q "..."
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
from rich.rule import Rule
from rich.status import Status

load_dotenv()
console = Console()


def require_api_key() -> str:
    key = os.getenv("LLM_API_KEY", "").strip()
    if not key:
        console.print(
            "[bold red]Error:[/] LLM_API_KEY is not set.\n"
            "Edit .env and add your key."
        )
        sys.exit(1)
    return key


def make_client(api_key: str):
    base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"HTTP-Referer": "https://github.com/local/search-agent"},
    )


def run_single_query(query: str, use_case, receipts_dir: str | None = None) -> None:
    from search_agent.config.profiles import get_profile

    console.print(Rule(f"[bold]Query:[/] {query}"))

    profile = get_profile(os.getenv("DEFAULT_PROFILE", "web"))
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
        console.print(f"    [dim]verdict={verdict} · independent_sources={source_count}[/dim]")

    console.print()
    console.print(Panel(Markdown(report.answer), title="[bold green]Answer[/]", border_style="green"))
    if report.audit_trail.receipt_path:
        console.print(f"\n[dim]Receipt: {report.audit_trail.receipt_path}[/dim]")


def run_research(client: OpenAI) -> None:
    from search_agent.infrastructure.llm import analyze_rag_papers
    from search_agent.infrastructure.searxng import fetch_rag_research

    console.print(Rule("[bold magenta]arXiv Pipeline Research[/]"))
    with Status("[magenta]Fetching arXiv papers...[/]", console=console):
        papers = fetch_rag_research(max_per_query=3)
    console.print(f"[dim]Found {len(papers)} papers.[/]\n")
    for i, paper in enumerate(papers, 1):
        console.print(f"  [{i}] {paper['title']}")
        console.print(f"       [dim]{paper['url']}[/dim]")
    console.print()
    with Status("[magenta]Analysing with LLM...[/]", console=console):
        analysis = analyze_rag_papers(papers, client)
    console.print(Panel(
        Markdown(analysis),
        title="[bold magenta]Techniques worth implementing[/]",
        border_style="magenta",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Assistant Prototype")
    parser.add_argument("--query", "-q", help="Single query (non-interactive, uses 'web' profile)")
    parser.add_argument("--research", action="store_true", help="Run arXiv research analysis")
    parser.add_argument("--eval", help="Run evaluation dataset JSONL")
    parser.add_argument("--receipts-dir", help="Persist audit receipts to this directory")
    args = parser.parse_args()

    require_api_key()
    client = make_client(os.getenv("LLM_API_KEY", "").strip())
    from search_agent import build_search_agent_use_case

    use_case = build_search_agent_use_case()

    if args.eval:
        from search_agent.evaluation import evaluate_dataset

        with Status("[cyan]Running evaluation...[/]", console=console):
            summary = evaluate_dataset(args.eval, client=client, receipts_dir=args.receipts_dir)
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
        return

    if args.research and not args.query:
        run_research(client)
        return

    if args.query:
        run_single_query(args.query, use_case, receipts_dir=args.receipts_dir)
        if args.research:
            console.print()
            run_research(client)
        return

    from search_agent.cli import SearchCLI

    SearchCLI(client, use_case).run()


if __name__ == "__main__":
    main()
