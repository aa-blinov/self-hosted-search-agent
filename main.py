#!/usr/bin/env python3
"""
Search Assistant Prototype.

Usage:
  uv run python main.py              # interactive CLI (recommended)
  uv run python main.py -q "..."     # single query, auto-selects 'web' profile
  uv run python main.py --research   # arXiv analysis only
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status

load_dotenv()
console = Console()


def require_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        console.print(
            "[bold red]Error:[/] OPENROUTER_API_KEY is not set.\n"
            "Edit .env and add your key."
        )
        sys.exit(1)
    return key


def make_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "https://github.com/local/search-agent"},
    )


def run_single_query(query: str, client: OpenAI) -> None:
    from evidence_loop import evidence_loop
    from llm import answer_with_sources
    from profiles import get_profile
    from search import search_web

    console.print(Rule(f"[bold]Query:[/] {query}"))

    profile = get_profile(os.getenv("DEFAULT_PROFILE", "web"))
    with Status("[cyan]Searching (EvidenceLoop)…[/]", console=console):
        sources, search_log = evidence_loop(
            query, search_fn=search_web, profile=profile, client=client, verbose=False
        )

    if len(search_log) > 1:
        console.print(
            f"\n[dim]EvidenceLoop: {len(search_log)} queries — "
            + ", ".join(f'"{q}"' for q in search_log) + "[/dim]"
        )

    console.print(f"\n[bold cyan]Sources ({len(sources)}):[/]")
    for i, s in enumerate(sources, 1):
        chars = len(s.get("snippet", ""))
        console.print(f"  [{i}] {s.get('title', '—')} [dim]({chars} chars)[/dim]")
        console.print(f"       [dim]{s.get('url', '')}[/dim]")

    console.print()
    with Status("[cyan]Generating grounded answer…[/]", console=console):
        answer = answer_with_sources(query, sources, client)

    console.print(Panel(Markdown(answer), title="[bold green]Answer[/]", border_style="green"))


def run_research(client: OpenAI) -> None:
    from llm import analyze_rag_papers
    from search import fetch_rag_research

    console.print(Rule("[bold magenta]arXiv Pipeline Research[/]"))
    with Status("[magenta]Fetching arXiv papers…[/]", console=console):
        papers = fetch_rag_research(max_per_query=3)
    console.print(f"[dim]Found {len(papers)} papers.[/]\n")
    for i, p in enumerate(papers, 1):
        console.print(f"  [{i}] {p['title']}")
        console.print(f"       [dim]{p['url']}[/dim]")
    console.print()
    with Status("[magenta]Analysing with LLM…[/]", console=console):
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
    args = parser.parse_args()

    client = make_client(require_api_key())

    if args.research and not args.query:
        run_research(client)
        return

    if args.query:
        run_single_query(args.query, client)
        if args.research:
            console.print()
            run_research(client)
        return

    # Interactive mode — profile selected per query
    from cli import SearchCLI
    SearchCLI(client).run()


if __name__ == "__main__":
    main()
