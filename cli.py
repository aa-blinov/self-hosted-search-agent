"""
Interactive CLI for the search agent.

Flow:
  1. User enters a query.
  2. CLI suggests a search profile.
  3. Agent runs claim-level search and verification.
  4. CLI renders verdicts and the grounded answer.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status

from agent import run_search_agent
from extractor import CRAWL4AI_TIMEOUT, EXTRACT_MAX_CHARS, shutdown as shutdown_crawler
from llm import analyze_rag_papers
from profiles import DEFAULT_PROFILE, PROFILES, SearchProfile, get_profile, list_profiles
from search import fetch_rag_research

console = Console()

_HISTORY_FILE = Path.home() / ".search_agent_history"
_COMMANDS = ["/help", "/config", "/profiles", "/research", "/clear", "/quit", "/exit", "/q"]
_STYLE = Style.from_dict({
    "prompt.bracket": "#888888",
    "prompt.arrow": "#ffcc00 bold",
})

_NEWS_KW = {
    "новости", "новость", "news", "latest", "today", "breaking",
    "вчера", "сегодня", "прямо сейчас", "last hour",
}
_SCI_KW = {
    "paper", "research", "study", "arxiv", "статья", "исследование",
    "научн", "pubmed", "journal", "abstract",
}
_TECH_KW = {
    "code", "error", "bug", "programming", "python", "javascript",
    "docker", "api", "github", "bash", "linux", "typescript", "npm",
    "kubectl", "terraform", "sql", "rust", "golang",
}
_TECH_QA_KW = {
    "error", "exception", "traceback", "stack trace", "fails", "failing",
    "bug", "why does", "how fix", "segfault", "crash", "not working",
}
_TECH_REPO_KW = {
    "github", "gitlab", "repo", "repository", "package", "library",
    "module", "crate", "npm", "pip", "pypi", "docker image",
}
_FACTUAL_RELEASE_KW = {
    "when was", "release date", "released", "announcement", "version",
    "date", "launch", "changelog", "what is", "who is",
}


def _suggest_profile(query: str) -> str:
    lowered = query.lower()
    has_cyrillic = bool(re.search(r"[а-яёА-ЯЁ]", query))

    if any(keyword in lowered for keyword in _NEWS_KW):
        return "ru_news" if has_cyrillic else "news"
    if any(keyword in lowered for keyword in _SCI_KW):
        return "science"
    if any(keyword in lowered for keyword in _TECH_QA_KW):
        return "it_qa"
    if any(keyword in lowered for keyword in _TECH_REPO_KW):
        return "it_repos"
    if any(keyword in lowered for keyword in _TECH_KW) and any(keyword in lowered for keyword in _FACTUAL_RELEASE_KW):
        return "web"
    if any(keyword in lowered for keyword in _TECH_KW):
        return "tech"
    if has_cyrillic:
        return "ru"
    return "web"


def _select_profile(query: str) -> SearchProfile:
    suggested_name = _suggest_profile(query)
    profile_items = list(PROFILES.items())
    suggested_idx = next((i for i, (name, _) in enumerate(profile_items, 1) if name == suggested_name), 1)

    console.print()
    console.print("[dim]  Search profile:[/dim]")
    console.print()
    for i, (name, profile) in enumerate(profile_items, 1):
        tag = " [bold cyan]< suggested[/bold cyan]" if name == suggested_name else ""
        console.print(
            f"  [bold white]{i}[/bold white]  [cyan]{name:<12}[/cyan] "
            f"[dim]{profile.description}[/dim]{tag}"
        )
    console.print()

    raw = console.input(
        f"  [dim]Profile [[/dim][bold cyan]{suggested_idx}[/bold cyan][dim]]: [/dim]"
    ).strip()

    if not raw:
        return get_profile(suggested_name)
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(profile_items):
            return profile_items[idx][1]
        console.print("[yellow]Invalid number, using suggestion.[/yellow]")
        return get_profile(suggested_name)
    try:
        return get_profile(raw.lower())
    except ValueError:
        console.print(f"[yellow]Unknown profile '{raw}', using suggestion.[/yellow]")
        return get_profile(suggested_name)


class SearchCLI:
    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self._last_profile_name = os.getenv("DEFAULT_PROFILE", DEFAULT_PROFILE)

    def _prompt(self):
        return HTML(
            "<ansibrightblack>[</ansibrightblack>"
            "search"
            "<ansibrightblack>]</ansibrightblack>"
            "<ansiyellow><b> > </b></ansiyellow>"
        )

    def _check_llm(self) -> bool:
        from llm import MODEL, _extra

        try:
            self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                extra_body=_extra(),
            )
            return True
        except Exception as exc:
            console.print(f"[dim red]LLM check failed: {exc}[/dim red]")
            return False

    def _run_query(self, query: str) -> None:
        profile = _select_profile(query)
        self._last_profile_name = profile.name

        console.print(Rule(f"[bold]{query}[/]"))
        console.print(
            f"[dim]Profile: {profile.name} · "
            f"categories: {'+'.join(profile.categories)} · "
            f"lang: {profile.language} · "
            f"fetch_top: {profile.fetch_top_n}[/dim]"
        )

        report = run_search_agent(
            query,
            profile=profile,
            client=self.client,
            receipts_dir=os.getenv("AGENT_RECEIPTS_DIR", "").strip() or None,
            log=console.print,
        )

        console.print(f"\n[bold cyan]Claims ({len(report.claims)}):[/]")
        for run in report.claims:
            verdict = (
                run.evidence_bundle.verification.verdict
                if run.evidence_bundle and run.evidence_bundle.verification
                else "unknown"
            )
            source_count = run.evidence_bundle.independent_source_count if run.evidence_bundle else 0
            console.print(f"  [[cyan]{run.claim.claim_id}[/]] {run.claim.claim_text}")
            console.print(f"       [dim]verdict={verdict} · independent_sources={source_count}[/dim]")

        console.print()
        console.print(Panel(
            Markdown(report.answer),
            title="[bold green]Answer[/]",
            border_style="green",
            width=min(console.width, 100),
        ))
        if report.audit_trail.receipt_path:
            console.print(f"[dim]Receipt: {report.audit_trail.receipt_path}[/dim]")

    def _cmd_help(self, _: str) -> None:
        console.print(Panel(
            "[bold cyan]/help[/]           Show this help\n"
            "[bold cyan]/config[/]         Show current settings\n"
            "[bold cyan]/profiles[/]       List all search profiles\n"
            "[bold cyan]/research[/]       Fetch arXiv papers and analyse pipeline\n"
            "[bold cyan]/clear[/]          Clear screen\n"
            "[bold cyan]/quit[/]           Exit\n\n"
            "[dim]Type any question to search.\n"
            "Profile is selected after each query.\n"
            "Tab completes commands. Up/down navigates history.[/dim]",
            title="[bold]Commands[/]",
            border_style="blue",
        ))

    def _cmd_config(self, _: str) -> None:
        console.print(Panel(
            f"Model         : [cyan]{os.getenv('LLM_MODEL', 'qwen/qwen3.5-35b-a3b')}[/]\n"
            f"Provider      : [cyan]{os.getenv('LLM_PROVIDER') or 'any (no routing)'}[/]\n"
            f"Search        : [cyan]SearXNG[/] @ [dim]{os.getenv('SEARXNG_URL', 'http://localhost:8888')}[/dim]\n"
            f"Extractor     : [cyan]crawl4ai[/] (timeout: {CRAWL4AI_TIMEOUT}s)\n"
            f"Extract limit : [cyan]{EXTRACT_MAX_CHARS}[/] chars\n"
            f"Agent fetch   : [cyan]{os.getenv('AGENT_FETCH_TOP_N', '4')}[/] deep docs per claim (min budget)\n"
            f"SERP gate     : [cyan]{os.getenv('SERP_GATE_MIN_URLS', '15')}..{os.getenv('SERP_GATE_MAX_URLS', '30')}[/] URLs\n\n"
            f"Receipts dir  : [cyan]{os.getenv('AGENT_RECEIPTS_DIR') or 'disabled'}[/]\n\n"
            f"[dim]Last profile used: {self._last_profile_name}[/dim]",
            title="[bold]Config[/]",
            border_style="blue",
            width=min(console.width, 80),
        ))

    def _cmd_profiles(self, _: str) -> None:
        console.print("\n[bold]Available profiles:[/]")
        console.print(list_profiles())
        console.print()

    def _cmd_research(self, _: str) -> None:
        console.print(Rule("[bold magenta]arXiv Pipeline Research[/]"))
        with Status("[magenta]Fetching arXiv papers...[/]", console=console):
            papers = fetch_rag_research(max_per_query=3)
        console.print(f"[dim]Found {len(papers)} papers.[/]\n")
        for i, paper in enumerate(papers, 1):
            console.print(f"  [{i}] {paper['title']}")
            console.print(f"       [dim]{paper['url']}[/dim]")
        console.print()
        with Status("[magenta]Analysing...[/]", console=console):
            analysis = analyze_rag_papers(papers, self.client)
        console.print(Panel(
            Markdown(analysis),
            title="[bold magenta]Techniques worth implementing[/]",
            border_style="magenta",
            width=min(console.width, 100),
        ))

    def _cmd_clear(self, _: str) -> None:
        console.clear()

    def _cmd_quit(self, _: str) -> None:
        raise SystemExit(0)

    _COMMAND_MAP = {
        "/help": _cmd_help,
        "/config": _cmd_config,
        "/profiles": _cmd_profiles,
        "/research": _cmd_research,
        "/clear": _cmd_clear,
        "/quit": _cmd_quit,
        "/exit": _cmd_quit,
        "/q": _cmd_quit,
    }

    def _dispatch(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        handler = self._COMMAND_MAP.get(cmd)
        if handler:
            handler(self, parts[1] if len(parts) > 1 else "")
        else:
            console.print(f"[red]Unknown command '{cmd}'.[/] Type [cyan]/help[/].")

    def run(self) -> None:
        session: PromptSession = PromptSession(
            history=FileHistory(str(_HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(_COMMANDS, sentence=True),
            complete_while_typing=False,
            style=_STYLE,
        )

        llm_ok = self._check_llm()
        model_line = (
            f"Model  : [cyan]{os.getenv('LLM_MODEL', 'qwen/qwen3.5-35b-a3b')}[/]"
            + (f" via [cyan]{os.getenv('LLM_PROVIDER')}[/]" if os.getenv("LLM_PROVIDER") else "")
            + ("  [green]OK[/]" if llm_ok else "  [red]unreachable[/]")
        )
        console.print(Panel(
            "[bold]Search Assistant[/]\n"
            f"{model_line}\n\n"
            "[dim]Type a question, choose a profile, and get a claim-verified answer.\n"
            "Tab completes commands (/help, /config, /profiles, /research, /quit).[/dim]",
            border_style="blue" if llm_ok else "red",
        ))

        while True:
            try:
                line = session.prompt(self._prompt()).strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Shutting down browser...[/]")
                shutdown_crawler()
                console.print("[dim]Bye.[/]")
                import os as _os
                _os._exit(0)

            if not line:
                continue

            if line.startswith("/"):
                try:
                    self._dispatch(line)
                except SystemExit:
                    console.print("[dim]Bye.[/]")
                    shutdown_crawler()
                    import os as _os
                    _os._exit(0)
            else:
                try:
                    self._run_query(line)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted.[/]")

            console.print()
