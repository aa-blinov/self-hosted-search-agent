"""
Interactive CLI for the search agent.

Flow:
  1. User types a query → Enter
  2. Profile selector appears (auto-suggested by query language/keywords)
  3. User picks a number (or Enter = suggestion)
  4. Search + EvidenceLoop + LLM answer

Commands (start with /):
  /help               — show this help
  /config             — show current settings
  /profiles           — list all available profiles
  /research           — fetch arXiv papers → analyse pipeline improvements
  /clear              — clear screen
  /quit               — exit
"""

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

from evidence_loop import evidence_loop
from extractor import shutdown as shutdown_crawler
from llm import analyze_rag_papers, answer_streaming
from profiles import DEFAULT_PROFILE, PROFILES, SearchProfile, get_profile, list_profiles
from search import fetch_rag_research, search_web

console = Console()

_HISTORY_FILE = Path.home() / ".search_agent_history"

_COMMANDS = [
    "/help", "/config", "/profiles", "/research", "/clear", "/quit", "/exit", "/q",
]

_STYLE = Style.from_dict({
    "prompt.bracket": "#888888",
    "prompt.arrow": "#ffcc00 bold",
})

# ── Profile auto-suggestion ──────────────────────────────────────────────────

_NEWS_KW  = {"новости", "новость", "news", "latest", "today", "breaking",
             "вчера", "сегодня", "прямо сейчас", "last hour"}
_SCI_KW   = {"paper", "research", "study", "arxiv", "статья", "исследование",
             "научн", "pubmed", "journal", "abstract"}
_TECH_KW  = {"code", "error", "bug", "programming", "python", "javascript",
             "docker", "api", "github", "bash", "linux", "typescript", "npm",
             "kubectl", "terraform", "sql", "rust", "golang"}


def _suggest_profile(query: str) -> str:
    q_low = query.lower()
    has_cyrillic = bool(re.search(r"[а-яёА-ЯЁ]", query))

    if any(kw in q_low for kw in _NEWS_KW):
        return "ru_news" if has_cyrillic else "news"
    if any(kw in q_low for kw in _SCI_KW):
        return "science"
    if any(kw in q_low for kw in _TECH_KW):
        return "tech"
    if has_cyrillic:
        return "ru"
    return "web"


# ── Profile selector ─────────────────────────────────────────────────────────

def _select_profile(query: str) -> SearchProfile:
    """Print numbered profile menu, return chosen SearchProfile."""
    suggested_name = _suggest_profile(query)
    profile_items = list(PROFILES.items())   # stable order

    # find suggested index (1-based)
    suggested_idx = next(
        (i for i, (n, _) in enumerate(profile_items, 1) if n == suggested_name), 1
    )

    console.print()
    console.print("[dim]  Search profile:[/dim]")
    console.print()
    for i, (name, p) in enumerate(profile_items, 1):
        tag = " [bold cyan]← suggested[/bold cyan]" if name == suggested_name else ""
        console.print(
            f"  [bold white]{i}[/bold white]  [cyan]{name:<12}[/cyan] "
            f"[dim]{p.description}[/dim]{tag}"
        )
    console.print()

    raw = console.input(
        f"  [dim]Profile [[/dim][bold cyan]{suggested_idx}[/bold cyan][dim]]: [/dim]"
    ).strip()

    if not raw:
        return get_profile(suggested_name)

    # numeric choice
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(profile_items):
            return profile_items[idx][1]
        console.print("[yellow]Invalid number, using suggestion.[/yellow]")
        return get_profile(suggested_name)

    # name typed directly
    try:
        return get_profile(raw.lower())
    except ValueError:
        console.print(f"[yellow]Unknown profile '{raw}', using suggestion.[/yellow]")
        return get_profile(suggested_name)


# ── SearchCLI ────────────────────────────────────────────────────────────────

class SearchCLI:
    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self._last_profile_name = os.getenv("DEFAULT_PROFILE", DEFAULT_PROFILE)

    def _prompt(self):
        return HTML(
            f'<ansibrightblack>[</ansibrightblack>'
            f'search'
            f'<ansibrightblack>]</ansibrightblack>'
            f'<ansiyellow><b> › </b></ansiyellow>'
        )

    # ── Query flow ────────────────────────────────────────────────────────

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

        sources, search_log = evidence_loop(
            query,
            search_fn=search_web,
            profile=profile,
            client=self.client,
            log=console.print,
        )

        if sources:
            console.print(f"\n[bold cyan]Sources ({len(sources)}):[/]")
            for i, s in enumerate(sources, 1):
                chars = len(s.get("snippet", ""))
                console.print(
                    f"  [[cyan]{i}[/]] {s.get('title', '—')} [dim]({chars} chars)[/dim]"
                )
                console.print(f"       [dim]{s.get('url', '')}[/dim]")
        else:
            console.print("[yellow]No sources found.[/]")

        console.print()
        console.print("[dim]▸ Generating grounded answer…[/dim]")
        answer_streaming(
            query, sources, self.client,
            console=console,
            panel_width=min(console.width, 100),
        )

    # ── Commands ──────────────────────────────────────────────────────────

    def _cmd_help(self, _: str) -> None:
        console.print(Panel(
            "[bold cyan]/help[/]           Show this help\n"
            "[bold cyan]/config[/]         Show current settings\n"
            "[bold cyan]/profiles[/]       List all search profiles\n"
            "[bold cyan]/research[/]       Fetch arXiv papers → analyse pipeline\n"
            "[bold cyan]/clear[/]          Clear screen\n"
            "[bold cyan]/quit[/]           Exit\n\n"
            "[dim]Type any question to search.\n"
            "Profile is selected [bold]after[/bold] each query.\n"
            "Tab completes commands. ↑↓ navigates history.[/dim]",
            title="[bold]Commands[/]", border_style="blue",
        ))

    def _cmd_config(self, _: str) -> None:
        from evidence_loop import MAX_ITERATIONS, MIN_CONTEXT_CHARS
        from extractor import CRAWL4AI_TIMEOUT, EXTRACT_MAX_CHARS
        console.print(Panel(
            f"Model         : [cyan]{os.getenv('OPENROUTER_MODEL', 'qwen/qwen3.5-35b-a3b')}[/]\n"
            f"Provider      : [cyan]Alibaba (no fallback)[/]\n"
            f"Thinking      : [cyan]off[/]\n"
            f"Search        : [cyan]SearXNG[/] @ [dim]{os.getenv('SEARXNG_URL', 'http://localhost:8888')}[/dim]\n"
            f"Extractor     : [cyan]crawl4ai[/] (timeout: {CRAWL4AI_TIMEOUT}s)\n"
            f"Extract limit : [cyan]{EXTRACT_MAX_CHARS}[/] chars\n"
            f"EvidenceLoop  : [cyan]on[/] [dim](max {MAX_ITERATIONS} iter, min {MIN_CONTEXT_CHARS} chars)[/dim]\n\n"
            f"[dim]Last profile used: {self._last_profile_name}[/dim]",
            title="[bold]Config[/]", border_style="blue",
            width=min(console.width, 80),
        ))

    def _cmd_profiles(self, _: str) -> None:
        console.print("\n[bold]Available profiles:[/]")
        console.print(list_profiles())
        console.print()

    def _cmd_research(self, _: str) -> None:
        console.print(Rule("[bold magenta]arXiv Pipeline Research[/]"))
        with Status("[magenta]Fetching arXiv papers…[/]", console=console):
            papers = fetch_rag_research(max_per_query=3)
        console.print(f"[dim]Found {len(papers)} papers.[/]\n")
        for i, p in enumerate(papers, 1):
            console.print(f"  [{i}] {p['title']}")
            console.print(f"       [dim]{p['url']}[/dim]")
        console.print()
        with Status("[magenta]Analysing…[/]", console=console):
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
        "/help":     _cmd_help,
        "/config":   _cmd_config,
        "/profiles": _cmd_profiles,
        "/research": _cmd_research,
        "/clear":    _cmd_clear,
        "/quit":     _cmd_quit,
        "/exit":     _cmd_quit,
        "/q":        _cmd_quit,
    }

    def _dispatch(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        handler = self._COMMAND_MAP.get(cmd)
        if handler:
            handler(self, parts[1] if len(parts) > 1 else "")
        else:
            console.print(f"[red]Unknown command '{cmd}'.[/] Type [cyan]/help[/].")

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        session: PromptSession = PromptSession(
            history=FileHistory(str(_HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(_COMMANDS, sentence=True),
            complete_while_typing=False,
            style=_STYLE,
        )

        console.print(Panel(
            f"[bold]Search Assistant[/]\n"
            f"Model  : [cyan]{os.getenv('OPENROUTER_MODEL', 'qwen/qwen3.5-35b-a3b')}[/] via Alibaba\n\n"
            "[dim]Type a question → choose a search profile → get a grounded answer.\n"
            "Tab completes commands (/help, /config, /profiles, /research, /quit).[/dim]",
            border_style="blue",
        ))

        while True:
            try:
                line = session.prompt(self._prompt).strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Shutting down browser…[/]")
                shutdown_crawler()
                console.print("[dim]Bye.[/]")
                # os._exit пропускает __del__ / GC — убирает шум от
                # ProactorEventLoop pipe-транспортов Playwright на Windows.
                # Chrome-процессы гасит OS автоматически вместе с родителем.
                import os as _os; _os._exit(0)

            if not line:
                continue

            if line.startswith("/"):
                try:
                    self._dispatch(line)
                except SystemExit:
                    console.print("[dim]Bye.[/]")
                    shutdown_crawler()
                    import os as _os; _os._exit(0)
            else:
                try:
                    self._run_query(line)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted.[/]")

            console.print()
