"""Compatibility facade for LLM-backed answer generation and paper analysis."""

from __future__ import annotations

from datetime import datetime
from time import sleep

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from search_agent.infrastructure.llm_tasks import build_task_runner
from search_agent.settings import get_settings

_SETTINGS = get_settings()
MODEL = _SETTINGS.llm_model
MAX_TOKENS = _SETTINGS.llm_max_tokens
_PROVIDER = (_SETTINGS.llm_provider or "").strip()


def _extra() -> dict:
    body: dict = {"reasoning": {"effort": "none"}}
    if _PROVIDER:
        body["provider"] = {
            "order": [_PROVIDER.lower()],
            "allow_fallbacks": False,
        }
    return body


def answer_with_sources(query: str, sources: list[dict], client=None) -> str:
    del client
    today = datetime.now().strftime("%d %B %Y, %A")
    return build_task_runner().answer_with_sources(query, sources, today=today)


def answer_streaming(
    query: str,
    sources: list[dict],
    client,
    console,
    panel_width: int = 100,
) -> str:
    del client
    answer = answer_with_sources(query, sources)
    if not answer:
        console.print("[yellow]No sources retrieved. Cannot answer without context.[/]")
        return ""

    rendered = ""
    with Live(
        Panel("[dim]…[/dim]", title="[bold green]Answer[/]", border_style="green", width=panel_width),
        console=console,
        refresh_per_second=12,
        vertical_overflow="visible",
    ) as live:
        for chunk in answer.split():
            rendered = f"{rendered} {chunk}".strip()
            live.update(
                Panel(
                    Markdown(rendered),
                    title="[bold green]Answer[/]",
                    border_style="green",
                    width=panel_width,
                )
            )
            sleep(0.01)
    return answer


def analyze_rag_papers(papers: list[dict], client=None) -> str:
    del client
    return build_task_runner().analyze_rag_papers(papers)
