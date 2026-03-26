"""
bench/latency_breakdown.py — полный прогон с таймингом на каждом шаге.

Патчит use_case на уровне step-методов, чтобы измерить, где именно теряется время:
  SERP search / SERP gate / routing / fetch / passage split / verify_claim / compose_answer

Вывод: таблица шагов с временем выполнения (мс) по каждому claim + iteration.

Запуск:
  uv run python bench/latency_breakdown.py -q "разница python 3.11 и 3.12" -S ddgs
  uv run python bench/latency_breakdown.py -q "who is guido van rossum" -S ddgs
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from search_agent.runtime_bootstrap import ensure_utf8_stdio
ensure_utf8_stdio()

from rich.console import Console
from rich.table import Table
from rich import box
from rich.rule import Rule

console = Console()

# Глобальный лог событий: list of {claim, iter, step, ms}
_events: list[dict[str, Any]] = []


@contextmanager
def _timed(label: str, claim_id: str = "-", iteration: int = 0):
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - start) * 1000
        _events.append({"claim": claim_id[:12], "iter": iteration, "step": label, "ms": ms})


def _patch_steps(steps, claim_id_getter):
    """Оборачивает методы LegacyAgentStepLibrary в таймеры."""
    original = {}
    for name in ("gate_serp_results", "route_claim_retrieval", "split_into_passages",
                 "cheap_passage_filter", "utility_rerank_passages", "build_evidence_bundle",
                 "should_stop_claim_loop", "compose_answer"):
        if not hasattr(steps, name):
            continue
        fn = getattr(steps, name)
        original[name] = fn

        def make_wrapper(n, f):
            def wrapper(*args, **kwargs):
                cid = claim_id_getter()
                with _timed(n, claim_id=cid):
                    return f(*args, **kwargs)
            return wrapper

        setattr(steps, name, make_wrapper(name, fn))
    return original


def _patch_search_gateway(gw, claim_id_getter, iter_getter):
    orig = gw.search_variant
    def wrapper(query, profile, **kwargs):
        cid = claim_id_getter()
        it = iter_getter()
        with _timed("search_variant", claim_id=cid, iteration=it):
            return orig(query, profile, **kwargs)
    gw.search_variant = wrapper


def _patch_fetch_gateway(fg, claim_id_getter, iter_getter):
    orig = fg.fetch_claim_documents
    def wrapper(*args, **kwargs):
        cid = claim_id_getter()
        it = iter_getter()
        with _timed("fetch_claim_documents", claim_id=cid, iteration=it):
            return orig(*args, **kwargs)
    fg.fetch_claim_documents = wrapper


def _patch_intelligence(intel, claim_id_getter):
    for name in ("classify_query", "decompose_claims", "verify_claim"):
        if not hasattr(intel, name):
            continue
        fn = getattr(intel, name)

        def make_wrapper(n, f):
            def wrapper(*args, **kwargs):
                cid = claim_id_getter()
                with _timed(n, claim_id=cid):
                    return f(*args, **kwargs)
            return wrapper

        setattr(intel, name, make_wrapper(name, fn))


def _show_results(query: str):
    console.print(Rule("Latency breakdown"))

    if not _events:
        console.print("[red]No events recorded.[/red]")
        return

    # Сводная таблица по шагам
    step_totals: dict[str, float] = {}
    step_counts: dict[str, int] = {}
    for e in _events:
        step_totals[e["step"]] = step_totals.get(e["step"], 0) + e["ms"]
        step_counts[e["step"]] = step_counts.get(e["step"], 0) + 1

    total_ms = sum(step_totals.values())

    t = Table(title="Step totals (all claims × all iterations)", box=box.SIMPLE_HEAD)
    t.add_column("step", style="bold")
    t.add_column("calls", justify="right")
    t.add_column("total ms", justify="right")
    t.add_column("avg ms", justify="right")
    t.add_column("% of total", justify="right")

    for step in sorted(step_totals, key=lambda s: -step_totals[s]):
        pct = step_totals[step] / total_ms * 100 if total_ms else 0
        bar = "█" * int(pct / 5)
        t.add_row(
            step,
            str(step_counts[step]),
            f"{step_totals[step]:.0f}",
            f"{step_totals[step] / step_counts[step]:.0f}",
            f"{pct:.1f}% {bar}",
        )
    t.add_row("[bold]TOTAL[/bold]", "", f"[bold]{total_ms:.0f}[/bold]", "", "100%")
    console.print(t)

    # Детальный лог
    console.print()
    t2 = Table(title="Per-event log", box=box.MINIMAL, show_lines=False)
    t2.add_column("claim", style="dim")
    t2.add_column("iter", justify="right", style="dim")
    t2.add_column("step")
    t2.add_column("ms", justify="right")

    for e in _events:
        ms_style = "red" if e["ms"] > 5000 else ("yellow" if e["ms"] > 1000 else "green")
        t2.add_row(e["claim"], str(e["iter"]) if e["iter"] else "-", e["step"], f"[{ms_style}]{e['ms']:.0f}[/]")
    console.print(t2)


def main():
    parser = argparse.ArgumentParser(description="Full run with step-level timing")
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("-S", "--search-provider", choices=("brave", "ddgs"), default=None)
    parser.add_argument("-p", "--profile", default="web")
    args = parser.parse_args()

    if args.search_provider:
        os.environ["SEARCH_PROVIDER_OVERRIDE"] = args.search_provider

    from search_agent.settings import get_settings
    from search_agent.infrastructure.gateway_factory import build_search_gateway
    from search_agent.infrastructure.fetch_gateway import LegacyFetchGateway
    from search_agent.infrastructure.intelligence import PydanticAIQueryIntelligence
    from search_agent.infrastructure.receipt_gateway import JsonReceiptWriter
    from search_agent.application.legacy_steps import LegacyAgentStepLibrary
    from search_agent.application.use_cases import SearchAgentUseCase
    from search_agent.config.profiles import get_profile
    from search_agent.infrastructure.telemetry import configure_logfire

    get_settings.cache_clear()
    settings = get_settings()
    configure_logfire(settings)

    intel = PydanticAIQueryIntelligence(settings)
    search_gw = build_search_gateway(settings)
    fetch_gw = LegacyFetchGateway()
    steps = LegacyAgentStepLibrary()

    # Контекст для патчей: текущий claim_id и iteration
    _ctx: dict[str, Any] = {"claim_id": "global", "iteration": 0}

    def claim_id_getter(): return _ctx["claim_id"]
    def iter_getter(): return _ctx["iteration"]

    _patch_intelligence(intel, claim_id_getter)
    _patch_search_gateway(search_gw, claim_id_getter, iter_getter)
    _patch_fetch_gateway(fetch_gw, claim_id_getter, iter_getter)
    _patch_steps(steps, claim_id_getter)

    # Патчим _run_claim чтобы обновлять контекст
    use_case = SearchAgentUseCase(
        intelligence=intel,
        search_gateway=search_gw,
        fetch_gateway=fetch_gw,
        receipt_writer=JsonReceiptWriter(),
        steps=steps,
    )

    orig_run_claim = use_case._run_claim

    def patched_run_claim(claim, classification, profile, *, search_gateway, log=None):
        _ctx["claim_id"] = claim.claim_id
        _ctx["iteration"] = 0

        # Оборачиваем loop через монки-патч search_variant чтобы узнать iteration
        orig_search = search_gateway.search_variant

        _iter_counter = [0]

        def search_with_iter(query, prof, **kw):
            # iteration инкрементируется при первом search в новой итерации
            return orig_search(query, prof, **kw)

        # Мы уже пропатчили search_gateway.search_variant выше, и он уже тянет iter_getter
        # Поэтому просто запускаем оригинальный _run_claim
        return orig_run_claim(claim, classification, profile, search_gateway=search_gateway, log=log)

    use_case._run_claim = patched_run_claim

    profile = get_profile(args.profile)

    console.print(Rule(f"[bold]Query:[/] {args.query}"))

    t_total_start = time.perf_counter()
    try:
        report = use_case.run(args.query, profile=profile, log=console.print)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    finally:
        t_total_ms = (time.perf_counter() - t_total_start) * 1000
        console.print(f"\n  [dim]Wall time: {t_total_ms:.0f} ms[/dim]\n")
        _show_results(args.query)


if __name__ == "__main__":
    main()
