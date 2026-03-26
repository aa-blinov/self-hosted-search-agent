"""
bench/routing_probe.py — SERP → gate → routing диагностика БЕЗ LLM и фетча.

Показывает:
  1. Таблицу URL со всеми метриками source_score
  2. Разбивку routing decision (certainty / consistency / sufficiency) с пороговыми значениями
  3. Почему consistency низкая: какие answer_candidates нашлись в каждом сниппете

Запуск:
  uv run python bench/routing_probe.py -q "разница python 3.11 и 3.12" -S ddgs
  uv run python bench/routing_probe.py -q "who created python" -S ddgs --claim "Who created Python programming language?"
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

# Корень проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from search_agent.runtime_bootstrap import ensure_utf8_stdio
ensure_utf8_stdio()

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()


def _make_claim(text: str) -> "Claim":
    from search_agent.domain.models import Claim
    return Claim(
        claim_id=str(uuid.uuid4())[:8],
        claim_text=text,
        priority=1,
        needs_freshness=False,
        entity_set=[],
        time_scope=None,
    )


def _probe_routing(claim, snapshots, *, verbose: bool = False):
    """Запускает gate + routing и возвращает (gated_results, routing_decision)."""
    from search_agent.application.agent_steps import gate_serp_results, route_claim_retrieval
    from search_agent import tuning

    limit = max(tuning.SERP_GATE_MIN_URLS, 20)
    gated = gate_serp_results(claim, snapshots, limit)
    decision = route_claim_retrieval(claim, gated)
    return gated, decision


def _show_url_table(gated_results):
    t = Table(
        title="SERP gate — per-URL metrics",
        box=box.SIMPLE_HEAD,
        show_lines=False,
        highlight=True,
    )
    t.add_column("#", style="dim", width=3)
    t.add_column("URL", no_wrap=True, max_width=52)
    t.add_column("score", justify="right")
    t.add_column("primary", justify="right")
    t.add_column("entity", justify="right")
    t.add_column("semantic", justify="right")
    t.add_column("freshness", justify="right")
    t.add_column("domain", style="dim")
    t.add_column("prior", justify="right", style="dim")
    t.add_column("spam", justify="right", style="dim")

    for i, g in enumerate(gated_results[:20], 1):
        a = g.assessment
        score_style = "green" if a.source_score >= 0.7 else ("yellow" if a.source_score >= 0.5 else "red")
        t.add_row(
            str(i),
            g.serp.url[:52],
            f"[{score_style}]{a.source_score:.3f}[/]",
            f"{a.primary_source_likelihood:.2f}",
            f"{a.entity_match_score:.2f}",
            f"{a.semantic_match_score:.2f}",
            f"{a.freshness_score:.2f}",
            a.domain_type,
            f"{a.source_prior:.2f}",
            f"{a.seo_spam_risk:.2f}",
        )
    console.print(t)


def _show_routing_decision(decision):
    from search_agent.application.agent_steps import _answer_type

    # Threshold table
    t = Table(title="Routing decision", box=box.SIMPLE_HEAD, show_lines=False)
    t.add_column("metric", style="bold")
    t.add_column("value", justify="right")
    t.add_column("short_path ≥", justify="right", style="dim")
    t.add_column("targeted ≥", justify="right", style="dim")
    t.add_column("pass?", justify="center")

    def check(val, short, targeted):
        if val >= short:
            return "[green]short_path[/]"
        if val >= targeted:
            return "[yellow]targeted[/]"
        return "[red]iterative[/]"

    t.add_row("certainty",    f"{decision.certainty:.3f}",           "0.800", "0.550", check(decision.certainty, 0.8, 0.55))
    t.add_row("consistency",  f"{decision.consistency:.3f}",          "0.650", "—",     ("[green]OK[/]" if decision.consistency >= 0.65 else "[red]FAIL (short_path blocked)[/]"))
    t.add_row("sufficiency",  f"{decision.evidence_sufficiency:.3f}", "0.600", "0.450", check(decision.evidence_sufficiency, 0.6, 0.45))

    mode_style = {"short_path": "green", "targeted_retrieval": "yellow", "iterative_loop": "red"}[decision.mode]
    console.print(t)
    console.print(f"  → route = [{mode_style}]{decision.mode}[/]")
    console.print(f"  rationale: {decision.rationale}\n")


def _show_consistency_breakdown(claim, gated_results):
    """Показывает, какие answer_candidates извлечены из каждого сниппета и почему consistency такая."""
    import re
    from search_agent.application.agent_steps import _extract_answer_candidates

    console.print(Rule("Consistency breakdown"))
    candidates_all: list[str] = []
    rows = []
    for g in gated_results[:5]:
        text = f"{g.serp.title} {g.serp.snippet}"
        cands = [c.casefold() for c in _extract_answer_candidates(claim, text)]
        candidates_all.extend(cands)
        rows.append((g.serp.url[:60], cands))

    for url, cands in rows:
        cand_str = ", ".join(cands) if cands else "[dim]none[/dim]"
        console.print(f"  [dim]{url}[/dim]\n    candidates: {cand_str}")

    if candidates_all:
        counts: dict[str, int] = {}
        for c in candidates_all:
            counts[c] = counts.get(c, 0) + 1
        top = sorted(counts.items(), key=lambda x: -x[1])[:5]
        console.print(f"\n  max_count={max(counts.values())}  total={len(candidates_all)}"
                      f"  -> consistency = {max(counts.values())}/{len(candidates_all)} = {max(counts.values())/len(candidates_all):.3f}")
        console.print("  top candidates: " + ", ".join(f"{k!r} x{v}" for k, v in top))
    else:
        # Fallback: semantic scores
        sem_scores = [g.assessment.semantic_match_score for g in gated_results[:5]]
        avg = sum(sem_scores) / len(sem_scores) if sem_scores else 0
        console.print(f"  No answer_candidates extracted -> consistency = avg(semantic_match) = {avg:.3f}")
        console.print(f"  Semantic scores: {[f'{s:.2f}' for s in sem_scores]}")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="SERP → gate → routing probe (no LLM, no fetch)")
    parser.add_argument("-q", "--query", required=True, help="User query")
    parser.add_argument("--claim", default=None, help="Override claim text (default: use query as-is)")
    parser.add_argument("-S", "--search-provider", choices=("brave", "ddgs"), default=None)
    parser.add_argument("-p", "--profile", default="web")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.search_provider:
        os.environ["SEARCH_PROVIDER_OVERRIDE"] = args.search_provider

    from search_agent.settings import get_settings
    from search_agent.infrastructure.gateway_factory import build_search_gateway
    from search_agent.config.profiles import get_profile
    from search_agent.application.agent_steps import build_query_variants
    from search_agent.domain.models import QueryClassification

    get_settings.cache_clear()
    settings = get_settings()
    gateway = build_search_gateway(settings)
    profile = get_profile(args.profile)

    claim_text = args.claim or args.query
    claim = _make_claim(claim_text)

    # Минимальная classification для build_query_variants
    classification = QueryClassification(
        query=args.query,
        normalized_query=args.query.lower(),
        intent="factual",
        complexity="simple",
        needs_freshness=False,
    )

    variants = build_query_variants(claim, classification)

    console.print(Rule(f"[bold]Query:[/] {args.query}"))
    console.print(f"  claim: [italic]{claim.claim_text}[/italic]")
    console.print(f"  variants ({len(variants)}):")
    for v in variants:
        console.print(f"    [dim]{v.strategy}:[/dim] {v.query_text}")
    console.print()

    # Выполняем только первые 3 варианта (как iteration 1)
    from search_agent.application.agent_steps import _retag_snapshot
    snapshots = []
    for variant in variants[:3]:
        console.print(f"  [dim]SERP:[/dim] {variant.query_text[:70]}")
        new_snaps = gateway.search_variant(variant.query_text, profile)
        for snap in new_snaps:
            snapshots.append(_retag_snapshot(snap, variant))
    console.print()

    gated, decision = _probe_routing(claim, snapshots, verbose=args.verbose)

    _show_url_table(gated)
    console.print()
    _show_routing_decision(decision)
    _show_consistency_breakdown(claim, gated)

    # Краткая сводка — что нужно изменить чтобы попасть в short_path
    console.print(Rule("Diagnosis"))
    blockers = []
    if decision.certainty < 0.8:
        blockers.append(f"certainty {decision.certainty:.3f} < 0.800")
    if decision.consistency < 0.65:
        blockers.append(f"consistency {decision.consistency:.3f} < 0.650 (main blocker for this query type)")
    if decision.evidence_sufficiency < 0.6:
        blockers.append(f"sufficiency {decision.evidence_sufficiency:.3f} < 0.600")
    if blockers:
        console.print("  [red]short_path blocked by:[/red]")
        for b in blockers:
            console.print(f"    • {b}")
    else:
        console.print("  [green]short_path would be taken[/green]")


if __name__ == "__main__":
    main()
