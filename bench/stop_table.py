"""
bench/stop_table.py — таблица истинности для should_stop_claim_loop.

Показывает все комбинации (verdict, confidence, independent_sources, has_primary_source, iteration)
и для каждой — остановится ли цикл или продолжится.

Запуск:
  uv run python bench/stop_table.py
  uv run python bench/stop_table.py --iter 1   # только iteration=1
  uv run python bench/stop_table.py --simple   # без итераций, только ключевые пороги
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from search_agent.runtime_bootstrap import ensure_utf8_stdio
ensure_utf8_stdio()

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def _stop(verdict, confidence, independent_sources, has_primary_source, iteration, max_iter=3):
    """Реплика логики should_stop_claim_loop без импорта tuning."""
    if verdict == "supported":
        if independent_sources >= 2 and has_primary_source:
            return True, "supported+2src+primary"
        if confidence >= 0.95 and independent_sources >= 2:
            return True, "supported+conf≥0.95+2src"
    return iteration >= max_iter, f"iter_cap({iteration}/{max_iter})"


def main():
    parser = argparse.ArgumentParser(description="Truth table for should_stop_claim_loop")
    parser.add_argument("--iter", type=int, default=None, help="Show only this iteration number")
    parser.add_argument("--simple", action="store_true", help="Just show key threshold rows")
    parser.add_argument("--max-iter", type=int, default=3, help="AGENT_MAX_CLAIM_ITERATIONS value")
    args = parser.parse_args()

    verdicts = ["supported", "contradicted", "insufficient_evidence"]
    confidences = [0.0, 0.50, 0.70, 0.80, 0.95, 1.0]
    source_counts = [0, 1, 2, 3]
    has_primaries = [False, True]
    iterations = [args.iter] if args.iter else [1, 2, 3]

    if args.simple:
        # Только строки где что-то интересное происходит
        confidences = [0.70, 0.95]
        source_counts = [1, 2]
        iterations = [1, 2, 3]

    t = Table(
        title=f"should_stop_claim_loop  (MAX_ITER={args.max_iter})",
        box=box.SIMPLE_HEAD,
        show_lines=False,
    )
    t.add_column("verdict", style="bold")
    t.add_column("conf", justify="right")
    t.add_column("sources", justify="right")
    t.add_column("primary", justify="center")
    t.add_column("iter", justify="right")
    t.add_column("STOP?", justify="center")
    t.add_column("reason", style="dim")

    rows_shown = 0
    for verdict, conf, sources, primary, it in product(verdicts, confidences, source_counts, has_primaries, iterations):
        stop, reason = _stop(verdict, conf, sources, primary, it, args.max_iter)

        # Фильтруем скучные строки: non-supported с stop=False показываем только на iter=max
        if args.simple and verdict != "supported" and not stop:
            continue

        stop_cell = "[green bold]YES[/]" if stop else "[red]no[/]"
        t.add_row(
            verdict,
            f"{conf:.2f}",
            str(sources),
            "yes" if primary else "no",
            str(it),
            stop_cell,
            reason,
        )
        rows_shown += 1

    console.print(t)
    console.print(f"\n[dim]{rows_shown} rows shown[/dim]\n")

    # Сводка: при каких условиях КРОМЕ iter_cap можно выйти рано
    console.print("[bold]Early-exit conditions (supported only):[/bold]")
    console.print("  * verdict=supported  AND  independent_sources>=2  AND  has_primary_source  -> STOP (any iteration)")
    console.print("  * verdict=supported  AND  confidence>=0.95  AND  independent_sources>=2    -> STOP (any iteration, no entity needed)")
    console.print()
    console.print("[bold]Practical problem:[/bold]")
    console.print("  For comparative queries (e.g. 'diff between X and Y'):")
    console.print("  - verify_claim often returns confidence ~0.70-0.85 -> misses 0.95 threshold")
    console.print("  - has_primary_source may be False if primary domain text is garbled (encoding issues)")
    console.print("  - Result: always runs to max iterations\n")


if __name__ == "__main__":
    main()
