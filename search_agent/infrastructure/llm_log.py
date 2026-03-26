"""Console logging around LLM calls (start, duration, failure) for debugging stalls."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from rich.markup import escape as _rich_escape

from search_agent.infrastructure.log_preview import preview_snippet

LogFn = Callable[[str], None] | None


@dataclass
class LLMLogMetrics:
    """Set ``output_chars`` inside the ``with`` block after the model returns."""

    input_chars: int
    output_chars: int = -1


def output_char_len(obj: object) -> int:
    """Character length of structured or string model output (for logs)."""
    if obj is None:
        return 0
    if isinstance(obj, str):
        return len(obj)
    dump = getattr(obj, "model_dump_json", None)
    if callable(dump):
        try:
            return len(dump())
        except Exception:
            pass
    return len(str(obj))


@contextmanager
def log_llm_call(
    log: LogFn,
    *,
    task: str,
    model: str,
    detail: str = "",
    input_chars: int,
) -> Iterator[LLMLogMetrics]:
    """Emit Rich-safe lines before/after the request; on failure, log duration and re-raise.

    After ``run_sync``, set ``metrics.output_chars = output_char_len(result.output)``.
    """
    metrics = LLMLogMetrics(input_chars=max(0, input_chars))
    if log is None:
        yield metrics
        return

    esc_model = _rich_escape(model)
    esc_detail = _rich_escape(preview_snippet(detail, max_len=96)) if detail.strip() else ""
    suffix = f" · {esc_detail}" if esc_detail else ""
    in_tok = metrics.input_chars // 4
    log(
        f"  [dim cyan]LLM[/] [cyan]→[/] [bold]{_rich_escape(task)}[/] "
        f"[dim]· {esc_model}{suffix} · in={metrics.input_chars} (~{in_tok}tok)[/dim]"
    )
    t0 = time.perf_counter()
    try:
        yield metrics
    except BaseException as exc:
        dt = time.perf_counter() - t0
        msg = _rich_escape(preview_snippet(str(exc), max_len=140))
        log(
            f"  [dim cyan]LLM[/] [red]ERR[/] [bold]{_rich_escape(task)}[/] "
            f"[dim]({dt:.1f}s) · in={metrics.input_chars} (~{in_tok}tok) · {msg}[/dim]"
        )
        raise
    dt = time.perf_counter() - t0
    out_n = metrics.output_chars
    out_disp = str(out_n) if out_n >= 0 else "?"
    out_tok = f"~{out_n // 4}tok" if out_n >= 0 else "?"
    log(
        f"  [dim cyan]LLM[/] [green]OK[/] [bold]{_rich_escape(task)}[/] "
        f"[dim]({dt:.1f}s) · in={metrics.input_chars} (~{in_tok}tok) · out={out_disp} ({out_tok})[/dim]"
    )
