from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Default directory for eval JSON artifacts (CLI --eval); override with --eval-out.
DEFAULT_EVAL_RUNS_DIR = "eval_runs"


def _git_revision(fallback: str | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or fallback
    except (OSError, subprocess.CalledProcessError):
        return fallback


def _slug_dataset(path: str) -> str:
    stem = Path(path).stem
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip("-")
    return slug or "dataset"


def merge_run_metadata(
    summary: dict[str, Any],
    *,
    label: str | None = None,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Attach reproducibility fields; keeps existing summary keys."""
    from search_agent.settings import get_settings

    s = get_settings()
    rev = _git_revision()
    meta: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_revision": rev,
        "git_revision_short": (rev[:8] if rev else None),
        "label": (label or "").strip() or None,
        "llm_model": s.llm_model,
        "llm_provider": s.llm_provider,
        "search_provider": s.resolved_search_provider(),
        "dataset_path": summary.get("dataset_path"),
        "case_count": summary.get("case_count"),
        "eval_runs_dir": DEFAULT_EVAL_RUNS_DIR,
    }
    if artifact_path is not None:
        ap = artifact_path.resolve()
        meta["artifact_filename"] = ap.name
        meta["artifact_path"] = ap.as_posix()
    return {"run_metadata": meta, **summary}


def default_artifact_filename(dataset_path: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = (_git_revision() or "nogit")[:8]
    return f"eval_{ts}_{short}_{_slug_dataset(dataset_path)}.json"


def save_eval_run(
    summary: dict[str, Any],
    output: str | Path,
    *,
    label: str | None = None,
) -> Path:
    """
    Write one JSON artifact. If ``output`` is a directory, picks a timestamped filename.

    Typical use: pass the dict returned by ``evaluate_dataset``, then open in diff/CI.
    """
    out = Path(output)
    if out.is_dir() or str(output).endswith(("/", "\\")):
        out = out / default_artifact_filename(str(summary.get("dataset_path", "dataset")))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = merge_run_metadata(summary, label=label, artifact_path=out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_eval_run(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def compare_metric_deltas(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    metric_keys: tuple[str, ...] | None = None,
) -> dict[str, dict[str, float | None]]:
    """
    Compare top-level ``metrics`` maps. Returns per-key previous, current, delta (current - previous).
    """
    keys = metric_keys
    if keys is None:
        pm = previous.get("metrics")
        cm = current.get("metrics")
        if not isinstance(pm, dict) or not isinstance(cm, dict):
            return {}
        keys = tuple(sorted(set(pm) & set(cm)))

    out: dict[str, dict[str, float | None]] = {}
    pm = previous.get("metrics") or {}
    cm = current.get("metrics") or {}
    for k in keys:
        a, b = pm.get(k), cm.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[k] = {"previous": float(a), "current": float(b), "delta": float(b) - float(a)}
        else:
            out[k] = {"previous": None, "current": None, "delta": None}
    return out
