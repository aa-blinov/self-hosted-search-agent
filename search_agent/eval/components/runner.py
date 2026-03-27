"""Shared runner infrastructure for component evals."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from search_agent.domain.models import (
    Claim,
    GatedSerpResult,
    Passage,
    SearchSnapshot,
    SerpResult,
    SourceAssessment,
)

DEFAULT_COMPONENT_RUNS_DIR = Path("eval_runs/components")
DEFAULT_DATASETS_DIR = Path("eval_data/components")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ComponentCaseResult:
    case_id: str
    passed: bool
    latency_ms: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentRunSummary:
    component: str
    dataset_path: str
    case_count: int
    pass_rate: float
    metrics: dict[str, Any]
    cases: list[ComponentCaseResult]


# ---------------------------------------------------------------------------
# Deserializers — JSON dict → domain dataclasses
# ---------------------------------------------------------------------------

def load_claim(d: dict) -> Claim:
    return Claim(
        claim_id=d["claim_id"],
        claim_text=d["claim_text"],
        priority=d.get("priority", 1),
        needs_freshness=d.get("needs_freshness", False),
        entity_set=list(d.get("entity_set") or []),
        time_scope=d.get("time_scope"),
    )


def load_passage(d: dict) -> Passage:
    return Passage(
        passage_id=d["passage_id"],
        url=d["url"],
        canonical_url=d.get("canonical_url", d["url"]),
        host=d.get("host", ""),
        title=d.get("title", ""),
        section=d.get("section"),
        published_at=d.get("published_at"),
        author=d.get("author"),
        extracted_at=d.get("extracted_at", ""),
        chunk_id=d.get("chunk_id", ""),
        text=d["text"],
        source_score=float(d.get("source_score", 0.5)),
        utility_score=float(d.get("utility_score", 0.0)),
    )


def load_serp_result(d: dict) -> SerpResult:
    return SerpResult(
        result_id=d["result_id"],
        query_variant_id=d.get("query_variant_id", ""),
        title=d.get("title", ""),
        url=d["url"],
        snippet=d.get("snippet", ""),
        canonical_url=d.get("canonical_url", d["url"]),
        host=d.get("host", ""),
        position=d.get("position", 0),
        score=d.get("score"),
        engine=d.get("engine"),
        published_at=d.get("published_at"),
        raw=dict(d.get("raw") or {}),
    )


def load_source_assessment(d: dict) -> SourceAssessment:
    return SourceAssessment(
        domain_type=d["domain_type"],
        source_prior=float(d["source_prior"]),
        primary_source_likelihood=float(d["primary_source_likelihood"]),
        freshness_score=float(d["freshness_score"]),
        seo_spam_risk=float(d["seo_spam_risk"]),
        entity_match_score=float(d["entity_match_score"]),
        semantic_match_score=float(d["semantic_match_score"]),
        source_score=float(d["source_score"]),
        duplicate_of=d.get("duplicate_of"),
        reasons=list(d.get("reasons") or []),
    )


def load_gated_serp_result(d: dict) -> GatedSerpResult:
    return GatedSerpResult(
        serp=load_serp_result(d["serp"]),
        assessment=load_source_assessment(d["assessment"]),
        matched_variant_ids=list(d.get("matched_variant_ids") or []),
    )


def load_search_snapshot(d: dict) -> SearchSnapshot:
    return SearchSnapshot(
        query=d["query"],
        suggestions=list(d.get("suggestions") or []),
        results=[load_serp_result(r) for r in (d.get("results") or [])],
        retrieved_at=d.get("retrieved_at", ""),
        profile_name=d.get("profile_name"),
        unresponsive_engines=list(d.get("unresponsive_engines") or []),
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


# ---------------------------------------------------------------------------
# Artifact saving (mirrors tracking.py pattern)
# ---------------------------------------------------------------------------

def _git_revision() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _default_artifact_filename(component: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = (_git_revision() or "nogit")[:8]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", component)
    return f"eval_comp_{ts}_{short}_{slug}.json"


def save_component_run(
    summary: ComponentRunSummary,
    output_dir: Path | None = None,
    *,
    label: str | None = None,
) -> Path:
    out_dir = output_dir or DEFAULT_COMPONENT_RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / _default_artifact_filename(summary.component)

    rev = _git_revision()
    payload: dict[str, Any] = {
        "component": summary.component,
        "dataset_path": summary.dataset_path,
        "case_count": summary.case_count,
        "pass_rate": summary.pass_rate,
        "metrics": summary.metrics,
        "cases": [
            {
                "case_id": c.case_id,
                "passed": c.passed,
                "latency_ms": c.latency_ms,
                **c.details,
            }
            for c in summary.cases
        ],
        "run_metadata": {
            "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "git_revision": rev,
            "git_revision_short": rev[:8] if rev else None,
            "label": (label or "").strip() or None,
        },
    }

    try:
        from search_agent.settings import get_settings
        s = get_settings()
        payload["run_metadata"].update({
            "llm_model": s.llm_model,
            "llm_provider": s.llm_provider,
        })
    except Exception:
        pass

    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact
