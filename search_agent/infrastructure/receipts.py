from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from search_agent.domain.models import AgentRunResult


DEFAULT_RECEIPTS_DIR = "receipts"


def _slugify(text: str, limit: int = 64) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").casefold()).strip("-")
    return slug[:limit] or "query"


def build_receipt_payload(report: AgentRunResult) -> dict:
    supported_claims = 0
    contradicted_claims = 0
    insufficient_claims = 0

    for run in report.claims:
        bundle = run.evidence_bundle
        verdict = bundle.verification.verdict if bundle and bundle.verification else None
        if verdict == "supported":
            supported_claims += 1
        elif verdict == "contradicted":
            contradicted_claims += 1
        elif verdict == "insufficient_evidence":
            insufficient_claims += 1

    payload = {
        "run_id": report.audit_trail.run_id,
        "profile_name": report.audit_trail.profile_name,
        "started_at": report.audit_trail.started_at,
        "completed_at": report.audit_trail.completed_at,
        "latency_ms": report.audit_trail.latency_ms,
        "estimated_search_cost": report.audit_trail.estimated_search_cost,
        "user_query": report.user_query,
        "classification": asdict(report.classification),
        "answer": report.answer,
        "summary": {
            "claim_count": len(report.claims),
            "supported_claims": supported_claims,
            "contradicted_claims": contradicted_claims,
            "insufficient_claims": insufficient_claims,
        },
        "claims": [asdict(run) for run in report.claims],
        "audit_trail": asdict(report.audit_trail),
    }
    return payload


def write_receipt(
    report: AgentRunResult,
    output_dir: str | None = None,
) -> str:
    base_dir = Path(output_dir or DEFAULT_RECEIPTS_DIR)
    timestamp = datetime.now(UTC).strftime("%Y%m%d")
    target_dir = base_dir / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)

    run_id = report.audit_trail.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{run_id}-{_slugify(report.user_query)}.json"
    path = target_dir / filename

    report.audit_trail.receipt_path = str(path)
    payload = build_receipt_payload(report)
    payload["receipt_path"] = str(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
