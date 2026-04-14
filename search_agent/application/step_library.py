from __future__ import annotations

from search_agent.application.agent_compose import _estimate_search_cost
from search_agent.application.agent_steps import (
    _build_run_id,
    _documents_for_passage_extraction,
    _retag_snapshot,
    _split_into_passages,
    gate_serp_results,
)


class AgentStepLibrary:
    """Thin facade over the helper functions the unified runner calls.

    Most other helpers in ``agent_steps`` / ``agent_scoring`` are still
    consumed directly by component evals and ``bench/routing_probe``, but they
    no longer need a Port method since the classic orchestrator is gone.
    """

    def build_run_id(self, query: str, started_at) -> str:
        return _build_run_id(query, started_at)

    def retag_snapshot(self, snapshot, variant):
        return _retag_snapshot(snapshot, variant)

    def gate_serp_results(self, claim, snapshots, limit):
        return gate_serp_results(claim, snapshots, limit)

    def documents_for_passage_extraction(self, documents):
        return _documents_for_passage_extraction(documents)

    def split_into_passages(self, document):
        return _split_into_passages(document)

    def estimate_search_cost(self, claim_runs):
        return _estimate_search_cost(claim_runs)
