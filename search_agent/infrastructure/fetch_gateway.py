from __future__ import annotations

import logfire

from search_agent.application.agent_steps import fetch_claim_documents


class LegacyFetchGateway:
    def fetch_claim_documents(
        self,
        claim,
        gated_results,
        profile,
        routing_decision,
        *,
        seen_urls,
        log=None,
    ):
        with logfire.span(
            "fetch_gateway.fetch_claim_documents",
            claim_id=claim.claim_id,
            route_mode=routing_decision.mode,
            gated_urls=len(gated_results),
        ):
            return fetch_claim_documents(
                claim,
                gated_results,
                profile,
                routing_decision,
                seen_urls=seen_urls,
                log=log,
            )
