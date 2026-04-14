"""Domain models for the unified iterative pipeline.

The unified runner collapses classify_intent + decompose + verify_claim + synthesize
into a single ``assess_and_answer`` LLM call that returns one ``Assessment`` per
iteration. ``KeyClaim`` captures the atomic factual assertions the answer makes so
that the stop condition can check independent-source coverage without re-running
verification per claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class KeyClaim:
    """A single factual assertion the answer makes, with its supporting citations.

    ``supporting_citation_numbers`` holds 1-based passage indices as they appear
    in the [N]-citation syntax inside ``Assessment.answer``.
    """

    text: str
    supporting_citation_numbers: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Assessment:
    """Output of ``assess_and_answer``: the answer plus self-evaluation.

    - ``answer``: user-visible markdown answer with [N] citations.
    - ``key_claims``: atomic assertions the answer makes (for sufficiency check).
    - ``confidence``: 0.0-1.0; the model's own rating of answer completeness.
    - ``gaps``: specific missing info that should drive the next iteration's queries.
    - ``contradicts_query``: True when the passages directly contradict an
      assumption in the user query (e.g. counterfactual date / attribution).
      Used as an immediate-stop signal.
    """

    answer: str
    key_claims: list[KeyClaim] = field(default_factory=list)
    confidence: float = 0.0
    gaps: list[str] = field(default_factory=list)
    contradicts_query: bool = False
