from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SourcePrior:
    source_prior: float = 0.0
    primary_boost: float = 0.0
    spam_penalty: float = 0.0
    verification_bonus: float = 0.0
    domain_type_override: str | None = None
    labels: list[str] = field(default_factory=list)


_EXACT_ROOT_PRIORS: dict[str, SourcePrior] = {
    "python.org": SourcePrior(
        source_prior=0.22,
        primary_boost=0.22,
        verification_bonus=0.24,
        domain_type_override="official",
        labels=["official_root"],
    ),
    "microsoft.com": SourcePrior(
        source_prior=0.22,
        primary_boost=0.22,
        verification_bonus=0.22,
        domain_type_override="official",
        labels=["official_root"],
    ),
    "openai.com": SourcePrior(
        source_prior=0.22,
        primary_boost=0.22,
        verification_bonus=0.22,
        domain_type_override="official",
        labels=["official_root"],
    ),
    "apple.com": SourcePrior(
        source_prior=0.22,
        primary_boost=0.22,
        verification_bonus=0.22,
        domain_type_override="official",
        labels=["official_root"],
    ),
    "github.com": SourcePrior(
        source_prior=0.08,
        primary_boost=0.08,
        verification_bonus=0.06,
        domain_type_override="vendor",
        labels=["code_host"],
    ),
    "medium.com": SourcePrior(
        source_prior=-0.06,
        primary_boost=-0.08,
        spam_penalty=0.05,
        verification_bonus=-0.08,
        labels=["ugc_platform"],
    ),
    "linkedin.com": SourcePrior(
        source_prior=-0.04,
        primary_boost=-0.04,
        spam_penalty=0.03,
        verification_bonus=-0.05,
        labels=["social_profile"],
    ),
    "youtube.com": SourcePrior(
        source_prior=-0.08,
        primary_boost=-0.10,
        spam_penalty=0.05,
        verification_bonus=-0.10,
        labels=["video_platform"],
    ),
    "x.com": SourcePrior(
        source_prior=-0.10,
        primary_boost=-0.12,
        spam_penalty=0.08,
        verification_bonus=-0.12,
        labels=["social_feed"],
    ),
    "news.ycombinator.com": SourcePrior(
        source_prior=-0.10,
        primary_boost=-0.14,
        spam_penalty=0.05,
        verification_bonus=-0.12,
        labels=["forum_discussion"],
    ),
    "reddit.com": SourcePrior(
        source_prior=-0.12,
        primary_boost=-0.16,
        spam_penalty=0.06,
        verification_bonus=-0.14,
        labels=["forum_discussion"],
    ),
    "wikipedia.org": SourcePrior(
        source_prior=0.02,
        primary_boost=-0.04,
        verification_bonus=0.0,
        labels=["secondary_reference"],
    ),
}

_SEGMENT_PRIORS: list[tuple[frozenset[str], SourcePrior]] = [
    (
        frozenset({"investor", "ir", "newsroom", "press", "media"}),
        SourcePrior(
            source_prior=0.16,
            primary_boost=0.18,
            verification_bonus=0.18,
            domain_type_override="official",
            labels=["official_subdomain"],
        ),
    ),
    (
        frozenset({"blog"}),
        SourcePrior(
            source_prior=0.06,
            primary_boost=0.10,
            verification_bonus=0.10,
            labels=["blog_subdomain"],
        ),
    ),
    (
        frozenset({"docs", "developer", "support", "help", "learn", "cloud"}),
        SourcePrior(
            source_prior=0.10,
            primary_boost=0.12,
            verification_bonus=0.12,
            domain_type_override="vendor",
            labels=["docs_subdomain"],
        ),
    ),
    (
        frozenset({"community", "forum", "discuss"}),
        SourcePrior(
            source_prior=-0.10,
            primary_boost=-0.14,
            spam_penalty=0.05,
            verification_bonus=-0.12,
            domain_type_override="forum",
            labels=["community_forum"],
        ),
    ),
]


def _host_root(host: str) -> str:
    parts = [part for part in host.lower().split(".") if part]
    if len(parts) <= 2:
        return ".".join(parts)
    if parts[-2] in {"co", "com", "org", "net"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _copy_prior(prior: SourcePrior) -> SourcePrior:
    return SourcePrior(
        source_prior=prior.source_prior,
        primary_boost=prior.primary_boost,
        spam_penalty=prior.spam_penalty,
        verification_bonus=prior.verification_bonus,
        domain_type_override=prior.domain_type_override,
        labels=list(prior.labels),
    )


def lookup_source_prior(host: str) -> SourcePrior:
    normalized_host = (host or "").lower().strip()
    root = _host_root(normalized_host)
    host_parts = {part for part in normalized_host.split(".") if part}

    if normalized_host in _EXACT_ROOT_PRIORS:
        return _copy_prior(_EXACT_ROOT_PRIORS[normalized_host])
    if root in _EXACT_ROOT_PRIORS:
        return _copy_prior(_EXACT_ROOT_PRIORS[root])

    merged = SourcePrior()
    for labels, prior in _SEGMENT_PRIORS:
        if labels & host_parts:
            merged.source_prior += prior.source_prior
            merged.primary_boost += prior.primary_boost
            merged.spam_penalty += prior.spam_penalty
            merged.verification_bonus += prior.verification_bonus
            if prior.domain_type_override:
                merged.domain_type_override = prior.domain_type_override
            merged.labels.extend(prior.labels)

    return merged
