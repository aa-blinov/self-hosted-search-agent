"""
Search profiles for the local runtime.

Important:
  - Search routing should lean on SearXNG-native tabs / groups / bangs.
  - Local profiles remain useful only for runtime budget and language defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchProfile:
    name: str
    description: str
    categories: list[str]
    language: str
    time_range: str | None
    fetch_top_n: int
    max_results: int
    bang_prefixes: list[str] = field(default_factory=list)
    engines: list[str] = field(default_factory=list)


PROFILES: dict[str, SearchProfile] = {
    "web": SearchProfile(
        name="web",
        description="SearXNG tab !general (default web search)",
        categories=["general"],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=20,
    ),
    "ru": SearchProfile(
        name="ru",
        description="SearXNG tab !general with Russian language bias",
        categories=["general"],
        language="ru",
        time_range=None,
        fetch_top_n=3,
        max_results=20,
    ),
    "news": SearchProfile(
        name="news",
        description="SearXNG tab !news for recent news lookups",
        categories=["news"],
        language="auto",
        time_range="week",
        fetch_top_n=2,
        max_results=15,
    ),
    "news_fresh": SearchProfile(
        name="news_fresh",
        description="SearXNG tab !news, last 24h",
        categories=["news"],
        language="auto",
        time_range="day",
        fetch_top_n=2,
        max_results=15,
    ),
    "ru_news": SearchProfile(
        name="ru_news",
        description="SearXNG tab !news with Russian language bias",
        categories=["news"],
        language="ru",
        time_range="week",
        fetch_top_n=2,
        max_results=15,
    ),
    "science": SearchProfile(
        name="science",
        description="SearXNG tab !science / group !scientific_publications",
        categories=["science"],
        language="auto",
        time_range=None,
        fetch_top_n=0,
        max_results=20,
        bang_prefixes=["!scientific_publications"],
    ),
    "tech": SearchProfile(
        name="tech",
        description="SearXNG tab !it with native bang routing",
        categories=["it"],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=20,
        bang_prefixes=["!it"],
    ),
    "it_qa": SearchProfile(
        name="it_qa",
        description="SearXNG group !q&a for troubleshooting / errors",
        categories=["it"],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=20,
        bang_prefixes=["!q&a"],
    ),
    "it_repos": SearchProfile(
        name="it_repos",
        description="SearXNG group !repos for code / packages / repos",
        categories=["it"],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=20,
        bang_prefixes=["!repos"],
    ),
    "deep": SearchProfile(
        name="deep",
        description="Wide search budget across multiple SearXNG tabs",
        categories=["general", "news", "science"],
        language="auto",
        time_range=None,
        fetch_top_n=5,
        max_results=30,
    ),
    "reference": SearchProfile(
        name="reference",
        description="Reference fallback via encyclopedia/knowledge engines",
        categories=["general"],
        language="auto",
        time_range=None,
        fetch_top_n=2,
        max_results=12,
        engines=["wikipedia", "wikidata"],
    ),
}

DEFAULT_PROFILE = "web"


def get_profile(name: str) -> SearchProfile:
    profile = PROFILES.get(name)
    if profile is None:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(PROFILES)}")
    return profile


def list_profiles() -> str:
    lines = []
    for name, profile in PROFILES.items():
        categories = "+".join(profile.categories)
        bangs = f" {' '.join(profile.bang_prefixes)}" if profile.bang_prefixes else ""
        lang = f" [{profile.language}]" if profile.language != "auto" else ""
        time_range = f" {profile.time_range}" if profile.time_range else ""
        lines.append(
            f"  [bold]{name:<12}[/bold] {profile.description} "
            f"[dim](cats={categories}{bangs})[/dim]{lang}{time_range}"
        )
    return "\n".join(lines)
