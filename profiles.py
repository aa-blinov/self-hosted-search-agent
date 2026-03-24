"""
Search profiles — predefined configurations for SearXNG queries.

Each profile sets: categories, engines, language, time_range, fetch_top_n, max_results.

Usage:
  from profiles import get_profile, PROFILES
  profile = get_profile("tech")
"""

from dataclasses import dataclass, field


@dataclass
class SearchProfile:
    name: str
    description: str
    categories: list[str]          # SearXNG categories
    engines: list[str]             # specific engines; empty = SearXNG picks from categories
    language: str                  # "auto" | "ru" | "en" | ...
    time_range: str | None         # None | "day" | "week" | "month" | "year"
    fetch_top_n: int               # how many URLs to extract full text via ReaderLM-v2
    max_results: int               # total results to request from SearXNG


PROFILES: dict[str, SearchProfile] = {

    # ── General ──────────────────────────────────────────────────────────────

    "web": SearchProfile(
        name="web",
        description="General web — Google + Brave + Startpage",
        categories=["general"],
        engines=[],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=5,
    ),

    # ── CIS / Russian ────────────────────────────────────────────────────────

    "ru": SearchProfile(
        name="ru",
        description="Russian/CIS — language=ru, Startpage + Google + Brave",
        categories=["general"],
        engines=["startpage", "google", "brave"],  # Yandex broken in SearXNG (parsing error)
        language="ru",
        time_range=None,
        fetch_top_n=3,
        max_results=5,
    ),

    "ru_news": SearchProfile(
        name="ru_news",
        description="Russian news — Bing News + Reuters + Startpage News, last week, ru",
        categories=["news"],
        engines=["bing news", "startpage news", "reuters"],
        language="ru",
        time_range="week",
        fetch_top_n=2,
        max_results=7,
    ),

    # ── News ─────────────────────────────────────────────────────────────────

    "news": SearchProfile(
        name="news",
        description="Recent news — Bing News + Reuters + Qwant News, last week",
        categories=["news"],
        engines=[],
        language="auto",
        time_range="week",
        fetch_top_n=2,
        max_results=7,
    ),

    "news_fresh": SearchProfile(
        name="news_fresh",
        description="Breaking news — last 24 hours",
        categories=["news"],
        engines=[],
        language="auto",
        time_range="day",
        fetch_top_n=2,
        max_results=7,
    ),

    # ── Academic ─────────────────────────────────────────────────────────────

    "science": SearchProfile(
        name="science",
        description="Academic — arXiv + OpenAIRE + PubMed (abstracts, no ReaderLM)",
        categories=["science"],
        engines=[],
        language="auto",
        time_range=None,
        fetch_top_n=0,   # abstracts already structured in content field — ReaderLM not needed
        max_results=7,
    ),

    # ── IT / Dev ─────────────────────────────────────────────────────────────

    "tech": SearchProfile(
        name="tech",
        description="IT/Dev — GitHub + StackOverflow + MDN + Docker Hub + AskUbuntu",
        categories=["it"],
        engines=[],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=5,
    ),

    # ── Deep ─────────────────────────────────────────────────────────────────

    "deep": SearchProfile(
        name="deep",
        description="Deep — general + tech + news combined, more results",
        categories=["general", "it", "news"],
        engines=[],
        language="auto",
        time_range=None,
        fetch_top_n=5,
        max_results=10,
    ),
}

DEFAULT_PROFILE = "web"


def get_profile(name: str) -> SearchProfile:
    p = PROFILES.get(name)
    if p is None:
        raise ValueError(
            f"Unknown profile '{name}'. Available: {list(PROFILES)}"
        )
    return p


def list_profiles() -> str:
    lines = []
    for name, p in PROFILES.items():
        cats = "+".join(p.categories)
        lang = f" [{p.language}]" if p.language != "auto" else ""
        tr = f" {p.time_range}" if p.time_range else ""
        lines.append(f"  [bold]{name:<12}[/bold] {p.description}{lang}{tr}")
    return "\n".join(lines)
