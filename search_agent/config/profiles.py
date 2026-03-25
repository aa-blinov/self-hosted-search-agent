"""
Search profiles for the local runtime.

Profiles map to search behaviour (web vs news, language, freshness) and query hints
(bangs → operator text, engines → site: filters) via `search_agent.infrastructure.serp_query.build_routed_query`.

- Brave: categories, Goggles, country/safesearch from env; `goggles` is Brave-only.
- DDGS: optional per-profile `ddgs_region`, `ddgs_safesearch`, `ddgs_timelimit` apply when the active provider is ddgs
  (`SEARCH_PROVIDER_OVERRIDE` or `SEARCH_PROVIDER`); timelimit otherwise follows `time_range` (day/week/month/year -> d/w/m/y).

`engines` shortcuts: wikipedia, wikidata, reddit, arxiv.
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
    # Brave Search API: hosted .goggle URLs and/or inline rules ($boost=...,site=...); multiple OK
    goggles: list[str] = field(default_factory=list)
    # DDGS (SEARCH_PROVIDER=ddgs): non-None overrides AppSettings ddgs_* / timelimit
    ddgs_region: str | None = None
    ddgs_safesearch: str | None = None
    # If set, passed to DDGS as timelimit; else map profile.time_range (day/week/month/year -> d/w/m/y)
    ddgs_timelimit: str | None = None


# Official Brave quickstart (register at search.brave.com/goggles if required by your plan)
GOGGLE_TECH_BLOGS = (
    "https://raw.githubusercontent.com/brave/goggles-quickstart/main/goggles/tech_blogs.goggle"
)
# Inline: small rules (no hosted file); newline separates instructions per Brave API docs
_GOGGLE_BOOST_WIKI = "$boost=5,site=wikipedia.org\n$boost=4,site=wikidata.org"
_GOGGLE_BOOST_REDDIT = "$boost=6,site=reddit.com"
_GOGGLE_BOOST_ARXIV = "$boost=5,site=arxiv.org\n$boost=3,site=semanticscholar.org"
_GOGGLE_BOOST_ARXIV_ONLY = "$boost=7,site=arxiv.org"
_GOGGLE_BOOST_SO = "$boost=5,site=stackoverflow.com\n$boost=3,site=serverfault.com\n$boost=2,site=superuser.com"
_GOGGLE_BOOST_REPOS = "$boost=5,site=github.com\n$boost=4,site=gitlab.com\n$boost=3,site=pypi.org"


PROFILES: dict[str, SearchProfile] = {
    "web": SearchProfile(
        name="web",
        description="General web search (Brave web index)",
        categories=["general"],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=20,
    ),
    "ru": SearchProfile(
        name="ru",
        description="General web search with Russian language bias",
        categories=["general"],
        language="ru",
        time_range=None,
        fetch_top_n=3,
        max_results=20,
        ddgs_region="ru-ru",
    ),
    "news": SearchProfile(
        name="news",
        description="News vertical (Brave news index), ~last week",
        categories=["news"],
        language="auto",
        time_range="week",
        fetch_top_n=2,
        max_results=15,
    ),
    "news_fresh": SearchProfile(
        name="news_fresh",
        description="News vertical, last 24h",
        categories=["news"],
        language="auto",
        time_range="day",
        fetch_top_n=2,
        max_results=15,
    ),
    "ru_news": SearchProfile(
        name="ru_news",
        description="News with Russian language bias",
        categories=["news"],
        language="ru",
        time_range="week",
        fetch_top_n=2,
        max_results=15,
    ),
    "ru_news_fresh": SearchProfile(
        name="ru_news_fresh",
        description="Russian-language news, last ~24h (Astana/today-style queries)",
        categories=["news"],
        language="ru",
        time_range="day",
        fetch_top_n=2,
        max_results=15,
    ),
    "science": SearchProfile(
        name="science",
        description="Scientific sources (Goggles + arxiv hints)",
        categories=["science"],
        language="auto",
        time_range=None,
        fetch_top_n=0,
        max_results=20,
        bang_prefixes=["!scientific_publications"],
        goggles=[_GOGGLE_BOOST_ARXIV],
    ),
    "tech": SearchProfile(
        name="tech",
        description="IT / software (Tech Blogs Goggle + query hints)",
        categories=["it"],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=20,
        bang_prefixes=["!it"],
        goggles=[GOGGLE_TECH_BLOGS],
    ),
    "it_qa": SearchProfile(
        name="it_qa",
        description="Q&A sites (inline Goggles boost SE family)",
        categories=["it"],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=20,
        bang_prefixes=["!q&a"],
        goggles=[_GOGGLE_BOOST_SO],
    ),
    "it_repos": SearchProfile(
        name="it_repos",
        description="Repositories and packages (Goggles boost GitHub / GitLab / PyPI)",
        categories=["it"],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=20,
        bang_prefixes=["!repos"],
        goggles=[_GOGGLE_BOOST_REPOS],
    ),
    "deep": SearchProfile(
        name="deep",
        description="Wide search budget (general + news + science categories in one query)",
        categories=["general", "news", "science"],
        language="auto",
        time_range=None,
        fetch_top_n=5,
        max_results=30,
    ),
    "reference": SearchProfile(
        name="reference",
        description="Reference fallback (site: + Goggles boost wiki/wikidata)",
        categories=["general"],
        language="auto",
        time_range=None,
        fetch_top_n=2,
        max_results=12,
        engines=["wikipedia", "wikidata"],
        goggles=[_GOGGLE_BOOST_WIKI],
    ),
    "reddit": SearchProfile(
        name="reddit",
        description="Reddit (site:reddit.com + Goggles boost)",
        categories=["general"],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=20,
        engines=["reddit"],
        goggles=[_GOGGLE_BOOST_REDDIT],
    ),
    "arxiv": SearchProfile(
        name="arxiv",
        description="arXiv.org only (papers; site + boost)",
        categories=["science"],
        language="auto",
        time_range=None,
        fetch_top_n=0,
        max_results=20,
        engines=["arxiv"],
        goggles=[_GOGGLE_BOOST_ARXIV_ONLY],
    ),
    "wikipedia": SearchProfile(
        name="wikipedia",
        description="Wikipedia + Wikidata (site: filters + Goggles boost)",
        categories=["general"],
        language="auto",
        time_range=None,
        fetch_top_n=3,
        max_results=15,
        engines=["wikipedia", "wikidata"],
        goggles=[_GOGGLE_BOOST_WIKI],
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
        gog = f" goggles={len(profile.goggles)}" if profile.goggles else ""
        lang = f" [{profile.language}]" if profile.language != "auto" else ""
        time_range = f" {profile.time_range}" if profile.time_range else ""
        lines.append(
            f"  [bold]{name:<12}[/bold] {profile.description} "
            f"[dim](cats={categories}{bangs}{gog})[/dim]{lang}{time_range}"
        )
    return "\n".join(lines)
