# Self-hosted search agent

An iterative web search agent that runs on your own machine and talks to any
OpenAI-compatible model endpoint. It searches DuckDuckGo or Brave, drops the
SEO filler, fetches the pages that survive, and asks the model for an answer
and for its own assessment of that answer in one call. The loop ends when
every claim in the answer rests on at least two independent domains, so a
question the pages cannot settle comes back with the list of what is missing
instead of a confident guess.

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

---

## Quick start

Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Put the model key in `.env`:

```
LLM_API_KEY=your_key_here
```

The default search backend is Brave, which needs a subscription token:

```
BRAVE_API_KEY=your_brave_subscription_token_here
```

DuckDuckGo needs no key. Set `SEARCH_PROVIDER=ddgs` in `.env`, or switch the
backend for one run with `-S ddgs`.

Then ask a question:

```bash
uv run search-agent -q "When was Python 3.13 released?" -p web
```

Without `-q` the program opens an interactive prompt where you pick a search
profile for every question. The commands there are `/help`, `/config`,
`/profiles`, `/research`, `/clear` and `/quit`.

The deep fetch path needs a browser, once:

```bash
uv run crawl4ai-setup
```

Without it the agent still works, but a page that returns no readable text
over HTTP is never retried through Playwright.

## What it does

- **Writes the query list.** One model call classifies the question and
  returns 3 to 5 keyword queries at the same time, so the first round of
  searching costs one round trip rather than two.
- **Reads pages, not snippets.** Search hits are scored, the pages are fetched
  over HTTP and stripped to text with trafilatura, and a page that yields too
  little text is fetched again through a headless browser.
- **Answers and grades itself in one call.** The model returns the markdown
  answer with `[N]` citations, the atomic claims behind it, a confidence value
  from 0 to 1, the gaps it noticed, and a flag for a question the evidence
  contradicts.
- **Stops on evidence.** The loop ends when the evidence contradicts the
  question, or when the confidence is at least 0.75 and every claim is backed
  by two independent domains, or when the iteration cap of 2 is reached.
- **Counts a host once.** Two pages on the same domain are one source, so a
  site that repeats itself cannot satisfy the two-domain rule.
- **Cites what it used.** Every `[N]` resolves to a URL, and the source list is
  appended under the answer.
- **Drops the junk.** The gate scores each hit on domain type, freshness,
  entity match, primary-source likelihood and spam cues, and keeps 10 to 20
  URLs.
- **Keeps one page from filling the window.** Passages are interleaved round
  by round across URLs, and the prompt takes at most 2 passages per URL, 4 per
  domain, and 14 in total.
- **Keeps the keys on your machine.** The model key, the search key and the
  `.env` file stay on your disk.
- **Ships an evaluation harness.** The datasets live in `eval_data/`, every
  run is saved as JSON in `eval_runs/`, and two runs can be diffed.

## How it works

```
question
  |
  v
normalize time references     <- model, only for relative dates
  |
  v
+-------------------------------------------------------------+
| iteration 1 .. 2                                            |
|                                                             |
|   queries                <- model; iteration 1 reuses the   |
|                            queries the classify call made   |
|   parallel search         <- CachingBudgetSearchGateway     |
|   gate                   -> 10 to 20 URLs                   |
|   fetch                  -> page_cache, shared              |
|   split into passages    -> at most 420 characters each     |
|   rank and interleave    -> top 25                          |
|   cap per URL and domain -> at most 14 passages             |
|   assess and answer      <- model, one call                 |
|                                                             |
|   stop?  contradicts_query, or confidence >= 0.75 and two  |
|          domains behind every claim, or the iteration cap  |
+-------------------------------------------------------------+
  |
  v
answer, source list, audit trail
```

The single model call at the end of an iteration carries the whole decision.
The same call that writes the answer returns the claims it made, so the
runner can check whether each claim has two independent domains behind it
without a second verification pass. The `[N]` numbers in the prompt and the
numbers in the source list come from the same list in the same order, which
is why the runner rebuilds the passage index from the capped list and not
from the ranked one.

The runner keeps a single synthetic claim, so the evaluation harness, which
was written for the older claim-per-claim pipeline, can still score it. The
verdict it reports comes from the assessment: `contradicts_query` becomes
`contradicted`, a confidence of at least 0.75 becomes `supported`, and
anything else becomes `insufficient_evidence`.

| Path | What it holds |
| --- | --- |
| `search_agent/application/unified_runner.py` | The loop. The only place the iteration is driven from. |
| `search_agent/application/agent_steps.py` | The helper steps the runner calls: gate, fetch, split, rank. |
| `search_agent/application/agent_sources.py` | Source scoring, the gate, the fetch plan, the passage split. |
| `search_agent/application/agent_compose.py` | The composed answer and the search-cost estimate. |
| `search_agent/application/contracts.py` | The ports. The boundary between the layers. |
| `search_agent/application/policy_tuning.py` | The weights of the scoring rules. |
| `search_agent/domain/assessment.py` | `Assessment` and `KeyClaim`, the output of the answer call. |
| `search_agent/domain/models.py` | Claims, passages, snapshots, the run result. |
| `search_agent/infrastructure/intelligence.py` | The model calls: classify, queries, answer. |
| `search_agent/infrastructure/caching_search_gateway.py` | The search cache and the per-run call budget. |
| `search_agent/infrastructure/ddgs_gateway.py` | DuckDuckGo, with retries on a transient backend failure. |
| `search_agent/infrastructure/brave_search.py` | Brave, with a fallback chain of profiles. |
| `search_agent/infrastructure/extractor.py` | Page text: HTTP first, headless browser as the fallback. |
| `search_agent/infrastructure/source_handlers.py` | Per-source extraction: Reddit, arXiv, Crossref, Semantic Scholar, GitHub. |
| `search_agent/config/profiles.py` | The 15 search profiles. |
| `search_agent/settings.py` | The environment settings. |
| `search_agent/tuning.py` | The internal constants. |

## Configuration

The keys, the provider and the endpoints live in the environment. Everything
else is a constant in `search_agent/tuning.py`, and the example file
`.env.example` holds the same list.

| Variable | Default | What it does |
| --- | --- | --- |
| `LLM_API_KEY` | none | The model key. The program exits with an error when it is missing. |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint. |
| `LLM_MODEL` | `qwen/qwen3.5-35b-a3b` | The model name sent to the endpoint. |
| `LLM_PROVIDER` | none | Routing hint for the endpoint. |
| `LLM_MAX_TOKENS` | `8192` | The ceiling for a single call. |
| `LLM_HTTP_REFERER` | `https://github.com/local/search-agent` | Sent as the `HTTP-Referer` header. |
| `SEARCH_PROVIDER` | `brave` | `brave` or `ddgs`. |
| `SEARCH_PROVIDER_OVERRIDE` | none | Wins over `SEARCH_PROVIDER`. The `-S` flag sets it. |
| `BRAVE_API_KEY` | none | The Brave Search subscription token. |
| `BRAVE_BASE_URL` | `https://api.search.brave.com` | The Brave endpoint. |
| `BRAVE_COUNTRY` | `US` | The country sent to Brave. |
| `BRAVE_SAFESEARCH` | `moderate` | `off`, `moderate` or `strict`. |
| `BRAVE_GOGGLES` | none | A JSON array, or newline-separated ranking rules. |
| `BRAVE_SEARCH_FALLBACK` | `true` | Falls back to other profiles when Brave returns nothing. |
| `SEARCH_BACKEND_FALLBACK_DELAY_SEC` | `1.0` | The wait before a fallback request. |
| `DDGS_REGION` | `wt-wt` | The DuckDuckGo region. |
| `DDGS_SAFESEARCH` | `moderate` | The DuckDuckGo safesearch setting. |
| `DDGS_TIMEOUT` | `15` | Seconds. |
| `DEFAULT_PROFILE` | `web` | The profile used when the run names none. |
| `AGENT_RECEIPTS_DIR` | none | Where the audit trail JSON is written. |
| `LOGFIRE_TOKEN` | none | Turns on Logfire tracing. |
| `GITHUB_TOKEN` | none | Raises the GitHub API rate limit for repository pages. |
| `EXTRACT_MAX_CHARS` | `4000` | Overrides `tuning.EXTRACT_MAX_CHARS`. |
| `COMPOSE_ANSWER_MAX_TOKENS` | `1600` | The cap on the composed answer. |
| `RAG_ANALYSIS_MAX_TOKENS` | `1400` | The cap on the arXiv analysis. |
| `CLAIM_DECOMPOSE_MAX_TOKENS` | `500` | The cap on a decomposition call. |
| `VERIFY_CLAIM_MAX_TOKENS` | `700` | The cap on a verification call. |
| `TIME_NORMALIZE_MAX_TOKENS` | `120` | The cap on the date normalization call. |
| `SYNTHESIZE_ANSWER_MAX_TOKENS` | `2000` | The cap on the answer call. |

A search profile decides the category, the language, the freshness window and
the query hints. The interactive prompt suggests one from the wording of the
question and lets you pick another.

| Profile | What it is for |
| --- | --- |
| `web` | General search. The default. |
| `ru` | General search with a Russian language bias. |
| `news` | The news vertical, last week. |
| `news_fresh` | The news vertical, last 24 hours. |
| `ru_news` | The news vertical, last week, Russian bias. |
| `ru_news_fresh` | The news vertical, last 24 hours, Russian bias. |
| `science` | Scientific sources, with an arXiv boost. |
| `tech` | Software and IT, with a tech-blog bias. |
| `it_qa` | Question and answer sites. |
| `it_repos` | Repositories and packages. |
| `deep` | A wide budget across the general, news and science categories. |
| `reference` | Wikipedia and Wikidata. |
| `reddit` | Reddit. |
| `arxiv` | arXiv only. |
| `wikipedia` | Wikipedia and Wikidata, with a wider result set. |

## Development

```bash
uv sync                                          # install
uv run python -m unittest discover -s tests -q   # the tests, 19 files, 85 tests
uv run search-agent --eval eval_data/quality_smoke.jsonl --eval-label smoke
uv run python -m search_agent.eval eval_runs/a.json eval_runs/b.json
```

The third command runs a dataset of 5 cases and saves the metrics to
`eval_runs/`; the fourth diffs two saved runs. There is no build step for
running it, and the package itself is a hatchling project, so `uv build`
produces a wheel.

| Path | What it holds |
| --- | --- |
| `search_agent/app.py` | The command line: one query, a dataset run, the arXiv mode. |
| `search_agent/cli.py` | The interactive prompt, the profile picker, the history file. |
| `search_agent/bootstrap.py` | The wiring. Builds the use case out of the ports. |
| `search_agent/eval/` | The run artifacts and the comparison between two runs. |
| `search_agent/evaluation.py` | The scoring of a dataset against the expectations. |
| `eval_data/` | The datasets and their schema. |
| `bench/` | Two scripts for measuring the routing and the stop condition. |
| `tests/` | The tests. |

`eval_data/README.md` holds the schema of a dataset row and the meaning of
every metric.

## License

MIT, © Alexander Blinov. See [LICENSE](LICENSE).
