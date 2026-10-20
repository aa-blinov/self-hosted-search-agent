# AGENTS.md — agent context for `self-hosted-search-agent`

A web search agent that runs on your own machine: it asks a language model
for search queries, fetches the pages they point to, and asks the same model
to write an answer and to grade its own completeness. It talks to any
OpenAI-compatible endpoint, and it searches through Brave or DuckDuckGo.

This file is what you need to work on the project without reading all of it.

## Run / build / test

```bash
uv sync                                          # install
uv run search-agent                              # the interactive prompt
uv run search-agent -q "your question" -p web    # one question
uv run python -m unittest discover -s tests -q   # the tests, 19 files, 85 tests
uv run search-agent --eval eval_data/quality_smoke.jsonl --eval-label smoke
uv run python -m search_agent.eval eval_runs/a.json eval_runs/b.json
uv run python -m search_agent.eval.components --all --no-llm
```

Python 3.11 or newer. The fifth command scores a dataset of 5 cases and
saves the metrics to `eval_runs/`; the sixth diffs two saved runs; the
last one runs the component checks that need neither a key nor a network.

`uv run crawl4ai-setup` installs the browser the deep fetch path uses. It is
needed once per machine, and only for pages that fail over to Playwright.

## Where things live

| Path | What it holds |
| --- | --- |
| `search_agent/app.py` | The command line: one query, a dataset run, the arXiv mode, the `-S` backend switch. |
| `search_agent/cli.py` | The interactive prompt, the profile picker, the keyword heuristics that suggest a profile, the history file. |
| `search_agent/bootstrap.py` | The wiring. Builds the use case out of the five ports. |
| `search_agent/settings.py` | The environment settings, and the `resolved_*` methods that let a few caps be overridden. |
| `search_agent/tuning.py` | The internal constants. Not configurable from the environment. |
| `search_agent/application/unified_runner.py` | The loop. The only place an iteration is driven from. |
| `search_agent/application/agent_steps.py` | The helper steps the runner calls, and the facade over the rest of the scoring code. |
| `search_agent/application/agent_sources.py` | Source scoring, the SERP gate, the fetch plan, the passage split. |
| `search_agent/application/agent_compose.py` | The answer composition of the older pipeline, and the search-cost estimate. |
| `search_agent/application/agent_passage_scoring.py` | The cheap TF-IDF passage filter and the utility rerank. |
| `search_agent/application/agent_evidence.py` | The evidence bundle and the old stop condition. |
| `search_agent/application/claim_policy.py` | The per-shape rules: how many sources a claim needs, whether it needs a primary source. |
| `search_agent/application/policy_tuning.py` | The weights of every scoring rule. |
| `search_agent/application/contracts.py` | The ports. The boundary between the layers. |
| `search_agent/domain/assessment.py` | `Assessment` and `KeyClaim`, the output of the answer call. |
| `search_agent/domain/models.py` | Claims, passages, snapshots, the run result, the audit trail. |
| `search_agent/domain/source_priors.py` | Per-domain priors for known hosts, and the SEO spam risk. |
| `search_agent/infrastructure/intelligence.py` | The model calls: classify, generate queries, answer and assess. |
| `search_agent/infrastructure/caching_search_gateway.py` | The search cache and the per-run call budget. |
| `search_agent/infrastructure/ddgs_gateway.py` | DuckDuckGo, with the retries. |
| `search_agent/infrastructure/brave_search.py` | Brave, with the fallback chain of profiles. |
| `search_agent/infrastructure/extractor.py` | Page text: HTTP and trafilatura first, a headless browser as the fallback. |
| `search_agent/infrastructure/source_handlers.py` | Per-source extraction: Reddit, arXiv, Crossref, Semantic Scholar, GitHub. |
| `search_agent/infrastructure/scholarly_sources.py` | The API calls behind those handlers. |
| `search_agent/config/profiles.py` | The 15 search profiles. |
| `search_agent/evaluation.py` | The scoring of a dataset against the expectations. |
| `search_agent/eval/` | The run artifacts and the comparison between two runs. |
| `eval_data/` | The datasets, and `eval_data/README.md` holds the schema of a row. |
| `bench/` | Two diagnostics: the scored search results with the routing breakdown, and the truth table of the older stop rule. |

## Traps

- **`CLAUDE.md` describes a pipeline that is gone.** It points at
  `search_agent/application/use_cases.py`,
  `search_agent/application/legacy_steps.py` and
  `search_agent/infrastructure/wikipedia_api.py`. None of those files exist
  any more. The orchestrator is
  `search_agent/application/unified_runner.py`, and the loop it runs has no
  intents, no decomposition and no per-claim verification. The Russian README
  matches the code; `CLAUDE.md` does not.

- **The `[N]` citation numbers are positional.** `assess_and_answer` takes the
  final capped passage list and does not re-cap or re-number it, and the
  runner builds its `passage_index` from that same list. Sorting, capping or
  filtering inside `intelligence.py` silently repoints every citation in the
  answer.

- **The stop condition counts hosts, and it needs two.** `_stop_check`
  returns False while any key claim has fewer than two distinct hosts behind
  it. An answer the model is completely sure about, drawn from a single site,
  still runs the loop to the iteration cap.

- **Iteration 1 makes no model call for the queries.** The queries come from
  the cache the classify call filled. A question that settles in one
  iteration therefore costs two model calls, and one that needs the second
  costs four. When the classify call fails or returns no queries,
  `generate_queries_unified` falls back to a call of its own.

- **A relative date in the question costs one more call.**
  `_normalize_time_references` tries the deterministic replacement first and
  calls the model only when that changes nothing and the question asks for
  freshness.

- **The report always holds exactly one claim.** `_build_synthetic_claim`
  creates a claim called `unified-1` whose text is the user's question, so the
  CLI prints "Claims (1)" whatever the question was. Nothing is being
  decomposed.

- **The fetch layer is called with `intent="synthesis"`.** That is how the
  runner gets the highest fetch ceiling, and it is also why the reference
  domains in `tuning.AUTHORITY_DOMAINS` get 15000 characters instead of
  4000. The synthetic classification uses the same intent so that the older
  evaluation gates stay out of the way.

- **The gate runs over every snapshot so far.** A hit from the first
  iteration can still be in the final 10 to 20 URLs. `_merge_gated` unions by
  canonical URL and keeps the entry with the higher source score, and a URL
  that was already fetched is never fetched again.

- **`get_settings()` and the use case are both cached.** The `-S` flag sets
  `SEARCH_PROVIDER_OVERRIDE` in the environment and then clears both caches
  by hand. Anything else that changes the environment in the same process has
  to do the same, or the old settings stay in force.

- **A passage is at most 420 characters, and one under 60 is dropped.** The
  split accumulates whole sentences, so the real size depends on where the
  sentence boundaries fall.

- **The deep fetch tries HTTP first.** A page whose HTTP text is under 1200
  characters is fetched again through Playwright. `FETCH_TRY_HTTP_FIRST` is
  left over and unused. Reddit, arXiv, Crossref, Semantic Scholar and GitHub
  have their own handlers and skip both paths.

- **The DuckDuckGo Wikipedia error is expected.** The library builds hostnames
  such as `wt.wikipedia.org` from the region prefix, and they do not resolve.
  The gateway turns that into an empty result and does not retry.

- **Importing the package pulls in the model stack.**
  `search_agent/__init__.py` imports the bootstrap, so `from search_agent
  import tuning` loads pydantic-ai and logfire with it. The tests need the
  dependencies installed, and Python 3.11, because the code uses
  `datetime.UTC`.

- **The Windows console needs code page 65001.** `ensure_utf8_stdio` sets it
  before Rich is built, so Cyrillic text does not arrive as mojibake.
  `PYTHONUTF8=1` does the same from the environment.

- **The source list inside the answer is headed "Источники:".** The footer is
  built in `assess_and_answer` and the heading is fixed, so an English answer
  carries a Russian one. The panel around the answer says "Answer" in
  English whatever the query language, because the panel title is set in
  `app.py` and `cli.py` rather than by `_compose_ui_labels`, which the
  unified runner never reaches.

- **`bench/stop_table.py` prints the truth table of the older stop rule.**
  It replicates `should_stop_claim_loop` by hand, without importing the
  tuning module, and the runner uses `_stop_check` in `unified_runner.py`
  instead. The two rules are not the same.

- **A stalled model call ends the iteration.** `LLM_REQUEST_TIMEOUT` is 30
  seconds, and `assess_and_answer` catches the failure and returns an empty
  assessment with a confidence of 0.0, which sends the loop into its next
  iteration.

## Style / safety rails

- **`search_agent/tuning.py` holds the constants and the environment holds the
  keys.** A new cap belongs in one of them, and only the few that genuinely
  need to change per machine get a `resolved_*` method on `AppSettings`.
- **One model call per iteration writes the answer and grades it.** Splitting
  it back into two calls costs a round trip and breaks the link between the
  answer and the claims it reports.
- **Prefer `get_settings()` over scattered `os.getenv`.**
- **Run `eval_data/quality_smoke.jsonl` when a change can affect the
  quality**, and consider `control_dataset.jsonl` before calling it done.
- **Do not commit `.env` or `eval_runs/`.**
- **Never paste a real API key into an issue, a commit or a chat log.**
- **Match the surrounding style, and leave unrelated code alone.** The
  scoring weights are the exception: they are tuned, and changing one needs
  a dataset run to justify it.
