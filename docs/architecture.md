# Architecture

How self-hosted-search-agent is put together.

Four layers inside one package: `domain` holds the data, `application` holds
the loop, `infrastructure` holds the calls to the outside world, and `config`
holds the profiles. The `eval` folder holds the harness that scores a run.

For what the agent does and how to run it, read the
[README](../README.md). For the traps, read [AGENTS.md](../AGENTS.md). For
the screen it writes to, read [../DESIGN.md](../DESIGN.md).

## The shape

| Module | What it is responsible for |
| --- | --- |
| `search_agent/application/unified_runner.py` | The loop. The only place an iteration is driven from. |
| `search_agent/application/agent_steps.py` | The helper steps the runner calls, and the facade over the rest of the scoring code. |
| `search_agent/application/agent_sources.py` | Source scoring, the SERP gate, the fetch plan and the passage split. |
| `search_agent/application/agent_compose.py` | The answer composition of the older pipeline, and the search-cost estimate. |
| `search_agent/application/agent_passage_scoring.py` | The cheap TF-IDF passage filter and the utility rerank. |
| `search_agent/application/agent_evidence.py` | The evidence bundle and the older stop condition. |
| `search_agent/application/claim_policy.py` | The per-shape rules: how many sources a claim needs, whether it needs a primary source. |
| `search_agent/application/policy_tuning.py` | The weights of every scoring rule. |
| `search_agent/application/contracts.py` | The five ports. The boundary between the layers. |
| `search_agent/domain/assessment.py` | `Assessment` and `KeyClaim`, the output of the answer call. |
| `search_agent/domain/models.py` | Claims, passages, snapshots, the run result and the audit trail. |
| `search_agent/domain/source_priors.py` | The per-domain priors for known hosts, and the SEO spam risk. |
| `search_agent/infrastructure/intelligence.py` | The model calls: classify, generate queries, answer and assess. |
| `search_agent/infrastructure/caching_search_gateway.py` | The search cache and the per-run call budget. |
| `search_agent/infrastructure/ddgs_gateway.py` | DuckDuckGo, with the retries. |
| `search_agent/infrastructure/brave_search.py` | Brave, with the fallback chain of profiles. |
| `search_agent/infrastructure/extractor.py` | The page text: HTTP and trafilatura first, a headless browser as the fallback. |
| `search_agent/infrastructure/source_handlers.py` | The per-source extraction: Reddit, arXiv, Crossref, Semantic Scholar, GitHub. |
| `search_agent/infrastructure/scholarly_sources.py` | The API calls behind those handlers. |
| `search_agent/infrastructure/llm_tasks.py` | The task-specific model calls with their own token caps. |
| `search_agent/config/profiles.py` | The 15 search profiles. |
| `search_agent/settings.py` | The environment settings, and the `resolved_*` methods that let a few caps be overridden. |
| `search_agent/tuning.py` | The internal constants. Not configurable from the environment. |
| `search_agent/bootstrap.py` | The wiring. It builds the use case out of the five ports. |
| `search_agent/app.py` | The command line: one query, a dataset run, the arXiv mode. |
| `search_agent/cli.py` | The interactive prompt, the profile picker, the keyword heuristics and the history file. |
| `search_agent/evaluation.py` | The scoring of a dataset against the expectations. |
| `search_agent/eval/` | The run artifacts and the comparison between two runs. |

## The data flow

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

The ports in `application/contracts.py` are the five the application needs:
`QueryIntelligencePort`, `SearchGatewayPort`, `FetchGatewayPort`,
`ReceiptWriterPort` and `StepLibraryPort`. `bootstrap.py` supplies the
implementations from `infrastructure` and hands the use case to `app.py` or to
`cli.py`.

The numbers in the flow come from `tuning.py`: `AGENT_MAX_CLAIM_ITERATIONS`
is 2, `SERP_GATE_MIN_URLS` and `SERP_GATE_MAX_URLS` are 10 and 20,
`SYNTHESIS_PASSAGE_LIMIT` is 25, `CHEAP_PASSAGE_LIMIT` is 12, and the fetch
limits are 5 shallow pages on the first iteration and 10 on the second, with
2 deep pages either way.

## Decisions worth knowing about

### The answer and its own assessment come from one call

`assess_and_answer` returns the markdown answer, the claims behind it, a
confidence, the gaps and the contradiction flag in one response. The runner
checks the two-domain rule against the claims of that same response, without
a second pass over the answer.

The cost is that the answer and the assessment cannot disagree: they are the
same object. Splitting them into two calls costs a round trip and breaks the
link.

### The citation numbers are positional

The model receives the final capped passage list and does not renumber it.
The runner builds its `passage_index` from that same list, so the `[N]` in the
answer and the `N` in the source list are the same number.

The cost is that sorting, capping or filtering inside `intelligence.py`
silently repoints every citation in the answer. The numbers in the prompt and
the numbers in the source list come from the same list in the same order, and
the runner rebuilds the index from the capped list and not from the ranked
one.

### The stop condition counts hosts and needs two

`_stop_check` returns False while any key claim has fewer than two distinct
hosts behind it. Two pages on the same host are one source, so a site that
repeats itself cannot satisfy the rule. The loop ends on the contradiction
flag, on a confidence of at least 0.75 together with two hosts behind every
claim, or on the iteration cap.

The cost is that an answer the model is completely sure about, drawn from a
single site, still runs the loop to the cap.

### Iteration one makes no model call for the queries

The queries come from the cache that the classify call filled. A question
that settles in one iteration costs two model calls, and one that needs the
second costs four. When the classify call fails or returns no queries,
`generate_queries_unified` falls back to a call of its own.

### The fetch layer is called with one intent

The intent decides the character cap, so the reference domains in
`tuning.AUTHORITY_DOMAINS` get 15000 characters instead of the 4000 of
`tuning.EXTRACT_MAX_CHARS`. The number of pages does not come from the
intent: it comes from the iteration number and the profile.

### The gate runs over every snapshot so far

A hit from the first iteration can still be in the final 10 to 20 URLs.
`_merge_gated` unions by canonical URL and keeps the entry with the higher
source score, and a URL that was already fetched is never fetched again.

### The report always holds exactly one claim

`_build_synthetic_claim` creates a claim called `unified-1` whose text is the
question. Nothing is decomposed, and the CLI prints "Claims (1)" whatever
the question was. The claim exists so that the evaluation harness, written
for the older claim-per-claim pipeline, can still score a run.

### The fetch tries HTTP first

A page whose HTTP text is under 1200 characters is fetched again through
Playwright. Reddit, arXiv, Crossref, Semantic Scholar and GitHub have their
own handlers and skip both paths. `FETCH_TRY_HTTP_FIRST` is left over and
unused.

### The settings and the use case are both cached

`get_settings()` caches, and the use case caches. The `-S` flag sets
`SEARCH_PROVIDER_OVERRIDE` in the environment and then clears both caches by
hand. Anything else that changes the environment in the same process has to
do the same, or the old settings stay in force.

## Boundaries

- **`domain` imports nothing from `application` or `infrastructure`.** It
  holds the data and the rules about it.
- **`application` talks to the outside world only through the five ports.**
  `contracts.py` is that whole boundary, and the runner is written against
  them.
- **`infrastructure` knows nothing about the loop.** It makes the calls and
  returns the data.
- **`config/profiles.py` holds the 15 profiles and nothing else.** The
  profile is a value that goes into a search, not a strategy.
- **`tuning.py` holds the constants and the environment holds the keys.** A
  new cap belongs in one of them, and only the few that genuinely change per
  machine get a `resolved_*` method on `AppSettings`.
- **`app.py` and `cli.py` are the only entry points.** Everything else is
  called by one of them or by the tests.
- **The evaluation harness does not import the runner.** It runs the use case
  through the same ports and scores the result it gets.
