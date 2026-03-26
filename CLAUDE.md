# Search Agent — Architecture Reference

## Quick commands

```bash
# Single query
uv run python -m search_agent -S ddgs -q "your query"

# Eval on control dataset
uv run python -m search_agent.eval --dataset eval_data/control_dataset.jsonl --label my-label

# Compare two eval runs
uv run python -m search_agent.eval compare eval_runs/eval_A.json eval_runs/eval_B.json
```

---

## File map

### Root
| File | Purpose |
|------|---------|
| `__main__.py` | CLI entrypoint → delegates to `cli.py` (interactive) or `app.py` (single/eval) |
| `app.py` | Non-interactive: single query, eval mode, research mode |
| `cli.py` | Interactive REPL; profile suggestion heuristics; history |
| `bootstrap.py` | **DI factory** — assembles all ports into `SearchAgentUseCase` |
| `settings.py` | Pydantic BaseSettings; env vars; fallback chains |
| `tuning.py` | Internal constants (not env-configurable) — see section below |
| `evaluation.py` | Eval harness: loads JSONL, runs agent, scores, aggregates metrics |
| `runtime_bootstrap.py` | UTF-8 stdio fix for Windows |

### Application layer (`application/`)
| File | Purpose |
|------|---------|
| `contracts.py` | **Port/Protocol interfaces** — the architectural boundary |
| `use_cases.py` | `SearchAgentUseCase` — main orchestrator; claim loop; parallelism |
| `agent_steps.py` | ~2200 lines — implements `StepLibraryPort`; all step functions |
| `legacy_steps.py` | Thin wrapper class delegating to `agent_steps.py` functions |
| `text_heuristics.py` | Entity extraction, time scope, comparison markers, freshness signals |

### Domain (`domain/`)
| File | Purpose |
|------|---------|
| `models.py` | All dataclasses: `Claim`, `SearchSnapshot`, `GatedSerpResult`, `Passage`, `RoutingDecision`, `VerificationResult`, `EvidenceBundle`, `ClaimRun`, `AgentRunResult`, `AuditTrail` |
| `source_priors.py` | Domain-type priors; SEO spam risk; `lookup_source_prior()` |

### Infrastructure (`infrastructure/`)
| File | Purpose |
|------|---------|
| `intelligence.py` | `PydanticAIQueryIntelligence` — LLM classify/decompose/verify/synthesize |
| `search_gateway.py` | `BraveSearchGateway` |
| `ddgs_gateway.py` | `DDGSSearchGateway` |
| `caching_search_gateway.py` | Decorator: thread-safe SERP cache + search budget; **double-check locking pattern** |
| `gateway_factory.py` | Picks brave vs ddgs per settings |
| `fetch_gateway.py` | `LegacyFetchGateway` — wraps `agent_steps.fetch_claim_documents()` |
| `extractor.py` | Page extraction: trafilatura (fast HTTP) + crawl4ai (Playwright); shallow/deep modes |
| `llm_tasks.py` | `PydanticAITaskRunner.answer_with_sources()` — grounded markdown answer |
| `llm.py` | Legacy LLM task runner |
| `pydantic_ai_factory.py` | OpenAI client builder; temperature/token config |
| `receipt_gateway.py` | `JsonReceiptWriter` — persists audit trails |
| `receipts.py` | JSON serialization of audit trail |
| `serp_query.py` | Query routing; bang expansion; site: filters |
| `source_handlers.py` | Domain-specific extraction (Reddit, etc.) |
| `url_utils.py` | URL normalization; canonical URL |
| `telemetry.py` | Logfire config |
| `llm_log.py` | LLM call logging (latency, token counts) |

### Config (`config/`)
| File | Purpose |
|------|---------|
| `profiles.py` | `SearchProfile` dataclass; predefined profiles: `web`, `news`, `wiki`, `arxiv`, `tech`, `ru`, etc. |

### Eval (`eval/`)
| File | Purpose |
|------|---------|
| `__main__.py` | Eval CLI |
| `tracking.py` | Save/load eval JSON artifacts; compute metric deltas |
| `compare_cli.py` | Diff two eval runs |

Eval results saved to: `eval_runs/eval_YYYYMMDDTHHMMSSZ_<git8>_<dataset>.json`

---

## Data flow

```
query: str  →  classify_query()  →  QueryClassification(intent, complexity, needs_freshness)
                                           │
                             decompose_claims()  [skipped if comparison intent]
                                           │
                               list[Claim]  (max 3, see DECOMPOSE_MAX_CLAIMS)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │   Per claim (parallel, max AGENT_MAX_PARALLEL_CLAIMS=4)  │
                    │                                              │
                    │  iteration 1..AGENT_MAX_CLAIM_ITERATIONS:   │
                    │    A. build_query_variants()                 │
                    │       → broad, entity_locked, exact_match    │
                    │         (iter1: capped at 3 variants)        │
                    │                                              │
                    │    B. search_variant() × N variants          │
                    │       (parallel via ThreadPoolExecutor)      │
                    │       [CachingBudgetSearchGateway: cache +   │
                    │        budget guard]                         │
                    │                                              │
                    │    C. gate_serp_results()                    │
                    │       → filter by domain type, SEO spam,     │
                    │         entity match, source prior           │
                    │         (10–20 URLs kept)                    │
                    │                                              │
                    │    D. route_claim_retrieval()                │
                    │       → short_path | targeted_retrieval |    │
                    │         iterative_loop                       │
                    │       (certainty × consistency × sufficiency)│
                    │                                              │
                    │    E. fetch_claim_documents()                │
                    │       (iter1: fast limits 3/4/5)             │
                    │       (iter2+: full limits 5/8/10)           │
                    │       [shared page cache across claims]      │
                    │                                              │
                    │    F. split_into_passages() → ~500-char chunks│
                    │    G. cheap_passage_filter() → top 12        │
                    │    H. utility_rerank_passages() → top K      │
                    │                                              │
                    │    I. verify_claim(claim, top_K_passages)    │
                    │       → verdict: supported | contradicted |  │
                    │         insufficient_evidence                │
                    │       [verify_claim cache: keyed on prompt]  │
                    │                                              │
                    │    J. build_evidence_bundle()                │
                    │    K. should_stop_claim_loop()               │
                    │       → stop if confident or max iterations  │
                    │                                              │
                    │    if not stop: refine_query_variants()       │
                    │      → freshness, source-restricted, aliases  │
                    └──────────────────────┬──────────────────────┘
                                           │
               [comparison intent] synthesize_answer(query, all_passages)
               [factual intent]    compose_answer(report)  ← LLM-grounded markdown
                                           │
                                  AgentRunResult(answer, claims, audit_trail)
```

---

## Port → Implementation map

| Port | Impl class | File |
|------|-----------|------|
| `QueryIntelligencePort` | `PydanticAIQueryIntelligence` | `infrastructure/intelligence.py` |
| `SearchGatewayPort` | `BraveSearchGateway` / `DDGSSearchGateway` | `infrastructure/search_gateway.py` / `ddgs_gateway.py` |
| *(decorator)* | `CachingBudgetSearchGateway` | `infrastructure/caching_search_gateway.py` |
| `FetchGatewayPort` | `LegacyFetchGateway` | `infrastructure/fetch_gateway.py` |
| `ReceiptWriterPort` | `JsonReceiptWriter` | `infrastructure/receipt_gateway.py` |
| `StepLibraryPort` | `LegacyAgentStepLibrary` | `application/legacy_steps.py` |

DI wired in `bootstrap.py → build_search_agent_use_case()`.

---

## tuning.py constants

### Agent loop
| Constant | Value | Notes |
|----------|-------|-------|
| `AGENT_MAX_CLAIM_ITERATIONS` | 2 | 1 = no refinement |
| `DECOMPOSE_MAX_CLAIMS` | 3 | 0 = unlimited |
| `AGENT_MAX_PARALLEL_CLAIMS` | 4 | 1 = sequential |
| `AGENT_MAX_SEARCH_CALLS_PER_RUN` | 0 | 0 = unlimited |
| `AGENT_MAX_QUERY_VARIANTS` | 6 | across all iterations |
| `AGENT_MAX_QUERY_VARIANTS_ITER1` | 3 | iter1 cap (broad+entity_locked+exact) |
| `AGENT_MAX_REFINE_VARIANTS` | 12 | available on iter2+ |
| `COMPARISON_SKIP_DECOMPOSE` | True | skip sub-claim decomp for comparison queries |

### Fetch limits
| Constant | Value | Notes |
|----------|-------|-------|
| `SHALLOW_FETCH_SHORT_FAST_LIMIT` | 3 | iter1, short_path |
| `SHALLOW_FETCH_TARGETED_FAST_LIMIT` | 4 | iter1, targeted_retrieval |
| `SHALLOW_FETCH_ITERATIVE_FAST_LIMIT` | 5 | iter1, iterative_loop |
| `SHALLOW_FETCH_SHORT_LIMIT` | 5 | iter2+, short_path |
| `SHALLOW_FETCH_TARGETED_LIMIT` | 8 | iter2+, targeted_retrieval |
| `SHALLOW_FETCH_ITERATIVE_LIMIT` | 10 | iter2+, iterative_loop |
| `DEEP_FETCH_SHORT_LIMIT` | 1 | deep cap for short_path |
| `DEEP_FETCH_TARGETED_LIMIT` | 2 | deep cap for targeted |
| `DEEP_FETCH_ITERATIVE_LIMIT` | 2 | deep cap for iterative |
| `FETCH_SHALLOW_CONCURRENCY` | 5 | parallel HTTP fetches per claim |

### Passages
| Constant | Value | Notes |
|----------|-------|-------|
| `CHEAP_PASSAGE_LIMIT` | 12 | TF-IDF filter keeps top N |
| `AGENT_PASSAGE_TOP_K` | 8 | sent to verify_claim |
| `SERP_GATE_MIN_URLS` | 10 | min after gating |
| `SERP_GATE_MAX_URLS` | 20 | max after gating |

### LLM token budgets
| Constant | Value | Notes |
|----------|-------|-------|
| `CLAIM_DECOMPOSE_MAX_TOKENS` | 500 | |
| `VERIFY_CLAIM_MAX_TOKENS` | 700 | |
| `SYNTHESIZE_ANSWER_MAX_TOKENS` | 2000 | comparison queries |
| `COMPOSE_ANSWER_MAX_TOKENS` | 1600 | factual queries |
| `TIME_NORMALIZE_MAX_TOKENS` | 120 | |

### Extractor
| Constant | Value | Notes |
|----------|-------|-------|
| `SHALLOW_FETCH_TIMEOUT` | 8 | HTTP timeout (sec) |
| `EXTRACT_MAX_CHARS` | 4000 | max chars per document |
| `CRAWL4AI_TIMEOUT` | 25 | Playwright timeout |
| `FETCH_HTTP_MIN_CHARS` | 1200 | below this → escalate to crawl4ai |

---

## Eval metrics

| Metric | What it measures |
|--------|-----------------|
| `claim_support_rate` | supported verdicts / expected supported |
| `contradiction_detection_rate` | contradicted hits / expected contradicted |
| `insufficient_detection_rate` | insufficient hits / expected insufficient |
| `route_match_rate` | actual route in expected_routes set |
| `source_requirement_rate` | independent_sources ≥ min_required |
| `primary_requirement_rate` | primary source present when required |
| `citation_validity_rate` | cited URLs found in actual passages |
| `unsupported_statement_rate` | answer bullets without claim support |
| `median_answer_latency` | ms (p50 across 11 cases) |
| `median_search_cost` | variants + 0.25×shallow + 1.0×deep + 0.5×claims |
| `avg_iterations_per_claim` | mean iterations |

### Baseline (pre-optimization)
`median_answer_latency: 20,743ms`

### Current best (parallel SERP + adaptive fetch + normalize cache + page cache + claim cap)
`median_answer_latency: ~11,400–13,400ms` (-35–45%), quality metrics at baseline or better.

---

## Key optimizations applied

1. **Adaptive fetch (iter1 fast limits)** — iter1 fetches 3/4/5 instead of 5/8/10 URLs
2. **Variant cap iter1 = 3** — no refinement variants on first pass
3. **Parallel SERP** — `ThreadPoolExecutor` in `use_cases.py`; `CachingBudgetSearchGateway` uses double-check locking (lock released during network I/O)
4. **normalize_time cache** — `intelligence.py._normalize_cache` keyed on raw query string
5. **verify_claim cache** — keyed on full prompt (claim + passage texts)
6. **Cross-claim page cache** — `FetchGatewayPort` shared `dict[url, str]`; prevents re-downloading same URL across claims
7. **Comparison intent bypass** — `COMPARISON_SKIP_DECOMPOSE=True`; single-claim search + `synthesize_answer` instead of 3–4 sub-claims + `verify_claim×N`

## Known regressions / behaviour notes

- `source_requirement_rate` varies 0.67–0.89 across runs due to DDGS non-determinism (different pages returned each call)
- DDGS Wikipedia connector errors (`wt.wikipedia.org ConnectError`) are expected and harmless — Wikipedia API is unreachable, DDGS falls back to web results
- qwen3.5-35b-a3b uses `<think>` tokens before output — this consumes part of `max_tokens`; `SYNTHESIZE_ANSWER_MAX_TOKENS=2000` accounts for ~800 thinking tokens
- Snippet-first optimization was tried and **reverted** — model confidence on short snippets always ~0.38, never reached 0.85 threshold

---

## Environment variables (`.env`)

```env
LLM_API_KEY=...
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen3.5-35b-a3b
SEARCH_PROVIDER=ddgs          # or brave
BRAVE_API_KEY=...             # if using brave
```

Override tuning at runtime (optional):
```env
EXTRACT_MAX_CHARS=6000
VERIFY_CLAIM_MAX_TOKENS=900
SYNTHESIZE_ANSWER_MAX_TOKENS=3000
```
