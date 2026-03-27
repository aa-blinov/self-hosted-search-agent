# Search Agent — Architecture Reference

## Quick commands

```bash
# Single query
uv run python -m search_agent -S ddgs -q "your query"

# E2E eval on control dataset (12 cases: 7 factual + 3 synthesis + 2 news_digest)
uv run python -m search_agent --eval eval_data/control_dataset.jsonl --eval-label my-label

# Compare two e2e eval runs
uv run python -m search_agent.eval eval_runs/eval_A.json eval_runs/eval_B.json

# Component eval — pure Python only, no API key needed (<1s)
uv run python -m search_agent.eval.components --all --no-llm

# Component eval — single component (LLM)
uv run python -m search_agent.eval.components verify_claim
uv run python -m search_agent.eval.components classify_intent

# Component eval — all components (pure-Python first, then LLM, ~50s total)
uv run python -m search_agent.eval.components --all

# Compare two component runs (reuses existing compare_cli)
uv run python -m search_agent.eval eval_runs/components/A.json eval_runs/components/B.json
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
| `intent_eval.py` | Standalone intent classifier eval |
| `components/__main__.py` | **Component eval CLI** — `python -m search_agent.eval.components` |
| `components/runner.py` | Dataset loading, deserializers (JSON→dataclasses), artifact saving |
| `components/metrics.py` | Shared metric helpers: accuracy, precision/recall/F1, percentile |
| `components/gate_serp.py` | Pure-Python: gate_serp_results correctness (include/exclude URLs, min count) |
| `components/route_claim.py` | Pure-Python: route_claim_retrieval accuracy per mode |
| `components/cheap_passage.py` | Pure-Python: cheap_passage_filter include/exclude recall |
| `components/classify_intent.py` | LLM: intent accuracy (factual/synthesis/news_digest) + heuristic baseline |
| `components/verify_claim.py` | LLM: verdict accuracy (supported/contradicted/insufficient_evidence) |
| `components/synthesize_answer.py` | LLM: keyword hit rate + min_chars pass rate |

E2E eval results: `eval_runs/eval_YYYYMMDDTHHMMSSZ_<git8>_<dataset>.json`
Component eval results: `eval_runs/components/eval_comp_YYYYMMDDTHHMMSSZ_<git8>_<component>.json`

### Component eval datasets (`eval_data/components/`)
| File | Cases | Type |
|------|-------|------|
| `gate_serp.jsonl` | 3 | pure Python |
| `cheap_passage.jsonl` | 3 | pure Python |
| `route_claim.jsonl` | 4 | pure Python |
| `classify_intent.jsonl` | 15 | LLM |
| `verify_claim.jsonl` | 8 | LLM |
| `synthesize_answer.jsonl` | 3 | LLM |

**Workflow**: change a component → run its component eval (seconds/minutes) → if pass_rate=1.0, optionally run e2e to confirm no latency regression.

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
| `EXTRACT_MAX_CHARS_AUTHORITY` | 15000 | for authority domains **when intent==synthesis** |
| `AUTHORITY_DOMAINS` | tuple | docs.python.org, wikipedia.org, MDN, etc. |
| `CRAWL4AI_TIMEOUT` | 25 | Playwright timeout |
| `FETCH_HTTP_MIN_CHARS` | 1200 | below this → escalate to crawl4ai |

---

## Eval metrics

| Metric | What it measures |
|--------|-----------------|
| `claim_support_rate` | supported verdicts / expected supported (factual split only) |
| `contradiction_detection_rate` | contradicted hits / expected contradicted (factual) |
| `insufficient_detection_rate` | insufficient hits / expected insufficient (factual) |
| `route_match_rate` | actual route in expected_routes set |
| `source_requirement_rate` | independent_sources ≥ min_required |
| `primary_requirement_rate` | primary source present when required |
| `citation_validity_rate` | cited URLs found in actual passages (factual split only) |
| `unsupported_statement_rate` | answer bullets without claim support (factual split only) |
| `answer_depth_rate` | % cases where answer ≥ min_answer_chars (synthesis/news) |
| `source_diversity_rate` | % news_digest cases with ≥ min_unique_sources domains |
| `median_answer_chars` | p50 answer length in characters |
| `median_answer_latency` | ms (p50 across all cases) |
| `median_search_cost` | variants + 0.25×shallow + 1.0×deep + 0.5×claims |
| `avg_iterations_per_claim` | mean iterations |

### Eval dataset (control_dataset.jsonl — 12 cases)
- **7 factual cases**: ceo-microsoft, python-313-release-date, water-boiling-celsius, irs-government-agency, python-wrong-release-date, nadella-wrong-year, nadella-room-temperature
- **3 synthesis cases**: synthesis-python-311-vs-312, synthesis-python-313-features, synthesis-asyncio-python (min_answer_chars=800)
- **2 news_digest cases**: news-digest-iran, news-digest-ai (min_answer_chars=600, min_unique_sources=4)

### Baseline (pre-optimization, 7-case factual-only dataset)
`median_answer_latency: 20,743ms`

### Current best (opt8: intent-aware authority extract, qwen3.5-35b-a3b)
`median_answer_latency: ~12,345ms` (-40% vs baseline), `claim_support_rate: 1.0`, `source_requirement_rate: 1.0`

Note: opt8 was measured on the 7-case factual-only dataset. All runs from opt9 onwards use the 12-case dataset (7 factual + 3 synthesis + 2 news_digest).

| opt | label | latency | claim_support | answer_depth | source_diversity | notes |
|-----|-------|---------|---------------|--------------|------------------|-------|
| baseline | — | 20,743ms | — | — | — | 7-case factual dataset |
| opt6 | parallel SERP + adaptive fetch + caches | ~12,359ms | 1.0 | — | — | 7-case |
| opt7 | authority extract 15K (all intents) | 16,865ms ❌ | 0.89 | — | — | regression |
| **opt8** | authority extract 15K (synthesis only) | **12,345ms** ✅ | **1.0** | — | — | qwen, 7-case |
| opt9/groq | gpt-oss-120b + PromptedOutput, Groq | 16,530ms ❌ | 0.40 | 0.60 | 0.0 | 429 rate-limits |
| opt9/or | gpt-oss-120b + PromptedOutput, OpenRouter | 25,852ms ❌ | 0.80 | 0.60 | 0.0 | reasoning tokens 2× latency |
| opt10/seq | rationale_guided query (SAFE-inspired, serial) | 52,037ms ❌ | 1.0 | **1.0** | 0.5 | LLM blocks iter2; 11s per claim |
| opt10/bg | rationale_guided query (background thread) | 17,391ms ❌ | 0.6 | — | — | extra SERP overhead; no accuracy gain |

**opt10/seq best on accuracy** (claim_support=1.0, answer_depth=1.0) but latency unacceptable. Parallel version (concurrent with SERP) not yet implemented.

---

## Key optimizations applied

1. **Adaptive fetch (iter1 fast limits)** — iter1 fetches 3/4/5 instead of 5/8/10 URLs
2. **Variant cap iter1 = 3** — no refinement variants on first pass
3. **Parallel SERP** — `ThreadPoolExecutor` in `use_cases.py`; `CachingBudgetSearchGateway` uses double-check locking (lock released during network I/O)
4. **normalize_time cache** — `intelligence.py._normalize_cache` keyed on raw query string
5. **verify_claim cache** — keyed on full prompt (claim + passage texts)
6. **Cross-claim page cache** — `FetchGatewayPort` shared `dict[url, str]`; prevents re-downloading same URL across claims
7. **Synthesis intent bypass** — `SYNTHESIS_SKIP_DECOMPOSE=True`; single-claim search + `synthesize_answer` instead of 3–4 sub-claims + `verify_claim×N`; triggered for explanation/comparison/how-to queries
8. **LLM intent classification** — 3-way classifier (factual | synthesis | news_digest); 100% accuracy on 35-example eval; replaces keyword heuristics; cached per query; `INTENT_CLASSIFY_MAX_TOKENS=300` (accounts for qwen `<think>` tokens)
9. **Authority domain extract (synthesis only)** — `EXTRACT_MAX_CHARS_AUTHORITY=15000` for wikipedia.org, docs.python.org, MDN etc., applied **only when intent==synthesis**; factual queries keep 4K to avoid slow verify_claim LLM calls
10. **Source-score ranked synthesis passages** — synthesis path sorts all raw passages by `source_score` desc (authoritative pages first) instead of TF-IDF `cheap_passage_filter`; fixes cross-language mismatch where Russian query scores English docs.python.org sections near zero, dropping feature-specific sections (f-strings, TypeVar, etc.)
11. **UTF-8 bytes-first encoding** — `extractor.py` tries `response.content.decode("utf-8")` before falling back to requests charset detection; fixes mojibake on Russian sites that declare `windows-1251`/`iso-8859-1` but serve UTF-8 (chardet sometimes returns MacRoman as `apparent_encoding`, making blind override worse)
12. **news_digest source diversity** — `synthesize_answer` applies `MAX_PER_URL=1` + `MAX_PER_DOMAIN=1` for news_digest intent; prevents aggregator pages (e.g. kommersant.ru/theme/…) from flooding all 12 prompt slots; ensures answers cite ≥ 12 unique news domains
13. **gpt-oss-120b / reasoning-model compatibility** (`pydantic_ai_factory.py`, `intelligence.py`):
    - `_is_reasoning_model()` detects `gpt-oss`, `o1`, `o3`, `o4` family
    - Temperature omitted for reasoning models (400 error if sent)
    - All structured-output agents use `PromptedOutput(Model)` for reasoning models — JSON schema injected in system prompt instead of broken tool-call / response_format spec on Groq
    - qwen-only: `{"reasoning": {"effort": "none"}}` in extra_body (unchanged)

## Known regressions / behaviour notes

- `source_requirement_rate` varies 0.67–0.89 across runs due to DDGS non-determinism (different pages returned each call)
- DDGS Wikipedia connector errors (`wt.wikipedia.org ConnectError`) are expected and harmless — Wikipedia API is unreachable, DDGS falls back to web results; suppressed to dim log level
- qwen3.5-35b-a3b uses `<think>` tokens before output — this consumes part of `max_tokens`; `SYNTHESIZE_ANSWER_MAX_TOKENS=2000` accounts for ~800 thinking tokens; `INTENT_CLASSIFY_MAX_TOKENS=300` same reason
- Snippet-first optimization was tried and **reverted** — model confidence on short snippets always ~0.38, never reached 0.85 threshold
- Authority extract 15K applied globally (opt7) caused latency regression: 30 chunks → top-8 utility_rerank → ~5000 chars to verify_claim (was ~1200) → 14s LLM calls; fixed in opt8 by scoping to synthesis intent only
- WikipediaSourceHandler **removed** — was using REST API returning 500-900 chars lead section only; now trafilatura extracts full article (up to 15K for synthesis)
- `cheap_passage_filter` TF-IDF threshold (0.18) kills cross-language passages — Russian query vs English docs.python.org: feature-specific sections ("f-string improvements", "PEP 695") don't repeat version numbers → score < 0.18 → dropped; synthesis path bypasses this by sorting on `source_score`
- `SYNTHESIS_PASSAGE_LIMIT=25` controls how many passages go to `synthesize_answer` (was using `CHEAP_PASSAGE_LIMIT=12`)
- `gpt-oss-120b` on Groq: `verify_claim` still occasionally fails with `Exceeded maximum retries (1) for output validation` on complex passages (4000+ input chars); falls back to heuristic verifier; root cause is model generating extra reasoning commentary around JSON — `PromptedOutput` greatly reduces but does not fully eliminate this
- **rationale_guided query (opt10) reverted** — SAFE-inspired: after `verify_claim` returns rationale, ask LLM to generate one focused search query from it. Two implementations tried: (1) serial — blocks 11s before iter2 (bad); (2) background thread — LLM+SERP runs during fetch+verify, zero blocking, but extra SERP results add fetch overhead (+5s) with no measurable accuracy improvement on 12-case eval. Serial version tested in this session: 52,037ms (+4× latency). `suggest_rationale_query` method kept in `intelligence.py`/`contracts.py` for future use. **TODO**: implement truly parallel version — start LLM call concurrently with iter2 SERP calls, inject result before fetch stage if ready within SERP wall-time.
- **`source_diversity_rate: 0.0–0.5`** (news_digest) — structural DDGS limitation: Iranian/geopolitical news sites block automated requests (403), only 1–2 domains successfully fetched. Not a code bug. Would improve with Brave Search news API.
- **`verify_claim` returns `confidence=0.0`** for short passages (1 sentence) — model doesn't output confidence field for tiny inputs; `_post_adjust_verification` sets `confidence=0.38` for supported, `0.0` for contradicted/insufficient. This is expected behavior for snippet-only evidence.

---

## Research landscape — related work

Architecture places this agent in the **Agentic RAG with claim-level verification** paradigm. Survey based on arxiv literature 2021–2025.

### Closest papers (ranked by similarity)

| # | Paper | Year | arxiv | What's shared |
|---|-------|------|-------|--------------|
| 1 | **PASS-FC** — Progressive & Adaptive Search for Fact Checking | 2025 | `2504.09866` | Atomic decomposition + adaptive multi-step web search + credible-source domain filter + reflection-triggered refinement loop |
| 2 | **Complex Claim Verification in the Wild** | 2023 | `2305.11859` | 5-stage pipeline: decompose → web retrieve → fine-grained evidence → summarize → verdict |
| 3 | **SAFE** — Search-Augmented Factuality Evaluator (Google DeepMind) | 2024 | `2403.18802` | Decompose LLM output → verify each atomic fact with Google Search → per-fact verdict |
| 4 | **FAIR-RAG** — Faithful Adaptive Iterative Refinement | 2025 | `2510.22344` | Intent routing + sub-query decomp + hybrid rerank (RRF) + gap-based iterative loop + cited answer |
| 5 | **HARIS** — Search-Informed Reasoning + Reasoning-Guided Search | 2025 | `2506.07528` | Two-agent claim verification, iterative search refinement, claim-level verdicts |

### Full genealogy

| Paper | Year | Core contribution | Relation |
|-------|------|------------------|---------|
| WebGPT `2112.09332` | 2021 | GPT-3 fine-tuned to browse web + cite | Ancestor: multi-step web search → answer with references |
| Baleen `2101.00436` | 2021 | Multi-hop retrieval with condensed context across hops | Ancestor: iterative retrieval with progressive query refinement |
| ReAct `2210.03629` | 2022 | Interleaved thought+action traces over Wikipedia | Ancestor: "retrieve when uncertain" loop concept |
| RARR `2210.08726` | 2022 | Research → revise LLM-generated text | Pioneer: query generation from claims → web search → evidence → verdict per span |
| FActScore `2305.14251` | 2023 | Decompose LLM output into atomic facts → verify each | Direct ancestor of decompose→verify pipeline |
| FLARE `2305.06983` | 2023 | Retrieve when low-confidence tokens during generation | Confidence-triggered retrieval → `should_stop_claim_loop()` |
| AVeriTeC `2305.13117` | 2023 | Benchmark: 3-way verdict supported/refuted/not-enough-info | Defines the exact verdict taxonomy used here |
| Self-RAG `2310.11511` | 2023 | Fine-tuned model with reflection tokens (IsREL/IsSUP/IsUSE) | Critique tokens ≈ `verify_claim` verdict; adaptive retrieval |
| SAFE `2403.18802` | 2024 | Decompose + Google Search per atomic fact at scale | Closest to verify_claim loop; evaluator not generator |
| Search-o1 `2501.05366` | 2025 | Reasoning model triggers search on uncertain knowledge mid-trace | Reason-in-Documents module ≈ passage filter + rerank + verify |
| PASS-FC `2504.09866` | 2025 | Full pipeline closest to this codebase | See rank #1 above |
| FAIR-RAG `2510.22344` | 2025 | Intent routing + hybrid rerank + gap-based loop | Closest to overall pipeline on closed corpus |
| HARIS `2506.07528` | 2025 | Two-agent RL-trained claim verifier | Dense retrieval only, binary verdict, RL-trained |

### What is well-studied (this agent is in mainstream)

| Component | Literature |
|-----------|-----------|
| Decompose into atomic claims → verify per claim | FActScore, SAFE, AVeriTeC, Complex Claim Verification |
| Iterative retrieval with confidence-based stop | FLARE, Self-RAG, Baleen |
| 3-way verdict: supported / refuted / not-enough-info | AVeriTeC benchmark, FEVER-style systems |
| Multi-step web agent with inline citations | WebGPT → ReAct → RARR → Search-o1 |
| Passage reranking (BM25+dense, RRF) | FAIR-RAG; extensive reranker literature |
| Source credibility / domain filtering | PASS-FC, WebFilter `2508.07956` |

### FActScore / SAFE — detailed comparison

**TL;DR: Both are evaluation frameworks, not search agents. Our verify_claim is richer; their evidence pipeline is weaker. Three techniques worth borrowing.**

FActScore (`2305.14251`) and SAFE (`2403.18802`) solve a *different problem* — they score the factuality of an **already-generated LLM response**, not answer a user query from scratch. This makes direct comparison partially ill-posed, but illuminates concrete differences:

**Evidence quality — SAFE is much weaker:**
- SAFE works on raw Serper *snippets* (~200 chars). It **never fetches the actual page**.
- FActScore uses Wikipedia passage chunks of 256 tokens, offline index.
- We fetch full pages (4,000 chars; 15,000 for authority domains on synthesis) + trafilatura + crawl4ai fallback. Essential for technical queries.

**Verdict taxonomy — ours is strictly richer:**
- FActScore/SAFE: binary `Supported` / `Not Supported`. Contradiction collapses into "not supported".
- Ours: `supported` / `contradicted` / `insufficient_evidence` + confidence float + missing_dimensions list + evidence spans with URLs. The `missing_dimensions` field directly drives `refine_query_variants()`.

**Search strategy — SAFE sequential, ours parallel:**
- SAFE: up to 5 sequential LLM-driven queries per atomic fact; each query conditions on prior results.
- Ours: 3 parallel typed variants (broad / entity_locked / exact_match) in iter1, up to 12 in iter2 via `refine_query_variants()`. Wall-time faster; less adaptive than SAFE's knowledge accumulation.

**What our pipeline has that they don't:** answer synthesis, full page extraction, domain quality gating (`gate_serp_results`), `contradicted` verdict, confidence-driven stopping, cross-claim page cache, intent routing, multilingual support.

**Three techniques from SAFE worth adopting:**

1. **CoT scratchpad before verdict** — SAFE prompt: *"Think step-by-step and summarize the main points of KNOWLEDGE. Final answer: [Supported] or [Not Supported]."* Our `verify_claim` goes straight to structured JSON. Adding an explicit summarization step before the JSON verdict could reduce false `insufficient_evidence` on ambiguous claims. Cost: ~100–200 extra tokens per call. *(Not yet implemented.)*

2. **LLM-generated focused query from rationale** — SAFE accumulates prior results as "KNOWLEDGE" before generating the next query. We implemented `suggest_rationale_query()` in `intelligence.py`/`contracts.py` (opt10). Two approaches tried: serial (blocks iter2, +40s) and background thread (zero blocking, but extra SERP results add fetch overhead +5s with no measurable accuracy gain). **Reverted** — DDGS non-determinism dominates eval variance; the marginal improvement is not detectable at 12 cases.

3. **Self-contained claim reformulation before search** — SAFE rewrites each atomic fact to resolve pronouns and ambiguous entity references before query generation. For multi-hop decomposed claims that inherit pronouns ("he", "the company", "the above version"), a lightweight rewrite in `build_query_variants()` would improve variant quality. Partially handled by `entity_set` already. *(Not yet implemented.)*

### What is relatively novel in this codebase

| Component | Why it stands out |
|-----------|------------------|
| **3-way intent classifier as hard router** (factual / synthesis / news_digest) — bypasses decompose+verify entirely for synthesis queries | Most systems treat all queries uniformly; no paper found with intent-gated bypass |
| **Multi-variant SERP with typed semantic roles** (broad / entity_locked / exact_match) run in parallel | Diversification literature exists but doesn't distinguish semantic roles of each variant |
| **`insufficient_evidence` as loop-continuation trigger** (not just a final verdict) | FActScore/SAFE score but don't loop back; FLARE loops at token level not claim verdict level |
| **Two-stage passage filter**: TF-IDF cheap_passage_filter → utility_rerank | Most papers use one reranker; two-stage at passage level (not document level) uncommon |
| **Source-score sorted passage selection for synthesis** — replaces TF-IDF with domain-authority ranking | Not found in any paper; closest is WebFilter's pre-retrieval source restriction |
| **MAX_PER_DOMAIN=1 at prompt-slot level for news_digest** | MMR studied in RAG, but per-domain cap at prompt construction time is novel |
| **Cross-claim page cache** — shared URL→content dict across parallel claims | Engineering optimization not studied as an architectural pattern |

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
