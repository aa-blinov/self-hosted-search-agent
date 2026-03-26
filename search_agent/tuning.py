"""
Internal tuning constants (not env-configurable).

Change here when profiling; keeps .env limited to keys, URLs, and provider choice.
"""

# --- Extractor / crawl ---
SHALLOW_FETCH_TIMEOUT = 8
# GET retries (429 / 5xx / transient network) before giving up on shallow fetch.
SHALLOW_FETCH_HTTP_ATTEMPTS = 3
SHALLOW_FETCH_RETRY_BACKOFF_SEC = 0.35
EXTRACT_MAX_CHARS = 4000
# Higher limit for authoritative reference pages where the key content is deep
# in the document (e.g. whatsnew pages, API docs, Wikipedia articles).
# Applied only for synthesis intent; synthesis path selects passages by
# source_score (not TF-IDF) so all sections of the page are reachable.
EXTRACT_MAX_CHARS_AUTHORITY = 15000
# Domain suffixes / substrings that qualify for the higher limit.
AUTHORITY_DOMAINS: tuple[str, ...] = (
    "docs.python.org",
    "peps.python.org",
    "realpython.com",
    "wikipedia.org",
    "developer.mozilla.org",
    "docs.djangoproject.com",
    "docs.pytest.org",
    "docs.rust-lang.org",
    "docs.github.com",
    "docs.aws.amazon.com",
    "cloud.google.com",
    "learn.microsoft.com",
    "docs.microsoft.com",
)
# Minimum extracted main text (trafilatura) before accepting vs legacy HTML heuristics.
TRAIFILATURA_MIN_MAIN_CHARS = 200
CRAWL4AI_TIMEOUT = 25
# Slightly longer for shallow browser-only path (e.g. vc.ru) — one Playwright load, no prior HTTP.
CRAWL4AI_BROWSER_ONLY_TIMEOUT = 30
CRAWL4AI_DELAY_BEFORE_HTML = 1.0
CRAWL4AI_PREFER_RAW = False
# Deprecated: deep fetch always tries HTTP (trafilatura + legacy) before crawl4ai.
FETCH_TRY_HTTP_FIRST = False
FETCH_HTTP_MIN_CHARS = 1200
# Host suffixes where plain ``requests`` often hits TLS/bot walls; shallow uses crawl4ai only.
SHALLOW_BROWSER_FIRST_HOST_SUFFIXES = (".vc.ru",)
# Lower than max workers reduces same-host rate limits and bot wall trips during SERP batch fetches.
FETCH_SHALLOW_CONCURRENCY = 5
FETCH_DEEP_CONCURRENCY = 2

# --- Agent loop ---
AGENT_MAX_CLAIM_ITERATIONS = 2
# Max claims to process per query (0 = unlimited). Caps LLM verify_claim calls for
# broad comparison/survey queries that decompose into many sub-claims.
DECOMPOSE_MAX_CLAIMS = 3
# Max claim-level workers per user query (1 = sequential). Fetch/LLM remain per-claim.
AGENT_MAX_PARALLEL_CLAIMS = 4
# Max backend search invocations per user query (0 = unlimited). Cache hits do not count.
AGENT_MAX_SEARCH_CALLS_PER_RUN = 0
AGENT_MAX_QUERY_VARIANTS = 6
# Variant cap for iteration 1 only (broad + entity_locked + exact_match at most).
# Freshness/source_restricted variants are added by refine_query_variants on iteration 2 if needed.
AGENT_MAX_QUERY_VARIANTS_ITER1 = 3
AGENT_MAX_REFINE_VARIANTS = 12
AGENT_FETCH_TOP_N = 4
AGENT_PASSAGE_TOP_K = 8
SERP_GATE_MIN_URLS = 10
SERP_GATE_MAX_URLS = 20
AGENT_SNIPPET_FALLBACK_DOCS = 2
SHALLOW_FETCH_SHORT_LIMIT = 5
SHALLOW_FETCH_TARGETED_LIMIT = 8
SHALLOW_FETCH_ITERATIVE_LIMIT = 10
# Fast limits for iteration 1 — scale up only if a second iteration is needed.
SHALLOW_FETCH_SHORT_FAST_LIMIT = 3
SHALLOW_FETCH_TARGETED_FAST_LIMIT = 4
SHALLOW_FETCH_ITERATIVE_FAST_LIMIT = 5
DEEP_FETCH_SHORT_LIMIT = 1
DEEP_FETCH_TARGETED_LIMIT = 2
DEEP_FETCH_ITERATIVE_LIMIT = 2
CHEAP_PASSAGE_LIMIT = 12
# For synthesis queries, max passages passed to synthesize_answer.
# Synthesis path bypasses TF-IDF cheap_passage_filter (cross-language queries
# score English content near zero) and instead sorts by source_score; this
# limit caps the final selection sent to the LLM.
SYNTHESIS_PASSAGE_LIMIT = 25
# Snippet-first: minimum confidence from SERP snippets to skip full-page fetch.
SNIPPET_VERIFY_CONFIDENCE_THRESHOLD = 0.85

# --- Query intelligence (PydanticAIQueryIntelligence) ---
CLAIM_DECOMPOSE_MAX_TOKENS = 500
VERIFY_CLAIM_MAX_TOKENS = 700
TIME_NORMALIZE_MAX_TOKENS = 120
# For synthesis intent: skip sub-claim decomposition, search with the original query as a
# single claim. Avoids N-claims × M-variants SERP fan-out for open-ended questions.
SYNTHESIS_SKIP_DECOMPOSE = True
# Tokens for LLM intent classification (factual / synthesis / news_digest).
# qwen3.5-35b-a3b generates <think>...</think> before the JSON answer — 300 is enough.
INTENT_CLASSIFY_MAX_TOKENS = 300
# Max tokens for the synthesize_answer LLM call (synthesis queries only).
# Higher than verify_claim because thinking models (qwen) consume budget on <think>...</think>
# before the actual answer, and synthesis needs to produce a complete bullet list.
SYNTHESIZE_ANSWER_MAX_TOKENS = 2000

# --- Composed answer (CLI / Panel) ---
# Grounded LLM answer (``llm_tasks.answer_with_sources``): capped by ``min(..., AppSettings.llm_max_tokens)``.
COMPOSE_ANSWER_MAX_TOKENS = 1600
# ``/research`` arXiv paper summaries (``llm_tasks.analyze_rag_papers``).
RAG_ANALYSIS_MAX_TOKENS = 1400
# Max characters per span/sentence line in compose_answer (verifier span or best sentence).
COMPOSE_ANSWER_MAX_SPAN_CHARS = 2000
# News-digest single line cap (slightly tighter for list readability).
COMPOSE_ANSWER_DIGEST_LINE_CHARS = 520

# --- Evaluation harness ---
EVAL_CASE_DELAY_SEC = 2.0

# --- Logfire (non-secret; token stays in AppSettings) ---
LOGFIRE_SERVICE_NAME = "self-hosted-search-agent"
LOGFIRE_ENVIRONMENT = "development"
LOGFIRE_SEND_TO_LOGFIRE = "if-token-present"
LOGFIRE_LOCAL = True
LOGFIRE_CONSOLE = False
