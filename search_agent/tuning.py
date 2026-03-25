"""
Internal tuning constants (not env-configurable).

Change here when profiling; keeps .env limited to keys, URLs, and provider choice.
"""

# --- Extractor / crawl ---
SHALLOW_FETCH_TIMEOUT = 8
EXTRACT_MAX_CHARS = 4000
# Minimum extracted main text (trafilatura) before accepting vs legacy HTML heuristics.
TRAIFILATURA_MIN_MAIN_CHARS = 200
CRAWL4AI_TIMEOUT = 25
CRAWL4AI_DELAY_BEFORE_HTML = 1.0
CRAWL4AI_PREFER_RAW = False
# Deprecated: deep fetch always tries HTTP (trafilatura + legacy) before crawl4ai.
FETCH_TRY_HTTP_FIRST = False
FETCH_HTTP_MIN_CHARS = 1200
FETCH_SHALLOW_CONCURRENCY = 8
FETCH_DEEP_CONCURRENCY = 2

# --- Agent loop ---
AGENT_MAX_CLAIM_ITERATIONS = 3
# Max claim-level workers per user query (1 = sequential). Fetch/LLM remain per-claim.
AGENT_MAX_PARALLEL_CLAIMS = 4
# Max backend search invocations per user query (0 = unlimited). Cache hits do not count.
AGENT_MAX_SEARCH_CALLS_PER_RUN = 0
AGENT_MAX_QUERY_VARIANTS = 6
AGENT_MAX_REFINE_VARIANTS = 12
AGENT_FETCH_TOP_N = 4
AGENT_PASSAGE_TOP_K = 8
SERP_GATE_MIN_URLS = 15
SERP_GATE_MAX_URLS = 30
AGENT_SNIPPET_FALLBACK_DOCS = 2
SHALLOW_FETCH_SHORT_LIMIT = 8
SHALLOW_FETCH_TARGETED_LIMIT = 12
SHALLOW_FETCH_ITERATIVE_LIMIT = 15
DEEP_FETCH_SHORT_LIMIT = 2
DEEP_FETCH_TARGETED_LIMIT = 3
DEEP_FETCH_ITERATIVE_LIMIT = 4
CHEAP_PASSAGE_LIMIT = 12

# --- Composed answer (CLI / Panel) ---
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
