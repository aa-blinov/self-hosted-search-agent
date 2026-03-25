# Agent instructions (humans & AI)

This file orients anyone (or any coding agent) working in this repository.

## Product

Grounded **web search agent**: iterative retrieval per **claim**, not one-shot global top‑K. Search backends: **Brave API** and **DuckDuckGo (ddgs)** via gateways. Page text via **HTTP shallow** and **crawl4ai** deep fetch. Answers must respect **verdicts** (`supported` / `contradicted` / `insufficient_evidence`) from the verifier layer.

## Where to look

| Area | Entry points |
|------|----------------|
| Config | `search_agent/settings.py` — env for keys/URLs/providers and optional caps (`EXTRACT_MAX_CHARS`, compose/RAG/intelligence token limits); tuning in `search_agent/tuning.py` |
| Agent loop | `search_agent/application/use_cases.py` — `_run_claim` iterations |
| Steps | `search_agent/application/agent_steps.py` — query variants, SERP gate, fetch, passages, compose answer |
| Contracts | `search_agent/application/contracts.py` |
| Domain types | `search_agent/domain/models.py` |
| Search | `search_agent/infrastructure/gateway_factory.py`, `brave_search.py`, `ddgs_gateway.py` |
| Fetch | `search_agent/infrastructure/extractor.py` |
| LLM / verify | `search_agent/infrastructure/intelligence.py`, `text_heuristics.py` |
| CLI / app | `search_agent/app.py`, `search_agent/cli.py` |
| Eval metrics | `search_agent/evaluation.py` |
| Eval artifacts | `search_agent/eval/tracking.py`, `search_agent/eval/compare_cli.py` |

## Commands

- Run app: `uv run search-agent` (see root `README.md`)
- Tests: `uv run python -m unittest discover -s tests -q` (includes `test_tuning.py` for `tuning` constants, `test_settings_layer` contract for env vs code)
- Eval (auto-saves JSON to `eval_runs/` unless `--eval-no-save`): `uv run search-agent --eval eval_data/quality_smoke.jsonl --eval-label smoke`
- Compare runs: `uv run python -m search_agent.eval <old.json> <new.json>`

## Conventions

- Match existing style: types, naming, minimal comments, no drive-by refactors unrelated to the task.
- Do not commit `.env` or `eval_runs/` (see `.gitignore`).
- Prefer `get_settings()` over scattered `os.getenv` for app settings.
- When changing behaviour that affects quality, run at least `eval_data/quality_smoke.jsonl` and consider `control_dataset.jsonl`.

## Eval datasets

- `eval_data/quality_smoke.jsonl` — short CI / local smoke (loose expectations, no routing constraints).
- `eval_data/quality_20.jsonl` + `quality_20_ground_truth.json` — 20 cases with human-verified reference answers and URLs for manual or future automated answer matching.
- `eval_data/control_dataset.jsonl` — broader regression (may include `expected_route`, `min_independent_sources`).
- `eval_data/sample_cases.jsonl` — small illustrative set.

Schema and field meanings: `eval_data/README.md`.

## Safety

Never paste real API keys into issues, commits, or chat logs. Rotate keys if exposed.
