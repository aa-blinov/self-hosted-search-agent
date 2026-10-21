# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

The product is a terminal application. It is written in Python, runs on the
operator's own machine, and reaches three kinds of remote service: a search
backend (Brave or DuckDuckGo), the pages those searches point at, and an
OpenAI-compatible model endpoint. There is no web interface, no mobile
client, and no service to deploy.

## Users

The person running the repository. The setup says so: a local install, a
`.env` holding the operator's own keys, and an OpenAI-compatible endpoint the
operator picks. The person is expected to read the console output, choose a
search profile for the question, and open the cited pages themselves when the
answer looks thin.

A second audience is not established. The repository page carries no
description, no topics, no releases and no stars, and neither the README nor
the code names a kind of user beyond the person at the keyboard.

## Product Purpose

The agent answers a question with pages it has read, and it says when the
pages were not enough.

Two decisions shape the rest. The model writes the answer and assesses it in
the same call, so the claims behind the answer and the confidence in it come
from one reading of one passage set. And the loop does not stop until every
claim in the answer has two independent domains behind it, which keeps the
common failure, a confident answer resting on one weak source, off the
console.

Success for the user: an answer whose `[N]` citations open pages that say
what the answer claims, and a clear list of what is missing when the question
was not settled. Success for the product: the answer cites the pages that
were fetched, and the run reports how many independent sources were
behind it.

## Positioning

A hosted assistant takes the question, hands it to a model, and returns prose
with a link list. This one puts a condition in front of the answer: the loop
keeps searching until each claim in the answer has two independent domains
behind it, or until the agent can state what is missing.

The second difference is ownership of the parts. The endpoint is any
OpenAI-compatible one, the search backend is Brave or DuckDuckGo, and the
project keeps nothing: the keys live in a `.env` on the operator's disk, and
the runs are read and diffed there.

The third difference sits in the shape of the loop. One procedure serves every
kind of question, with no separate branch for a fact, an overview or a news
roundup, and the two-domain rule applies to all of them.

## Operating Context

The agent runs from a terminal, in a checkout, with a `.env` beside it. Two
ways in: an interactive prompt where a profile is picked for each question,
and a one-shot `-q` run, which is what the evaluation harness and any script
use.

The search profile is part of the work rather than a detail. It decides the
category, the language, the freshness window and the query hints, and the 15
of them range from a general web search to arXiv only. The interactive prompt
suggests one from the wording of the question and lets the operator pick
another.

Every dataset run writes a JSON artifact to `eval_runs/` holding the metrics,
the per-case results, the git revision and the model name, and two artifacts
can be diffed. A single run can write its audit trail to a file with
`--receipts-dir`.

The operator is expected to check the citations. The answer prints its source
list underneath, with the URLs that were fetched, and the independent-source
count sits next to the verdict.

## Capabilities and Constraints

Confirmed by the code:

- One loop for every kind of question, with a cap of 2 iterations.
- One model call per iteration writes the answer and grades it, and returns
  the answer, the atomic claims, a confidence from 0 to 1, the gaps and a
  flag for a question the evidence contradicts.
- The loop stops when the evidence contradicts the question, or when the
  confidence is at least 0.75 and every claim has two independent hosts
  behind it, or when the iteration cap is reached.
- The first iteration reuses the 3 to 5 queries that the classify call made,
  and the second asks for up to 4 queries aimed at the gaps of the first.
- Search through Brave or DuckDuckGo. Brave falls back through the
  reference, web and deep profiles when it comes back empty.
- The gate scores every hit on domain type, freshness, entity match,
  primary-source likelihood and spam cues, and keeps 10 to 20 URLs.
- Page text over HTTP with trafilatura, a headless browser when the HTTP
  text comes back under 1200 characters, and dedicated handlers for Reddit,
  arXiv, Crossref, Semantic Scholar and GitHub.
- 4000 characters per page, 15000 for the reference domains.
- Passages of at most 420 characters, at most 2 per URL, 4 per domain and 14
  in the prompt.
- 15 search profiles, and a profile suggestion from the wording of the
  question.
- An evaluation harness: datasets, metrics, JSON artifacts, a comparison
  between two runs, and component checks that run without a key.
- Logfire tracing when a token is set.

Deliberately not true, and worth knowing before a change:

- The interface is the terminal. There is no web page and no mobile client.
- There is no account, no sharing and no server-side history. The only record
  of past questions is a history file in the home directory.
- The loop runs at most 2 iterations, so a question that needs three rounds
  of searching comes back with gaps instead of an answer.
- The two-domain rule counts hosts, not publishers. Two hosts under one owner
  satisfy it, and two pages on one host do not.
- The search backend is not deterministic. DuckDuckGo returns a different set
  of pages on each call, so a dataset run does not repeat to the decimal.
- The deep fetch path needs a browser, installed separately with
  `uv run crawl4ai-setup`.
- The source list inside the answer text is headed "Источники:" in every
  language, and the answer panel around it says "Answer".

## Brand Commitments

**Name:** Self-hosted search agent. **Voice:** plain and technical. The
console reports what happened, including when it did not work.

The interface states the state of the run: the query that was sent, the number
of URLs that survived the gate, the number of passages, the confidence, the
count of gaps, and the verdict with the number of independent sources next
to it. A failed search is a line in the output, and a model call that failed
returns an answer with a confidence of 0.0 rather than a silent empty.

The answer is written in the same language as the question, and its shape
follows the question: a sentence or two for a fact, a structured answer for an
explanation, a breakdown by dimension for a comparison, and a list of
separate developments for news.

## Evidence on Hand

- `README.md` describes the loop, the stop condition and the per-run caches.
- `search_agent/application/unified_runner.py` is the loop, and the constants
  that bound it sit next to it in `search_agent/tuning.py`.
- `eval_data/` holds the datasets: `quality_smoke.jsonl` with 5 cases,
  `quality_20.jsonl` with 20 cases and `quality_20_ground_truth.json` with
  the reference answers, `control_dataset.jsonl` with 12 cases,
  `sample_cases.jsonl` with 5 and `intent_classification_dataset.jsonl` with
  35.
- `eval_data/components/` holds the per-component datasets: `gate_serp` with
  3 cases, `cheap_passage` with 3, `route_claim` with 4, `classify_intent`
  with 40, `verify_claim` with 8 and `synthesize_answer` with 3.
- `eval_data/replay_fixtures/` holds 12 recorded runs of the previous
  pipeline, and `eval_data/replay_smoke.json` a recorded smoke run.
- `tests/` holds 19 test files, 85 tests, and they pass without an API key.
- `CLAUDE.md` holds the research notes, the optimisation history and the
  measured latency of each optimisation.
- `bench/` holds two diagnostics: one prints the scored search results and
  the routing breakdown, the other prints a truth table for the older stop
  condition.
- `LICENSE` carries the MIT terms.

Absent, and not to be invented: no usage numbers, no download counts, no
testimonials, and no comparison with a named product. The repository has no
stars, no releases and no description, so there is nothing to count.

## Product Principles

1. **The answer and its assessment come from the same reading.** One model
   call writes the answer and reports the claims behind it, so the claims
   and the confidence describe the passages the answer was written from.
2. **A claim needs two independent domains.** One source is a lead, not
   evidence. The rule holds for every kind of question, and a question that
   cannot meet it comes back with its gaps.
3. **The pages are read.** A search snippet can stand in for a page that
   would not load, and nothing else does.
4. **The evidence is shown.** The answer carries the citation numbers, the
   source list is printed under it, and the independent-source count sits
   next to the verdict.
5. **The parts belong to the operator.** The model endpoint, the search
   backend and the `.env` file are chosen locally, and the project stores
   nothing of its own.

## Accessibility & Inclusion

- **The answer comes back in the same language as the question.** The model
  is instructed to answer in that language, and the query generator adds an
  English query when the question is in another one.
- **The source list inside the answer is in Russian.** The footer heading
  "Источники:" is fixed in the code, and the panel around it says "Answer"
  in English, so a non-Russian reader gets an answer in their own language
  under a Russian heading.
- **The profile suggestion is a keyword heuristic.** It reads Russian and
  English cue words, and the operator picks the profile by hand when the
  guess is wrong.
- **No standard has been set.** There is no contrast or screen-reader
  requirement for the terminal output, and the console styling comes from
  Rich defaults.
