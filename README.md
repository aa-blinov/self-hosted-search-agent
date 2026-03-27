# Self-hosted search agent

Итеративный веб-поисковый агент: поиск (Brave / DuckDuckGo), SERP-gate, shallow/deep fetch, извлечение passages, верификация утверждений и ответ только на основе подтверждённых claims. Рассчитан на локальный запуск и относительно дешёвые LLM через OpenAI-совместимый API (например OpenRouter).

## Требования

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) (рекомендуется)

## Установка

```bash
uv sync
cp .env.example .env
# Заполните LLM_API_KEY и при необходимости BRAVE_API_KEY / SEARCH_PROVIDER
```

Первый запуск crawl4ai (если используете deep fetch):

```bash
uv run crawl4ai-setup
```

## Запуск

Интерактивный CLI:

```bash
uv run search-agent
```

Один запрос:

```bash
uv run search-agent -q "Your question" -p web
```

### Windows и кириллица в консоли

При старте приложение переключает консоль на **UTF-8** (кодовая страница 65001) и задаёт UTF-8 для `stdout`/`stderr`, чтобы Rich и русский текст не превращались в «кракозябры» (`â…`, `Ð…`). Если в очень старом `cmd` всё ещё искажения, откройте **Windows Terminal** или выполните перед запуском: `chcp 65001`. Дополнительно можно задать `PYTHONUTF8=1` в среде.

Переопределение поискового бэкенда на один прогон:

```bash
uv run search-agent -S ddgs -q "..."
```

## Конфигурация

Переменные окружения — ключи, провайдеры и связанные URL (`search_agent/settings.py`). Дополнительно: **`EXTRACT_MAX_CHARS`**, лимиты LLM для grounded-ответа и arXiv (**`COMPOSE_ANSWER_MAX_TOKENS`**, **`RAG_ANALYSIS_MAX_TOKENS`**), а также для intelligence (**`CLAIM_DECOMPOSE_MAX_TOKENS`**, **`VERIFY_CLAIM_MAX_TOKENS`**, **`TIME_NORMALIZE_MAX_TOKENS`**); итог по токенам не выше **`LLM_MAX_TOKENS`**. Остальные таймауты и лимиты агента — в `search_agent/tuning.py`. См. `.env.example`.

## Оценка качества (eval)

Датасеты лежат в `eval_data/`. Подробнее: [`eval_data/README.md`](eval_data/README.md).

Быстрый smoke (5 кейсов). Результат **автоматически** пишется в `eval_runs/` (имя файла с UTC-временем и git short hash), см. `run_metadata` внутри JSON:

```bash
uv run search-agent --eval eval_data/quality_smoke.jsonl --eval-label smoke
```

Набор из 20 кейсов с эталонными ответами в `eval_data/quality_20_ground_truth.json`:

```bash
uv run search-agent --eval eval_data/quality_20.jsonl --eval-label q20
```

Полный контрольный набор:

```bash
uv run search-agent --eval eval_data/control_dataset.jsonl --eval-label control
```

Отключить сохранение файла: `--eval-no-save`. Другой путь: `--eval-out path.json` или `--eval-out my_dir/`.

Сравнение двух сохранённых отчётов:

```bash
uv run python -m search_agent.eval eval_runs/run_a.json eval_runs/run_b.json
```

## Архитектура

### Как работает агент — кратко

1. **Нормализация времени** — если в запросе есть относительные временны́е ссылки («сегодня», «на этой неделе»), LLM заменяет их на конкретные даты.

2. **Классификация + генерация запросов** *(1 LLM-вызов)* — модель определяет тип вопроса и сразу генерирует 3–5 поисковых запросов:
   - `factual` — конкретный проверяемый факт («кто CEO Microsoft?», «когда вышел Python 3.13?»)
   - `synthesis` — объяснение, сравнение, обзор, список изменений («как работает asyncio?», «отличия Python 3.11 и 3.12»)
   - `news_digest` — последние новости по теме

3. **Декомпозиция** — сложный factual-запрос разбивается на 2–3 атомарных под-вопроса (каждый со своими поисковыми запросами). Synthesis и news_digest не декомпозируются — ищутся как единый claim.

4. **Поиск** *(параллельно)* — для каждого claim'а параллельно запускаются 3 SERP-запроса через DDGS или Brave. Результаты кешируются, соблюдается бюджет вызовов.

5. **Фильтрация URL** — из 20+ результатов SERP отбираются 10–20 качественных: отсеивается SEO-спам, нерелевантные домены, проверяется совпадение сущностей.

6. **Маршрутизация** — исходя из уверенности, согласованности источников и полноты определяется режим загрузки: `short_path` / `targeted_retrieval` / `iterative_loop`.

7. **Загрузка страниц** — скачиваются 3–10 страниц (в зависимости от маршрута): сначала быстрый HTTP (trafilatura), при нехватке текста — Playwright (crawl4ai). Страницы кешируются и переиспользуются между параллельными claim'ами.

8. **Фильтрация пассажей** — страницы режутся на куски ~500 символов. TF-IDF отбирает топ-12, utility reranker оставляет топ-8.

9. **Верификация** *(LLM)* — модель получает claim + до 8 пассажей и возвращает:
   - **вердикт**: `supported` / `contradicted` / `insufficient_evidence`
   - **цитаты**: конкретные фрагменты, подтверждающие или опровергающие
   - **missing_dimensions**: чего не хватило (дата, источник, число, локация) — используется для уточнения запросов
   - **rationale**: объяснение вердикта текстом
   - **confidence**: уверенность (0.0–1.0)

10. **Вторая итерация** — если вердикт `insufficient_evidence`, генерируются уточнённые запросы на основе `missing_dimensions` (добавляется год, точная цитата, ограничение по домену) и весь цикл повторяется.

11. **Финальный ответ**:
    - `synthesis` / `news_digest` → LLM строит связный ответ из всех собранных пассажей, источники ранжируются по авторитетности домена; для news_digest — максимум 1 пассаж с одного домена
    - `factual` → LLM пишет grounded-ответ с цитатами из верифицированных источников

### Поток данных

```
query
  │
  ▼
normalize_time_references()          ← LLM (только если есть временны́е ссылки)
  │                                    кеш: raw_query → normalized
  ▼
classify_intent + generate_queries() ← 1 LLM-вызов (intent + 3–5 запросов)
  │                                    кеш: normalized_query → (intent, queries)
  ▼
QueryClassification(intent, complexity, needs_freshness, time_scope, ...)
  │
  ├── synthesis / news_digest  ──────────────────────────────────────┐
  │   queries из кеша, один claim = весь запрос                      │
  │                                                                  │
  ├── factual, простой (not should_decompose) ──────────────────────►┤
  │   queries из кеша, один claim = весь запрос                      │
  │                                                                  │
  └── factual, сложный (should_decompose)                           │
      LLM: декомпозиция на 2–3 под-вопроса, каждый со своими queries │
                                                                     │
  ◄──────────────────────────────────────────────────────────────────┘
  │  list[Claim]  (обрабатываются параллельно, до 4 штук)
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Цикл на claim (max 2 итерации)                     │
│                                                     │
│  build_query_variants()  ← claim.search_queries[:3] │
│  search_variant() × N    ← параллельно              │
│  gate_serp_results()     → 10–20 URL                │
│  route_claim_retrieval() → short|targeted|iterative │
│  fetch_claim_documents() ← shared page cache        │
│  split_into_passages()   → ~500 char chunks         │
│  cheap_passage_filter()  → top 12  (TF-IDF)         │
│  utility_rerank_passages() → top 8                  │
│  verify_claim()          ← LLM, кеш по промпту      │
│    → supported / contradicted / insufficient        │
│  should_stop_claim_loop()?                          │
│    нет → refine_query_variants() → iter2            │
└─────────────────────────────────────────────────────┘
  │
  ├── synthesis / news_digest → synthesize_answer()  ← LLM
  └── factual               → compose_answer()       ← LLM
```

### Кеши (in-process, per-run)

| Кеш | Ключ | Что хранит |
|-----|------|------------|
| `_normalize_cache` | raw query | нормализованная строка запроса |
| `_intent_cache` | normalized query | `factual` / `synthesis` / `news_digest` |
| `_query_cache` | normalized query | список поисковых запросов (из того же LLM-вызова) |
| `_verify_cache` | полный промпт | `VerificationResult` |
| page cache | URL | извлечённый текст страницы |

### LLM-вызовы на типичный run

| Вызов | Когда | Примерная стоимость |
|-------|-------|---------------------|
| normalize_time | только при временны́х ссылках | ~120 токенов |
| classify_intent + generate_queries | каждый run | ~500 токенов |
| decompose_claims | только сложные factual | ~500 токенов |
| verify_claim | каждая итерация × claim | ~700 токенов |
| synthesize_answer / compose_answer | финал | ~1600–2000 токенов |

Для простого factual-запроса без декомпозиции — **3 LLM-вызова**: classify+queries → verify → compose.

## Тесты

```bash
uv run python -m unittest discover -s tests -q
```

## Структура репозитория

| Путь | Назначение |
|------|------------|
| `search_agent/settings.py` | Конфигурация приложения |
| `search_agent/application/` | Сценарии агента, шаги, use case |
| `search_agent/domain/` | Модели (claims, evidence, verdicts) |
| `search_agent/infrastructure/` | Поиск, fetch, LLM, receipts |
| `search_agent/eval/` | Сохранение и сравнение eval-прогонов |
| `eval_data/` | JSONL-датасеты и документация по eval |

Для ассистентов и контрибьюторов см. [`AGENTS.md`](AGENTS.md).
