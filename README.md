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

Переопределение поискового бэкенда на один прогон:

```bash
uv run search-agent -S ddgs -q "..."
```

## Конфигурация

Переменные окружения — только то, что нужно для ключей и выбора провайдеров (`search_agent/settings.py`). Таймауты, лимиты агента, crawl4ai и т.п. — константы в `search_agent/tuning.py`. См. `.env.example`.

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
