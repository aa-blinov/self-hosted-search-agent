# Evaluation datasets & quality control

Наборы в формате **JSONL**: одна строка = один JSON-объект (один тестовый кейс).

## Файлы

| Файл | Назначение |
|------|------------|
| `quality_smoke.jsonl` | **Быстрый smoke** (5 кейсов): single-hop, multi-hop, conflict, insufficient evidence. Без жёстких `expected_route` — меньше флаков при смене роутера. |
| `quality_20.jsonl` | **20 кейсов** для контроля качества: разные `split`, multi-hop, конфликты, guardrail. |
| `quality_20_ground_truth.json` | Эталонные ответы и ссылки на источники (ручная сверка / будущий answer-level scoring). |
| `control_dataset.jsonl` | **Расширенный регресс**: маршруты, `min_independent_sources`, primary source. Дольше и строже. |
| `sample_cases.jsonl` | Короткий демо-набор для ручных прогонов. |

Рекомендуемый порядок: сначала `quality_smoke.jsonl`, затем `quality_20.jsonl` (сверка с `quality_20_ground_truth.json`), перед релизом — `control_dataset.jsonl`.

## Схема строки

Обязательные поля верхнего уровня:

| Поле | Тип | Описание |
|------|-----|----------|
| `case_id` | string | Уникальный id кейса |
| `split` | string | Бакет для метрик (`factual_single-hop`, `factual_multi-hop`, `freshness-sensitive`, `entity-disambiguation`, `conflicting-web-sources`, `unsupported-guardrail`, …) |
| `query` | string | Пользовательский запрос |
| `profile` | string | Имя профиля из `search_agent/config/profiles.py` (часто `web`) |
| `expected_claims` | array | Ожидания по под-утверждениям |

Элемент `expected_claims`:

| Поле | Обязательно | Описание |
|------|---------------|----------|
| `match` | да | Подстрока для сопоставления с текстом claim из прогона (case-insensitive) |
| `expected_verdict` | да | `supported` \| `contradicted` \| `insufficient_evidence` |
| `requires_primary_source` | нет | Если true — в метриках проверяется primary source при `supported` |
| `expected_route` | нет | Один режим или массив: `short_path`, `targeted_retrieval`, `iterative_loop` (может быть нестабильно — для smoke лучше не задавать) |
| `min_independent_sources` | нет | Минимум независимых источников при `supported` |

Подробности scoring: `score_reports()` в `search_agent/evaluation.py`.

## Запуск

По умолчанию каждый прогон **`--eval` сохраняет** полный JSON в каталог **`eval_runs/`** (хардкод в коде). Имя файла: `eval_<UTC timestamp>_<githash>_<dataset>.json`. Внутри — метрики, кейсы и блок `run_metadata` (время, git, модель, путь к артефакту).

```bash
uv run search-agent --eval eval_data/quality_smoke.jsonl --eval-label smoke
uv run search-agent --eval eval_data/control_dataset.jsonl --eval-label control
```

Явный путь к файлу или каталогу:

```bash
uv run search-agent --eval eval_data/quality_smoke.jsonl --eval-out eval_runs/baseline.json
uv run search-agent --eval eval_data/quality_smoke.jsonl --eval-out D:/metrics/eval/
```

Не писать файл на диск: `--eval-no-save`.

Метаданные формируются в `search_agent/eval/tracking.py` (`merge_run_metadata`). Пауза между кейсами eval: константа `EVAL_CASE_DELAY_SEC` в `search_agent/tuning.py`.

## Сравнение прогонов

```bash
uv run python -m search_agent.eval eval_runs/older.json eval_runs/newer.json
uv run python -m search_agent.eval eval_runs/a.json eval_runs/b.json --json
```

## Метрики (итоговый объект)

Среди прочего: `claim_support_rate`, `citation_validity_rate`, `unsupported_statement_rate`, `primary_source_coverage`, `contradiction_detection_rate`, `median_search_cost`, `median_answer_latency`, разрез `by_split`.

## Примечания

- Веб-результаты меняются со временем: часть кейсов может редко «плавать»; для стабильности используйте smoke без лишних ограничений и смотрите `backend_issue_rate` в отчёте.
- Пауза между кейсами: `EVAL_CASE_DELAY_SEC` в `search_agent/tuning.py`.
