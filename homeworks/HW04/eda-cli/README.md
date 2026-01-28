# S04 – eda_cli: HTTP Service

Проект HW04. HTTP-сервис на базе FastAPI для анализа качества данных.

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (HW04/eda-cli):

```bash
uv sync
```

Эта команда:
- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

---

## Запуск HTTP API (Сервис)

Для запуска веб-сервера используйте команду:

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Сервис будет доступен по адресу: [http://127.0.0.1:8000](http://127.0.0.1:8000).
Интерактивная документация (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### API Endpoints

1. **GET /health**
   - Проверка статуса сервиса.
2. **POST /quality**
   - Быстрая оценка качества по метаданным (число строк, колонок, пропусков).
3. **POST /quality-from-csv**
   - Принимает файл, возвращает общий скор качества и базовые метрики.

### Дополнительный эндпоинт (HW04)

**POST /quality-flags-from-csv**

Этот эндпоинт реализует расширенную логику проверок из HW03.
- **Вход:** CSV-файл (Multipart/Form-Data).
- **Выход:** JSON с детальным списком сработавших эвристик.
  - `has_constant_columns`: наличие колонок с одним значением.
  - `has_many_zero_values`: наличие колонок, где более 30% нулей.
  - `has_suspicious_id_duplicates`: проверка на дубликаты в ID-подобных колонках.

---

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports --title "My Analysis"
```

В результате в каталоге `reports/` появятся Markdown-отчёт, графики и CSV-таблицы.

## Тесты

```bash
uv run pytest -q
```