# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

В этой папке находится итоговый мини-проект по курсу.  
Проект должен демонстрировать применение методов и инструментов инженерии ИИ: работу с данными, модели, пайплайны, сервис, эксперименты и (по возможности) воспроизводимость.

Заполните данный файл под ваш конкретный проект.

---

## 1. Паспорт проекта

- **Название проекта:** `Прогноз спроса на велопрокат`
- **Автор:** `Некрасов Глеб Андреевич`
- **Группа:** `ИКБО-28-22`
- **Контакт:** `lfazz@mail.ru`
- **Задача:** предсказать значение `cnt`, то есть количество аренд велосипедов за час.
- **Результат:** REST API с endpoint `/predict`, который использует обученную модель из `artifacts/model.joblib`.

---

## 2. Структура проекта

```text
project/
  README.md
  report.md
  self-checklist.md
  SECURITY.md
  requirements.txt
  Dockerfile
  configs/
  data/
  notebooks/
  src/
  tests/
  artifacts/
  scripts/
```

Ключевые части:

- `src/data/` - загрузка UCI Bike Sharing Dataset.
- `src/features/` - список признаков и подготовка обучающей таблицы.
- `src/models/` - обучение, сравнение моделей и инференс.
- `src/service/` - FastAPI-сервис с `/health`, `/model-info`, `/predict`, `/predict-batch`.
- `notebooks/` - EDA и эксперименты.
- `artifacts/` - обученная модель, метрики, графики и предсказания на тесте.

---

## 3. Требования и установка

Требуется Python 3.10 или выше. Проверено на Python 3.11.

Windows:

```powershell
cd project
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
cd project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Как запустить обучение

```bash
python -m src.train
```

Команда скачает открытый датасет UCI, подготовит данные, обучит модели и сохранит артефакты:

- `artifacts/model.joblib` - финальная модель;
- `artifacts/metrics.csv` и `artifacts/metrics.json` - сравнение моделей;
- `artifacts/feature_importance.csv` - важность признаков;
- `artifacts/*.png` - графики для отчёта и защиты.

---

## 5. Как запустить сервис

```bash
python -m src.service
```

По умолчанию сервис поднимается на `http://127.0.0.1:8000`.

Полезные страницы и endpoints:

- `GET /` - простой пользовательский интерфейс с формой прогноза;
- `GET /health` - проверка работоспособности;
- `GET /model-info` - информация о модели и метриках;
- `POST /predict` - одно предсказание;
- `POST /predict-batch` - пакет до 100 объектов;
- `GET /docs` - Swagger UI для технической проверки API.

Для обычной демонстрации откройте `http://127.0.0.1:8000`, выберите дату, час, погоду, температуру, влажность и нажмите «Рассчитать спрос». Страница сама преобразует дату в месяц, день недели и сезон, а затем отправляет запрос в `/predict`.

Однострочный вариант через `curl.exe` для Windows:

```powershell
curl.exe -s -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@data/sample_predict.json"
```

Для Linux/macOS:

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@data/sample_predict.json"
```

Ожидаемый формат ответа:

```json
{
  "prediction": 836.718,
  "rounded_prediction": 837,
  "model": "random_forest"
}
```

---

## 6. Запуск сервиса через Docker

```bash
docker build -t bike-demand-project .
docker run -p 8000:8000 bike-demand-project
```

Если модель отсутствует, перед сборкой или запуском выполните `python -m src.train`.

---

## 7. Тесты

```bash
python -m pytest tests
```

Запускайте тесты из активированного виртуального окружения. Команда через `python -m pytest` важна: так используется именно тот Python, в который установлены зависимости проекта.

Тесты проверяют подготовку признаков, работу инференса, наличие пользовательского интерфейса и базовые endpoints сервиса.

---

## 8. Данные

Используется открытый UCI Bike Sharing Dataset. В проекте нет персональных данных, токенов, паролей или закрытых выгрузок. Для демонстрации сохраняется небольшая выборка `data/sample_hour.csv`, а полный датасет можно заново скачать командой обучения.

Основные признаки:

- календарные: `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`;
- погодные: `weathersit`, `temp`, `atemp`, `hum`, `windspeed`;
- целевая переменная: `cnt`.

Признаки `casual` и `registered` исключены, потому что они являются компонентами `cnt` и создают утечку целевой переменной.

---

## 9. Демонстрация на защите

1. Показать структуру проекта и файлы `README.md`, `report.md`, `self-checklist.md`.
2. Запустить `python -m src.train` или показать готовые `artifacts/metrics.csv`.
3. Запустить `python -m src.service`.
4. Открыть пользовательский интерфейс `http://127.0.0.1:8000`.
5. Заполнить форму и показать прогноз количества аренд.
6. При необходимости открыть `http://127.0.0.1:8000/docs` и выполнить запрос к `/predict` с примером из `data/sample_predict.json`.
7. Показать, что `/predict` использует `artifacts/model.joblib`, а не заглушку.

---

## 10. Ограничения и дальнейшая работа

Модель обучена на данных 2011-2012 годов и не учитывает городские события, цены, наличие велосипедов на станциях и современные изменения поведения пользователей. Для реального продакшена стоило бы добавить свежие данные, мониторинг качества, историю запросов и периодическое переобучение.

---