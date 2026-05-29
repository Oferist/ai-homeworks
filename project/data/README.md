# Данные проекта

Основной источник - открытый UCI Bike Sharing Dataset:

https://archive.ics.uci.edu/ml/datasets/bike%2Bsharing%2Bdataset

Файлы:

- `raw/hour.csv` - почасовые данные, используются для обучения;
- `raw/day.csv` - дневная агрегация из исходного датасета, в модели не используется;
- `sample_hour.csv` - небольшая демонстрационная выборка;
- `sample_predict.json` - пример тела запроса для `/predict`;
- `sample_batch_predict.json` - пример тела запроса для `/predict-batch`.

Данные скачиваются автоматически командой:

```bash
python -m src.train
```

В обучении не используются `casual` и `registered`, потому что эти поля являются компонентами целевой переменной `cnt`.
