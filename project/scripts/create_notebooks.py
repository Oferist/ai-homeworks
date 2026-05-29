from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=2), encoding="utf-8")


eda_cells = [
    markdown_cell(
        "# 01. EDA: спрос на велопрокат\n"
        "\n"
        "Цель ноутбука - быстро проверить структуру UCI Bike Sharing Dataset, "
        "распределение целевой переменной `cnt` и основные зависимости спроса от часа, "
        "сезона, погоды и рабочих дней."
    ),
    code_cell(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "from matplotlib import pyplot as plt\n"
        "\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "data = pd.read_csv(ROOT / 'data' / 'raw' / 'hour.csv')\n"
        "data.head()"
    ),
    code_cell(
        "data[['cnt', 'temp', 'atemp', 'hum', 'windspeed']].describe().round(3)"
    ),
    code_cell(
        "hourly = data.groupby('hr', as_index=False)['cnt'].mean()\n"
        "sns.lineplot(data=hourly, x='hr', y='cnt', marker='o')\n"
        "plt.title('Средний спрос по часам')\n"
        "plt.show()"
    ),
    code_cell(
        "weather = data.groupby('weathersit', as_index=False)['cnt'].mean()\n"
        "season = data.groupby('season', as_index=False)['cnt'].mean()\n"
        "display(weather)\n"
        "display(season)"
    ),
    markdown_cell(
        "Выводы: спрос заметно зависит от часа дня, температуры, рабочего дня и погоды. "
        "В обучении не используются `casual` и `registered`, потому что они являются "
        "компонентами целевой переменной `cnt` и дали бы утечку."
    ),
]

model_cells = [
    markdown_cell(
        "# 02. Эксперименты с моделями\n"
        "\n"
        "Цель ноутбука - воспроизвести обучение из модульного кода и сравнить baseline "
        "с улучшенной моделью по MAE, RMSE и R2."
    ),
    code_cell(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "metrics = pd.read_csv(ROOT / 'artifacts' / 'metrics.csv')\n"
        "metrics"
    ),
    code_cell(
        "importance = pd.read_csv(ROOT / 'artifacts' / 'feature_importance.csv')\n"
        "importance.head(15)"
    ),
    code_cell(
        "predictions = pd.read_csv(ROOT / 'artifacts' / 'test_predictions.csv')\n"
        "predictions[['actual', 'prediction', 'absolute_error']].head(10)"
    ),
    markdown_cell(
        "Финальной выбрана модель `random_forest`: она даёт минимальный RMSE на validation "
        "и сохраняется в `artifacts/model.joblib`. Сервис `/predict` загружает этот файл, "
        "поэтому endpoint использует реальную обученную модель."
    ),
]


def main() -> None:
    write_notebook(ROOT / "notebooks" / "01_eda.ipynb", eda_cells)
    write_notebook(ROOT / "notebooks" / "02_model_experiments.ipynb", model_cells)


if __name__ == "__main__":
    main()
