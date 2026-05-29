from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import __version__
from src.config import load_config, resolve_path
from src.models.predict import load_model, predict_batch, predict_one


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("bike_demand_service")

config = load_config()
artifact_cfg = config["artifacts"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
        logger.info("Model loaded successfully")
    except FileNotFoundError as exc:
        logger.warning("%s", exc)
    yield


app = FastAPI(
    title="Bike Demand Forecast API",
    description="Predict hourly bike sharing demand using a trained ML model.",
    version=__version__,
    lifespan=lifespan,
)
app.mount("/artifacts", StaticFiles(directory=resolve_path("artifacts")), name="artifacts")


UI_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Прогноз спроса на велопрокат</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f7f6;
      --panel: #ffffff;
      --panel-strong: #ecf4f2;
      --text: #17211f;
      --muted: #5e6f6b;
      --line: #cbd9d5;
      --accent: #0f766e;
      --accent-dark: #0b5f59;
      --warning: #8f4c16;
      --shadow: 0 16px 40px rgba(24, 44, 39, 0.10);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, Segoe UI, system-ui, -apple-system, sans-serif;
      background:
        linear-gradient(90deg, rgba(15, 118, 110, 0.06) 1px, transparent 1px),
        linear-gradient(0deg, rgba(15, 118, 110, 0.05) 1px, transparent 1px),
        var(--bg);
      background-size: 44px 44px;
      color: var(--text);
    }

    header {
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
      backdrop-filter: blur(10px);
    }

    .topbar {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 18px 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
      font-weight: 760;
      letter-spacing: 0;
    }

    .model-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }

    .status-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #20a672;
      box-shadow: 0 0 0 3px rgba(32, 166, 114, 0.16);
    }

    main {
      width: min(1180px, calc(100% - 32px));
      margin: 24px auto 36px;
      display: grid;
      grid-template-columns: minmax(0, 1.16fr) minmax(320px, 0.84fr);
      gap: 20px;
    }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .form-panel { padding: 22px; }

    .section-title {
      margin: 0 0 18px;
      font-size: 17px;
      line-height: 1.25;
      font-weight: 720;
      letter-spacing: 0;
    }

    form {
      display: grid;
      gap: 18px;
    }

    fieldset {
      margin: 0;
      padding: 0;
      border: 0;
      display: grid;
      gap: 12px;
    }

    legend {
      padding: 0;
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .grid.two {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    label {
      display: grid;
      gap: 7px;
      min-width: 0;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }

    input,
    select {
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      background: #ffffff;
      color: var(--text);
      font: inherit;
      font-size: 15px;
      outline: none;
    }

    input:focus,
    select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.13);
    }

    input[type="range"] {
      padding: 0;
      min-height: 28px;
      accent-color: var(--accent);
    }

    .range-value {
      color: var(--text);
      font-weight: 720;
    }

    .derived {
      min-height: 42px;
      display: flex;
      align-items: center;
      padding: 9px 10px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: var(--panel-strong);
      color: var(--text);
      font-size: 15px;
      font-weight: 680;
    }

    .actions {
      display: flex;
      align-items: center;
      gap: 12px;
      padding-top: 2px;
    }

    button {
      min-height: 44px;
      border: 0;
      border-radius: 8px;
      padding: 0 18px;
      background: var(--accent);
      color: #ffffff;
      font: inherit;
      font-size: 15px;
      font-weight: 760;
      cursor: pointer;
    }

    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: 0.7; cursor: wait; }

    .hint {
      color: var(--muted);
      font-size: 13px;
    }

    .side {
      display: grid;
      gap: 20px;
    }

    .result-panel,
    .chart-panel,
    .metrics-panel {
      padding: 20px;
    }

    .prediction {
      display: grid;
      gap: 8px;
      min-height: 134px;
      align-content: center;
      border-radius: 8px;
      background: linear-gradient(135deg, #e2f4ef, #fff7ec);
      border: 1px solid #d4e5df;
      padding: 18px;
    }

    .prediction-label {
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .prediction-value {
      font-size: 44px;
      line-height: 1;
      font-weight: 820;
      letter-spacing: 0;
    }

    .prediction-extra {
      color: var(--muted);
      font-size: 14px;
    }

    .chart {
      display: block;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .metric strong {
      display: block;
      margin-top: 6px;
      font-size: 20px;
      letter-spacing: 0;
    }

    .error {
      color: var(--warning);
      font-size: 14px;
      font-weight: 700;
    }

    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .grid,
      .grid.two,
      .metric-grid { grid-template-columns: 1fr; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .prediction-value { font-size: 38px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <h1>Прогноз спроса на велопрокат</h1>
      <div class="model-pill"><span class="status-dot"></span><span id="modelName">model.joblib</span></div>
    </div>
  </header>

  <main>
    <section class="form-panel">
      <h2 class="section-title">Параметры поездки</h2>
      <form id="predictForm">
        <fieldset>
          <legend>Дата и время</legend>
          <div class="grid">
            <label>
              Дата
              <input id="date" type="date" required />
            </label>
            <label>
              Час
              <select id="hr" required></select>
            </label>
            <label>
              Сезон
              <div class="derived" id="seasonLabel">-</div>
            </label>
          </div>
        </fieldset>

        <fieldset>
          <legend>Календарь</legend>
          <div class="grid">
            <label>
              День недели
              <div class="derived" id="weekdayLabel">-</div>
            </label>
            <label>
              Рабочий день
              <select id="workingday">
                <option value="1">Да</option>
                <option value="0">Нет</option>
              </select>
            </label>
            <label>
              Праздник
              <select id="holiday">
                <option value="0">Нет</option>
                <option value="1">Да</option>
              </select>
            </label>
          </div>
        </fieldset>

        <fieldset>
          <legend>Погода</legend>
          <div class="grid">
            <label>
              Состояние
              <select id="weathersit">
                <option value="1">Ясно</option>
                <option value="2">Облачно</option>
                <option value="3">Дождь/снег</option>
                <option value="4">Сильная непогода</option>
              </select>
            </label>
            <label>
              Температура <span class="range-value" id="tempValue">24°C</span>
              <input id="tempC" type="range" min="-8" max="39" step="1" value="24" />
            </label>
            <label>
              Ощущается как <span class="range-value" id="atempValue">26°C</span>
              <input id="atempC" type="range" min="-16" max="50" step="1" value="26" />
            </label>
          </div>
          <div class="grid two">
            <label>
              Влажность <span class="range-value" id="humValue">45%</span>
              <input id="humPct" type="range" min="0" max="100" step="1" value="45" />
            </label>
            <label>
              Ветер <span class="range-value" id="windValue">18 км/ч</span>
              <input id="windKmh" type="range" min="0" max="67" step="1" value="18" />
            </label>
          </div>
        </fieldset>

        <div class="actions">
          <button id="submitButton" type="submit">Рассчитать спрос</button>
          <span class="hint">API: POST /predict</span>
        </div>
      </form>
    </section>

    <div class="side">
      <section class="result-panel">
        <h2 class="section-title">Результат</h2>
        <div class="prediction">
          <div class="prediction-label">Ожидаемое количество аренд за час</div>
          <div class="prediction-value" id="predictionValue">-</div>
          <div class="prediction-extra" id="predictionExtra">Заполните параметры и запустите расчёт.</div>
          <div class="error" id="errorBox"></div>
        </div>
      </section>

      <section class="chart-panel">
        <h2 class="section-title">Средний спрос по часам</h2>
        <img class="chart" src="/artifacts/hourly_demand.png" alt="График среднего спроса на велопрокат по часам" />
      </section>

      <section class="metrics-panel">
        <h2 class="section-title">Качество модели</h2>
        <div class="metric-grid">
          <div class="metric"><span>MAE</span><strong>39.44</strong></div>
          <div class="metric"><span>RMSE</span><strong>59.51</strong></div>
          <div class="metric"><span>R2</span><strong>0.888</strong></div>
        </div>
      </section>
    </div>
  </main>

  <script>
    const form = document.querySelector("#predictForm");
    const dateInput = document.querySelector("#date");
    const hourSelect = document.querySelector("#hr");
    const seasonLabel = document.querySelector("#seasonLabel");
    const weekdayLabel = document.querySelector("#weekdayLabel");
    const tempC = document.querySelector("#tempC");
    const atempC = document.querySelector("#atempC");
    const humPct = document.querySelector("#humPct");
    const windKmh = document.querySelector("#windKmh");
    const tempValue = document.querySelector("#tempValue");
    const atempValue = document.querySelector("#atempValue");
    const humValue = document.querySelector("#humValue");
    const windValue = document.querySelector("#windValue");
    const modelName = document.querySelector("#modelName");
    const predictionValue = document.querySelector("#predictionValue");
    const predictionExtra = document.querySelector("#predictionExtra");
    const errorBox = document.querySelector("#errorBox");
    const submitButton = document.querySelector("#submitButton");

    const seasonNames = {1: "Зима", 2: "Весна", 3: "Лето", 4: "Осень"};
    const weekdayNames = ["Воскресенье", "Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота"];

    for (let hour = 0; hour < 24; hour += 1) {
      const option = document.createElement("option");
      option.value = String(hour);
      option.textContent = `${String(hour).padStart(2, "0")}:00`;
      if (hour === 17) option.selected = true;
      hourSelect.appendChild(option);
    }

    function setToday() {
      const today = new Date();
      dateInput.value = today.toISOString().slice(0, 10);
    }

    function monthToSeason(month) {
      if ([12, 1, 2].includes(month)) return 1;
      if ([3, 4, 5].includes(month)) return 2;
      if ([6, 7, 8].includes(month)) return 3;
      return 4;
    }

    function dateParts() {
      const selected = new Date(`${dateInput.value}T00:00:00`);
      const month = selected.getMonth() + 1;
      const weekday = selected.getDay();
      const season = monthToSeason(month);
      return {month, weekday, season};
    }

    function normalizedTemp(valueC) {
      return Number(((valueC + 8) / 47).toFixed(4));
    }

    function normalizedFeelsLike(valueC) {
      return Number(((valueC + 16) / 66).toFixed(4));
    }

    function normalizedWind(valueKmh) {
      return Number((valueKmh / 67).toFixed(4));
    }

    function updateDerived() {
      const parts = dateParts();
      seasonLabel.textContent = seasonNames[parts.season];
      weekdayLabel.textContent = weekdayNames[parts.weekday];
      tempValue.textContent = `${tempC.value}°C`;
      atempValue.textContent = `${atempC.value}°C`;
      humValue.textContent = `${humPct.value}%`;
      windValue.textContent = `${windKmh.value} км/ч`;
    }

    async function loadModelInfo() {
      try {
        const response = await fetch("/model-info");
        const data = await response.json();
        modelName.textContent = data.final_model ? `${data.final_model} · R2 ${data.test_metrics.r2}` : "model.joblib";
      } catch (_) {
        modelName.textContent = "model.joblib";
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      errorBox.textContent = "";
      submitButton.disabled = true;
      predictionExtra.textContent = "Расчёт...";

      const parts = dateParts();
      const payload = {
        season: parts.season,
        mnth: parts.month,
        hr: Number(hourSelect.value),
        holiday: Number(document.querySelector("#holiday").value),
        weekday: parts.weekday,
        workingday: Number(document.querySelector("#workingday").value),
        weathersit: Number(document.querySelector("#weathersit").value),
        yr: dateInput.value.slice(0, 4) >= "2012" ? 1 : 0,
        temp: normalizedTemp(Number(tempC.value)),
        atemp: normalizedFeelsLike(Number(atempC.value)),
        hum: Number((Number(humPct.value) / 100).toFixed(4)),
        windspeed: normalizedWind(Number(windKmh.value))
      };

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        predictionValue.textContent = data.rounded_prediction;
        predictionExtra.textContent = `Модель: ${data.model}. Точное значение: ${data.prediction}.`;
      } catch (error) {
        predictionValue.textContent = "-";
        predictionExtra.textContent = "Не удалось получить прогноз.";
        errorBox.textContent = String(error.message || error);
      } finally {
        submitButton.disabled = false;
      }
    });

    [dateInput, tempC, atempC, humPct, windKmh].forEach((element) => {
      element.addEventListener("input", updateDerived);
    });

    setToday();
    updateDerived();
    loadModelInfo();
  </script>
</body>
</html>
"""


class PredictionRequest(BaseModel):
    season: int = Field(..., ge=1, le=4, description="1=winter, 2=spring, 3=summer, 4=fall")
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4)
    yr: int = Field(..., ge=0, le=1, description="0=2011-like period, 1=2012-like period")
    temp: float = Field(..., ge=0.0, le=1.0)
    atemp: float = Field(..., ge=0.0, le=1.0)
    hum: float = Field(..., ge=0.0, le=1.0)
    windspeed: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    prediction: float
    rounded_prediction: int
    model: str


class BatchPredictionRequest(BaseModel):
    records: list[PredictionRequest] = Field(..., min_length=1, max_length=100)


@app.get("/", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    return HTMLResponse(UI_HTML)


def read_metadata() -> dict[str, Any]:
    path = resolve_path(artifact_cfg["metadata_path"])
    if not path.exists():
        return {"model_loaded": False, "message": "metadata file is not found"}
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "method=%s path=%s status=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    model_path = resolve_path(artifact_cfg["model_path"])
    metadata_path = resolve_path(artifact_cfg["metadata_path"])
    return {
        "status": "ok" if model_path.exists() else "model_missing",
        "version": __version__,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "model_loaded": model_path.exists(),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return read_metadata()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    metadata = read_metadata()
    prediction = predict_one(request.model_dump())
    logger.info("prediction=%.3f hour=%s weather=%s", prediction, request.hr, request.weathersit)
    return PredictionResponse(
        prediction=round(prediction, 3),
        rounded_prediction=int(round(prediction)),
        model=str(metadata.get("final_model", "unknown")),
    )


@app.post("/predict-batch")
def batch_predict(request: BatchPredictionRequest) -> dict[str, Any]:
    metadata = read_metadata()
    values = predict_batch([record.model_dump() for record in request.records])
    return {
        "model": metadata.get("final_model", "unknown"),
        "predictions": [
            {"prediction": round(value, 3), "rounded_prediction": int(round(value))}
            for value in values
        ],
    }
