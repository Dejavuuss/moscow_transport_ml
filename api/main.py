"""FastAPI-сервис: прогноз пассажиропотока станций метро, каталог аномалий и качество данных.

Запуск:  uvicorn api.main:app --reload      Документация: http://localhost:8000/docs

Модель загружается ОДИН раз при старте (lifespan), а не на каждый запрос. Источник модели:
  MODEL_SOURCE=local   -> папка artifacts/latest (по умолчанию)
  MODEL_SOURCE=mlflow  -> бандл версии с алиасом 'champion' из MLflow Model Registry
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from src.config import setup_logging
from src.inference import (check_observation, filter_catalog, forecast_station, load_bundle,
                           load_bundle_from_mlflow)

from .schemas import PERIOD, ForecastResponse, Observation, Verdict

setup_logging()
log = logging.getLogger("api")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["bundle"] = load_bundle_from_mlflow() if os.getenv("MODEL_SOURCE", "local") == "mlflow" else load_bundle()
        log.info("Модель загружена: %s", state["bundle"].meta.get("run_id"))
    except Exception:       # сервис поднимается и без модели: /health покажет проблему, остальное вернёт 503
        log.exception("Не удалось загрузить модель")
    yield
    state.clear()


app = FastAPI(title="Moscow Metro ML", version="2.0.0", lifespan=lifespan,
              description="Прогноз квартального пассажиропотока станций метро и детекция аномалий "
                          "на открытых данных data.mos.ru")


def _bundle():
    if "bundle" not in state:
        raise HTTPException(503, "Модель не загружена. Обучите её: python -m src.training --stage all")
    return state["bundle"]


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\"")) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/health")
def health() -> dict:
    """Liveness/readiness-проба для Docker/Kubernetes."""
    b = state.get("bundle")
    return {"status": "ok" if b else "degraded", "model_loaded": b is not None,
            "model_version": b.meta.get("run_id") if b else None,
            "data_periods": b.meta.get("data_periods") if b else None}


@app.get("/model")
def model_info() -> dict:
    """Метаданные модели: метрики, бейзлайны, версии наборов данных, на которых она обучена."""
    m = _bundle().meta
    return {k: m[k] for k in ("run_id", "candidate", "model_kind", "val_wape", "test_wape", "baseline_val_wape",
                              "train_end", "val_end", "data_periods", "forecast_period", "datasets") if k in m}


@app.get("/stations")
def stations(line: str | None = None) -> list[dict]:
    p = _bundle().panel
    last = [p.period(int(np.nonzero(np.isfinite(row))[0].max())) for row in p.incoming]
    out = [{"station": s, "line": l, "last_period": lp} for s, l, lp in zip(p.stations, p.lines, last)]
    return [o for o in out if line is None or o["line"] == line]


@app.get("/forecast", response_model=ForecastResponse)
def forecast(station: str, line: str | None = None,
             period: str | None = Query(None, pattern=PERIOD, description="По умолчанию — следующий квартал")):
    """Прогноз входов на квартал. Для прошедших кварталов — бэктест (модель видит только прошлое)."""
    return _call(forecast_station, _bundle(), station, line, period.upper() if period else None)


@app.get("/forecast/next-quarter")
def forecast_next(line: str | None = None, top: int = Query(20, ge=1, le=500)) -> list[dict]:
    """Готовые прогнозы на следующий квартал (самые загруженные станции первыми)."""
    f = _bundle().forecasts
    if line:
        f = f[f["line"] == line]
    return f.sort_values("forecast", ascending=False).head(top).to_dict(orient="records")


@app.post("/anomaly/check", response_model=Verdict)
def anomaly_check(obs: Observation):
    """Проверить наблюдение станции за квартал: тип аномалии, серьёзность и объяснение."""
    return _call(check_observation, _bundle(), obs.station, obs.line, obs.period.upper(), obs.incoming, obs.outgoing)


@app.get("/anomalies")
def anomalies(period: str | None = Query(None, pattern=PERIOD), line: str | None = None,
              anomaly_type: str | None = Query(None, alias="type"), severity: str | None = None,
              limit: int = Query(100, ge=1, le=2000)) -> list[dict]:
    """Каталог найденных станционных аномалий (val + test периоды)."""
    return _call(filter_catalog, _bundle(), period, line, anomaly_type, severity, True, limit)


@app.get("/anomalies/events")
def events() -> dict:
    """События уровня линии и города: одно событие вместо десятков станционных тревог."""
    b = _bundle()
    return {"line_events": b.line_events.to_dict(orient="records"),
            "city_events": b.city_events.to_dict(orient="records")}


@app.get("/anomalies/city-monthly")
def city_monthly(attribution: str | None = Query(None, description="event | data_error")) -> list[dict]:
    """Аномалии месячных рядов по видам транспорта с атрибуцией «реальное событие / ошибка данных»."""
    c = _bundle().city_monthly
    c = c[c["is_anomaly"]]
    if attribution:
        c = c[c["attribution"] == attribution]
    c = c.assign(date=c["date"].dt.strftime("%Y-%m"))
    return c.round({"change": 4, "z": 2, "expected": 0}).to_dict(orient="records")


@app.get("/quality")
def quality() -> dict:
    """Отчёт о качестве данных последнего обучения: сводка и найденные проблемы."""
    dq = _bundle().dq
    return {"summary": dq["summary"], "issues": dq["issues"]}
