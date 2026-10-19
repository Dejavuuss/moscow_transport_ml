"""Инференс: «бандл» модели и функции, которые использует API.

Бандл — папка, которая целиком описывает версию модели:
    model/                       модель прогноза, дообученная на всех данных (GBM или PyTorch)
    detectors/                   станционный детектор (+ Isolation Forest) и городской детектор
    panel.csv                    история «станция × квартал» — из неё строятся признаки
    station_anomalies.csv        каталог: каждая станция-квартал val/test с типом и объяснением
    line_events.csv, city_quarter_events.csv, city_monthly_anomalies.csv
    forecast_next_quarter.csv    готовые прогнозы на следующий квартал
    dq_report.json, meta.json    отчёт о качестве данных, версии наборов, метрики
Бандл логируется в MLflow как артефакт run'а и копируется в artifacts/latest при повышении до champion.
Обучение и API используют ОДНИ И ТЕ ЖЕ функции признаков (src.features) -> нет train/serving skew.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import settings
from .data_loader import parse_period, station_key
from .features import Panel, build_features, frame_to_panel
from .models.anomaly_detection import CityMonthlyDetector, StationAnomalyDetector
from .models.forecasting import load_model

log = logging.getLogger(__name__)


@dataclass
class Bundle:
    model: object
    detector: StationAnomalyDetector
    city_detector: CityMonthlyDetector
    panel: Panel
    meta: dict
    catalog: pd.DataFrame
    line_events: pd.DataFrame
    city_events: pd.DataFrame
    city_monthly: pd.DataFrame
    forecasts: pd.DataFrame
    dq: dict

    @property
    def vocab(self) -> list[str]:
        return self.meta["line_vocab"]


def load_bundle(d: Path | None = None) -> Bundle:
    d = Path(d or settings.bundle_dir)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"В {d} нет обученной модели. Запустите: python -m src.training --stage all")
    read = lambda name, **kw: pd.read_csv(d / name, **kw)
    panel_df = read("panel.csv")
    return Bundle(
        model=load_model(d / "model"),
        detector=StationAnomalyDetector.load(d / "detectors"),
        city_detector=CityMonthlyDetector.load(d / "detectors"),
        panel=frame_to_panel(panel_df),
        meta=json.loads((d / "meta.json").read_text(encoding="utf-8")),
        catalog=read("station_anomalies.csv"),
        line_events=read("line_events.csv"),
        city_events=read("city_quarter_events.csv"),
        city_monthly=read("city_monthly_anomalies.csv", parse_dates=["date"]),
        forecasts=read("forecast_next_quarter.csv"),
        dq=json.loads((d / "dq_report.json").read_text(encoding="utf-8")),
    )


def load_bundle_from_mlflow(alias: str | None = None) -> Bundle:
    """Бандл версии модели с алиасом champion из MLflow Model Registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mv = MlflowClient().get_model_version_by_alias(settings.model_name, alias or settings.champion_alias)
    return load_bundle(Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="bundle")))


def _station_index(b: Bundle, station: str, line: str | None) -> int:
    p = b.panel
    if line:
        return p.index(station_key(station, line))
    hits = [i for i, s in enumerate(p.stations) if s.lower() == station.lower()]
    if not hits:
        raise KeyError(f"Неизвестная станция: {station}")
    if len(hits) > 1:
        raise KeyError(f"Станция «{station}» есть на нескольких линиях — укажите line: "
                       + ", ".join(p.lines[i] for i in hits))
    return hits[0]


def _predict_log(b: Bundle, s: int, t: int) -> tuple[float, dict]:
    f = build_features(b.panel, np.array([s]), np.array([t]), b.vocab)
    if not np.isfinite(f.anchor[0]):
        raise ValueError("Недостаточно истории: нужен хотя бы один из 5 предыдущих кварталов")
    anchor_lag = next(k for k in (4, 1, 2, 3, 5) if np.isfinite(f.lags[0, k - 1]))
    return float(f.anchor[0] + b.model.predict_delta(f)[0]), {"anchor_lag_quarters": anchor_lag}


def forecast_station(b: Bundle, station: str, line: str | None = None, period: str | None = None) -> dict:
    """Прогноз входов на квартал period (по умолчанию — следующий после данных).

    Для прошлых кварталов это честный бэктест: признаки строятся только из кварталов ДО period.
    """
    p = b.panel
    s = _station_index(b, station, line)
    t = p.Q if period is None else p.t_of(period)
    if not 0 < t <= p.Q:
        raise ValueError(f"Квартал вне диапазона данных: доступно {p.period(1)}..{p.period(p.Q)}")
    log_pred, info = _predict_log(b, s, t)
    pred = float(np.exp(log_pred))
    iv = b.meta["interval"]
    hist = [{"period": p.period(i), "incoming": float(p.incoming[s, i]), "closed": bool(p.closed[s, i])}
            for i in range(max(0, t - 8), t) if np.isfinite(p.incoming[s, i])]
    actual = float(p.incoming[s, t]) if t < p.Q and np.isfinite(p.incoming[s, t]) else None
    return {"station": p.stations[s], "line": p.lines[s], "period": p.period(t), "forecast": round(pred),
            "low": round(pred * np.exp(iv["q10"])), "high": round(pred * np.exp(iv["q90"])), "actual": actual,
            "model_version": b.meta.get("run_id"), "history": hist, **info}


def check_observation(b: Bundle, station: str, line: str | None, period: str, incoming: float,
                      outgoing: float) -> dict:
    """Новое (или исправленное) наблюдение станции за квартал -> вердикт детектора.

    Контекст города/линии берётся из каталога, если для этого квартала он уже посчитан по всем станциям;
    для нового квартала контекста ещё нет — оценка идёт относительно прогноза станции.
    """
    p = b.panel
    s = _station_index(b, station, line)
    t = p.t_of(period)
    if not 0 < t <= p.Q:
        raise ValueError(f"Квартал вне диапазона: доступно {p.period(1)}..{p.period(p.Q)}")
    closed = incoming == 0 or outgoing == 0
    try:
        expected = float(np.exp(_predict_log(b, s, t)[0]))
    except ValueError:
        if not closed:          # без истории прогноза нет, но нулевой поток — закрытие и без него
            raise
        expected = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(p.incoming[s, :t] / p.outgoing[s, :t])
    past = lr[np.isfinite(lr) & ~p.closed[s, :t]]
    usual = np.median(past) if len(past) >= 2 else 0.0
    imb_dev = (np.log(incoming / outgoing) - usual) if not closed else np.nan

    ctx = b.catalog[(b.catalog["period"] == period) & (b.catalog["line"] == p.lines[s])]
    r_city = float(ctx["r_city"].iloc[0]) if len(ctx) else np.nan
    r_line = float(ctx["r_line"].iloc[0]) if len(ctx) else np.nan
    row = pd.DataFrame([{"key": p.keys[s], "station": p.stations[s], "line": p.lines[s], "period": period, "t": t,
                         "actual": float(incoming), "outgoing": float(outgoing), "expected": expected,
                         "is_closed": closed, "r": np.log(max(incoming, 1.0) / expected), "imb_dev": imb_dev,
                         "r_city": r_city, "r_line": r_line}])
    r = b.detector.detect(row).iloc[0]
    _f = lambda x: None if not np.isfinite(x) else round(float(x), 3)
    return {"station": p.stations[s], "line": p.lines[s], "period": period, "actual": float(incoming),
            "expected": round(expected) if np.isfinite(expected) else None,
            "change": round(incoming / expected - 1, 4) if np.isfinite(expected) else None,
            "status": "anomaly" if r["is_anomaly"] else "ok", "anomaly_type": r["anomaly_type"],
            "severity": r["severity"], "explanation": r["explanation"], "context_known": bool(len(ctx)),
            "scores": {"z_rel": _f(r["z_rel"]), "z_line": _f(r["z_line"]), "z_imbalance": _f(r["z_imb"]),
                       "isolation_forest": _f(r["if_score"])}}


def filter_catalog(b: Bundle, period: str | None = None, line: str | None = None, anomaly_type: str | None = None,
                   severity: str | None = None, only_anomalies: bool = True, limit: int = 100) -> list[dict]:
    c = b.catalog
    if only_anomalies:
        c = c[c["is_anomaly"]]
    if period:
        parse_period(period)               # валидация формата
        c = c[c["period"] == period.upper()]
    if line:
        c = c[c["line"] == line]
    if anomaly_type:
        c = c[c["anomaly_type"] == anomaly_type]
    if severity:
        c = c[c["severity"] == severity]
    c = c.assign(_abs=c["z_rel"].abs()).sort_values(["severity", "_abs"], ascending=[True, False]).head(limit)
    cols = ["station", "line", "period", "actual", "expected", "anomaly_type", "severity", "explanation"]
    return json.loads(c[cols].to_json(orient="records", force_ascii=False))
