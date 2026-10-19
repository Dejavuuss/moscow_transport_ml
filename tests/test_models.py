"""Тесты моделей прогноза и детекторов аномалий."""
import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.models.anomaly_detection import (CityMonthlyDetector, StationAnomalyDetector, evaluate_by_injection,
                                          line_events)
from src.models.forecasting import GBMForecaster, TorchForecaster, load_model


@pytest.mark.parametrize("kind", ["gbm", "torch"])
def test_forecaster_fit_predict_save_load(clean_metro_df, tmp_path, kind):
    p = F.frame_to_panel(clean_metro_df)
    vocab = sorted(set(p.lines))
    f, d = F.training_samples(p, vocab)
    if kind == "gbm":
        m = GBMForecaster(len(vocab), max_iter=50)
    else:
        m = TorchForecaster(len(p.keys), len(vocab), epochs=30, patience=5)
    info = m.fit(f, d, f, d)
    pred = m.predict_delta(f)
    assert pred.shape == d.shape and np.isfinite(pred).all() and info["n_iter"] >= 1
    # на гладких данных модель должна быть не хуже «нулевого» прогноза delta = 0 (т.е. бейзлайна-якоря)
    assert np.mean(np.abs(pred - d)) <= np.mean(np.abs(d)) + 1e-3
    m.save(tmp_path / "m")
    np.testing.assert_allclose(load_model(tmp_path / "m").predict_delta(f), pred, rtol=1e-5, atol=1e-5)


def _frame(n_lines=4, per_line=8, quarters=6, seed=0) -> pd.DataFrame:
    """Кадр детекции: остатки r ~ N(0, 0.03), без аномалий."""
    rng = np.random.default_rng(seed)
    rows = [{"key": f"S{l}-{s}", "station": f"S{l}-{s}", "line": f"L{l}", "t": t, "period": f"2025Q{t % 4 + 1}",
             "actual": 1e6, "outgoing": 1e6, "expected": 1e6, "is_closed": False,
             "r": rng.normal(0, 0.03), "imb_dev": rng.normal(0, 0.01)}
            for l in range(n_lines) for s in range(per_line) for t in range(quarters)]
    df = pd.DataFrame(rows)
    df["actual"] = df["expected"] * np.exp(df["r"])
    return df


def test_detector_separates_station_line_and_closure():
    det = StationAnomalyDetector().fit(_frame(seed=1))
    df = _frame(seed=2)
    one = (df["key"] == "S0-0") & (df["t"] == 2)
    df.loc[one, "r"] += np.log(0.5)                          # одна станция −50%
    line = (df["line"] == "L1") & (df["t"] == 3)
    df.loc[line, "r"] += np.log(0.7)                         # вся линия −30%
    closed = (df["key"] == "S2-0") & (df["t"] == 4)
    df.loc[closed, ["actual", "is_closed"]] = [0.0, True]
    df["actual"] = np.where(df["is_closed"], 0.0, df["expected"] * np.exp(df["r"]))
    res = det.detect(df)
    assert res.loc[one, "anomaly_type"].item() == "station_drop"
    assert (res.loc[line, "anomaly_type"] == "line_event").all()   # одно событие линии, а не 8 станционных
    assert res.loc[closed, "anomaly_type"].item() == "closure"
    assert res["is_anomaly"].sum() <= one.sum() + line.sum() + closed.sum() + 1
    ev = line_events(res)
    assert len(ev) == 1 and ev["n_stations"].item() == 8


def test_city_wide_shift_is_not_a_station_anomaly():
    det = StationAnomalyDetector().fit(_frame(seed=1))
    df = _frame(seed=3)
    df.loc[df["t"] == 1, "r"] += np.log(0.85)                  # весь город −15%
    df["actual"] = df["expected"] * np.exp(df["r"])
    res = det.detect(df)
    assert not res.loc[df["t"] == 1, "is_anomaly"].any()
    assert "2025Q2" in det.city_events(res)["period"].tolist()


def test_injection_evaluation_reports_metrics():
    det = StationAnomalyDetector().fit(_frame(seed=1))
    m = evaluate_by_injection(det, _frame(n_lines=6, per_line=10, seed=4), rounds=3)
    assert {"precision", "recall", "false_alarm_rate", "recall_line_event"} <= set(m)
    assert m["recall"] > 0.8 and m["false_alarm_rate"] < 0.05


def test_city_detector_finds_typo_and_attributes_data_error():
    dates = pd.date_range("2019-01-01", "2025-12-01", freq="MS")
    noise = np.random.default_rng(0).normal(1, 0.02, len(dates))          # реальные ряды не идеально гладкие
    y = 2e8 * (1 + 0.1 * np.sin(dates.month / 2)) * 1.02 ** (dates.year - 2019) * noise
    monthly = pd.DataFrame({"transport": "Московский метрополитен", "date": dates, "year": dates.year,
                            "month": dates.month, "passengers": y})
    monthly.loc[monthly["date"] == "2025-01-01", "passengers"] /= 10        # потерян разряд
    dq = {"issues": [{"check": "yearly_mismatch", "severity": "warning", "message": "",
                      "details": {"transport": "Московский метрополитен", "year": 2025, "gap": -0.07}}]}
    det = CityMonthlyDetector().fit(monthly, fit_end_year=2024)
    res = det.detect(monthly, dq)
    hit = res[res["date"] == "2025-01-01"].iloc[0]
    assert hit["is_anomaly"] and hit["attribution"] == "data_error"
    # медиана уровня: одна ошибка не порождает ложных тревог в следующие месяцы
    assert not res[(res["date"] > "2025-01-01")]["is_anomaly"].any()
