"""Интеграционные тесты API на обученном бандле (artifacts/latest). Пропускаются, если модель не обучена."""
import pytest

from src.config import settings

pytestmark = pytest.mark.skipif(not (settings.bundle_dir / "meta.json").exists(),
                                reason="нет обученной модели: python -m src.training --stage all")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["model_loaded"] and r["status"] == "ok"


def test_forecast_next_quarter(client):
    st = client.get("/stations").json()[0]
    r = client.get("/forecast", params={"station": st["station"], "line": st["line"]})
    assert r.status_code == 200
    body = r.json()
    assert body["low"] <= body["forecast"] <= body["high"]


def test_unknown_station_404_and_bad_period_422(client):
    assert client.get("/forecast", params={"station": "Нет такой станции"}).status_code == 404
    st = client.get("/stations").json()[0]
    assert client.get("/forecast", params={"station": st["station"], "line": st["line"],
                                           "period": "2025-03"}).status_code == 422


def test_anomaly_check_detects_closure_and_drop(client):
    st = client.get("/stations").json()[0]
    period = client.get("/health").json()["data_periods"][1]
    body = {"station": st["station"], "line": st["line"], "period": period, "incoming": 0, "outgoing": 0}
    assert client.post("/anomaly/check", json=body).json()["anomaly_type"] == "closure"
    fc = client.get("/forecast", params={"station": st["station"], "line": st["line"], "period": period}).json()
    body.update(incoming=fc["forecast"] * 0.1, outgoing=fc["forecast"] * 0.1)
    v = client.post("/anomaly/check", json=body).json()
    assert v["status"] == "anomaly" and v["anomaly_type"] in ("partial_closure", "station_drop")


def test_catalog_quality_and_city(client):
    assert client.get("/anomalies", params={"limit": 5}).status_code == 200
    assert "line_events" in client.get("/anomalies/events").json()
    assert client.get("/quality").json()["summary"]["n_rows_clean"] > 0
    assert client.get("/anomalies/city-monthly").status_code == 200
