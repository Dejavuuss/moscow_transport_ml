"""Общие фикстуры: маленькие таблицы в формате ответа API data.mos.ru (без сети и без скачанных данных)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.ingest import MONTHS

QNAMES = ["I квартал", "II квартал", "III квартал", "IV квартал"]
SEASON = np.array([0.95, 1.0, 0.9, 1.15])


def metro_cells(n_lines: int = 3, per_line: int = 5, years=range(2021, 2025), seed: int = 0) -> list[dict]:
    """Квартальный поток: сезонность + небольшой рост + шум; вход ≈ выход."""
    rng = np.random.default_rng(seed)
    rows, gid = [], 1
    for li in range(n_lines):
        for si in range(per_line):
            base = rng.uniform(0.5e6, 3e6)
            for yi, y in enumerate(years):
                for q in range(4):
                    v = base * SEASON[q] * (1.03 ** yi) * rng.normal(1, 0.01)
                    rows.append({"global_id": gid, "NameOfStation": f"Станция {li}-{si}", "Line": f"Линия {li}",
                                 "Year": y, "Quarter": QNAMES[q], "IncomingPassengers": int(v),
                                 "OutgoingPassengers": int(v * rng.normal(1, 0.005))})
                    gid += 1
    return rows


def monthly_cells(years=range(2019, 2025)) -> list[dict]:
    names = list(MONTHS)
    return [{"Year": y, "Month": names[m], "TransportType": "Московский метрополитен",
             "PassengerTraffic": int(2e8 * (1 + 0.1 * np.sin(m / 2)) * (1.02 ** (y - 2019)))}
            for y in years for m in range(12)]


@pytest.fixture
def raw_dir(tmp_path):
    """Снимки API в tmp_path: метро (с дублями и закрытием), месячный и годовой наборы."""
    metro = metro_cells()
    metro += [dict(metro[0]), dict(metro[1])]                              # 2 точных дубликата
    metro.append({**metro[2], "global_id": 10_000, "IncomingPassengers": 1})  # конфликтный дубль (новее)
    metro[5]["IncomingPassengers"] = 0                                      # закрытие станции
    monthly = monthly_cells()
    yearly = pd.DataFrame(monthly).groupby(["Year", "TransportType"])["PassengerTraffic"].sum().astype(float).reset_index()
    yearly.loc[yearly["Year"] == 2024, "PassengerTraffic"] *= 1.10       # годовой итог не сходится с месяцами
    for name, rows in {"metro_quarterly": metro, "transport_monthly": monthly,
                       "transport_yearly": yearly.to_dict("records")}.items():
        snap = {"meta": {"Id": 0, "VersionNumber": "test"}, "fetched_at": "2026-01-01", "rows": rows}
        (tmp_path / f"{name}.json").write_text(json.dumps(snap, ensure_ascii=False, default=int), encoding="utf-8")
    return tmp_path


@pytest.fixture
def clean_metro_df(raw_dir):
    from src.data_loader import clean_metro
    from src.ingest import read_metro_raw
    return clean_metro(read_metro_raw(raw_dir))[0]
