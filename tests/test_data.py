"""Тесты очистки и контроля качества данных."""
import shutil
import subprocess

import pandas as pd
import pytest

from src import data_loader as dl
from src import quality as q
from src.ingest import read_metro_raw, read_monthly_raw, read_yearly_raw


def test_cleaning_removes_duplicates_and_flags_closures(raw_dir):
    df, stats = dl.clean_metro(read_metro_raw(raw_dir))
    assert stats["n_exact_duplicates"] == 2
    assert stats["n_conflicting_duplicates"] == 1
    assert not df.duplicated(["key", "qi"]).any()
    # из конфликтных дублей остаётся более новая версия (больший global_id)
    first = df[(df["key"] == "Станция 0-0 | Линия 0") & (df["period"] == "2021Q3")]
    assert first["incoming"].item() == 1
    assert stats["n_closed_station_quarters"] == 1
    assert df["is_closed"].sum() == 1


def test_period_helpers_roundtrip():
    assert dl.period_index(2021, 1) == 8084
    assert dl.period_label(8084) == "2021Q1"
    assert dl.parse_period("2025q3") == dl.period_index(2025, 3)


def test_coverage_flags_incomplete_quarter_and_suspect_monthly(raw_dir):
    metro, _ = dl.clean_metro(read_metro_raw(raw_dir))
    monthly, _ = dl.clean_monthly(read_monthly_raw(raw_dir))
    total = metro.groupby("qi")["incoming"].sum()
    # подгоняем месячный итог метро так, чтобы станции покрывали ровно 100% каждого квартала
    monthly["qi"] = dl.period_index(monthly["year"], monthly["quarter"])
    monthly["passengers"] = monthly["qi"].map(total / 3)
    monthly = monthly.dropna(subset=["passengers"])
    q1 = dl.parse_period("2022Q1")
    metro_incomplete = metro[~((metro["qi"] == q1) & (metro["line"] == "Линия 0"))]  # убрали треть станций
    monthly.loc[(monthly["qi"] == dl.parse_period("2023Q2")) & (monthly["month"] == 4), "passengers"] /= 10
    cov = q.quarter_coverage(metro_incomplete, monthly).set_index("period")
    assert cov.loc["2022Q1", "status"] == "incomplete"
    assert cov.loc["2023Q2", "status"] == "monthly_suspect"
    assert cov.loc["2021Q1", "status"] == "ok"


def test_yearly_reconciliation_detects_mismatch(raw_dir):
    monthly, _ = dl.clean_monthly(read_monthly_raw(raw_dir))
    rec = q.yearly_reconciliation(monthly, read_yearly_raw(raw_dir)).set_index("year")
    assert rec.loc[2024, "status"] == "mismatch"
    assert rec.loc[2023, "status"] == "ok"


def test_quality_gate(raw_dir, tmp_path):
    stats = dl.clean_all(raw_dir, tmp_path / "processed")
    report = q.build_report(*dl.load_clean(tmp_path / "processed"), stats)
    q.check_quality_gate(report)                      # фикстура — «нормальные» данные
    assert any(i["check"] == "yearly_mismatch" for i in report["issues"])
    report["issues"].append({"check": "x", "severity": "critical", "message": "сломано", "details": {}})
    with pytest.raises(ValueError, match="сломано"):
        q.check_quality_gate(report)


def _java_ok() -> bool:
    if not shutil.which("java"):
        return False
    out = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr
    return '"1.8' not in out and '"11' not in out       # PySpark 4 требует Java 17+


@pytest.mark.skipif(not _java_ok(), reason="PySpark 4 требует Java 17+ (есть в Docker-образе)")
def test_spark_cleaning_matches_pandas(raw_dir):
    pdf, _ = dl.clean_metro(read_metro_raw(raw_dir))
    sdf, stats = dl.clean_metro_spark(raw_dir)
    cols = ["key", "qi", "incoming", "outgoing", "is_closed"]
    pd.testing.assert_frame_equal(pdf[cols].reset_index(drop=True), sdf[cols].reset_index(drop=True),
                                  check_dtype=False)
    assert stats["n_exact_duplicates"] == 2
