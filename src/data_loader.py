"""Очистка данных: Pandas (основной путь локально) + PySpark (тот же шаг для больших объёмов / Docker).

Что делает очистка квартальных данных метро (ключевой пункт вакансии «обеспечение качества данных»):
  1. нормализация строк (пробелы) и справочник кварталов -> номер квартала, индекс периода;
  2. дубликаты по ключу (станция, линия, год, квартал): точные — удаляем, конфликтные
     (разные значения) — оставляем последнюю версию по global_id и считаем их отдельно;
  3. невалидные значения (NULL / отрицательные) -> NaN;
  4. нулевой вход или выход — НЕ ошибка, а закрытие станции на ремонт: ставим флаг is_closed
     (в обучении такие кварталы не используются как цель, а детектор аномалий их показывает).

Ключ станции — «название + линия»: у пересадочных узлов (например, «Авиамоторная») на разных
линиях разные ряды, и при переносе станции на другую линию (Некрасовская -> БКЛ) начинается
новый ряд. Это честнее, чем склеивать их по названию.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import settings
from .ingest import MONTHS, QUARTERS, read_metro_raw, read_monthly_raw, read_yearly_raw

log = logging.getLogger(__name__)
KEY = ["station", "line", "year", "quarter"]


def period_index(year, quarter):
    """Сквозной номер квартала: 2021Q1 -> 8084. Удобно для лагов (t-1, t-4) без дат."""
    return np.asarray(year) * 4 + np.asarray(quarter) - 1


def period_label(qi: int) -> str:
    return f"{int(qi) // 4}Q{int(qi) % 4 + 1}"


def parse_period(label: str) -> int:
    """'2025Q3' -> 8102."""
    y, q = label.upper().split("Q")
    return int(y) * 4 + int(q) - 1


def station_key(station: str, line: str) -> str:
    return f"{station} | {line}"


# --------------------------------------------------------------------------------------
# Pandas
# --------------------------------------------------------------------------------------
def clean_metro(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Сырые квартальные строки -> чистая таблица (station, line, key, year, quarter, qi, incoming, outgoing,
    is_closed) и счётчики для DQ-отчёта."""
    df = raw.copy()
    stats = {"n_raw_rows": len(df)}
    df["quarter"] = df["quarter_name"].map(QUARTERS)
    stats["n_unknown_quarter"] = int(df["quarter"].isna().sum())
    df = df.dropna(subset=["quarter"]).astype({"quarter": int})

    exact = df.duplicated(KEY + ["incoming", "outgoing"])
    stats["n_exact_duplicates"] = int(exact.sum())
    df = df[~exact]
    conflict = df.duplicated(KEY, keep=False)
    stats["n_conflicting_duplicates"] = int(df.loc[conflict].duplicated(KEY).sum())
    df = df.sort_values("global_id").drop_duplicates(KEY, keep="last")   # последняя версия записи

    for col in ("incoming", "outgoing"):
        bad = df[col].isna() | (df[col] < 0)
        stats[f"n_invalid_{col}"] = int(bad.sum())
        df[col] = df[col].where(~bad).astype(float)
    df["is_closed"] = ((df["incoming"] == 0) | (df["outgoing"] == 0)).astype(int)
    stats["n_closed_station_quarters"] = int(df["is_closed"].sum())

    df["qi"] = period_index(df["year"], df["quarter"])
    df["period"] = [period_label(q) for q in df["qi"]]
    df["key"] = [station_key(s, l) for s, l in zip(df["station"], df["line"])]
    cols = ["key", "station", "line", "year", "quarter", "qi", "period", "incoming", "outgoing", "is_closed"]
    df = df[cols].sort_values(["key", "qi"]).reset_index(drop=True)
    stats["n_rows_clean"] = len(df)
    stats["n_stations"] = int(df["key"].nunique())
    stats["n_lines"] = int(df["line"].nunique())
    stats["periods"] = [period_label(df["qi"].min()), period_label(df["qi"].max())]
    return df, stats


def clean_monthly(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df["month"] = df["month_name"].map(MONTHS)
    stats = {"n_unknown_month": int(df["month"].isna().sum())}
    df = df.dropna(subset=["month"]).astype({"month": int})
    stats["n_duplicates"] = int(df.duplicated(["year", "month", "transport"]).sum())
    df = df.drop_duplicates(["year", "month", "transport"], keep="last")
    df["passengers"] = df["passengers"].where(df["passengers"] >= 0).astype(float)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df["quarter"] = (df["month"] - 1) // 3 + 1
    return df[["transport", "date", "year", "month", "quarter", "passengers"]].sort_values(
        ["transport", "date"]).reset_index(drop=True), stats


def clean_all(raw_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    """Полный шаг очистки (Pandas). Пишет parquet в data/processed, возвращает счётчики."""
    out_dir = out_dir or settings.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metro, s_metro = clean_metro(read_metro_raw(raw_dir))
    monthly, s_month = clean_monthly(read_monthly_raw(raw_dir))
    yearly = read_yearly_raw(raw_dir).drop_duplicates(["year", "transport"], keep="last")
    metro.to_parquet(out_dir / "metro_quarterly.parquet", index=False)
    monthly.to_parquet(out_dir / "transport_monthly.parquet", index=False)
    yearly.to_parquet(out_dir / "transport_yearly.parquet", index=False)
    stats = {"metro": s_metro, "monthly": s_month, "engine": "pandas"}
    log.info("Очистка: %s", stats)
    return stats


def load_clean(processed_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = processed_dir or settings.processed_dir
    path = d / "metro_quarterly.parquet"
    if not path.exists():
        raise FileNotFoundError("Нет очищенных данных. Запустите: python -m src.data_loader (или make clean)")
    return (pd.read_parquet(path), pd.read_parquet(d / "transport_monthly.parquet"),
            pd.read_parquet(d / "transport_yearly.parquet"))


# --------------------------------------------------------------------------------------
# PySpark: та же логика для квартальных данных метро
# --------------------------------------------------------------------------------------
def get_spark(app_name: str = "moscow_metro"):
    """SparkSession. Импорт внутри функции: без PySpark остальной проект продолжает работать.
    PySpark 4 требует Java 17+ (в Docker-образе она есть)."""
    from pyspark.sql import SparkSession

    return (SparkSession.builder.appName(app_name)
            .master(os.getenv("SPARK_MASTER", "local[*]"))
            # 200 партиций шаффла по умолчанию — для локального запуска лишние накладные расходы
            .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
            .getOrCreate())


def clean_metro_spark(raw_dir: Path | None = None, spark=None) -> tuple[pd.DataFrame, dict]:
    """Очистка квартальных данных на Spark. Тест сверяет результат с Pandas-версией.

    На реальных объёмах ДИТ (события валидаторов — миллиарды строк) меняется только SPARK_MASTER
    и источник (JDBC из PostgreSQL / parquet в S3), код шагов тот же.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    own = spark is None
    spark = spark or get_spark()
    path = str((raw_dir or settings.raw_dir) / "metro_quarterly.json")
    rows = (spark.read.option("multiLine", True).json(path)
            .select(F.explode("rows").alias("r")).select("r.*"))
    q_map = F.create_map(*[x for k, v in QUARTERS.items() for x in (F.lit(k), F.lit(v))])
    df = rows.select(F.col("global_id").cast("long"), F.trim("NameOfStation").alias("station"),
                     F.trim("Line").alias("line"), F.col("Year").cast("int").alias("year"),
                     q_map[F.trim("Quarter")].alias("quarter"),
                     F.col("IncomingPassengers").cast("double").alias("incoming"),
                     F.col("OutgoingPassengers").cast("double").alias("outgoing")).cache()
    n_raw = df.count()
    df = df.filter(F.col("quarter").isNotNull())
    n_known = df.count()
    exact = df.dropDuplicates(KEY + ["incoming", "outgoing"])
    n_exact = n_known - exact.count()
    # Конфликтные дубли: оставляем последнюю версию по global_id (оконная функция row_number)
    w = Window.partitionBy(*KEY).orderBy(F.col("global_id").desc())
    dedup = exact.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")
    n_conflict = exact.count() - dedup.count()
    out = (dedup
           .withColumn("incoming", F.when(F.col("incoming") >= 0, F.col("incoming")))
           .withColumn("outgoing", F.when(F.col("outgoing") >= 0, F.col("outgoing")))
           .withColumn("is_closed", ((F.col("incoming") == 0) | (F.col("outgoing") == 0)).cast("int"))
           .withColumn("qi", F.col("year") * 4 + F.col("quarter") - 1)
           .withColumn("key", F.concat_ws(" | ", "station", "line")))
    pdf = out.toPandas()
    if own:
        spark.stop()
    pdf["period"] = [period_label(q) for q in pdf["qi"]]
    cols = ["key", "station", "line", "year", "quarter", "qi", "period", "incoming", "outgoing", "is_closed"]
    pdf = pdf[cols].sort_values(["key", "qi"]).reset_index(drop=True)
    stats = {"n_raw_rows": n_raw, "n_unknown_quarter": n_raw - n_known, "n_exact_duplicates": n_exact,
             "n_conflicting_duplicates": n_conflict, "n_rows_clean": len(pdf), "engine": "spark"}
    return pdf, stats


if __name__ == "__main__":
    import argparse
    import json

    from .config import setup_logging
    from .quality import build_report, check_quality_gate, save_report

    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["pandas", "spark"], default="pandas")
    a = ap.parse_args()
    stats = clean_all()
    if a.engine == "spark":       # Spark-версия перезаписывает квартальную таблицу метро
        metro, s = clean_metro_spark()
        metro.to_parquet(settings.processed_dir / "metro_quarterly.parquet", index=False)
        stats["metro"].update(s)
        stats["engine"] = "spark"
    report = build_report(*load_clean(), stats)
    save_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    check_quality_gate(report)
