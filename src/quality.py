"""Контроль качества данных (Data Quality): проверки, отчёт и «ворота качества» перед обучением.

Главная идея — **сверка независимых источников** (reconciliation). У портала есть три набора,
которые описывают одно и то же с разной детализацией:

    станции × квартал  ──Σ по станциям──►  метро за квартал  ◄──Σ 3 месяцев──  месячный набор
                                                                                 │ Σ 12 месяцев
                                                                                 ▼
                                                                             годовой набор

Если суммы не сходятся, значит один из источников неполон или содержит ошибку. Так находятся
проблемы, которых не видно внутри одной таблицы:
  * **неполный квартал** — в станционном наборе опубликована только часть станций
    (покрытие месячного итога < 90%): такие станции в модели — пропуски, а не «падение потока»;
  * **ошибка ввода в месячном наборе** — сумма месяцев не равна годовому итогу, а станции
    покрывают больше 100% месячного итога (январь 2025: 17.3 млн вместо ~172 млн — потерян разряд).

Ещё проверки: дубликаты (точные/конфликтные), невалидные значения, закрытые станции (нули),
дисбаланс «вход/выход», разрывы рядов, прекращённые ряды (троллейбус после 2020 г.).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import settings
from .data_loader import period_index, period_label

log = logging.getLogger(__name__)

# Какие линии станционного набора соответствуют какой строке месячного набора
MCC_LINE = "Московское центральное кольцо"
EXCLUDED_FROM_METRO = (MCC_LINE, "Московская монорельсовая транспортная система")
METRO_MONTHLY, MCC_MONTHLY = "Московский метрополитен", "Московское центральное кольцо"


def _issue(check: str, severity: str, message: str, **details) -> dict:
    return {"check": check, "severity": severity, "message": message, "details": details}


def quarter_coverage(metro: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Покрытие: Σ входов по станциям квартала / Σ месячного итога за те же 3 месяца."""
    st = metro[~metro["line"].isin(EXCLUDED_FROM_METRO)].groupby("qi").agg(
        n_stations=("key", "nunique"), stations_sum=("incoming", "sum"))
    mm = monthly[monthly["transport"] == METRO_MONTHLY].copy()
    mm["qi"] = period_index(mm["year"], mm["quarter"])
    m = mm.groupby("qi").agg(monthly_sum=("passengers", "sum"), n_months=("month", "count"))
    cov = st.join(m, how="left")
    cov["coverage"] = cov["stations_sum"] / cov["monthly_sum"]
    cov["status"] = np.select(
        [cov["n_months"] < 3, cov["coverage"] > 1.05, cov["coverage"] < settings.min_coverage],
        ["no_monthly", "monthly_suspect", "incomplete"], "ok")
    cov = cov.reset_index()
    cov.insert(1, "period", [period_label(q) for q in cov["qi"]])
    return cov


def yearly_reconciliation(monthly: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    """Σ 12 месяцев vs годовой итог по каждому виду транспорта."""
    m = monthly.groupby(["year", "transport"]).agg(monthly_sum=("passengers", "sum"),
                                                     n_months=("month", "count")).reset_index()
    j = m.merge(yearly.rename(columns={"passengers": "yearly"}), on=["year", "transport"], how="left")
    j["gap"] = j["monthly_sum"] / j["yearly"] - 1
    j["status"] = np.select([j["yearly"].isna(), j["n_months"] < 12, j["gap"].abs() > settings.max_reconcile_gap],
                            ["no_yearly", "partial_year", "mismatch"], "ok")
    return j


def station_gaps(metro: pd.DataFrame) -> pd.DataFrame:
    """Для каждой станции: первый/последний квартал, число пропущенных кварталов внутри ряда."""
    g = metro.groupby("key")["qi"].agg(["min", "max", "count"])
    g["missing_inside"] = g["max"] - g["min"] + 1 - g["count"]
    return g


def imbalance(metro: pd.DataFrame) -> pd.DataFrame:
    """log(вход/выход): в норме ≈ 0 (сколько вошло — столько вышло за квартал)."""
    ok = metro[(metro["incoming"] > 0) & (metro["outgoing"] > 0)].copy()
    ok["log_in_out"] = np.log(ok["incoming"] / ok["outgoing"])
    return ok


def build_report(metro: pd.DataFrame, monthly: pd.DataFrame, yearly: pd.DataFrame, clean_stats: dict) -> dict:
    s = clean_stats["metro"]
    issues: list[dict] = []
    n = max(s["n_raw_rows"], 1)
    if s["n_exact_duplicates"]:
        issues.append(_issue("duplicates", "warning", f"Точных дубликатов: {s['n_exact_duplicates']} "
                             f"({s['n_exact_duplicates'] / n:.1%} строк) — удалены", count=s["n_exact_duplicates"]))
    if s["n_conflicting_duplicates"]:
        sev = "critical" if s["n_conflicting_duplicates"] / n > 0.05 else "warning"
        issues.append(_issue("conflicting_duplicates", sev, "Дубли ключа с разными значениями — оставлена "
                             "последняя версия по global_id", count=s["n_conflicting_duplicates"]))
    if s["n_unknown_quarter"] / n > 0.01:
        issues.append(_issue("schema", "critical", "Нераспознанные значения квартала", count=s["n_unknown_quarter"]))
    closed = metro[metro["is_closed"] == 1]
    if len(closed):
        issues.append(_issue("closed_stations", "info", f"{len(closed)} станций-кварталов с нулевым потоком "
                             "(закрытие на ремонт) — не используются как цель обучения",
                             count=len(closed), stations=int(closed["key"].nunique())))

    cov = quarter_coverage(metro, monthly)
    for _, r in cov[cov["status"] == "incomplete"].iterrows():
        issues.append(_issue("incomplete_quarter", "warning",
                             f"{r['period']}: станции покрывают {r['coverage']:.0%} месячного итога метро "
                             f"({int(r['n_stations'])} станций) — квартал опубликован не полностью",
                             period=r["period"], coverage=round(float(r["coverage"]), 3)))
    for _, r in cov[cov["status"] == "monthly_suspect"].iterrows():
        issues.append(_issue("monthly_suspect", "warning",
                             f"{r['period']}: станции дают {r['coverage']:.0%} месячного итога — ошибка в "
                             "месячном наборе", period=r["period"], coverage=round(float(r["coverage"]), 3)))

    rec = yearly_reconciliation(monthly, yearly)
    for _, r in rec[rec["status"] == "mismatch"].iterrows():
        sev = "warning" if abs(r["gap"]) > 0.02 else "info"
        issues.append(_issue("yearly_mismatch", sev, f"{r['transport']} {r['year']}: Σ месяцев отличается от "
                             f"годового итога на {r['gap']:+.2%}", year=int(r["year"]), transport=r["transport"],
                             gap=round(float(r["gap"]), 4)))

    # Прекращённые ряды: последний год < 1% медианы ряда (троллейбус в Москве закрыт в 2020 г.)
    for tr, g in monthly.groupby("transport"):
        if g["passengers"].tail(12).median() < 0.01 * g["passengers"].median():
            issues.append(_issue("series_discontinued", "info", f"Ряд «{tr}» фактически прекращён "
                                 f"(последнее ненулевое значение: {g.loc[g['passengers'] > 0, 'date'].max():%Y-%m})",
                                 transport=tr))

    gaps = station_gaps(metro)
    imb = imbalance(metro)
    strong = imb[imb["log_in_out"].abs() > 0.5]
    if len(strong):
        issues.append(_issue("in_out_imbalance", "info", f"{len(strong)} станций-кварталов: вход и выход "
                             "отличаются больше чем в 1.6 раза", count=len(strong)))

    sev_count = pd.Series([i["severity"] for i in issues]).value_counts().to_dict()
    summary = {
        **{k: v for k, v in s.items() if k != "periods"}, "periods": s.get("periods"),
        "incomplete_periods": cov.loc[cov["status"] == "incomplete", "period"].tolist(),
        "monthly_suspect_periods": cov.loc[cov["status"] == "monthly_suspect", "period"].tolist(),
        "stations_with_gaps": int((gaps["missing_inside"] > 0).sum()),
        "issues": {k: sev_count.get(k, 0) for k in ("critical", "warning", "info")},
    }
    return {"summary": summary, "issues": issues,
            "quarter_coverage": json.loads(cov.to_json(orient="records", force_ascii=False)),
            "yearly_reconciliation": json.loads(rec.to_json(orient="records", force_ascii=False))}


def save_report(report: dict, path: Path | None = None) -> Path:
    path = path or settings.processed_dir / "dq_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for i in report["issues"]:
        (log.warning if i["severity"] != "info" else log.info)("[DQ %s] %s", i["severity"], i["message"])
    return path


def load_report(path: Path | None = None) -> dict:
    return json.loads((path or settings.processed_dir / "dq_report.json").read_text(encoding="utf-8"))


def check_quality_gate(report: dict, min_periods: int = 8) -> None:
    """Ворота качества: при критичных проблемах пайплайн падает ДО обучения (champion остаётся прежним)."""
    crit = [i["message"] for i in report["issues"] if i["severity"] == "critical"]
    if crit:
        raise ValueError("Критичные проблемы качества данных: " + "; ".join(crit))
    if report["summary"]["n_rows_clean"] == 0:
        raise ValueError("После очистки не осталось данных")
    first, last = report["summary"]["periods"]
    if period_index(int(last[:4]), int(last[-1])) - period_index(int(first[:4]), int(first[-1])) + 1 < min_periods:
        raise ValueError(f"Слишком короткая история: {first}..{last}")
