"""Airflow DAG: регулярное обновление данных с data.mos.ru и переобучение.

    скачать наборы (API) -> загрузить в PostgreSQL -> очистка -> ворота качества -> обучение + детекторы
    -> регистрация (champion/challenger) -> сводка экспериментов

  * schedule="@monthly": месячный набор обновляется раз в месяц, квартальный — раз в квартал;
    если новых данных нет, обучение даст ту же модель и она не станет новым champion'ом;
  * catchup=False — не догоняем пропущенные запуски (данные всегда скачиваются целиком);
  * max_active_runs=1 — два переобучения не должны одновременно писать в одну модель/реестр;
  * retries — внешний API и БД иногда временно недоступны;
  * ворота качества: если данные плохие, DAG падает ДО обучения, а действующий champion остаётся
    в проде (а не заменяется моделью, обученной на мусоре);
  * MOS_API_KEY берётся из окружения воркера (в проде — Airflow Connection/Variable или Vault).

Требование: в окружении воркера установлены зависимости проекта (requirements.txt) и проект в PYTHONPATH.
Совместим с Airflow 2.x и 3.x.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

try:                                     # Airflow 3
    from airflow.sdk import dag, task
except ImportError:                      # Airflow 2
    from airflow.decorators import dag, task

DEFAULT_ARGS = {"owner": "ml-team", "retries": 2, "retry_delay": timedelta(minutes=10)}


@dag(dag_id="moscow_metro_retrain", schedule="@monthly", start_date=datetime(2026, 1, 1), catchup=False,
     max_active_runs=1, default_args=DEFAULT_ARGS, tags=["ml", "transport", "data.mos.ru"])
def retrain_pipeline():
    @task
    def fetch() -> dict:
        from src.ingest import fetch_all
        return fetch_all()                              # {набор: число строк} -> XCom

    @task
    def load_postgres(_: dict) -> None:
        if os.getenv("LOAD_POSTGRES", "1") == "1":
            from src.ingest import load_to_postgres
            load_to_postgres()

    @task
    def clean_and_check() -> dict:
        from src.training import run_data_stage
        report = run_data_stage(os.getenv("CLEAN_ENGINE", "spark"))   # внутри — ворота качества (ValueError)
        return report["summary"]

    @task
    def train_and_register(_: dict) -> dict:
        from src.training import CANDIDATES, run_pipeline, setup_mlflow
        setup_mlflow()
        result = run_pipeline(CANDIDATES)
        return {k: result[k] for k in ("run_id", "version", "promoted", "anom_f1", "n_station_anomalies")}

    @task
    def summarize(result: dict) -> None:
        from src.training import compare_runs, setup_mlflow
        setup_mlflow()
        compare_runs()
        logging.getLogger(__name__).info("Переобучение завершено: %s", result)

    counts = fetch()
    loaded = load_postgres(counts)
    summary = clean_and_check()
    loaded >> summary
    summarize(train_and_register(summary))


retrain_pipeline()
