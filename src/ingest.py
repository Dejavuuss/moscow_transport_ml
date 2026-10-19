"""Сбор данных: API Портала открытых данных Правительства Москвы (apidata.mos.ru) -> data/raw -> PostgreSQL.

Что здесь важно с точки зрения инженерии данных:
  * **Пагинация.** API отдаёт не больше 1000 строк за запрос ($top/$skip), поэтому сначала
    спрашиваем /count, потом листаем страницы.
  * **Ретраи с экспоненциальной паузой.** Внешний API иногда отвечает 5xx/таймаутом —
    пайплайн не должен падать из-за одной сетевой ошибки.
  * **Снимок «как есть» (raw snapshot) + метаданные версии набора.** Сохраняем ответ API без
    изменений и номер версии/релиза набора: всегда можно понять, на каких данных обучена модель,
    и переиграть очистку без повторного скачивания (lineage / воспроизводимость).
  * **Посадочная зона в PostgreSQL** без ограничений уникальности: реальная выгрузка содержит
    дубликаты, принимаем её как есть, чистим на следующем шаге.
  * **Ключ — только из окружения** (MOS_API_KEY в .env), и он не попадает в логи.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DATASETS, settings

log = logging.getLogger(__name__)


class MosDataClient:
    """Тонкий клиент API data.mos.ru (только чтение)."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: int = 120,
                 retries: int = 4) -> None:
        self.api_key = api_key or settings.mos_api_key
        if not self.api_key:
            raise RuntimeError("Не задан MOS_API_KEY. Скопируйте .env.example в .env и впишите ключ API.")
        self.base_url = (base_url or settings.mos_api_url).rstrip("/")
        self.timeout, self.retries = timeout, retries

    def _get(self, path: str, **params) -> object:
        query = urllib.parse.urlencode({"api_key": self.api_key, **params})
        url = f"{self.base_url}/{path}?{query}"
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                if attempt == self.retries:
                    raise
                pause = 2 ** attempt
                log.warning("%s: попытка %d/%d не удалась (%s), повтор через %d с",
                            path, attempt, self.retries, type(e).__name__, pause)
                time.sleep(pause)
        raise AssertionError("unreachable")

    def dataset_meta(self, dataset_id: int) -> dict:
        meta = self._get(f"datasets/{dataset_id}")
        keep = ("Id", "Caption", "VersionNumber", "VersionDate", "ItemsCount", "DepartmentCaption")
        return {k: meta.get(k) for k in keep}

    def count(self, dataset_id: int) -> int:
        return int(self._get(f"datasets/{dataset_id}/count"))

    def rows(self, dataset_id: int) -> list[dict]:
        """Все строки набора (поле Cells), постранично."""
        n = self.count(dataset_id)
        out: list[dict] = []
        for skip in range(0, n, settings.page_size):
            page = self._get(f"datasets/{dataset_id}/rows", **{"$top": settings.page_size, "$skip": skip})
            out.extend(r["Cells"] for r in page)
        if len(out) != n:   # проверка полноты выгрузки — первая проверка качества данных
            raise ValueError(f"Набор {dataset_id}: ожидалось {n} строк, получено {len(out)}")
        return out


def fetch_all(raw_dir: Path | None = None, client: MosDataClient | None = None) -> dict[str, int]:
    """Скачивает все наборы из DATASETS в raw_dir/<имя>.json. Возвращает число строк по наборам."""
    raw_dir = raw_dir or settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = client or MosDataClient()
    counts = {}
    for name, ds_id in DATASETS.items():
        rows = client.rows(ds_id)
        snapshot = {"meta": client.dataset_meta(ds_id),
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "rows": rows}
        (raw_dir / f"{name}.json").write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
        counts[name] = len(rows)
        log.info("%s (набор %d, версия %s от %s): %d строк", name, ds_id, snapshot["meta"]["VersionNumber"],
                 snapshot["meta"]["VersionDate"], len(rows))
    return counts


# --------------------------------------------------------------------------------------
# Чтение снимков в DataFrame (единые имена колонок для всего проекта)
# --------------------------------------------------------------------------------------
QUARTERS = {"I квартал": 1, "II квартал": 2, "III квартал": 3, "IV квартал": 4}
MONTHS = {m: i for i, m in enumerate(["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август",
                                      "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"], start=1)}


def _rows(name: str, raw_dir: Path | None) -> list[dict]:
    path = (raw_dir or settings.raw_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}. Сначала скачайте данные: python -m src.ingest")
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def read_metro_raw(raw_dir: Path | None = None) -> pd.DataFrame:
    df = pd.DataFrame(_rows("metro_quarterly", raw_dir))
    return pd.DataFrame({
        "global_id": df["global_id"].astype("int64"),
        "station": df["NameOfStation"].astype(str).str.strip(),
        "line": df["Line"].astype(str).str.strip(),
        "year": df["Year"].astype(int),
        "quarter_name": df["Quarter"].astype(str).str.strip(),
        "incoming": pd.to_numeric(df["IncomingPassengers"], errors="coerce"),
        "outgoing": pd.to_numeric(df["OutgoingPassengers"], errors="coerce"),
    })


def read_monthly_raw(raw_dir: Path | None = None) -> pd.DataFrame:
    df = pd.DataFrame(_rows("transport_monthly", raw_dir))
    return pd.DataFrame({
        "year": df["Year"].astype(int),
        "month_name": df["Month"].astype(str).str.strip(),
        "transport": df["TransportType"].astype(str).str.strip(),
        "passengers": pd.to_numeric(df["PassengerTraffic"], errors="coerce"),
    })


def read_yearly_raw(raw_dir: Path | None = None) -> pd.DataFrame:
    df = pd.DataFrame(_rows("transport_yearly", raw_dir))
    return pd.DataFrame({
        "year": df["Year"].astype(int),
        "transport": df["TransportType"].astype(str).str.strip(),
        "passengers": pd.to_numeric(df["PassengerTraffic"], errors="coerce"),
    })


def load_to_postgres(raw_dir: Path | None = None) -> None:
    """Пишет сырые таблицы в посадочную зону PostgreSQL (таблицы создаёт data/schema.sql)."""
    from sqlalchemy import create_engine, text

    eng = create_engine(f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}"
                        f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}")
    tables = {"metro_quarterly_raw": read_metro_raw(raw_dir), "transport_monthly_raw": read_monthly_raw(raw_dir),
              "transport_yearly_raw": read_yearly_raw(raw_dir)}
    with eng.begin() as conn:            # одна транзакция: либо загрузились все таблицы, либо ни одной
        for table, df in tables.items():
            # числа приходят как float с NaN; NaN нельзя записать в BIGINT -> целые с NULL (Int64)
            for col in ("incoming", "outgoing", "passengers"):
                if col in df:
                    df[col] = df[col].round().astype("Int64")
            conn.execute(text(f"TRUNCATE {table}"))     # полная перезагрузка: наборы небольшие
            df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1000)
            log.info("PostgreSQL: %s <- %d строк", table, len(df))


if __name__ == "__main__":
    import argparse

    from .config import setup_logging

    setup_logging()
    ap = argparse.ArgumentParser(description="Скачать наборы data.mos.ru в data/raw")
    ap.add_argument("--to-postgres", action="store_true", help="после скачивания загрузить в PostgreSQL")
    ap.add_argument("--skip-download", action="store_true", help="использовать уже скачанные снимки")
    a = ap.parse_args()
    if not a.skip_download:
        fetch_all()
    if a.to_postgres:
        load_to_postgres()
