"""Единая конфигурация проекта.

Все настройки лежат в одном месте и переопределяются переменными окружения
(удобно для Docker / Airflow: один и тот же код, разные окружения).
Секреты (API-ключ портала, пароль БД) НИКОГДА не хардкодим — только из окружения / .env.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Минимальный загрузчик .env (без лишней зависимости): KEY=VALUE, # — комментарий.

    Переменные, уже заданные в окружении, не перетираются (окружение важнее файла).
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str) -> str:
    """Читает переменную окружения; если нет — возвращает значение по умолчанию."""
    return os.getenv(name, default)


# Наборы данных Портала открытых данных Правительства Москвы (data.mos.ru).
DATASETS = {
    "metro_quarterly": 62743,    # Пассажиропоток по станциям метрополитена: квартал, вход/выход
    "transport_monthly": 62521,  # Месячный пассажиропоток по видам общественного транспорта
    "transport_yearly": 62523,   # Годовой пассажиропоток по видам общественного транспорта
}


@dataclass(frozen=True)
class Settings:
    # ---- Пути ----
    raw_dir: Path = ROOT / "data" / "raw"              # снимки ответов API «как есть»
    processed_dir: Path = ROOT / "data" / "processed"  # результат очистки
    bundle_dir: Path = Path(_env("BUNDLE_DIR", str(ROOT / "artifacts" / "latest")))

    # ---- API портала открытых данных ----
    mos_api_url: str = _env("MOS_API_URL", "https://apidata.mos.ru/v1")
    mos_api_key: str = _env("MOS_API_KEY", "")
    page_size: int = 1000          # максимум, который разрешает API ($top ≤ 1000)

    # ---- PostgreSQL ----
    pg_host: str = _env("POSTGRES_HOST", "localhost")
    pg_port: str = _env("POSTGRES_PORT", "5432")
    pg_db: str = _env("POSTGRES_DB", "transport")
    pg_user: str = _env("POSTGRES_USER", "transport")
    pg_password: str = _env("POSTGRES_PASSWORD", "transport")

    # ---- MLflow ----
    # По умолчанию — локальный SQLite (поддерживает Model Registry без сервера).
    mlflow_tracking_uri: str = _env("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow_experiment: str = _env("MLFLOW_EXPERIMENT", "moscow_metro_quarterly")
    model_name: str = "moscow_metro_forecaster"
    champion_alias: str = "champion"
    seed: int = 42

    # ---- Разбиение по ВРЕМЕНИ (по кварталу-цели; никакого random split!) ----
    train_end: str = "2024Q2"      # train: цели до 2024Q2 включительно
    val_end: str = "2025Q2"        # val: 2024Q3..2025Q2; test: 2025Q3 и позже

    # ---- Модель прогноза ----
    n_lags: int = 5                # лаги 1..5 кварталов (5-й нужен для годового тренда)
    hidden: int = 64
    emb_dim: int = 8
    epochs: int = 400
    batch_size: int = 256
    lr: float = 3e-3
    patience: int = 40             # early stopping по val-loss

    # ---- Качество данных ----
    min_coverage: float = 0.90       # квартал «полный», если станции покрывают ≥90% месячного итога метро
    max_reconcile_gap: float = 0.01  # допустимое расхождение «сумма месяцев vs годовой итог»

    # ---- Детекция аномалий ----
    z_threshold: float = 3.5       # порог модифицированного z-score (Iglewicz–Hoaglin)
    closure_ratio: float = 0.25    # факт < 25% прогноза -> подозрение на закрытие станции/вестибюля
    min_line_stations: int = 3     # «событие линии» считаем, если на линии есть хотя бы 3 станции
    # Минимальный ПРАКТИЧЕСКИЙ размер эффекта (кроме статистической значимости): входы/выходы на
    # реальных данных очень стабильны, и без порога детектор поднимал бы тревогу из-за отклонения на 3%.
    min_station_effect: float = 0.20   # станция отклонилась от города и линии больше чем на ~20%
    min_line_effect: float = 0.07      # линия отклонилась от города больше чем на ~7%
    min_city_effect: float = 0.05      # весь город отклонился от прогноза больше чем на ~5%
    min_imbalance_effect: float = 0.10 # вход/выход отличаются от обычного больше чем на ~10%


settings = Settings()


def setup_logging(level: str = "INFO") -> None:
    """Единый формат логов (logging вместо print — можно писать в файл/агрегатор)."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
