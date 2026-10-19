"""Обучение, оценка и трекинг экспериментов (MLflow) + Model Registry.

Этапы (запуск: python -m src.training --stage all):
  data       очистка -> DQ-отчёт -> ворота качества
  baselines  три бейзлайна без обучения (прошлый квартал, год назад, год назад × тренд) — планка качества
  train      кандидаты (GBM, PyTorch MLP) -> выбор лучшего по VAL -> детекторы аномалий ->
             дообучение на всех данных -> бандл для API -> регистрация модели
  sweep      то же, но с перебором гиперпараметров
  compare    таблица всех запусков, отсортированная по val_wape

ВАЖНО про выбор модели: champion выбирается по VAL-метрике. Тест трогаем только для итогового
отчёта — иначе выбор «лучшей» модели по тесту = утечка и завышенная оценка.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from . import data_loader
from .config import settings, setup_logging
from .features import (BASELINES, FEATURE_NAMES, Features, Panel, baseline_log, build_features, forecast_metrics,
                       frame_to_panel, split_masks, training_samples)
from .models.anomaly_detection import (CityMonthlyDetector, StationAnomalyDetector, detection_frame,
                                       evaluate_by_injection, line_events)
from .models.forecasting import MODELS
from .quality import build_report, check_quality_gate, load_report, save_report

log = logging.getLogger(__name__)

CANDIDATES = {
    "gbm": dict(kind="gbm", learning_rate=0.1, max_iter=1500, max_leaf_nodes=31, min_samples_leaf=20),
    "torch_mlp": dict(kind="torch_mlp", hidden=32, lr=5e-4),
}
SWEEP = {
    "gbm_small_leaves": dict(kind="gbm", learning_rate=0.05, max_iter=2000, max_leaf_nodes=7, min_samples_leaf=50),
    "torch_mlp_wide": dict(kind="torch_mlp", hidden=128, dropout=0.3, lr=1e-3),
    "torch_mlp_64": dict(kind="torch_mlp", hidden=64, lr=1e-3),
}


@dataclass
class Data:
    panel: Panel
    vocab: list[str]               # справочник линий
    f: Features                    # все обучающие пары (станция, квартал)
    d: np.ndarray                  # цель: delta к якорю (log)
    masks: dict[str, np.ndarray]   # train / val / test по времени цели
    monthly: pd.DataFrame
    dq: dict

    def part(self, name: str) -> tuple[Features, np.ndarray]:
        m = self.masks[name]
        return self.f.subset(m), self.d[m]

    def actual(self, f: Features) -> np.ndarray:
        return self.panel.incoming[f.s, f.t]


def run_data_stage(engine: str = "pandas") -> dict:
    stats = data_loader.clean_all()
    if engine == "spark":
        metro, s = data_loader.clean_metro_spark()
        metro.to_parquet(settings.processed_dir / "metro_quarterly.parquet", index=False)
        stats["metro"].update(s)
    report = build_report(*data_loader.load_clean(), stats)
    save_report(report)
    check_quality_gate(report)
    return report


def prepare_data() -> Data:
    metro, monthly, _ = data_loader.load_clean()
    panel = frame_to_panel(metro)
    vocab = sorted(set(panel.lines))
    f, d = training_samples(panel, vocab)
    masks = split_masks(panel, f.t)
    log.info("Панель %s (станции × кварталы %s..%s); обучающих пар: %s", panel.incoming.shape, panel.period(0),
             panel.period(panel.Q - 1), {k: int(v.sum()) for k, v in masks.items()})
    return Data(panel, vocab, f, d, masks, monthly, load_report())


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)


def _split_metrics(data: Data, predict_log, prefix_splits=("train", "val", "test")) -> dict:
    out = {}
    for split in prefix_splits:
        f, _ = data.part(split)
        for k, v in forecast_metrics(data.actual(f), np.exp(predict_log(f))).items():
            out[f"{split}_{k}"] = v
    return out


def _common_params() -> dict:
    return {"train_end": settings.train_end, "val_end": settings.val_end, "n_lags": settings.n_lags,
            "n_features": len(FEATURE_NAMES)}


# --------------------------------------------------------------------------------------
# Бейзлайны
# --------------------------------------------------------------------------------------
def run_baselines(data: Data) -> dict:
    results = {}
    for name in BASELINES:
        with mlflow.start_run(run_name=f"baseline_{name}"):
            mlflow.set_tag("model_type", "baseline")
            mlflow.log_params({"model": name, **_common_params()})
            metrics = _split_metrics(data, lambda f, n=name: baseline_log(f, n))
            mlflow.log_metrics(metrics)
            results[name] = metrics
            log.info("baseline %-15s val_wape=%.4f test_wape=%.4f", name, metrics["val_wape"], metrics["test_wape"])
    return results


# --------------------------------------------------------------------------------------
# Кандидаты
# --------------------------------------------------------------------------------------
def make_model(cfg: dict, data: Data):
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    if kind == "gbm":
        return MODELS[kind](n_lines=len(data.vocab), seed=settings.seed, **cfg)
    return MODELS[kind](n_stations=len(data.panel.keys), n_lines=len(data.vocab), seed=settings.seed,
                        epochs=settings.epochs, batch_size=settings.batch_size, patience=settings.patience, **cfg)


def run_candidate(data: Data, name: str, cfg: dict) -> dict:
    """Обучение на train с подбором числа итераций по val; метрики train/val/test -> MLflow."""
    f_tr, d_tr = data.part("train")
    f_va, d_va = data.part("val")
    model = make_model(cfg, data)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("model_type", model.kind)
        mlflow.log_params({**model.hparams, **_common_params(), "candidate": name})
        info = model.fit(f_tr, d_tr, f_va, d_va,
                         on_epoch=lambda e, tr, va: mlflow.log_metrics(
                             {k: v for k, v in {"train_loss": tr, "val_mae_log": va}.items() if v is not None}, step=e))
        mlflow.log_param("best_n_iter", info["n_iter"])
        metrics = _split_metrics(data, lambda f: f.anchor + model.predict_delta(f))
        mlflow.log_metrics(metrics)
        log.info("%-18s val_wape=%.4f test_wape=%.4f (n_iter=%d)", name, metrics["val_wape"], metrics["test_wape"],
                 info["n_iter"])
    return {"name": name, "cfg": cfg, "model": model, "run_id": run.info.run_id, "n_iter": info["n_iter"], **metrics}


# --------------------------------------------------------------------------------------
# Champion: детекторы, дообучение на всех данных, бандл, реестр
# --------------------------------------------------------------------------------------
def finalize_champion(data: Data, best: dict, baselines: dict) -> dict:
    p, model = data.panel, best["model"]
    predict_log = lambda f: f.anchor + model.predict_delta(f)   # модель, обученная ТОЛЬКО на train
    t_val, t_test = p.t_of(_next_period(settings.train_end)), p.t_of(_next_period(settings.val_end))
    fv = detection_frame(p, predict_log, t_val, t_test, data.vocab)
    ft = detection_frame(p, predict_log, t_test, p.Q, data.vocab)

    # --- 1. Детекторы аномалий: обучение без меток на val, оценка инъекцией на test ---
    det = StationAnomalyDetector().fit(fv)
    m_inj = evaluate_by_injection(det, ft)
    m_inj_noctx = evaluate_by_injection(StationAnomalyDetector(use_context=False).fit(fv), ft)
    catalog = det.detect(pd.concat([fv, ft], ignore_index=True))
    cities = det.city_events(catalog)
    lines = line_events(catalog)
    skip = tuple(i["details"]["transport"] for i in data.dq["issues"] if i["check"] == "series_discontinued")
    city_det = CityMonthlyDetector().fit(data.monthly, fit_end_year=int(settings.train_end[:4]), skip=skip)
    city = city_det.detect(data.monthly, data.dq)

    # Прогноз без аномальных кварталов: какая часть ошибки объясняется событиями, а не моделью
    flagged = set(zip(catalog.loc[catalog["is_anomaly"], "key"], catalog.loc[catalog["is_anomaly"], "t"]))
    metrics = {}
    for split in ("val", "test"):
        f, _ = data.part(split)
        keep = np.array([(p.keys[s], t) not in flagged for s, t in zip(f.s, f.t)])
        pred = np.exp(predict_log(f))
        metrics[f"{split}_wape_clean"] = forecast_metrics(data.actual(f)[keep], pred[keep])["wape"]
        metrics[f"{split}_anomaly_share"] = float(1 - keep.mean())
    metrics.update({f"anom_{k}": v for k, v in m_inj.items()})
    metrics.update({f"anom_noctx_{k}": v for k, v in m_inj_noctx.items()})
    metrics.update({"n_station_anomalies": int(catalog["is_anomaly"].sum()), "n_city_events": len(cities),
                    "n_line_events": len(lines), "n_city_monthly_anomalies": int(city["is_anomaly"].sum()),
                    "n_city_data_errors": int((city["attribution"] == "data_error").sum())})

    # Интервал прогноза из остатков val: [q10, q90] отношения факт/прогноз
    f_va, _ = data.part("val")
    resid = np.log(data.actual(f_va)) - predict_log(f_va)
    interval = {"q10": float(np.quantile(resid, 0.1)), "q90": float(np.quantile(resid, 0.9))}

    # --- 2. Production refit: та же конфигурация на ВСЕХ данных, число итераций — найденное по val ---
    prod = make_model(best["cfg"], data)
    prod.fit(data.f, data.d, n_iter=best["n_iter"])
    forecasts = next_quarter_forecasts(p, prod, data.vocab, interval)

    # --- 3. Бандл и логирование в run'е champion'а ---
    meta = {"run_id": best["run_id"], "candidate": best["name"], "model_kind": prod.kind,
            "val_wape": best["val_wape"], "test_wape": best["test_wape"],
            "baseline_val_wape": {k: v["val_wape"] for k, v in baselines.items()},
            "train_end": settings.train_end, "val_end": settings.val_end,
            "data_periods": [p.period(0), p.period(p.Q - 1)], "forecast_period": p.period(p.Q),
            "interval": interval, "line_vocab": data.vocab, "n_iter": best["n_iter"],
            "datasets": {k: _dataset_version(k) for k in ("metro_quarterly", "transport_monthly", "transport_yearly")}}
    tmp = Path(tempfile.mkdtemp()) / "bundle"
    save_bundle(tmp, prod, det, city_det, meta, p, catalog, cities, lines, city, forecasts, data.dq)

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("champion_candidate", "true")
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
        mlflow.log_dict(data.dq, "data_quality/dq_report.json")
        mlflow.log_dict(meta, "bundle_meta.json")
        mlflow.log_artifacts(str(tmp), artifact_path="bundle")
        version = _log_registered_model(prod, f_va.subset(np.arange(len(f_va.s)) < 8))
    promoted = promote_if_better(version, best["val_wape"])
    if promoted:
        if settings.bundle_dir.exists():
            shutil.rmtree(settings.bundle_dir)
        shutil.copytree(tmp, settings.bundle_dir)        # то, что отдаёт API
        log.info("Модель %s v%s повышена до '%s' -> %s", settings.model_name, version, settings.champion_alias,
                 settings.bundle_dir)
    shutil.rmtree(tmp.parent, ignore_errors=True)
    log.info("Аномалии: %d станционных, %d событий линий, %d городских кварталов, %d месячных (%d — ошибки данных)",
             metrics["n_station_anomalies"], metrics["n_line_events"], metrics["n_city_events"],
             metrics["n_city_monthly_anomalies"], metrics["n_city_data_errors"])
    log.info("Инъекция (test): precision=%.3f recall=%.3f false_alarm=%.4f | без контекста: precision=%.3f recall=%.3f "
             "recall_line_event: %.3f vs %.3f (тип угадан: %.3f vs %.3f)", m_inj["precision"], m_inj["recall"],
             m_inj["false_alarm_rate"], m_inj_noctx["precision"], m_inj_noctx["recall"],
             m_inj.get("recall_line_event", np.nan), m_inj_noctx.get("recall_line_event", np.nan),
             m_inj.get("type_acc_line_event", np.nan), m_inj_noctx.get("type_acc_line_event", np.nan))
    return {"run_id": best["run_id"], "version": version, "promoted": promoted, **metrics}


def _next_period(label: str) -> str:
    from .data_loader import parse_period, period_label
    return period_label(parse_period(label) + 1)


def _dataset_version(name: str) -> dict:
    path = settings.raw_dir / f"{name}.json"
    if not path.exists():
        return {}
    snap = json.loads(path.read_text(encoding="utf-8"))
    return {**snap["meta"], "fetched_at": snap["fetched_at"]}


def next_quarter_forecasts(p: Panel, model, vocab: list[str], interval: dict) -> pd.DataFrame:
    """Прогноз на квартал после последнего в данных — для всех станций, работавших в последние 4 квартала."""
    y = p.y
    active = np.nonzero(np.isfinite(y[:, -4:]).any(axis=1))[0]
    f = build_features(p, active, np.full(len(active), p.Q), vocab)
    ok = np.isfinite(f.anchor)
    f = f.subset(ok)
    pred = np.exp(f.anchor + model.predict_delta(f))
    return pd.DataFrame({"key": [p.keys[s] for s in f.s], "station": [p.stations[s] for s in f.s],
                         "line": [p.lines[s] for s in f.s], "period": p.period(p.Q), "forecast": pred.round(),
                         "low": (pred * np.exp(interval["q10"])).round(), "high": (pred * np.exp(interval["q90"])).round()})


def save_bundle(d: Path, model, det: StationAnomalyDetector, city_det: CityMonthlyDetector, meta: dict, panel: Panel,
                catalog: pd.DataFrame, cities: pd.DataFrame, lines: pd.DataFrame, city: pd.DataFrame,
                forecasts: pd.DataFrame, dq: dict) -> None:
    """Бандл = всё, что нужно API для одной версии модели (без обращения к сырым данным)."""
    d.mkdir(parents=True, exist_ok=True)
    model.save(d / "model")
    det.save(d / "detectors")
    city_det.save(d / "detectors")
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (d / "dq_report.json").write_text(json.dumps(dq, indent=2, ensure_ascii=False), encoding="utf-8")
    panel.to_frame().to_csv(d / "panel.csv", index=False)
    catalog.drop(columns=["t"]).to_csv(d / "station_anomalies.csv", index=False)
    cities.to_csv(d / "city_quarter_events.csv", index=False)
    lines.to_csv(d / "line_events.csv", index=False)
    city.to_csv(d / "city_monthly_anomalies.csv", index=False)
    forecasts.to_csv(d / "forecast_next_quarter.csv", index=False)


def _log_registered_model(model, example: Features) -> str:
    """Регистрирует модель в MLflow Model Registry, возвращает номер версии.

    Пример входа нужен MLflow для сигнатуры модели (и для трассировки графа в формате pt2).
    """
    if model.kind == "gbm":
        import mlflow.sklearn
        info = mlflow.sklearn.log_model(model.model, name="model", registered_model_name=settings.model_name,
                                        input_example=model._X(example))
    else:
        import mlflow.pytorch

        from .models.forecasting import FlowNetServing
        info = mlflow.pytorch.log_model(FlowNetServing(model.net).eval(), name="model",
                                        registered_model_name=settings.model_name,
                                        input_example=model.serving_input(example))
    return str(info.registered_model_version)


def promote_if_better(version: str, val_wape: float, min_improvement: float = 0.01) -> bool:
    """Challenger vs champion: новая версия получает алиас 'champion', только если она ЗАМЕТНО лучше по VAL.

    Порог min_improvement (1% относительного улучшения WAPE) защищает от «переключений на шум»:
    на ~1000 val-точках разница 0.0733 vs 0.0732 статистически ничего не значит, а каждая смена
    модели в проде — это риск и работа (мониторинг, сравнение, откат).
    Алиасы заменили устаревшие «стадии» (Staging/Production) в MLflow ≥ 2.9.
    """
    client = MlflowClient()
    name = settings.model_name
    try:
        champ = client.get_model_version_by_alias(name, settings.champion_alias)
        champ_wape = client.get_run(champ.run_id).data.metrics.get("val_wape", float("inf"))
    except Exception:       # алиаса ещё нет — первая модель автоматически становится champion
        champ_wape = float("inf")
    if val_wape < champ_wape * (1 - min_improvement) or not settings.bundle_dir.exists():
        client.set_registered_model_alias(name, settings.champion_alias, version)
        return True
    log.info("Новая модель (val_wape=%.4f) не лучше champion (%.4f) хотя бы на %.0f%% — остаётся прежняя",
             val_wape, champ_wape, 100 * min_improvement)
    return False


def compare_runs() -> pd.DataFrame:
    """Сводная таблица экспериментов (то же, что вкладка Compare в MLflow UI)."""
    exp = mlflow.get_experiment_by_name(settings.mlflow_experiment)
    runs = mlflow.search_runs([exp.experiment_id], order_by=["metrics.val_wape ASC"])
    cols = [c for c in ["tags.mlflow.runName", "tags.model_type", "metrics.val_wape", "metrics.test_wape",
                        "metrics.val_mape", "metrics.test_mape", "metrics.test_wape_clean", "metrics.anom_f1",
                        "metrics.anom_false_alarm_rate"] if c in runs.columns]
    table = runs[cols].rename(columns=lambda c: c.rsplit(".", 1)[-1]).drop_duplicates("runName", keep="first")
    out = settings.bundle_dir.parent / "experiments.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    log.info("\n%s", table.round(4).to_string(index=False))
    return table


def run_pipeline(candidates: dict) -> dict:
    data = prepare_data()
    baselines = run_baselines(data)
    results = [run_candidate(data, name, cfg) for name, cfg in candidates.items()]
    best = min(results, key=lambda r: r["val_wape"])
    best_base = min(v["val_wape"] for v in baselines.values())
    log.info("Лучший кандидат по val: %s (val_wape=%.4f; лучший бейзлайн %.4f)", best["name"], best["val_wape"],
             best_base)
    return finalize_champion(data, best, baselines)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "baselines", "train", "sweep", "compare", "all"], default="all")
    ap.add_argument("--engine", choices=["pandas", "spark"], default="pandas")
    a = ap.parse_args()
    if a.stage in ("data", "all"):
        run_data_stage(a.engine)
    setup_mlflow()
    if a.stage == "baselines":
        run_baselines(prepare_data())
    if a.stage in ("train", "all"):
        run_pipeline(CANDIDATES)
    if a.stage == "sweep":
        run_pipeline({**CANDIDATES, **SWEEP})
    if a.stage in ("compare", "sweep", "all"):
        compare_runs()


if __name__ == "__main__":
    main()
