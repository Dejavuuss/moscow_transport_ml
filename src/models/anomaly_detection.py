"""Детекция аномалий на реальных данных: «что произошло на станции» и «событие это или ошибка данных».

ИДЕЯ. Аномалия = факт сильно отличается от того, что ОЖИДАЛА модель прогноза (построенного только
по прошлому). Но на квартальных данных важно ещё и ОБЪЯСНИТЬ отклонение: виновата станция, линия
или весь город (а может, это вообще ошибка данных)? Поэтому отклонение раскладывается по иерархии:

    r       = log(факт / прогноз)                          — отклонение станции от ожидания
    r_city  = медиана r по всем станциям квартала           — «весь город» (сезон, погода, неполные данные)
    r_line  = медиана (r − r_city) по станциям линии        — «вся линия» (закрытие участка, новая линия рядом)
    r_rel   = r − r_city − r_line                           — отклонение именно этой станции
    imb     = log(вход/выход) − обычный для станции уровень — дисбаланс учёта

1. **Станция** (StationAnomalyDetector): робастные z-score (медиана/MAD — устойчивы к самим аномалиям)
   + минимальный практический размер эффекта + Isolation Forest на [z_rel, z_line, z_imb] как второе
   мнение для severity. Типы:
     closure          нулевой вход/выход — станция закрыта (ремонт, реконструкция)
     partial_closure  факт < 25% прогноза — закрыт вестибюль / станция работала часть квартала
     station_drop     станция просела сильнее города и линии (закрытие выхода, ремонт эскалаторов, …)
     station_surge    станция выросла сильнее города и линии (открылась пересадка/ТПУ рядом)
     line_event       просела/выросла вся линия относительно города
     imbalance        вход ≠ выход сильнее обычного (сбой учёта, односторонний вестибюль)
   Плюс **city_event** — квартал, в котором отклонился весь город (отдельная таблица city_events).
   Зачем иерархия: когда весь город «ниже прогноза» на 12%, детектор без контекста поднимет сотни
   тревог; с контекстом — одно городское событие. Когда закрыт участок линии — одно событие линии,
   а не 10 станционных. И наоборот: общий рост линии не маскирует проблему отдельной станции.

2. **Город** (CityMonthlyDetector) — месячные ряды по видам транспорта: сезонный профиль + робастный
   уровень (медиана трёх прошлых месяцев) -> z-score остатка. Находит локдаун весны 2020 г. и
   «провалы», которые на деле являются ошибками ввода.

3. **Атрибуция «событие или ошибка данных»** — сверка с независимыми источниками из DQ-отчёта
   (src.quality): если месячное значение не сходится с годовым итогом и со станционными данными —
   это ошибка данных (январь 2025: 17.3 млн вместо ~172 млн), а не реальное падение спроса.

Оценка без разметки: на реальных данных нет меток аномалий, поэтому качество детектора меряется
**инъекцией** контролируемых аномалий в реальный test-период (evaluate_by_injection): известно, куда
вставили, — считаем recall, precision и долю ложных тревог. Это стандартная практика для unsupervised.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..config import settings
from ..features import Features, Panel, build_features

log = logging.getLogger(__name__)
MAD_K = 1.4826          # MAD·1.4826 ≈ σ для нормального распределения


def robust_center_scale(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    med = float(np.median(x))
    return med, float(MAD_K * np.median(np.abs(x - med))) or 1e-6


# --------------------------------------------------------------------------------------
# Кадр для детекции: факт, прогноз, дисбаланс — по всем станциям-кварталам периода
# --------------------------------------------------------------------------------------
def detection_frame(panel: Panel, predict_log: Callable[[Features], np.ndarray], t_lo: int, t_hi: int,
                    line_vocab: list[str]) -> pd.DataFrame:
    """Все наблюдённые станции-кварталы t∈[t_lo, t_hi) (включая закрытые), для которых есть прогноз."""
    obs = np.isfinite(panel.incoming[:, t_lo:t_hi])
    s, t = np.nonzero(obs)
    t = t + t_lo
    f = build_features(panel, s, t, line_vocab)
    ok = np.isfinite(f.anchor)
    f = f.subset(ok)
    expected = np.exp(predict_log(f))
    actual, outgoing = panel.incoming[f.s, f.t], panel.outgoing[f.s, f.t]
    return pd.DataFrame({
        "key": [panel.keys[i] for i in f.s], "station": [panel.stations[i] for i in f.s],
        "line": [panel.lines[i] for i in f.s], "period": [panel.period(x) for x in f.t], "t": f.t,
        "actual": actual, "outgoing": outgoing, "expected": expected,
        "is_closed": panel.closed[f.s, f.t],
        "r": np.log(np.maximum(actual, 1.0) / expected),
        "imb_dev": _imbalance_dev(panel, f.s, f.t),
    })


def _imbalance_dev(panel: Panel, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """log(вход/выход) минус медиана этой станции по ПРОШЛЫМ кварталам (без закрытых)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(panel.incoming / panel.outgoing)
    lr[panel.closed | ~np.isfinite(lr)] = np.nan
    city = float(np.nanmedian(lr))
    out = np.full(len(s), np.nan)
    for i, (si, ti) in enumerate(zip(s, t)):
        past = lr[si, :ti]
        past = past[np.isfinite(past)]
        out[i] = lr[si, ti] - (np.median(past) if len(past) >= 2 else city)
    return out


def hierarchy(df: pd.DataFrame, min_line: int, min_city: int = 10) -> pd.DataFrame:
    """Разложение отклонения: r_city (медиана по городу), r_line (медиана по линии после вычета города).

    Считается по открытым станциям; если станций мало — компонента NaN (контекста нет).
    """
    open_ = df[~df["is_closed"]]
    city = open_.groupby("t")["r"].agg(["median", "count"])
    city.loc[city["count"] < min_city, "median"] = np.nan
    r_city = df["t"].map(city["median"]).astype(float)
    tmp = open_.assign(_x=open_["r"] - r_city.loc[open_.index].fillna(0.0))
    line = tmp.groupby(["line", "t"])["_x"].agg(["median", "count"])
    line.loc[line["count"] < min_line, "median"] = np.nan
    r_line = pd.Series(df.set_index(["line", "t"]).index.map(line["median"]), index=df.index, dtype=float)
    return pd.DataFrame({"r_city": r_city, "r_line": r_line}, index=df.index)


# --------------------------------------------------------------------------------------
# Станционный детектор
# --------------------------------------------------------------------------------------
TYPE_RU = {"closure": "закрытие станции", "partial_closure": "частичное закрытие", "station_drop": "падение потока",
           "station_surge": "рост потока", "line_event": "событие линии", "imbalance": "дисбаланс вход/выход"}


@dataclass
class StationAnomalyDetector:
    z_threshold: float = settings.z_threshold
    closure_ratio: float = settings.closure_ratio
    min_line_stations: int = settings.min_line_stations
    min_station_effect: float = settings.min_station_effect
    min_line_effect: float = settings.min_line_effect
    min_city_effect: float = settings.min_city_effect
    min_imbalance_effect: float = settings.min_imbalance_effect
    use_context: bool = True          # False — детектор «без иерархии» (для сравнения в отчёте)
    seed: int = settings.seed
    stats: dict = field(default_factory=dict)
    iforest: IsolationForest | None = None

    # --- признаки ---
    def _decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.use_context and {"r_city", "r_line"} <= set(df.columns):
            h = df[["r_city", "r_line"]].astype(float).copy()   # контекст уже известен (проверка одной строки в API)
        elif self.use_context:
            h = hierarchy(df, self.min_line_stations)
        else:
            h = pd.DataFrame({"r_city": np.nan, "r_line": np.nan}, index=df.index)
        h["r_rel"] = df["r"] - h["r_city"].fillna(0.0) - h["r_line"].fillna(0.0)
        return h

    def _z(self, df: pd.DataFrame) -> pd.DataFrame:
        st = self.stats
        out = self._decompose(df)
        out["z_rel"] = (out["r_rel"] - st["rel"][0]) / st["rel"][1]
        out["z_line"] = ((out["r_line"] - st["line"][0]) / st["line"][1]).fillna(0.0)
        out["z_imb"] = ((df["imb_dev"] - st["imb"][0]) / st["imb"][1]).fillna(0.0)
        return out

    def _if_features(self, z: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(z[["z_rel", "z_line", "z_imb"]].to_numpy(), nan=0.0).clip(-50, 50)

    def fit(self, df: pd.DataFrame) -> "StationAnomalyDetector":
        """Без меток: робастные центры/масштабы и Isolation Forest по остаткам val-периода."""
        work = df[~df["is_closed"]]
        h = self._decompose(work)
        line_vals = work.assign(_l=h["r_line"]).dropna(subset=["_l"]).drop_duplicates(["line", "t"])["_l"]
        self.stats = {"rel": robust_center_scale(h["r_rel"]),
                      "line": robust_center_scale(line_vals) if len(line_vals) else (0.0, 1.0),
                      "imb": robust_center_scale(work["imb_dev"])}
        z = self._z(work)
        self.iforest = IsolationForest(n_estimators=300, contamination="auto", random_state=self.seed)
        self.iforest.fit(self._if_features(z))
        self.stats["if_threshold"] = float(np.quantile(-self.iforest.score_samples(self._if_features(z)), 0.975))
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        z = self._z(df)
        thr = self.z_threshold
        if_score = -self.iforest.score_samples(self._if_features(z))
        ratio = df["actual"] / df["expected"]
        # Значимо (|z| > порога) И заметно на практике (эффект больше минимального)
        rel_big = z["r_rel"].abs() > np.log1p(self.min_station_effect)
        line_big = z["r_line"].abs() > np.log1p(self.min_line_effect)
        imb_big = df["imb_dev"].abs() > np.log1p(self.min_imbalance_effect)
        conds = [df["is_closed"].to_numpy(),
                 (ratio < self.closure_ratio).to_numpy(),
                 ((z["z_rel"] < -thr) & rel_big).to_numpy(), ((z["z_rel"] > thr) & rel_big).to_numpy(),
                 ((z["z_line"].abs() > thr) & line_big).to_numpy(),
                 ((z["z_imb"].abs() > thr) & imb_big).to_numpy()]
        types = np.select(conds, ["closure", "partial_closure", "station_drop", "station_surge", "line_event",
                                  "imbalance"], "ok")
        out = df.drop(columns=[c for c in z.columns if c in df.columns]).join(z)
        out["if_score"] = if_score
        out["anomaly_type"] = types
        out["is_anomaly"] = types != "ok"
        zmax = np.maximum(z["z_rel"].abs(), z["z_imb"].abs()).to_numpy()
        critical = np.isin(types, ["closure", "partial_closure"]) | ((types != "ok") & (zmax > 2 * thr)) | \
            ((types != "ok") & (if_score > self.stats["if_threshold"]))
        out["severity"] = np.where(types == "ok", "none", np.where(critical, "critical", "warning"))
        out["explanation"] = [_explain(r) for r in out.itertuples()]
        return out

    def city_events(self, detected: pd.DataFrame) -> pd.DataFrame:
        """Кварталы, в которых весь город отклонился от прогноза (одно событие вместо сотен алертов)."""
        g = detected.groupby("period").agg(r_city=("r_city", "first"), n_stations=("key", "count")).reset_index()
        g = g[g["r_city"].abs() > np.log1p(self.min_city_effect)].copy()
        g["city_change"] = np.expm1(g["r_city"])
        return g.drop(columns="r_city")

    # --- сериализация ---
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        cfg = {k: v for k, v in asdict(self).items() if k != "iforest"}
        (d / "station_detector.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        joblib.dump(self.iforest, d / "iforest.joblib")

    @classmethod
    def load(cls, d: Path) -> "StationAnomalyDetector":
        cfg = json.loads((d / "station_detector.json").read_text(encoding="utf-8"))
        cfg["stats"] = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg["stats"].items()}
        return cls(**cfg, iforest=joblib.load(d / "iforest.joblib"))


def _fmt(x: float) -> str:
    return f"{x / 1e6:.2f} млн"


def _explain(r) -> str:
    chg = r.actual / r.expected - 1
    base = f"Вход {_fmt(r.actual)} при ожидаемых {_fmt(r.expected)} ({chg:+.0%})"
    ctx = []
    if np.isfinite(r.r_city):
        ctx.append(f"город {np.expm1(r.r_city):+.0%}")
    if np.isfinite(r.r_line):
        ctx.append(f"линия относительно города {np.expm1(r.r_line):+.0%}")
    line = f" [{', '.join(ctx)}]" if ctx else ""
    t = r.anomaly_type
    if t == "ok":
        return base + line
    if t == "closure":
        return "Нулевой поток — станция закрыта (ремонт/реконструкция)"
    if t == "partial_closure":
        return base + " — вероятно, закрыт вестибюль или станция работала часть квартала"
    if t == "line_event":
        return base + line + " — изменилась вся линия, а не только станция"
    if t == "imbalance":
        now = r.actual / r.outgoing
        usual = now / np.exp(r.imb_dev)
        return (f"Вход/выход = {now:.2f} (вход {_fmt(r.actual)}, выход {_fmt(r.outgoing)}), обычно для станции "
                f"{usual:.2f} — проверить учёт/работу вестибюлей")
    return base + line + (" — станция просела сильнее города и линии" if t == "station_drop" else
                          " — станция выросла сильнее города и линии")


def line_events(detected: pd.DataFrame) -> pd.DataFrame:
    """Агрегат по линиям: одно событие на линию-квартал вместо десятка станционных алертов."""
    ev = detected[detected["anomaly_type"] == "line_event"]
    if ev.empty:
        return pd.DataFrame(columns=["line", "period", "n_stations", "line_change"])
    return (ev.groupby(["line", "period"]).agg(n_stations=("key", "count"), r_line=("r_line", "first"))
            .assign(line_change=lambda x: np.expm1(x["r_line"])).drop(columns="r_line").reset_index())


# --------------------------------------------------------------------------------------
# Оценка детектора инъекцией аномалий в реальные данные
# --------------------------------------------------------------------------------------
def inject(df: pd.DataFrame, base: pd.DataFrame, rng: np.random.Generator, station_rate: float = 0.04,
           imb_rate: float = 0.03, n_line_events: int = 2) -> tuple[pd.DataFrame, pd.Series]:
    """Вставляет аномалии в «нормальные» строки (по мнению детектора до инъекции). Возвращает кадр и метки."""
    df = df.copy()
    label = pd.Series("none", index=df.index)
    normal = base.index[(base["anomaly_type"] == "ok") & ~base["is_closed"]]

    # 1) события линии: весь поток линии в квартале × f
    groups = base.loc[normal].groupby(["line", "t"]).size()
    groups = groups[groups >= 5]
    for (line, t) in groups.sample(min(n_line_events, len(groups)), random_state=int(rng.integers(1e9))).index:
        f = rng.choice([rng.uniform(0.55, 0.75), rng.uniform(1.35, 1.7)])
        idx = df.index[(df["line"] == line) & (df["t"] == t) & ~df["is_closed"]]
        df.loc[idx, ["actual", "outgoing"]] *= f
        df.loc[idx, "r"] += np.log(f)
        label[idx.intersection(normal)] = "line_event"

    # 2) станционные падения/рост (вход и выход вместе — дисбаланс не меняется)
    free = normal[label[normal] == "none"]
    st_idx = rng.choice(free, size=int(station_rate * len(free)), replace=False)
    f = np.where(rng.random(len(st_idx)) < 0.5, rng.uniform(0.4, 0.7, len(st_idx)), rng.uniform(1.5, 2.5, len(st_idx)))
    df.loc[st_idx, ["actual", "outgoing"]] = df.loc[st_idx, ["actual", "outgoing"]].to_numpy() * f[:, None]
    df.loc[st_idx, "r"] += np.log(f)
    label[st_idx] = np.where(f < 1, "station_drop", "station_surge")

    # 3) дисбаланс: меняется только выход
    free = normal[label[normal] == "none"]
    im_idx = rng.choice(free, size=int(imb_rate * len(free)), replace=False)
    g = np.where(rng.random(len(im_idx)) < 0.5, rng.uniform(1.4, 2.0, len(im_idx)), 1 / rng.uniform(1.4, 2.0, len(im_idx)))
    df.loc[im_idx, "outgoing"] *= g
    df.loc[im_idx, "imb_dev"] -= np.log(g)
    label[im_idx] = "imbalance"
    return df, label


def evaluate_by_injection(det: StationAnomalyDetector, df: pd.DataFrame, rounds: int = 10,
                          seed: int = settings.seed) -> dict:
    """Recall по типам, precision и доля ложных тревог на «нормальных» строках. Усреднение по раундам."""
    rng = np.random.default_rng(seed)
    base = det.detect(df)
    normal = (base["anomaly_type"] == "ok") & ~base["is_closed"]
    acc: dict[str, list] = {}
    for _ in range(rounds):
        inj, label = inject(df, base, rng)
        res = det.detect(inj)
        ev = normal                                   # оцениваем только строки, нормальные до инъекции
        pos, flag = (label != "none") & ev, res["is_anomaly"] & ev
        tp = (pos & flag).sum()
        m = {"precision": tp / max(flag.sum(), 1), "recall": tp / max(pos.sum(), 1),
             "false_alarm_rate": (flag & ~pos).sum() / max((ev & ~pos).sum(), 1)}
        m["f1"] = 2 * m["precision"] * m["recall"] / max(m["precision"] + m["recall"], 1e-9)
        for kind in ("station_drop", "station_surge", "line_event", "imbalance"):
            k = label == kind
            if k.any():
                m[f"recall_{kind}"] = res.loc[k, "is_anomaly"].mean()
                m[f"type_acc_{kind}"] = (res.loc[k, "anomaly_type"] == kind).mean()
        for k, v in m.items():
            acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


# --------------------------------------------------------------------------------------
# Городской уровень: месячные ряды по видам транспорта + атрибуция «событие / ошибка данных»
# --------------------------------------------------------------------------------------
KNOWN_CONTEXT = [  # внешний контекст для подписи найденных аномалий (НЕ используется детектором)
    ("2020-03-01", "2021-06-30", "ограничения COVID-19"),
]


@dataclass
class CityMonthlyDetector:
    z_threshold: float = settings.z_threshold
    level_window: int = 3
    season: dict = field(default_factory=dict)      # transport -> 12 сезонных поправок (log)
    scale: dict = field(default_factory=dict)       # transport -> MAD-масштаб остатка

    def _residuals(self, g: pd.DataFrame, season: np.ndarray) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        g["log_y"] = np.log(g["passengers"].where(g["passengers"] > 0))
        g["deseason"] = g["log_y"] - season[g["month"].to_numpy() - 1]
        # Уровень — медиана прошлых месяцев: одна ошибка ввода не «сдвигает» ожидание на следующие месяцы
        level = g["deseason"].shift(1).rolling(self.level_window, min_periods=2).median()
        g["expected"] = np.exp(level + season[g["month"].to_numpy() - 1])
        g["r"] = g["log_y"] - np.log(g["expected"])
        return g

    def fit(self, monthly: pd.DataFrame, fit_end_year: int, skip: tuple[str, ...] = ()) -> "CityMonthlyDetector":
        for tr, g in monthly[monthly["year"] <= fit_end_year].groupby("transport"):
            if tr in skip:
                continue
            g = g[g["passengers"] > 0].copy()
            g["ly"] = np.log(g["passengers"])
            full = g.groupby("year")["month"].transform("count") >= 10
            dev = g.loc[full, "ly"] - g.loc[full].groupby("year")["ly"].transform("median")
            # медиана по годам: локдаун 2020 г. — один год из нескольких и не искажает профиль
            season = dev.groupby(g.loc[full, "month"]).median().reindex(range(1, 13)).fillna(0.0).to_numpy()
            self.season[tr] = season.tolist()
            self.scale[tr] = robust_center_scale(self._residuals(g, season)["r"])[1]
        return self

    def detect(self, monthly: pd.DataFrame, dq_report: dict | None = None) -> pd.DataFrame:
        frames = []
        for tr, g in monthly.groupby("transport"):
            if tr not in self.season:
                continue
            g = self._residuals(g, np.asarray(self.season[tr]))
            g["z"] = g["r"] / self.scale[tr]
            frames.append(g)
        df = pd.concat(frames, ignore_index=True)
        df["is_anomaly"] = df["z"].abs() > self.z_threshold
        df["direction"] = np.where(df["z"] < 0, "drop", "surge")
        df["change"] = df["passengers"] / df["expected"] - 1
        df["attribution"], df["comment"] = zip(*[_attribute(r, dq_report) for r in df.itertuples()])
        cols = ["transport", "date", "passengers", "expected", "change", "z", "is_anomaly", "direction",
                "attribution", "comment"]
        return df[cols]

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "city_detector.json").write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "CityMonthlyDetector":
        return cls(**json.loads((d / "city_detector.json").read_text(encoding="utf-8")))


def _attribute(r, dq: dict | None) -> tuple[str, str]:
    """Событие реального мира или ошибка данных? Сверяемся с независимыми источниками (DQ-отчёт)."""
    if not r.is_anomaly:
        return "none", ""
    if dq:
        q = f"{r.date.year}Q{(r.date.month - 1) // 3 + 1}"
        for i in dq["issues"]:
            d = i["details"]
            if (i["check"] == "yearly_mismatch" and d.get("transport") == r.transport and d.get("year") == r.date.year
                    and np.sign(d["gap"]) == np.sign(r.z)):
                return "data_error", (f"Σ месяцев {r.date.year} г. не сходится с годовым итогом ({d['gap']:+.1%}) — "
                                      "вероятна ошибка ввода, а не реальное изменение спроса")
            if (i["check"] == "monthly_suspect" and d.get("period") == q and r.transport == "Московский метрополитен"
                    and r.z < 0):
                return "data_error", (f"Станционные данные за {q} дают {d['coverage']:.0%} месячного итога — "
                                      "месячное значение занижено ошибочно")
    for lo, hi, text in KNOWN_CONTEXT:
        if pd.Timestamp(lo) <= r.date <= pd.Timestamp(hi):
            return "event", f"Реальное событие (контекст: {text})"
    return "event", "Отклонение подтверждается независимыми источниками / не объясняется ошибкой данных"
