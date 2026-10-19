"""Панель «станция × квартал» и признаки для прогноза на следующий квартал.

Особенности реальных данных, под которые сделаны признаки:
  * **Короткие ряды** (≤ 22 квартала) и **дыры** (неполные публикации кварталов) — поэтому одна
    ГЛОБАЛЬНАЯ модель на все станции, а не 300 отдельных; лаги допускают пропуски (NaN + маска).
  * **Сезонность** квартала (лето ниже, IV квартал выше) — «якорь» прогноза по возможности берётся
    из того же квартала год назад (лаг 4); если его нет — из прошлого квартала (лаг 1) и т.д.
  * Модель предсказывает **отклонение от якоря в log-шкале** (≈ процентное изменение), а не сам
    поток: станции отличаются в 100 раз, а относительные изменения сопоставимы.
  * **Контекст линии и города**: медианный годовой рост по линии/городу за прошлый квартал
    (например, открытие БКЛ перераспределило поток на целых линиях).

Всё строится ТОЛЬКО из данных до целевого квартала: признаки для квартала t используют
кварталы ≤ t-1 (проверяется тестом test_features_use_only_past).
Одни и те же функции используются в обучении и в API -> нет train/serving skew.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import settings
from .data_loader import parse_period, period_label
from .quality import MCC_LINE

N_LAGS = settings.n_lags
ANCHOR_ORDER = (4, 1, 2, 3, 5)      # предпочтение якоря: тот же квартал год назад, затем прошлый


@dataclass
class Panel:
    keys: list[str]           # S ключей «станция | линия»
    stations: list[str]
    lines: list[str]          # линия каждой станции
    qi0: int                  # сквозной номер первого квартала панели
    incoming: np.ndarray      # (S, Q) входы, NaN = нет данных
    outgoing: np.ndarray      # (S, Q)
    closed: np.ndarray        # (S, Q) bool — нулевой поток (закрытие)

    @property
    def Q(self) -> int:
        return self.incoming.shape[1]

    @property
    def y(self) -> np.ndarray:
        """log(вход) для «рабочих» кварталов; закрытые и отсутствующие -> NaN."""
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log(self.incoming)
        v[self.closed | ~np.isfinite(v)] = np.nan
        return v

    def period(self, t: int) -> str:
        return period_label(self.qi0 + t)

    def t_of(self, period: str) -> int:
        return parse_period(period) - self.qi0

    def index(self, key: str) -> int:
        try:
            return self.keys.index(key)
        except ValueError:
            raise KeyError(f"Неизвестная станция: {key}") from None

    def to_frame(self) -> pd.DataFrame:
        s, t = np.nonzero(np.isfinite(self.incoming))
        return pd.DataFrame({"key": [self.keys[i] for i in s], "station": [self.stations[i] for i in s],
                             "line": [self.lines[i] for i in s], "qi": self.qi0 + t,
                             "incoming": self.incoming[s, t], "outgoing": self.outgoing[s, t],
                             "is_closed": self.closed[s, t].astype(int)})


def frame_to_panel(df: pd.DataFrame) -> Panel:
    keys = sorted(df["key"].unique())
    first = df.drop_duplicates("key").set_index("key")
    qi0, qi1 = int(df["qi"].min()), int(df["qi"].max())
    S, Q = len(keys), qi1 - qi0 + 1
    pos = {k: i for i, k in enumerate(keys)}
    s, t = df["key"].map(pos).to_numpy(), (df["qi"] - qi0).to_numpy()
    inc, out = np.full((S, Q), np.nan), np.full((S, Q), np.nan)
    closed = np.zeros((S, Q), dtype=bool)
    inc[s, t], out[s, t] = df["incoming"].to_numpy(float), df["outgoing"].to_numpy(float)
    closed[s, t] = df["is_closed"].to_numpy(bool)
    return Panel(keys, first.loc[keys, "station"].tolist(), first.loc[keys, "line"].tolist(), qi0, inc, out, closed)


def _lag(y: np.ndarray, s: np.ndarray, t: np.ndarray, k: int) -> np.ndarray:
    tt = t - k
    ok = (tt >= 0) & (tt < y.shape[1])
    out = np.full(len(s), np.nan)
    out[ok] = y[s[ok], tt[ok]]
    return out


def _group_trend(y: np.ndarray, groups: np.ndarray, t: np.ndarray, s: np.ndarray, min_n: int) -> np.ndarray:
    """Медиана годового роста log y[:, t-1] − y[:, t-5] внутри группы станции (линия или весь город)."""
    out = np.full(len(s), np.nan)
    for tv in np.unique(t):
        if tv - 5 < 0 or tv - 1 >= y.shape[1]:
            continue
        g = y[:, tv - 1] - y[:, tv - 5]
        sel = np.nonzero(t == tv)[0]
        for gv in np.unique(groups[s[sel]]):
            vals = g[(groups == gv) & np.isfinite(g)]
            if len(vals) >= min_n:
                out[sel[groups[s[sel]] == gv]] = np.median(vals)
    return out


FEATURE_NAMES = ([f"rel_lag{k}" for k in range(1, N_LAGS + 1)] + ["anchor", "yoy", "line_trend", "city_trend"]
                 + [f"q{i}" for i in range(1, 5)] + ["age", "is_mcc"]
                 + [f"anchor_is_lag{k}" for k in ANCHOR_ORDER] + [f"has_lag{k}" for k in range(1, N_LAGS + 1)])


@dataclass
class Features:
    s: np.ndarray          # индекс станции
    t: np.ndarray          # индекс целевого квартала
    X: np.ndarray          # (N, F) числовые признаки, NaN = нет значения
    line_idx: np.ndarray   # (N,) индекс линии (для embedding / категориального признака)
    anchor: np.ndarray     # (N,) log-якорь; прогноз = anchor + delta
    lags: np.ndarray       # (N, N_LAGS) log-лаги (для бейзлайнов)

    def subset(self, m: np.ndarray) -> "Features":
        return Features(self.s[m], self.t[m], self.X[m], self.line_idx[m], self.anchor[m], self.lags[m])


def build_features(panel: Panel, s: np.ndarray, t: np.ndarray, line_vocab: list[str]) -> Features:
    """Признаки для пар (станция s, целевой квартал t). t может быть = Q (прогноз «следующего» квартала)."""
    s, t = np.asarray(s, dtype=int), np.asarray(t, dtype=int)
    y = panel.y
    lags = np.stack([_lag(y, s, t, k) for k in range(1, N_LAGS + 1)], axis=1)
    anchor = np.full(len(s), np.nan)
    anchor_k = np.zeros(len(s), dtype=int)
    for k in ANCHOR_ORDER[::-1]:                       # последний записанный = самый приоритетный
        v = lags[:, k - 1]
        anchor = np.where(np.isfinite(v), v, anchor)
        anchor_k = np.where(np.isfinite(v), k, anchor_k)

    line_pos = {l: i for i, l in enumerate(line_vocab)}
    station_line = np.array([line_pos.get(l, len(line_vocab)) for l in panel.lines])  # неизвестная линия -> OOV
    rel = lags - anchor[:, None]
    q_of_year = (panel.qi0 + t) % 4
    seen = np.isfinite(y)
    age = np.array([seen[si, :min(ti, panel.Q)].sum() for si, ti in zip(s, t)])
    X = np.column_stack([
        rel, anchor - 14.0,                                    # log(1.2 млн) ≈ 14 — центрирование
        lags[:, 0] - lags[:, 4],                               # годовой тренд станции
        _group_trend(y, station_line, t, s, settings.min_line_stations),
        _group_trend(y, np.zeros(len(panel.keys), dtype=int), t, s, 10),
        np.eye(4)[q_of_year], np.log1p(age),
        np.array([panel.lines[i] == MCC_LINE for i in s], dtype=float),
        np.column_stack([anchor_k == k for k in ANCHOR_ORDER]).astype(float),
        np.isfinite(lags).astype(float),
    ])
    return Features(s, t, X.astype(np.float32), station_line[s], anchor, lags)


def training_samples(panel: Panel, line_vocab: list[str]) -> tuple[Features, np.ndarray]:
    """Все пары (s, t), у которых есть и цель (рабочий квартал), и якорь. Возвращает признаки и цель delta."""
    y = panel.y
    s, t = np.nonzero(np.isfinite(y))
    f = build_features(panel, s, t, line_vocab)
    ok = np.isfinite(f.anchor)
    f = f.subset(ok)
    return f, (y[f.s, f.t] - f.anchor).astype(np.float32)


def split_masks(panel: Panel, t: np.ndarray) -> dict[str, np.ndarray]:
    """Разбиение по ВРЕМЕНИ целевого квартала."""
    qi = panel.qi0 + t
    tr, va = parse_period(settings.train_end), parse_period(settings.val_end)
    return {"train": qi <= tr, "val": (qi > tr) & (qi <= va), "test": qi > va}


# --------------------------------------------------------------------------------------
# Бейзлайны и метрики
# --------------------------------------------------------------------------------------
def _first_finite(*cols: np.ndarray) -> np.ndarray:
    out = np.full(len(cols[0]), np.nan)
    for c in cols[::-1]:
        out = np.where(np.isfinite(c), c, out)
    return out


def baseline_log(f: Features, name: str) -> np.ndarray:
    """Прогноз в log-шкале для простых бейзлайнов (без обучения)."""
    L = [f.lags[:, k] for k in range(N_LAGS)]
    if name == "naive_last":                  # «как в прошлом квартале»
        return _first_finite(L[0], L[1], L[2], L[3], L[4])
    if name == "seasonal_naive":              # «как год назад в том же квартале»
        return f.anchor
    if name == "seasonal_trend":              # год назад × годовой тренд станции
        trend = L[0] - L[4]
        return np.where(np.isfinite(L[3]) & np.isfinite(trend), L[3] + trend, f.anchor)
    raise ValueError(name)


BASELINES = ("naive_last", "seasonal_naive", "seasonal_trend")


def forecast_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    """Метрики в пассажирах. WAPE — главная: устойчива к масштабу станций и понятна бизнесу."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    m = np.isfinite(a) & np.isfinite(p) & (a > 0)
    a, p = a[m], p[m]
    e = p - a
    return {"wape": float(np.abs(e).sum() / a.sum()), "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())), "mape": float(np.mean(np.abs(e) / a)),
            "bias": float(e.sum() / a.sum()), "n": int(m.sum())}
