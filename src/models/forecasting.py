"""Модели прогноза пассажиропотока станции на следующий квартал.

Обе модели предсказывают delta = log(факт) − log(якорь), см. src.features. Прогноз = якорь · e^delta.

1. **GBMForecaster** — градиентный бустинг (sklearn HistGradientBoosting): нативно работает с NaN
   (пропуски лагов из-за неполных кварталов), категориальный признак «линия». Сильный выбор для
   табличных данных такого размера (~5 тыс. строк).
2. **TorchForecaster** — MLP на PyTorch с embedding'ами станции и линии. Embedding станции — это
   «индивидуальность» ряда (например, станция-ТПУ растёт быстрее); новая станция получает
   OOV-вектор. Huber-loss: в данных есть закрытия/перераспределения потока, MSE дал бы им
   квадратичный вес.

Общий интерфейс: fit(...) -> dict, predict_delta(features) -> np.ndarray, save(dir), load(dir).
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

from ..features import FEATURE_NAMES, Features

log = logging.getLogger(__name__)


class GBMForecaster:
    kind = "gbm"

    def __init__(self, n_lines: int, max_iter: int = 600, learning_rate: float = 0.03, max_leaf_nodes: int = 15,
                 min_samples_leaf: int = 30, l2_regularization: float = 1.0, seed: int = 42) -> None:
        self.hparams = dict(n_lines=n_lines, max_iter=max_iter, learning_rate=learning_rate,
                            max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                            l2_regularization=l2_regularization, seed=seed)
        self.model = None

    def _X(self, f: Features) -> np.ndarray:
        return np.column_stack([f.X, f.line_idx.astype(np.float32)])

    def _make(self, max_iter: int, early: bool):
        from sklearn.ensemble import HistGradientBoostingRegressor

        h = self.hparams
        return HistGradientBoostingRegressor(
            loss="absolute_error",                  # медиана изменения — устойчиво к выбросам
            max_iter=max_iter, learning_rate=h["learning_rate"], max_leaf_nodes=h["max_leaf_nodes"],
            min_samples_leaf=h["min_samples_leaf"], l2_regularization=h["l2_regularization"],
            categorical_features=[len(FEATURE_NAMES)], early_stopping=early, validation_fraction=None,
            n_iter_no_change=50, random_state=h["seed"])

    def fit(self, f_tr: Features, d_tr: np.ndarray, f_va: Features | None = None, d_va: np.ndarray | None = None,
            n_iter: int | None = None, on_epoch: Callable | None = None) -> dict:
        """С валидацией — подбираем число деревьев по val; без неё (refit) — обучаем ровно n_iter деревьев."""
        if f_va is None:
            self.model = self._make(n_iter or self.hparams["max_iter"], early=False).fit(self._X(f_tr), d_tr)
            return {"n_iter": self.model.n_iter_}
        # Ручной подбор числа деревьев по val (sklearn early stopping берёт случайный кусок train,
        # а для временных рядов валидация должна быть «из будущего»)
        self.model = self._make(self.hparams["max_iter"], early=False).fit(self._X(f_tr), d_tr)
        val_curve = [float(np.mean(np.abs(p - d_va))) for p in self.model.staged_predict(self._X(f_va))]
        best = int(np.argmin(val_curve)) + 1
        if on_epoch:
            for i in range(0, len(val_curve), 10):
                on_epoch(i + 1, None, val_curve[i])
        self.model = self._make(best, early=False).fit(self._X(f_tr), d_tr)
        return {"n_iter": best, "best_val_mae_log": val_curve[best - 1]}

    def predict_delta(self, f: Features) -> np.ndarray:
        return self.model.predict(self._X(f))

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, d / "gbm.joblib")
        (d / "model.json").write_text(json.dumps({"kind": self.kind, "hparams": self.hparams}), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "GBMForecaster":
        m = cls(**json.loads((d / "model.json").read_text(encoding="utf-8"))["hparams"])
        m.model = joblib.load(d / "gbm.joblib")
        return m


# --------------------------------------------------------------------------------------
# PyTorch
# --------------------------------------------------------------------------------------
try:                                   # torch нужен только нейросети; GBM работает и без него
    import torch
    from torch import nn
except ImportError:                    # pragma: no cover
    torch = nn = None


if nn is not None:
    class FlowNet(nn.Module):
        """[числовые признаки, emb(станции), emb(линии)] -> MLP -> delta."""

        def __init__(self, n_features: int, n_stations: int, n_lines: int, hidden: int, emb_dim: int,
                     dropout: float) -> None:
            super().__init__()
            self.n_features = n_features
            self.st = nn.Embedding(n_stations + 1, emb_dim)     # +1: OOV для новых станций
            self.ln = nn.Embedding(n_lines + 1, emb_dim // 2)
            self.mlp = nn.Sequential(
                nn.Linear(n_features + emb_dim + emb_dim // 2, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x, s, l):
            return self.mlp(torch.cat([x, self.st(s), self.ln(l)], dim=1)).squeeze(1)

    class FlowNetServing(nn.Module):
        """Обёртка для MLflow/сервинга: ОДИН тензор [признаки..., индекс станции, индекс линии]."""

        def __init__(self, net: FlowNet) -> None:
            super().__init__()
            self.net = net

        def forward(self, inp):
            k = self.net.n_features
            return self.net(inp[:, :k], inp[:, k].long(), inp[:, k + 1].long())


def _net(n_features: int, n_stations: int, n_lines: int, hidden: int, emb_dim: int, dropout: float):
    return FlowNet(n_features, n_stations, n_lines, hidden, emb_dim, dropout)


class TorchForecaster:
    kind = "torch_mlp"

    def __init__(self, n_stations: int, n_lines: int, hidden: int = 64, emb_dim: int = 8, dropout: float = 0.2,
                 lr: float = 3e-3, weight_decay: float = 1e-3, epochs: int = 400, batch_size: int = 256,
                 patience: int = 40, seed: int = 42) -> None:
        self.hparams = dict(n_stations=n_stations, n_lines=n_lines, hidden=hidden, emb_dim=emb_dim, dropout=dropout,
                            lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size,
                            patience=patience, seed=seed)
        self.net = None
        self.x_mean = self.x_std = None

    def _tensors(self, f: Features):
        x = np.nan_to_num((f.X - self.x_mean) / self.x_std, nan=0.0)  # NaN -> 0 (маски has_lag* в признаках)
        s = np.where(f.s < self.hparams["n_stations"], f.s, self.hparams["n_stations"])
        return (torch.from_numpy(x.astype(np.float32)), torch.from_numpy(s.astype(np.int64)),
                torch.from_numpy(f.line_idx.astype(np.int64)))

    def fit(self, f_tr: Features, d_tr: np.ndarray, f_va: Features | None = None, d_va: np.ndarray | None = None,
            n_iter: int | None = None, on_epoch: Callable | None = None) -> dict:
        h = self.hparams
        torch.manual_seed(h["seed"])
        rng = np.random.default_rng(h["seed"])
        # Статистики стандартизации — ТОЛЬКО из train
        self.x_mean = np.nanmean(f_tr.X, axis=0)
        self.x_std = np.nanstd(f_tr.X, axis=0) + 1e-6
        self.x_mean, self.x_std = np.nan_to_num(self.x_mean), np.nan_to_num(self.x_std, nan=1.0)
        self.net = _net(f_tr.X.shape[1], h["n_stations"], h["n_lines"], h["hidden"], h["emb_dim"], h["dropout"])
        opt = torch.optim.AdamW(self.net.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])
        loss_fn = nn.HuberLoss(delta=0.1)          # 0.1 в log ≈ 10% — дальше ошибка считается линейно
        xt, st, lt = self._tensors(f_tr)
        yt = torch.from_numpy(d_tr.astype(np.float32))
        if f_va is not None:
            xv, sv, lv = self._tensors(f_va)
            yv = torch.from_numpy(d_va.astype(np.float32))
        epochs = n_iter or h["epochs"]
        best, best_state, best_epoch, bad = float("inf"), None, epochs, 0
        for epoch in range(1, epochs + 1):
            self.net.train()
            total = 0.0
            for idx in np.array_split(rng.permutation(len(yt)), max(1, len(yt) // h["batch_size"])):
                idx = torch.from_numpy(idx)
                loss = loss_fn(self.net(xt[idx], st[idx], lt[idx]), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if f_va is None:
                continue
            self.net.eval()
            with torch.no_grad():
                val = float(torch.mean(torch.abs(self.net(xv, sv, lv) - yv)))
            if on_epoch and epoch % 5 == 0:
                on_epoch(epoch, total / len(yt), val)
            if val < best - 1e-5:
                best, best_state, best_epoch, bad = val, copy.deepcopy(self.net.state_dict()), epoch, 0
            else:
                bad += 1
                if bad >= h["patience"]:
                    log.info("Early stopping на эпохе %d (лучшая %d)", epoch, best_epoch)
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return {"n_iter": best_epoch, "best_val_mae_log": best if f_va is not None else None}

    def serving_input(self, f: Features) -> np.ndarray:
        """Вход для FlowNetServing (то, что принимает зарегистрированная в MLflow модель)."""
        x, s, l = self._tensors(f)
        return torch.cat([x, s[:, None].float(), l[:, None].float()], dim=1).numpy()

    def predict_delta(self, f: Features) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return self.net(*self._tensors(f)).numpy()

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), d / "net.pt")
        (d / "model.json").write_text(json.dumps({"kind": self.kind, "hparams": self.hparams,
                                                  "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "TorchForecaster":
        meta = json.loads((d / "model.json").read_text(encoding="utf-8"))
        m = cls(**meta["hparams"])
        m.x_mean, m.x_std = np.asarray(meta["x_mean"], np.float32), np.asarray(meta["x_std"], np.float32)
        h = m.hparams
        m.net = _net(len(m.x_mean), h["n_stations"], h["n_lines"], h["hidden"], h["emb_dim"], h["dropout"])
        m.net.load_state_dict(torch.load(d / "net.pt", map_location="cpu", weights_only=True))
        m.net.eval()
        return m


MODELS = {GBMForecaster.kind: GBMForecaster, TorchForecaster.kind: TorchForecaster}


def load_model(d: Path):
    return MODELS[json.loads((d / "model.json").read_text(encoding="utf-8"))["kind"]].load(d)
