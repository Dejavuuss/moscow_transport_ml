"""Тесты признаков: главное — отсутствие утечки из будущего и корректный якорь прогноза."""
import numpy as np

from src import features as F


def _panel(clean_metro_df):
    p = F.frame_to_panel(clean_metro_df)
    return p, sorted(set(p.lines))


def test_features_use_only_past(clean_metro_df):
    p, vocab = _panel(clean_metro_df)
    s, t = np.arange(len(p.keys)), np.full(len(p.keys), 9)
    before = F.build_features(p, s, t, vocab)
    p.incoming[:, 9:] *= 100                      # «портим» целевой квартал и всё будущее
    after = F.build_features(p, s, t, vocab)
    np.testing.assert_array_equal(np.nan_to_num(before.X), np.nan_to_num(after.X))
    np.testing.assert_array_equal(before.anchor, after.anchor)


def test_anchor_prefers_same_quarter_last_year(clean_metro_df):
    p, vocab = _panel(clean_metro_df)
    y = p.y
    f = F.build_features(p, np.array([3]), np.array([8]), vocab)
    assert f.anchor[0] == y[3, 4]                 # лаг 4
    p.incoming[3, 4] = np.nan                     # нет «год назад» -> берём прошлый квартал
    f = F.build_features(p, np.array([3]), np.array([8]), vocab)
    assert f.anchor[0] == p.y[3, 7]


def test_closed_quarters_are_not_targets(clean_metro_df):
    p, vocab = _panel(clean_metro_df)
    f, d = F.training_samples(p, vocab)
    closed = set(zip(*np.nonzero(p.closed)))
    assert closed and not closed & set(zip(f.s, f.t))
    assert np.isfinite(d).all()


def test_split_is_by_time(clean_metro_df):
    p, vocab = _panel(clean_metro_df)
    f, _ = F.training_samples(p, vocab)
    m = F.split_masks(p, f.t)
    assert not (m["train"] & m["val"]).any()
    if m["val"].any():
        assert f.t[m["train"]].max() < f.t[m["val"]].min()


def test_seasonal_naive_is_lag4_and_metrics(clean_metro_df):
    p, vocab = _panel(clean_metro_df)
    f, _ = F.training_samples(p, vocab)
    has4 = np.isfinite(f.lags[:, 3])
    np.testing.assert_allclose(F.baseline_log(f, "seasonal_naive")[has4], f.lags[has4, 3])
    m = F.forecast_metrics(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
    assert np.isclose(m["wape"], 20 / 300) and np.isclose(m["mae"], 10)
