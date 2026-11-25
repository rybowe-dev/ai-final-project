"""Tests for the leakage safeguards and the saved/repeatable experiment framework.

All tests use synthetic prices, so they run offline and in seconds.
"""

import json

import numpy as np
import pandas as pd
import pytest

import experiment
import features


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 1400
    close = 50 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    spread = np.abs(rng.normal(0.01, 0.003, n))
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.003, n)), "High": close * (1 + spread),
        "Low": close * (1 - spread), "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=pd.bdate_range("2015-01-01", periods=n, name="Date"))


# ------------------------------------------------------ leakage safeguards
def test_features_are_causal(prices):
    full = features.make_features(prices)
    part = features.make_features(prices.iloc[:900])
    pd.testing.assert_frame_equal(full.loc[part.index], part)


def test_labels_use_future_close_and_last_rows_unlabeled(prices):
    y, fut = features.make_labels(prices["Close"], 5)
    assert y.iloc[-5:].isna().all() and y.iloc[:-5].notna().all()
    i = 100
    assert fut.iloc[i] == pytest.approx(prices["Close"].iloc[i + 5] / prices["Close"].iloc[i] - 1)
    assert y.iloc[i] == float(fut.iloc[i] > 0)


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_windows_have_no_label_overlap_and_are_disjoint(prices, horizon):
    ds = features.build_dataset(prices, horizon)
    wins = experiment.make_windows(len(ds.X) + horizon, len(ds.X), 4, 0.5, horizon)
    for train_end, start, end in wins:
        # the last training label looks `horizon` rows ahead and must stay before the block
        assert train_end - 1 + horizon < start
        assert end <= len(ds.X)
    assert all(a[2] <= b[1] for a, b in zip(wins, wins[1:]))


def test_window_dates_identical_across_horizons(prices):
    starts = {}
    for h in (1, 5, 10):
        ds = features.build_dataset(prices, h)
        wins = experiment.make_windows(len(ds.X) + h, len(ds.X), 4, 0.5, h)
        starts[h] = [ds.X.index[s] for _, s, _ in wins]
    assert starts[1] == starts[5] == starts[10]


def test_drop_incomplete_bar(prices):
    day = prices.index[-1]
    before_close = pd.Timestamp(day.date().isoformat() + " 11:00", tz="America/New_York")
    after_close = pd.Timestamp(day.date().isoformat() + " 17:00", tz="America/New_York")
    assert len(features.drop_incomplete_bar(prices, before_close)) == len(prices) - 1
    assert len(features.drop_incomplete_bar(prices, after_close)) == len(prices)
    next_day = after_close + pd.Timedelta(days=1)
    assert len(features.drop_incomplete_bar(prices, next_day)) == len(prices)


# ----------------------------------------------------- experiment framework
@pytest.fixture(scope="module")
def saved_run(prices, tmp_path_factory):
    root = tmp_path_factory.mktemp("exp")
    cfg = experiment.Config(ticker="TEST", n_windows=3, n_seeds=2, tune=False, n_bootstrap=100)
    snap = experiment.save_snapshot(prices, "TEST", root / "snap")
    experiment.execute_run(cfg, snap, root / "runs" / "a", say=lambda *_: None)
    return root, cfg, snap


def test_run_saves_expected_outputs(saved_run):
    root, cfg, _ = saved_run
    run = root / "runs" / "a"
    for f in ["config.json", "manifest.json", "metrics.csv", "predictions.csv.gz", "grid_results.csv",
              "best_params.csv", "summary.csv", "window_summary.csv", "regime_summary.csv", "report.txt"]:
        assert (run / f).exists(), f
    metrics = pd.read_csv(run / "metrics.csv")
    # baseline and Naive Bayes are deterministic (1 fit/window); Perceptron and MLP use every seed
    counts = metrics.groupby("Model").size()
    assert counts["Naive Bayes"] == cfg.n_windows and counts["MLP"] == cfg.n_windows * cfg.n_seeds
    assert json.loads((run / "config.json").read_text())["n_windows"] == 3


def test_repeat_reproduces_and_detects_changes(saved_run):
    root, cfg, snap = saved_run
    experiment.execute_run(cfg, snap, root / "runs" / "b", say=lambda *_: None)
    assert experiment.verify_repeat(root / "runs" / "a", root / "runs" / "b") == []

    tampered = pd.read_csv(root / "runs" / "b" / "metrics.csv")
    tampered.loc[0, "Accuracy"] += 0.01
    tampered.to_csv(root / "runs" / "b" / "metrics.csv", index=False)
    assert any("Accuracy" in p for p in experiment.verify_repeat(root / "runs" / "a", root / "runs" / "b"))


def test_snapshot_hash_is_verified(saved_run):
    _, _, snap = saved_run
    experiment.load_snapshot(snap, experiment._sha256(snap.read_bytes()))
    with pytest.raises(ValueError):
        experiment.load_snapshot(snap, "0" * 64)


def test_compare_flags_config_and_summarises(saved_run):
    root, cfg, snap = saved_run
    other = experiment.Config(ticker="TEST", horizon=10, n_windows=3, n_seeds=2, tune=False, n_bootstrap=100)
    experiment.execute_run(other, snap, root / "runs" / "c", say=lambda *_: None)
    text, table = experiment.compare_runs(["a", "c"], root / "runs")
    assert "horizon=5 vs 10" in text
    assert "different test windows" not in text  # identical boundaries across horizons for the same data
    assert set(table.index.get_level_values("Model")) == set(experiment.ALL_NAMES)


def test_existing_run_name_fails_before_any_work(tmp_path, monkeypatch):
    (tmp_path / "runs" / "taken").mkdir(parents=True)
    monkeypatch.setattr("sys.argv", ["experiment.py", "--runs-dir", str(tmp_path / "runs"), "run", "--name", "taken"])
    monkeypatch.setattr(features, "load_prices", lambda *a, **k: pytest.fail("data must not be downloaded"))
    with pytest.raises(SystemExit, match="already exists"):
        experiment.main()


def test_baseline_has_zero_difference_and_balanced_accuracy_half(saved_run):
    root, _, _ = saved_run
    s = pd.read_csv(root / "runs" / "a" / "summary.csv", index_col="Model")
    base = s.loc[experiment.BASELINE_NAME]
    assert base["Acc - baseline"] == 0 and base["Balanced Accuracy"] == pytest.approx(0.5)
