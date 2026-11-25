"""Tests for loading saved experiment runs and for the dashboard's experiment views.

Runs are produced from synthetic prices with the real experiment code, so the artifact schemas
under test are the ones experiment.py actually writes. Everything runs offline in seconds.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import experiment
import run_results as rr

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory) -> Path:
    rng = np.random.default_rng(1)
    n = 1400
    close = 50 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    spread = np.abs(rng.normal(0.01, 0.003, n))
    prices = pd.DataFrame({
        "Open": close, "High": close * (1 + spread), "Low": close * (1 - spread), "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=pd.bdate_range("2015-01-01", periods=n, name="Date"))
    root = tmp_path_factory.mktemp("dash")
    snap = experiment.save_snapshot(prices, "TEST", root / "snap")
    for name, horizon in [("h5", 5), ("h10", 10)]:
        cfg = experiment.Config(ticker="TEST", horizon=horizon, n_windows=3, n_seeds=3, tune=False,
                                n_bootstrap=100)
        experiment.execute_run(cfg, snap, root / "runs" / name, say=lambda *_: None)
    return root / "runs"


@pytest.fixture
def copy_of_runs(runs_dir, tmp_path) -> Path:
    shutil.copytree(runs_dir, tmp_path / "runs")
    return tmp_path / "runs"


# ---------------------------------------------------------------- loading
def test_discover_loads_valid_runs_and_skips_broken_ones(copy_of_runs):
    (copy_of_runs / "empty").mkdir()
    shutil.copytree(copy_of_runs / "h5", copy_of_runs / "bad_summary")
    (copy_of_runs / "bad_summary" / "summary.csv").write_text("not,a,summary\n")
    shutil.copytree(copy_of_runs / "h5", copy_of_runs / "bad_json")
    (copy_of_runs / "bad_json" / "manifest.json").write_text("{broken")
    shutil.copytree(copy_of_runs / "h5", copy_of_runs / "no_config")
    (copy_of_runs / "no_config" / "config.json").unlink()

    runs, problems = rr.discover_runs(copy_of_runs)
    assert sorted(r.name for r in runs) == ["h10", "h5"]
    assert set(problems) == {"empty", "bad_summary", "bad_json", "no_config"}
    assert "config.json" in problems["no_config"]


def test_no_runs_directory_is_not_an_error(tmp_path):
    assert rr.discover_runs(tmp_path / "missing") == ([], {})


def test_missing_optional_files_only_add_notes(copy_of_runs):
    for f in ["metrics.csv", "regime_summary.csv", "report.txt"]:
        (copy_of_runs / "h5" / f).unlink()
    run = rr.load_run(copy_of_runs / "h5")
    assert run.metrics is None and run.regime_summary is None and run.report is None
    assert any("metrics.csv" in n for n in run.notes)
    assert rr.per_fit_metrics(run.metrics).empty and rr.seed_spread(run.metrics, "Accuracy").empty


def test_signature_changes_when_a_run_is_modified(copy_of_runs):
    before = rr.runs_signature(copy_of_runs)
    assert rr.runs_signature(copy_of_runs) == before
    (copy_of_runs / "h5" / "report.txt").write_text("changed")
    assert rr.runs_signature(copy_of_runs) != before


# --------------------------------------------------------------- analysis
def test_per_fit_metrics_matches_metrics_csv(runs_dir):
    run = rr.load_run(runs_dir / "h5")
    fit = rr.per_fit_metrics(run.metrics)
    expected = run.metrics[run.metrics["Model"] == "MLP"].groupby("Window")[["F1", "Recall"]].mean().mean()
    assert fit.loc["MLP", ["F1", "Recall"]].to_numpy() == pytest.approx(expected.to_numpy())
    assert {"Precision", "Recall", "F1", "MCC", "AUC"} <= set(fit.columns)
    assert list(fit.index) == rr.ALL_NAMES


def test_seed_spread_covers_only_stochastic_models(runs_dir):
    run = rr.load_run(runs_dir / "h5")
    spread = rr.seed_spread(run.metrics, "Balanced Accuracy")
    assert set(spread.index.get_level_values("Model")) == set(rr.STOCHASTIC)
    assert (spread["Min"] <= spread["Mean"]).all() and (spread["Mean"] <= spread["Max"]).all()


def test_verdicts_follow_the_confidence_intervals():
    s = pd.DataFrame({"Acc - baseline": [0, 0.1, -0.1, 0.0], "Diff CI low": [0, 0.05, -0.2, -0.05],
                      "Diff CI high": [0, 0.2, -0.05, 0.05], "BA CI low": [0.5, 0.52, 0.4, 0.48],
                      "BA CI high": [0.5, 0.6, 0.48, 0.55]},
                     index=pd.Index(rr.ALL_NAMES, name="Model"))
    v = rr.verdicts(s)
    assert v.loc["Naive Bayes", "Accuracy vs. majority baseline"].startswith("Above")
    assert v.loc["Naive Bayes", "Balanced accuracy vs. 0.5"].startswith("Above")
    assert v.loc["Perceptron", "Accuracy vs. majority baseline"].startswith("Below")
    assert v.loc["Perceptron", "Balanced accuracy vs. 0.5"].startswith("Below")
    assert v.loc["MLP", "Accuracy vs. majority baseline"].startswith("Not distinguishable")
    assert rr.BASELINE_NAME not in v.index


# ------------------------------------------------------------- comparison
def test_same_data_different_horizon_is_comparable_with_a_note(runs_dir):
    runs = [rr.load_run(runs_dir / n) for n in ("h5", "h10")]
    levels = [lvl for lvl, _ in rr.comparability(runs)]
    assert "error" not in levels and "warning" not in levels
    assert list(rr.config_differences(runs).index) == ["horizon"]


def test_different_data_windows_and_procedure_are_flagged(runs_dir):
    a, b = (rr.load_run(runs_dir / n) for n in ("h5", "h10"))

    other_data = rr.load_run(runs_dir / "h10")
    other_data.manifest = json.loads(json.dumps(b.manifest))
    other_data.manifest["data"]["sha256"] = "0" * 64
    assert "warning" in [lvl for lvl, _ in rr.comparability([a, other_data])]

    other_windows = rr.load_run(runs_dir / "h10")
    other_windows.manifest = json.loads(json.dumps(b.manifest))
    other_windows.manifest["windows"][1][1] = "2099-01-01"
    assert "error" in [lvl for lvl, _ in rr.comparability([a, other_windows])]

    other_proc = rr.load_run(runs_dir / "h10")
    other_proc.config = {**b.config, "n_windows": 9}
    msgs = [m for lvl, m in rr.comparability([a, other_proc]) if lvl == "warning"]
    assert any("n_windows" in m for m in msgs)

    other_ticker = rr.load_run(runs_dir / "h10")
    other_ticker.config = {**b.config, "ticker": "AAPL"}
    assert any("tickers" in m for _, m in rr.comparability([a, other_ticker]))


# -------------------------------------------------------------- dashboard
def test_dashboard_renders_runs_without_training(copy_of_runs, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("the dashboard must never run an experiment")

    monkeypatch.setattr(experiment, "execute_run", forbidden)
    monkeypatch.setattr(experiment, "run_windows", forbidden)
    monkeypatch.chdir(copy_of_runs.parent)  # no artifacts/ here, so the Experiments view is the default

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Summary", "Model performance", "Walk-forward windows",
                                          "Volatility regimes", "Compare experiments"]

    at.sidebar.selectbox[0].set_value("h10").run()  # switching experiments only reloads saved files
    assert not at.exception, [e.value for e in at.exception]
    keys = [r.key for r in at.radio if r.label != "Results from"]
    for key in keys:
        for option in at.radio(key=key).options:
            at.radio(key=key).set_value(option).run()
            assert not at.exception, (key, option, [e.value for e in at.exception])
    assert any("Horizons differ" in i.value for i in at.info)


def test_dashboard_with_no_runs_shows_instructions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.sidebar.radio[0].value.startswith("Experiments")
    assert any("No completed experiments" in w.value for w in at.warning)
