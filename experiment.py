"""
experiment.py - saved, repeatable experiment runs for comparing the three models.

One *run* evaluates Gaussian Naive Bayes, a Perceptron and an MLP (plus the
majority-class baseline) over several chronological walk-forward windows and
saves everything needed to inspect, repeat and compare it.

Commands
--------
    python experiment.py run --name nvda_h5 --horizon 5          # download data, run, save
    python experiment.py run --name nvda_h10 --horizon 10 --snapshot data/snapshots/<file>.csv
    python experiment.py repeat nvda_h5                          # re-run from the saved config
                                                                 # + data and verify identical metrics
    python experiment.py compare nvda_h5 nvda_h10                # side-by-side summary
    python experiment.py list

Design
------
* Data: the downloaded prices are written to data/snapshots/ (named by SHA-256) and the
  run always reads them back from that file. Pass the same --snapshot to runs you
  want to compare so they see identical data. Today's incomplete bar is dropped.
* Windows: the first `initial_train_fraction` of the history is only ever training data.
  The rest is cut into `n_windows` consecutive test blocks. Each window trains on
  everything before it (expanding), minus a `horizon`-row gap so no training label
  reaches into the block. Window boundaries depend only on the number of feature
  rows, so every horizon is tested on the same dates.
* Tuning: GridSearchCV (walk-forward CV, balanced accuracy) runs inside each window
  on that window's training data only. Perceptron and MLP are then refit with
  `n_seeds` different random seeds at the chosen hyperparameters to expose
  seed-to-seed variance.
* Statistics: pooled out-of-sample predictions get moving-block-bootstrap 95% CIs
  (blocks keep the autocorrelation that overlapping labels create).
* Output (runs/<run_id>/): config.json, manifest.json (data hash, versions, git
  commit, windows), metrics.csv, predictions.csv.gz, grid_results.csv, best_params.csv,
  window_summary.csv, regime_summary.csv, summary.csv, report.txt.
"""

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

import features
import predictor
from evaluation import compute_metrics
from predictor import BASELINE_NAME, MODEL_NAMES

RUNS_DIR = Path("runs")
SNAPSHOT_DIR = Path("data/snapshots")
ALL_NAMES = [BASELINE_NAME] + MODEL_NAMES
STOCHASTIC = {"Perceptron", "MLP"}  # models whose result depends on the random seed
CI_LEVEL = 95


# ---------------------------------------------------------------- config
@dataclass
class Config:
    ticker: str = "NVDA"
    start: str = "2010-01-01"
    end: str | None = None            # optional last date (inclusive) used from the snapshot
    horizon: int = 5
    n_windows: int = 5
    initial_train_fraction: float = 0.5
    n_seeds: int = 10
    cv_folds: int = 4
    tune: bool = True
    n_bootstrap: int = 2000
    bootstrap_seed: int = 0

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]


# ------------------------------------------------------------------ data
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_snapshot(prices: pd.DataFrame, ticker: str, directory: Path = SNAPSHOT_DIR) -> Path:
    """Write prices to a content-addressed CSV and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    data = prices.to_csv().encode()
    path = directory / f"{ticker}_{prices.index[-1]:%Y%m%d}_{_sha256(data)[:8]}.csv"
    path.write_bytes(data)
    return path


def load_snapshot(path: Path, expected_sha: str | None = None) -> pd.DataFrame:
    data = Path(path).read_bytes()
    if expected_sha and _sha256(data) != expected_sha:
        raise ValueError(f"Snapshot {path} does not match its recorded SHA-256; refusing to use it.")
    return pd.read_csv(Path(path), index_col=0, parse_dates=True)


def prepare_prices(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return prices.loc[:cfg.end] if cfg.end else prices


# --------------------------------------------------------------- windows
def make_windows(n_valid: int, n_labeled: int, n_windows: int, initial_fraction: float,
                 horizon: int) -> list[tuple[int, int, int]]:
    """Return (train_end, test_start, test_end) row positions into the labeled data.

    `n_valid` is the number of rows with complete features. Boundaries depend only on it,
    so all horizons share the same test dates; the last window is cut short by the rows
    whose label is not yet known. Training stops `horizon` rows before the block starts.
    """
    first = int(n_valid * initial_fraction)
    edges = np.linspace(first, n_valid, n_windows + 1).astype(int)
    windows = []
    for k in range(n_windows):
        start, end = int(edges[k]), min(int(edges[k + 1]), n_labeled)
        train_end = start - horizon
        if end - start < 20 or train_end < 100:
            raise ValueError("Too little data for this many windows; lower --windows or the horizon.")
        windows.append((train_end, start, end))
    return windows


# ---------------------------------------------------------------- running
def _scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _mcc(y, p) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(matthews_corrcoef(y, p))


def _fit_info(model) -> tuple[float, float]:
    clf = model.named_steps["clf"]
    n_iter = getattr(clf, "n_iter_", np.nan)
    max_iter = getattr(clf, "max_iter", np.nan)
    return float(n_iter), float(n_iter < max_iter) if not np.isnan(n_iter) else np.nan


def _fit_one(w: int, name: str, params: dict, seed: int, X_tr, y_tr, X_te, y_te, regime):
    """Fit one model with one seed on one window; return its metrics row and predictions."""
    fit_params = {**params, "clf__random_state": seed} if name in STOCHASTIC else params
    model = predictor.fit_model(name, X_tr, y_tr, fit_params or None)
    pred, score = model.predict(X_te), _scores(model, X_te)
    n_iter, converged = _fit_info(model)
    metric_row = {
        "Window": w, "Model": name, "Seed": seed,
        "Train rows": len(X_tr), "Train end": X_tr.index[-1], "Test start": X_te.index[0],
        "Test end": X_te.index[-1], "Test rows": len(X_te), "Test up share": float(y_te.mean()),
        **compute_metrics(y_te, pred), "MCC": _mcc(y_te, pred),
        "AUC": float(roc_auc_score(y_te, score)) if y_te.nunique() == 2 else np.nan,
        "Predicted up share": float(pred.mean()), "n_iter": n_iter, "Converged": converged}
    pred_frame = pd.DataFrame({"Window": w, "Date": X_te.index, "Model": name, "Seed": seed,
                               "y": y_te.to_numpy(), "pred": pred, "score": score, "regime": regime})
    return metric_row, pred_frame


def run_windows(cfg: Config, prices: pd.DataFrame, n_jobs: int = 1, say=print) -> dict:
    ds = features.build_dataset(prices, cfg.horizon)
    X, y = ds.X, ds.y
    windows = make_windows(len(X) + cfg.horizon, len(X), cfg.n_windows,
                           cfg.initial_train_fraction, cfg.horizon)
    preds, metrics, grid, best = [], [], [], []

    for w, (train_end, start, end) in enumerate(windows, 1):
        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_te, y_te = X.iloc[start:end], y.iloc[start:end]
        regime = ds.regime.iloc[start:end].to_numpy()
        span = f"{X_te.index[0]:%Y-%m-%d} -> {X_te.index[-1]:%Y-%m-%d}"
        say(f"Window {w}/{len(windows)}: train {len(X_tr)} rows, test {len(X_te)} rows ({span})")

        for name in ALL_NAMES:
            params: dict = {}
            if name in MODEL_NAMES and cfg.tune:
                t0 = time.time()
                search = predictor.tune_search(name, X_tr, y_tr, cfg.horizon, cfg.cv_folds, n_jobs)
                params = dict(search.best_params_)
                res = pd.DataFrame(search.cv_results_)
                for _, r in res.iterrows():
                    grid.append({"Window": w, "Model": name,
                                 **{k.replace("clf__", ""): str(v) for k, v in r["params"].items()},
                                 "mean_cv_score": r["mean_test_score"], "std_cv_score": r["std_test_score"],
                                 "rank": int(r["rank_test_score"])})
                say(f"  tuned {name}: {predictor.format_params(params)} ({time.time() - t0:.0f}s)")
            if name in MODEL_NAMES:
                best.append({"Window": w, "Model": name, "params": predictor.format_params(params)})

            seeds = range(cfg.n_seeds) if name in STOCHASTIC else [0]
            fits = Parallel(n_jobs=n_jobs)(  # results come back in seed order
                delayed(_fit_one)(w, name, params, seed, X_tr, y_tr, X_te, y_te, regime) for seed in seeds)
            for metric_row, pred_frame in fits:
                metrics.append(metric_row)
                preds.append(pred_frame)
    return {"metrics": pd.DataFrame(metrics), "predictions": pd.concat(preds, ignore_index=True),
            "grid": pd.DataFrame(grid), "best_params": pd.DataFrame(best), "windows": windows,
            "dataset": ds}


# ------------------------------------------------------------- statistics
def _block_indices(n: int, n_boot: int, block: int, rng) -> np.ndarray:
    block = min(block, n)
    k = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    return (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]


def _ci(x: np.ndarray) -> tuple[float, float]:
    lo = (100 - CI_LEVEL) / 2
    return tuple(np.percentile(x, [lo, 100 - lo]))


def _balanced(c: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Balanced accuracy from per-row correctness (last axis) and a boolean 'actual up' mask."""
    return 0.5 * ((c * up).sum(-1) / up.sum(-1) + (c * ~up).sum(-1) / (~up).sum(-1))


def summarize(res: dict, cfg: Config) -> dict:
    """Pooled out-of-sample statistics, per-window tables and regime breakdown."""
    preds, metrics = res["predictions"], res["metrics"]

    # Per-row correctness, averaged over seeds (a fraction for stochastic models).
    correct = (preds.assign(correct=(preds["pred"] == preds["y"]).astype(float))
               .groupby(["Window", "Date", "Model"])["correct"].mean().unstack("Model"))
    labels = preds.drop_duplicates(["Window", "Date"]).set_index(["Window", "Date"])
    up = labels["y"].reindex(correct.index).to_numpy() == 1
    n = len(correct)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    idx = _block_indices(n, cfg.n_bootstrap, max(10, 2 * cfg.horizon), rng)
    base = correct[BASELINE_NAME].to_numpy()

    seed_ba = {name: [balanced_accuracy_score(g["y"], g["pred"]) for _, g in grp.groupby("Seed")]
               for name, grp in preds.groupby("Model")}
    wm = metrics.groupby(["Window", "Model"])[["Accuracy", "Balanced Accuracy", "MCC", "AUC"]].mean()
    win_acc = wm["Accuracy"].unstack("Model")
    pooled_mcc = {name: np.mean([_mcc(g["y"], g["pred"]) for _, g in grp.groupby("Seed")])
                  for name, grp in preds.groupby("Model")}

    rows = []
    for name in ALL_NAMES:
        c = correct[name].to_numpy()
        acc, ba, d_acc = c.mean(), _balanced(c, up), (c - base).mean()
        acc_b, ba_b, d_b = c[idx].mean(1), _balanced(c[idx], up[idx]), (c[idx] - base[idx]).mean(1)
        wba = wm["Balanced Accuracy"].unstack("Model")[name]
        rows.append({
            "Model": name, "Accuracy": acc, "Acc CI low": _ci(acc_b)[0], "Acc CI high": _ci(acc_b)[1],
            "Balanced Accuracy": ba, "BA CI low": _ci(ba_b)[0], "BA CI high": _ci(ba_b)[1],
            "Acc - baseline": d_acc, "Diff CI low": _ci(d_b)[0], "Diff CI high": _ci(d_b)[1],
            "MCC": pooled_mcc[name], "AUC (window mean)": wm["AUC"].unstack("Model")[name].mean(),
            "Windows beating baseline": int((win_acc[name] > win_acc[BASELINE_NAME]).sum()),
            "Windows": len(win_acc), "Window BA mean": wba.mean(), "Window BA sd": wba.std(ddof=1),
            "Seed BA sd": float(np.std(seed_ba[name], ddof=1)) if len(seed_ba[name]) > 1 else np.nan,
            "Predicted up share": metrics.loc[metrics["Model"] == name, "Predicted up share"].mean(),
        })
    summary = pd.DataFrame(rows).set_index("Model")

    reg = labels["regime"].reindex(correct.index).to_numpy()
    reg_rows = []
    for regime in ("Low volatility", "High volatility"):
        mask = reg == regime
        if mask.sum() < 20 or up[mask].all() or (~up[mask]).all():
            continue
        for name in ALL_NAMES:
            c = correct[name].to_numpy()[mask]
            reg_rows.append({"Regime": regime, "Model": name, "Rows": int(mask.sum()),
                             "Up share": up[mask].mean(), "Accuracy": c.mean(),
                             "Balanced Accuracy": _balanced(c, up[mask])})
    return {"summary": summary, "window_summary": wm.reset_index(),
            "regime_summary": pd.DataFrame(reg_rows)}


# ----------------------------------------------------------------- report
def _ci_text(v, lo, hi) -> str:
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Accuracy [95% CI]": [_ci_text(r["Accuracy"], r["Acc CI low"], r["Acc CI high"]) for _, r in summary.iterrows()],
        "Bal. acc. [95% CI]": [_ci_text(r["Balanced Accuracy"], r["BA CI low"], r["BA CI high"]) for _, r in summary.iterrows()],
        "Acc - baseline [95% CI]": [f"{r['Acc - baseline']:+.3f} [{r['Diff CI low']:+.3f}, {r['Diff CI high']:+.3f}]"
                                    for _, r in summary.iterrows()],
        "MCC": summary["MCC"].map("{:+.3f}".format),
        "AUC": summary["AUC (window mean)"].map("{:.3f}".format),
        "Win vs base": [f"{int(r['Windows beating baseline'])}/{int(r['Windows'])}" for _, r in summary.iterrows()],
        "Window BA sd": summary["Window BA sd"].map("{:.3f}".format),
        "Seed BA sd": summary["Seed BA sd"].map("{:.3f}".format),
    }, index=summary.index)


def build_report(cfg: Config, manifest: dict, stats: dict, res: dict) -> str:
    wba = stats["window_summary"].pivot(index="Window", columns="Model", values="Balanced Accuracy")[ALL_NAMES]
    conv = res["metrics"].groupby("Model")["Converged"].mean().dropna()
    lines = [
        f"Run {manifest['run_id']}  |  {cfg.ticker}  horizon={cfg.horizon}  windows={cfg.n_windows}  "
        f"seeds={cfg.n_seeds}  tuned={cfg.tune}",
        f"Data: {manifest['data']['first']} -> {manifest['data']['last']} ({manifest['data']['rows']} rows), "
        f"sha256 {manifest['data']['sha256'][:12]}",
        "", "Pooled out-of-sample results (all windows; stochastic models averaged over seeds)",
        format_summary(stats["summary"]).to_string(),
        "", "Balanced accuracy by window (seed mean)", wba.round(3).to_string(),
        "", "Window test periods",
        "\n".join(f"  {w}: {s} -> {e}" for w, s, e in manifest["windows"]),
        "", "Fraction of fits that converged before max_iter",
        conv.round(2).to_string() if len(conv) else "  n/a",
    ]
    if len(stats["regime_summary"]):
        lines += ["", "By volatility regime (pooled)",
                  stats["regime_summary"].round(3).to_string(index=False)]
    lines += ["", f"CIs: moving-block bootstrap, {cfg.n_bootstrap} resamples, block length "
              f"{max(10, 2 * cfg.horizon)}. A CI for 'Acc - baseline' that contains 0 (or a balanced-accuracy "
              "CI that contains 0.5) means no detectable skill over the baseline.",
              "Research output only; not trading advice."]
    return "\n".join(lines)


# --------------------------------------------------------------- manifest
def _git_info() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "*.py"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}


def _grids() -> dict:
    return json.loads(json.dumps(predictor.PARAM_GRIDS))


# ------------------------------------------------------------ run / repeat
def execute_run(cfg: Config, snapshot: Path, run_dir: Path, n_jobs: int = 1, say=print) -> dict:
    """Run one experiment from a data snapshot and save all outputs to run_dir."""
    snapshot = Path(snapshot)
    sha = _sha256(snapshot.read_bytes())
    prices = prepare_prices(load_snapshot(snapshot, sha), cfg)
    started = time.time()
    res = run_windows(cfg, prices, n_jobs, say)
    stats = summarize(res, cfg)

    ds = res["dataset"]
    windows = [(w, f"{ds.X.index[s]:%Y-%m-%d}", f"{ds.X.index[e - 1]:%Y-%m-%d}")
               for w, (_, s, e) in enumerate(res["windows"], 1)]
    manifest = {"run_id": run_dir.name, "created": dt.datetime.now().isoformat(timespec="seconds"),
                "config_digest": cfg.digest(),
                "data": {"snapshot": str(snapshot), "sha256": sha, "rows": len(prices),
                         "first": f"{prices.index[0]:%Y-%m-%d}", "last": f"{prices.index[-1]:%Y-%m-%d}"},
                "environment": _environment(), "git": _git_info(), "param_grids": _grids(),
                "windows": windows, "n_jobs": n_jobs, "runtime_seconds": round(time.time() - started, 1)}

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    res["metrics"].to_csv(run_dir / "metrics.csv", index=False)
    res["predictions"].to_csv(run_dir / "predictions.csv.gz", index=False)
    res["grid"].to_csv(run_dir / "grid_results.csv", index=False)
    res["best_params"].to_csv(run_dir / "best_params.csv", index=False)
    stats["summary"].to_csv(run_dir / "summary.csv")
    stats["window_summary"].to_csv(run_dir / "window_summary.csv", index=False)
    stats["regime_summary"].to_csv(run_dir / "regime_summary.csv", index=False)
    report = build_report(cfg, manifest, stats, res)
    (run_dir / "report.txt").write_text(report + "\n")
    return {"report": report, "manifest": manifest}


def resolve_run(ref: str, runs_dir: Path = RUNS_DIR) -> Path:
    path = Path(ref) if Path(ref).is_dir() else runs_dir / ref
    if not (path / "manifest.json").exists():
        raise SystemExit(f"No saved run found for '{ref}' (looked in {path}).")
    return path


def load_config(run_dir: Path) -> Config:
    return Config(**json.loads((run_dir / "config.json").read_text()))


def verify_repeat(original: Path, repeated: Path, rtol: float = 1e-9) -> list[str]:
    """Compare two runs' metrics.csv. Returns a list of problems (empty = reproduced)."""
    a, b = (pd.read_csv(p / "metrics.csv") for p in (original, repeated))
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return ["metrics tables have different shapes/columns"]
    problems = []
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            if not np.allclose(a[col], b[col], rtol=rtol, atol=1e-12, equal_nan=True):
                problems.append(f"column '{col}' differs (max abs diff "
                                f"{np.nanmax(np.abs(a[col] - b[col])):.3g})")
        elif not a[col].equals(b[col]):
            problems.append(f"column '{col}' differs")
    return problems


def repeat_run(ref: str, name: str | None, n_jobs: int, runs_dir: Path, say=print) -> int:
    original = resolve_run(ref, runs_dir)
    cfg, manifest = load_config(original), json.loads((original / "manifest.json").read_text())
    notes = []
    if manifest["environment"] != _environment():
        notes.append("software environment differs from the original run")
    if manifest["param_grids"] != _grids():
        notes.append("hyperparameter grids in predictor.py differ from the original run")
    if manifest["git"]["commit"] != _git_info()["commit"]:
        notes.append("git commit differs from the original run")

    new_dir = runs_dir / (name or f"{original.name}-repeat")
    out = execute_run(cfg, Path(manifest["data"]["snapshot"]), new_dir, n_jobs, say)
    problems = verify_repeat(original, new_dir)
    say("\n" + out["report"])
    say("\nREPRODUCED: metrics identical to " + original.name if not problems else
        "\nNOT REPRODUCED:\n  " + "\n  ".join(problems))
    for n in notes:
        say(f"  note: {n}")
    return 0 if not problems else 1


# ---------------------------------------------------------------- compare
def compare_runs(refs: list[str], runs_dir: Path = RUNS_DIR) -> tuple[str, pd.DataFrame]:
    dirs = [resolve_run(r, runs_dir) for r in refs]
    cfgs = {d.name: asdict(load_config(d)) for d in dirs}
    mans = {d.name: json.loads((d / "manifest.json").read_text()) for d in dirs}
    keys = [k for k in next(iter(cfgs.values())) if len({json.dumps(c[k]) for c in cfgs.values()}) > 1]

    lines = ["Runs compared:"]
    for d in dirs:
        m = mans[d.name]
        lines.append(f"  {d.name}: data sha {m['data']['sha256'][:8]} ({m['data']['first']} -> {m['data']['last']}), "
                     f"commit {(m['git']['commit'] or 'n/a')[:8]}{'*' if m['git']['dirty'] else ''}")
    lines.append("Config differences: " + (", ".join(f"{k}=" + " vs ".join(str(c[k]) for c in cfgs.values())
                                                     for k in keys) or "none"))
    if len({m["data"]["sha256"] for m in mans.values()}) > 1:
        lines.append("WARNING: runs use different data snapshots; differences may come from the data.")
    if len({tuple((w, s) for w, s, _ in m["windows"]) for m in mans.values()}) > 1:
        lines.append("WARNING: runs use different test windows; metrics are not computed on the same dates.")

    frames = []
    for d in dirs:
        s = pd.read_csv(d / "summary.csv", index_col="Model")
        f = format_summary(s)
        f.insert(0, "Run", d.name)
        frames.append(f.reset_index())
    table = pd.concat(frames)
    table["Model"] = pd.Categorical(table["Model"], ALL_NAMES, ordered=True)
    table["Run"] = pd.Categorical(table["Run"], [d.name for d in dirs], ordered=True)
    table = table.sort_values(["Model", "Run"], kind="stable").set_index(["Model", "Run"])
    lines += ["", table.to_string()]
    return "\n".join(lines), table


# -------------------------------------------------------------------- CLI
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a new experiment and save it")
    r.add_argument("--name", help="run id (default: <TICKER>_h<H>_<timestamp>)")
    r.add_argument("--ticker", default="NVDA")
    r.add_argument("--start", default="2010-01-01")
    r.add_argument("--end", default=None, help="last date to use (YYYY-MM-DD)")
    r.add_argument("--horizon", type=int, default=5)
    r.add_argument("--windows", type=int, default=5, dest="n_windows")
    r.add_argument("--initial-train-fraction", type=float, default=0.5)
    r.add_argument("--seeds", type=int, default=10, dest="n_seeds")
    r.add_argument("--folds", type=int, default=4, dest="cv_folds")
    r.add_argument("--no-tune", action="store_true", help="use default hyperparameters")
    r.add_argument("--bootstrap", type=int, default=2000, dest="n_bootstrap")
    r.add_argument("--snapshot", type=Path, help="reuse this saved price snapshot instead of downloading")
    r.add_argument("--n-jobs", type=int, default=1, help="parallel workers for grid search and seed refits")

    rp = sub.add_parser("repeat", help="re-run a saved run and verify identical metrics")
    rp.add_argument("run")
    rp.add_argument("--name")
    rp.add_argument("--n-jobs", type=int, default=1)

    c = sub.add_parser("compare", help="compare saved runs side by side")
    c.add_argument("runs", nargs="+")
    c.add_argument("--csv", type=Path, help="also write the table to this CSV file")

    sub.add_parser("list", help="list saved runs")
    return p.parse_args()


def main() -> int:
    a = _parse()
    if a.cmd == "run":
        cfg = Config(ticker=a.ticker.upper().strip(), start=a.start, end=a.end, horizon=a.horizon,
                     n_windows=a.n_windows, initial_train_fraction=a.initial_train_fraction,
                     n_seeds=a.n_seeds, cv_folds=a.cv_folds, tune=not a.no_tune,
                     n_bootstrap=a.n_bootstrap)
        run_id = a.name or f"{cfg.ticker}_h{cfg.horizon}_{dt.datetime.now():%Y%m%d-%H%M%S}"
        if (a.runs_dir / run_id).exists():  # fail before the download and the long computation
            raise SystemExit(f"Run '{run_id}' already exists in {a.runs_dir}. Choose another --name "
                             "(runs are never overwritten).")
        snapshot = a.snapshot
        if snapshot is None:
            prices = features.load_prices(cfg.ticker, cfg.start)
            snapshot = save_snapshot(prices, cfg.ticker)
            print(f"Saved data snapshot {snapshot}")
        out = execute_run(cfg, snapshot, a.runs_dir / run_id, a.n_jobs)
        print("\n" + out["report"])
        print(f"\nSaved to {a.runs_dir / run_id}")
        return 0
    if a.cmd == "repeat":
        return repeat_run(a.run, a.name, a.n_jobs, a.runs_dir)
    if a.cmd == "compare":
        text, table = compare_runs(a.runs, a.runs_dir)
        print(text)
        if a.csv:
            table.to_csv(a.csv)
        return 0
    rows = []
    for m in sorted(a.runs_dir.glob("*/manifest.json")):
        man, cfg = json.loads(m.read_text()), json.loads((m.parent / "config.json").read_text())
        rows.append({"run": m.parent.name, "ticker": cfg["ticker"], "horizon": cfg["horizon"],
                     "windows": cfg["n_windows"], "seeds": cfg["n_seeds"], "tuned": cfg["tune"],
                     "data_last": man["data"]["last"], "data_sha": man["data"]["sha256"][:8],
                     "created": man["created"]})
    print(pd.DataFrame(rows).to_string(index=False) if rows else "No saved runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
