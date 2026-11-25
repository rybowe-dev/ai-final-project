"""
run_results.py - read-only loading and analysis of the runs saved by experiment.py.

Nothing here trains a model or imports experiment.py: everything is derived from the files in
runs/<name>/ (config.json, manifest.json, summary.csv, window_summary.csv, regime_summary.csv,
metrics.csv, best_params.csv, grid_results.csv, report.txt). Only config.json, manifest.json and
summary.csv are required; a missing optional file only disables the view that needs it.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from predictor import BASELINE_NAME, MODEL_NAMES

RUNS_DIR = Path("runs")
ALL_NAMES = [BASELINE_NAME] + MODEL_NAMES
STOCHASTIC = ["Perceptron", "MLP"]
REQUIRED_FILES = ["config.json", "manifest.json", "summary.csv"]
OPTIONAL_FILES = ["window_summary.csv", "regime_summary.csv", "metrics.csv", "best_params.csv",
                  "grid_results.csv", "report.txt"]
FIT_METRICS = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC", "AUC"]
# summary.csv columns holding (value, CI low, CI high) for each quantity that has a bootstrap CI
CI_COLUMNS = {"Accuracy": ("Accuracy", "Acc CI low", "Acc CI high"),
              "Balanced Accuracy": ("Balanced Accuracy", "BA CI low", "BA CI high"),
              "Accuracy - baseline": ("Acc - baseline", "Diff CI low", "Diff CI high")}
# config fields that only identify what was studied; any other difference changes the procedure
IDENTITY_FIELDS = {"ticker", "horizon"}


class RunLoadError(Exception):
    """A saved run is missing required files or one of them cannot be parsed."""


@dataclass
class RunData:
    name: str
    path: Path
    config: dict
    manifest: dict
    summary: pd.DataFrame
    window_summary: pd.DataFrame | None = None
    regime_summary: pd.DataFrame | None = None
    metrics: pd.DataFrame | None = None
    best_params: pd.DataFrame | None = None
    grid: pd.DataFrame | None = None
    report: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_ci(self) -> bool:
        return all(c in self.summary.columns for cols in CI_COLUMNS.values() for c in cols)


# ---------------------------------------------------------------- loading
def run_signature(path: Path) -> tuple:
    """Cheap fingerprint of a run directory (file names, sizes, mtimes), used as a cache key."""
    path = Path(path)
    if not path.is_dir():
        return ()
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.iterdir()) if p.is_file())


def runs_signature(runs_dir: Path = RUNS_DIR) -> tuple:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return ()
    return tuple((d.name, run_signature(d)) for d in sorted(runs_dir.iterdir()) if d.is_dir())


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise RunLoadError(f"{path.name} is unreadable ({e})") from e
    if not isinstance(data, dict):
        raise RunLoadError(f"{path.name} does not contain a JSON object")
    return data


def _read_csv(path: Path, required: bool, **kw) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, **kw)
    except FileNotFoundError:
        if required:
            raise RunLoadError(f"{path.name} is missing") from None
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        if required:
            raise RunLoadError(f"{path.name} is unreadable ({e})") from e
    return None


def load_run(path: Path) -> RunData:
    """Load one saved run. Raises RunLoadError if a required file is missing or malformed."""
    path = Path(path)
    missing = [f for f in REQUIRED_FILES if not (path / f).exists()]
    if missing:
        raise RunLoadError("missing " + ", ".join(missing))
    config, manifest = _read_json(path / "config.json"), _read_json(path / "manifest.json")
    if not {"ticker", "horizon"} <= config.keys():
        raise RunLoadError("config.json lacks ticker/horizon")
    if not {"data", "windows"} <= manifest.keys() or not isinstance(manifest["data"], dict):
        raise RunLoadError("manifest.json lacks data/windows")
    summary = _read_csv(path / "summary.csv", True)
    if summary is None or summary.empty or not {"Model", "Accuracy"} <= set(summary.columns):
        raise RunLoadError("summary.csv has no usable results")
    summary = summary.set_index("Model")

    run = RunData(path.name, path, config, manifest, summary)
    for fname, attr in [("window_summary.csv", "window_summary"), ("regime_summary.csv", "regime_summary"),
                        ("metrics.csv", "metrics"), ("best_params.csv", "best_params"),
                        ("grid_results.csv", "grid")]:
        df = _read_csv(path / fname, False)
        if df is None:
            if (path / fname).exists():
                run.notes.append(f"{fname} could not be read")
            else:
                run.notes.append(f"{fname} not found")
        setattr(run, attr, df)
    try:
        run.report = (path / "report.txt").read_text()
    except OSError:
        run.notes.append("report.txt not found")
    if not run.has_ci:
        run.notes.append("summary.csv lacks some bootstrap confidence-interval columns")
    return run


def discover_runs(runs_dir: Path = RUNS_DIR) -> tuple[list[RunData], dict[str, str]]:
    """Load every run under runs_dir. Returns (usable runs, {folder name: reason it was skipped})."""
    runs_dir, runs, problems = Path(runs_dir), [], {}
    if not runs_dir.is_dir():
        return runs, problems
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        try:
            runs.append(load_run(d))
        except RunLoadError as e:
            problems[d.name] = str(e)
    runs.sort(key=lambda r: r.manifest.get("created", ""), reverse=True)
    return runs, problems


# --------------------------------------------------------------- analysis
def ci_table(run: RunData, quantity: str) -> pd.DataFrame:
    """Columns Value / CI low / CI high (one row per model) for a quantity in CI_COLUMNS."""
    value, lo, hi = CI_COLUMNS[quantity]
    if not all(c in run.summary.columns for c in (value, lo, hi)):
        return pd.DataFrame(columns=["Value", "CI low", "CI high"])
    out = run.summary[[value, lo, hi]].copy()
    out.columns = ["Value", "CI low", "CI high"]
    return out.reindex([m for m in ALL_NAMES if m in out.index])


def verdicts(summary: pd.DataFrame) -> pd.DataFrame:
    """Plain-language reading of the bootstrap CIs for each non-baseline model."""
    rows = {}
    for name in [m for m in MODEL_NAMES if m in summary.index]:
        r = summary.loc[name]
        d_lo, d_hi = r.get("Diff CI low"), r.get("Diff CI high")
        b_lo, b_hi = r.get("BA CI low"), r.get("BA CI high")
        if pd.isna(d_lo) or pd.isna(d_hi):
            acc = "n/a"
        elif d_lo > 0:
            acc = "Above baseline (CI excludes 0)"
        elif d_hi < 0:
            acc = "Below baseline (CI excludes 0)"
        else:
            acc = "Not distinguishable from baseline"
        if pd.isna(b_lo) or pd.isna(b_hi):
            bal = "n/a"
        elif b_lo > 0.5:
            bal = "Above chance (CI excludes 0.5)"
        elif b_hi < 0.5:
            bal = "Below chance (CI excludes 0.5)"
        else:
            bal = "Not distinguishable from chance"
        rows[name] = {"Accuracy vs. majority baseline": acc, "Balanced accuracy vs. 0.5": bal}
    return pd.DataFrame.from_dict(rows, orient="index")


def per_fit_metrics(metrics: pd.DataFrame | None) -> pd.DataFrame:
    """Mean of each metric over windows (seed-averaged inside a window), from metrics.csv.

    Unlike summary.csv (pooled predictions), this is the average of per-window scores, so
    Precision, Recall and F1 are available too.
    """
    if metrics is None or not {"Window", "Model"} <= set(metrics.columns):
        return pd.DataFrame()
    cols = [c for c in FIT_METRICS if c in metrics.columns]
    per_window = metrics.groupby(["Window", "Model"])[cols].mean()
    out = per_window.groupby("Model").mean()
    return out.reindex([m for m in ALL_NAMES if m in out.index])


def window_labels(manifest: dict) -> dict[int, str]:
    labels = {}
    for w in manifest.get("windows", []):
        if len(w) == 3:
            labels[int(w[0])] = f"W{w[0]}: {w[1][:7]} → {w[2][:7]}"
    return labels


def window_info(metrics: pd.DataFrame | None) -> pd.DataFrame:
    """One row per test window: dates, rows and class balance."""
    if metrics is None or "Window" not in metrics.columns:
        return pd.DataFrame()
    cols = [c for c in ["Train rows", "Test start", "Test end", "Test rows", "Test up share"] if c in metrics]
    return metrics.groupby("Window")[cols].first()


def window_metric(run: RunData, metric: str) -> pd.DataFrame:
    """Window x Model table of a metric (seed means), or an empty frame if unavailable."""
    ws = run.window_summary
    if ws is None or not {"Window", "Model", metric} <= set(ws.columns):
        return pd.DataFrame()
    piv = ws.pivot(index="Window", columns="Model", values=metric)
    return piv[[m for m in ALL_NAMES if m in piv.columns]]


def seed_metrics(metrics: pd.DataFrame | None, metric: str) -> pd.DataFrame:
    """Per-seed scores of the stochastic models (Window, Model, Seed, metric); empty if no spread."""
    need = {"Window", "Model", "Seed", metric}
    if metrics is None or not need <= set(metrics.columns):
        return pd.DataFrame()
    s = metrics[metrics["Model"].isin(STOCHASTIC)][list(need)]
    return s if s["Seed"].nunique() > 1 else pd.DataFrame()


def seed_spread(metrics: pd.DataFrame | None, metric: str) -> pd.DataFrame:
    """Min / mean / max / sd of a metric across seeds, per model and window."""
    s = seed_metrics(metrics, metric)
    if s.empty:
        return s
    g = s.groupby(["Model", "Window"])[metric].agg(["min", "mean", "max", "std"])
    return g.rename(columns={"min": "Min", "mean": "Mean", "max": "Max", "std": "SD"})


def convergence(metrics: pd.DataFrame | None) -> pd.DataFrame:
    if metrics is None or not {"Model", "Converged"} <= set(metrics.columns):
        return pd.DataFrame()
    c = metrics.dropna(subset=["Converged"]).groupby("Model")["Converged"].agg(["mean", "size"])
    return c.rename(columns={"mean": "Fits converged", "size": "Fits"})


# ------------------------------------------------------------- comparison
def _window_starts(run: RunData) -> list:
    return [(w[0], w[1]) for w in run.manifest.get("windows", []) if len(w) >= 2]


def _window_ends(run: RunData) -> list:
    return [(w[0], w[2]) for w in run.manifest.get("windows", []) if len(w) >= 3]


def config_differences(runs: list[RunData]) -> pd.DataFrame:
    """Rows = settings that differ between the runs (config fields, data hash, window dates, code version)."""
    items = {}
    for key in runs[0].config:
        vals = [r.config.get(key) for r in runs]
        if len({json.dumps(v) for v in vals}) > 1:
            items[key] = vals
    for label, getter in [("data sha256", lambda r: str(r.manifest["data"].get("sha256", "n/a"))[:8]),
                          ("data range", lambda r: f"{r.manifest['data'].get('first')} → "
                                                   f"{r.manifest['data'].get('last')}"),
                          ("test window starts", lambda r: str([s for _, s in _window_starts(r)])),
                          ("git commit", lambda r: str(r.manifest.get("git", {}).get("commit") or "n/a")[:8]),
                          ("hyperparameter grids", lambda r: r.manifest.get("param_grids"))]:
        vals = [getter(r) for r in runs]
        if len({json.dumps(v, sort_keys=True, default=str) for v in vals}) > 1:
            items[label] = vals if label != "hyperparameter grids" else ["differs"] * len(vals)
    return pd.DataFrame(items, index=[r.name for r in runs]).T.astype(str)


def comparability(runs: list[RunData]) -> list[tuple[str, str]]:
    """Return [(level, message)] with level in {"ok", "info", "warning", "error"} for a set of runs."""
    if len(runs) < 2:
        return []
    out = []
    tickers = {r.config.get("ticker") for r in runs}
    shas = {r.manifest["data"].get("sha256") for r in runs}
    starts = {tuple(_window_starts(r)) for r in runs}
    ends = {tuple(_window_ends(r)) for r in runs}
    procedure = [k for k in runs[0].config if k not in IDENTITY_FIELDS
                 and len({json.dumps(r.config.get(k)) for r in runs}) > 1]
    grids = {json.dumps(r.manifest.get("param_grids"), sort_keys=True) for r in runs}

    if len(tickers) > 1:
        out.append(("warning", "Different tickers: each stock has its own price history, class balance and "
                               "test dates, so the numbers describe different problems. Compare patterns, "
                               "not individual scores."))
    elif len(shas) > 1:
        out.append(("warning", "The runs used different data snapshots; differences may come from the data "
                               "rather than from the settings you changed."))
    if len(tickers) == 1 and len(shas) == 1 and len(starts) > 1:
        out.append(("error", "The runs use different test windows, so their metrics are computed on "
                             "different dates and are not directly comparable."))
    elif len(starts) > 1 and len(tickers) > 1:
        out.append(("info", "Test windows start on different dates."))
    if procedure:
        out.append(("warning", "The evaluation procedure differs (" + ", ".join(procedure) + "); "
                               "differences may reflect that rather than the model or horizon."))
    if len(grids) > 1:
        out.append(("warning", "The hyperparameter search grids differ between the runs."))
    if len({r.config.get("horizon") for r in runs}) > 1 and len(starts) == 1 and len(shas) == 1:
        note = "Horizons differ but the runs share the same data and test-window start dates."
        if len(ends) > 1:
            note += " The last window ends earlier for longer horizons because the newest rows have no label yet."
        out.append(("info", note + " Longer horizons have heavily overlapping labels, so their effective "
                              "sample size is smaller."))
    if not out:
        out.append(("ok", "Same data snapshot, test windows and evaluation settings: directly comparable."))
    return out
