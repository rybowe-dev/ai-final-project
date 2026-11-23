"""
evaluation.py - validation, metrics and analysis, plus the end-to-end pipeline.

Workflow of run_pipeline()
--------------------------
1. Build features/labels (features.py).
2. Chronological hold-out split. The last `horizon` rows of the training set are
   removed so no training label looks into the test period.
3. Tune each model with walk-forward CV on the TRAINING period only (predictor.py).
4. Evaluate the tuned evaluation-models once on the untouched hold-out test set.
5. Analyse the test predictions: permutation importance, volatility regimes.
6. Compare prediction horizons (re-using the tuned hyperparameters, no re-tuning).
7. Train separate FINAL models on all labeled data for the latest prediction.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

import predictor
from features import Dataset, build_dataset
from predictor import BASELINE_NAME, MODEL_NAMES, RANDOM_STATE

ALL_NAMES = [BASELINE_NAME] + MODEL_NAMES
REGIMES = ["Low volatility", "High volatility"]


# ----------------------------------------------------------------- split
def chronological_split(X, y, test_fraction: float, gap: int):
    """Train on the past, test on the future, with no label overlap.

    Training rows end `gap` rows before the test set starts. A training label at
    row i uses the price at row i+gap, so the last training label only reaches the
    day before the first test row.
    """
    n = len(X)
    split = int(n * (1 - test_fraction))
    train_end = split - gap
    if train_end < 100:
        raise ValueError("Training set too small: lower the test fraction or horizon.")
    return X.iloc[:train_end], X.iloc[split:], y.iloc[:train_end], y.iloc[split:]


# --------------------------------------------------------------- metrics
def _balanced_accuracy(y_true, y_pred) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return balanced_accuracy_score(y_true, y_pred)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": _balanced_accuracy(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_on_test(models: dict, X_test, y_test):
    preds = {n: pd.Series(m.predict(X_test), index=X_test.index) for n, m in models.items()}
    metrics = pd.DataFrame({n: compute_metrics(y_test, p) for n, p in preds.items()}).T
    confusion = {n: confusion_matrix(y_test, p, labels=[0, 1]) for n, p in preds.items()}
    return preds, metrics, confusion


# -------------------------------------------------------------- analysis
def permutation_importances(models: dict, X_test, y_test, n_repeats: int = 10) -> dict:
    """Drop in balanced accuracy on the hold-out test set when one feature is shuffled."""
    out = {}
    for name in MODEL_NAMES:
        r = permutation_importance(models[name], X_test, y_test, scoring="balanced_accuracy",
                                   n_repeats=n_repeats, random_state=RANDOM_STATE)
        out[name] = (pd.DataFrame({"Importance": r.importances_mean, "Std": r.importances_std},
                                  index=X_test.columns)
                     .sort_values("Importance", ascending=False))
    return out


def regime_analysis(y_test, preds: dict, regime: pd.Series) -> pd.DataFrame:
    regime = regime.reindex(y_test.index)
    rows = []
    for reg in REGIMES:
        mask = (regime == reg).to_numpy()
        if mask.sum() == 0:
            continue
        yt = y_test.to_numpy()[mask]
        for name in ALL_NAMES:
            yp = preds[name].to_numpy()[mask]
            rows.append({"Regime": reg, "Model": name, "Accuracy": accuracy_score(yt, yp),
                         "Balanced Accuracy": _balanced_accuracy(yt, yp),
                         "Samples": int(mask.sum()), "Up share": float(yt.mean())})
    return pd.DataFrame(rows)


def horizon_comparison(prices, horizons, test_fraction: float, best_params: dict) -> pd.DataFrame:
    """Re-run split/fit/test for several horizons. Hyperparameters tuned at the main
    horizon are re-used (no re-tuning) to keep this comparison cheap."""
    rows = []
    for h in sorted(set(horizons)):
        ds = build_dataset(prices, h)
        X_tr, X_te, y_tr, y_te = chronological_split(ds.X, ds.y, test_fraction, h)
        for name in ALL_NAMES:
            model = predictor.fit_model(name, X_tr, y_tr, best_params.get(name))
            m = compute_metrics(y_te, model.predict(X_te))
            rows.append({"Horizon": h, "Model": name, "Accuracy": m["Accuracy"],
                         "Balanced Accuracy": m["Balanced Accuracy"], "F1": m["F1"],
                         "Up share (test)": float(y_te.mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- pipeline
@dataclass
class ExperimentResult:
    horizon: int
    test_fraction: float
    tuned: bool
    n_labeled: int
    train_range: tuple
    test_range: tuple
    train_up_share: float
    test_up_share: float
    metrics: pd.DataFrame
    confusion: dict
    predictions: pd.DataFrame       # Actual + each model, indexed by test date
    cv_scores: pd.DataFrame         # walk-forward balanced accuracy, one row per fold
    best_params: dict
    importance: dict
    regime_metrics: pd.DataFrame
    horizon_metrics: pd.DataFrame
    final_predictions: pd.DataFrame
    latest_date: pd.Timestamp
    latest_features: pd.DataFrame


def run_pipeline(prices, horizon: int = 5, test_fraction: float = 0.2, tune: bool = True,
                 n_splits: int = 4, horizons=(1, 5, 10, 20), progress=None) -> ExperimentResult:
    say = progress or (lambda msg: None)

    say("Building features and labels...")
    ds: Dataset = build_dataset(prices, horizon)
    X_tr, X_te, y_tr, y_te = chronological_split(ds.X, ds.y, test_fraction, horizon)

    eval_models, best_params, cv = {}, {}, {}
    say("Fitting baseline...")
    eval_models[BASELINE_NAME] = predictor.fit_model(BASELINE_NAME, X_tr, y_tr)
    cv[BASELINE_NAME] = predictor.walk_forward_scores(
        predictor.build_model(BASELINE_NAME), X_tr, y_tr, horizon, n_splits)

    for name in MODEL_NAMES:
        if tune:
            say(f"Tuning {name} with walk-forward GridSearchCV...")
            eval_models[name], best_params[name], cv[name] = predictor.tune_model(
                name, X_tr, y_tr, horizon, n_splits)
        else:
            say(f"Fitting {name}...")
            best_params[name] = {}
            eval_models[name] = predictor.fit_model(name, X_tr, y_tr)
            cv[name] = predictor.walk_forward_scores(
                predictor.build_model(name), X_tr, y_tr, horizon, n_splits)

    say("Evaluating on the hold-out test set...")
    preds, metrics, confusion = evaluate_on_test(eval_models, X_te, y_te)
    predictions = pd.DataFrame({"Actual": y_te, **preds})

    say("Computing permutation importance...")
    importance = permutation_importances(eval_models, X_te, y_te)

    say("Analysing volatility regimes...")
    regimes = regime_analysis(y_te, preds, ds.regime)

    say("Comparing prediction horizons...")
    horizon_metrics = horizon_comparison(prices, set(horizons) | {horizon}, test_fraction, best_params)

    say("Training final models on all labeled data...")
    final_models = predictor.train_final_models(ds.X, ds.y, best_params)
    final_predictions = predictor.predict_latest(final_models, ds.X_latest)

    return ExperimentResult(
        horizon=horizon, test_fraction=test_fraction, tuned=tune, n_labeled=len(ds.X),
        train_range=(X_tr.index[0], X_tr.index[-1]), test_range=(X_te.index[0], X_te.index[-1]),
        train_up_share=float(y_tr.mean()), test_up_share=float(y_te.mean()),
        metrics=metrics, confusion=confusion, predictions=predictions,
        cv_scores=pd.DataFrame(cv, index=[f"Fold {i + 1}" for i in range(n_splits)]),
        best_params=best_params, importance=importance, regime_metrics=regimes,
        horizon_metrics=horizon_metrics, final_predictions=final_predictions,
        latest_date=ds.X_latest.index[0], latest_features=ds.X_latest,
    )