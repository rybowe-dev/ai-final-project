"""
predictor.py - model definitions, hyperparameter tuning, final training and prediction.

The three required algorithms are Gaussian Naive Bayes, a Perceptron and an MLP.
Every model is wrapped in a Pipeline with a StandardScaler, so scaling statistics
are learned from the training data only (also inside every cross-validation fold).

Design notes
------------
* MLP early stopping is disabled. scikit-learn's built-in early stopping carves a
  *random* validation set out of the training data, which mixes future and past
  rows. Regularisation (alpha) and a fixed iteration cap are used instead, and
  alpha / architecture are tuned with chronological walk-forward CV.
* Evaluation models (trained on the training period only) are kept separate from
  final models (trained on all labeled data) that are used for the live prediction.
* Run `python predictor.py` for a quick command-line report.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODEL_NAMES = ["Naive Bayes", "Perceptron", "MLP"]
BASELINE_NAME = "Baseline (majority)"

PARAM_GRIDS = {
    "Naive Bayes": {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]},
    "Perceptron": {"clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2]},
    "MLP": {
        "clf__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
        "clf__alpha": [1e-3, 1e-2, 1e-1],
    },
}


@contextmanager
def quiet():
    """Silence ConvergenceWarning noise during fitting."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        yield


def build_model(name: str) -> Pipeline:
    """Untrained model with default hyperparameters."""
    if name == BASELINE_NAME:
        clf = DummyClassifier(strategy="most_frequent")
    elif name == "Naive Bayes":
        clf = GaussianNB()
    elif name == "Perceptron":
        clf = Perceptron(penalty="l2", alpha=1e-4, max_iter=1000, tol=1e-3,
                         random_state=RANDOM_STATE)
    elif name == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=500,
                            early_stopping=False, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def fit_model(name: str, X, y, params: dict | None = None) -> Pipeline:
    model = build_model(name)
    if params:
        model.set_params(**params)
    with quiet():
        model.fit(X, y)
    return model


# ------------------------------------------------------------ validation
def walk_forward_scores(estimator, X, y, horizon: int, n_splits: int) -> np.ndarray:
    """Balanced accuracy on each walk-forward fold (expanding window, with a
    `horizon`-sized gap so training labels never overlap the validation period)."""
    cv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    with quiet():
        return cross_val_score(clone(estimator), X, y, cv=cv, scoring="balanced_accuracy")


def tune_search(name: str, X_train, y_train, horizon: int, n_splits: int,
                n_jobs: int = 1) -> GridSearchCV:
    """Fitted GridSearchCV (walk-forward CV with a `horizon` gap). Keeps cv_results_
    so the score of every grid point can be inspected, not just the winner."""
    cv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    search = GridSearchCV(build_model(name), PARAM_GRIDS[name], cv=cv,
                          scoring="balanced_accuracy", refit=True, n_jobs=n_jobs)
    with quiet():
        search.fit(X_train, y_train)
    return search


def tune_model(name: str, X_train, y_train, horizon: int, n_splits: int):
    """GridSearchCV with walk-forward (TimeSeriesSplit + gap) cross-validation.

    Returns (best fitted model, best params, per-fold CV scores of the best params).
    The grid search only ever sees the training period.
    """
    search = tune_search(name, X_train, y_train, horizon, n_splits)
    best = search.best_index_
    folds = np.array([search.cv_results_[f"split{i}_test_score"][best] for i in range(n_splits)])
    return search.best_estimator_, search.best_params_, folds


# ------------------------------------------------------- final inference
def train_final_models(X, y, best_params: dict) -> dict:
    """Fit fresh models on ALL labeled data (used only for the live prediction)."""
    return {name: fit_model(name, X, y, best_params.get(name)) for name in MODEL_NAMES}


def predict_latest(models: dict, X_latest: pd.DataFrame) -> pd.DataFrame:
    """Predicted price direction for the newest feature row."""
    rows = []
    for name, model in models.items():
        prob = float(model.predict_proba(X_latest)[0, 1]) if hasattr(model, "predict_proba") else np.nan
        rows.append({"Model": name, "Predicted direction": int(model.predict(X_latest)[0]),
                     "P(up)": prob})
    return pd.DataFrame(rows).set_index("Model")


def format_params(params: dict) -> str:
    return ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in params.items()) or "defaults"


# ------------------------------------------------------------------- CLI
def main(ticker: str = "NVDA", start: str = "2010-01-01", horizon: int = 5):
    from evaluation import run_pipeline  # imported here to avoid a circular import
    from features import load_prices

    prices = load_prices(ticker, start)
    print(f"{ticker}: {len(prices)} rows, {prices.index[0].date()} -> {prices.index[-1].date()}")
    result = run_pipeline(prices, horizon=horizon, progress=print)

    pd.set_option("display.width", 120)
    print("\nHold-out test metrics")
    print(result.metrics.round(3))
    print("\nLatest predicted price direction "
          f"({result.latest_date.date()}, next {horizon} trading days)")
    print(result.final_predictions.replace({"Predicted direction": {1: "UP", 0: "DOWN"}}))
    print("\nThis is a direction forecast from a research model, not a trading recommendation.")


if __name__ == "__main__":
    main()