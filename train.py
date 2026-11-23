"""
train.py - train and evaluate from the console, then save results for the dashboard.

Examples
--------
    python train.py                                  # NVDA, 5-day horizon, tuned
    python train.py --ticker AAPL --horizon 10
    python train.py --no-tune                        # faster, default hyperparameters

Results are saved to artifacts/<TICKER>_h<HORIZON>.joblib. Then run:
    streamlit run app.py
"""

import argparse
import datetime as dt
from pathlib import Path

import joblib
import pandas as pd

import evaluation
import features
import predictor

ARTIFACT_DIR = Path("artifacts")


def parse_args():
    p = argparse.ArgumentParser(description="Train stock-direction models and save results.")
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--start", default="2010-01-01", help="History start date (YYYY-MM-DD)")
    p.add_argument("--horizon", type=int, default=5, help="Prediction horizon in trading days")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--folds", type=int, default=4, help="Walk-forward CV folds")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20],
                   help="Horizons to compare in the Market Conditions tab")
    p.add_argument("--no-tune", action="store_true", help="Skip GridSearchCV")
    return p.parse_args()


def main():
    a = parse_args()
    ticker = a.ticker.upper().strip()
    prices = features.load_prices(ticker, a.start)
    print(f"{ticker}: {len(prices)} rows, {prices.index[0].date()} -> {prices.index[-1].date()}")

    result = evaluation.run_pipeline(
        prices, horizon=a.horizon, test_fraction=a.test_fraction, tune=not a.no_tune,
        n_splits=a.folds, horizons=a.horizons, progress=print)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    path = ARTIFACT_DIR / f"{ticker}_h{a.horizon}.joblib"
    joblib.dump({"ticker": ticker, "prices": prices, "result": result,
                 "created": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}, path)

    pd.set_option("display.width", 120)
    print("\nHold-out test metrics")
    print(result.metrics.round(3))
    print("\nSelected hyperparameters")
    for name, params in result.best_params.items():
        print(f"  {name}: {predictor.format_params(params)}")
    print(f"\nLatest predicted direction ({result.latest_date.date()}, next {a.horizon} trading days)")
    print(result.final_predictions.replace({"Predicted direction": {1: "UP", 0: "DOWN"}}))
    print(f"\nSaved to {path}. Now run: streamlit run app.py")
    print("These are direction forecasts from a research model, not trading recommendations.")


if __name__ == "__main__":
    main()