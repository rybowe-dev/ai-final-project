"""
Buy/Sell classifier for a stock using Naive Bayes, Perceptron, and MLP.

Setup:
    pip install yfinance pandas numpy scikit-learn matplotlib

Run:
    python nvda_buy_sell.py
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

TICKER = "NVDA"          # note: the ticker is NVDA (company: NVIDIA)
START = "2010-01-01"
HORIZON = 5              # predict whether price is higher HORIZON trading days from now
TEST_FRACTION = 0.2      # last 20% of history is held out for testing


# ---------------------------------------------------------------- data
def load_prices(ticker=TICKER, start=START):
    import yfinance as yf
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    # newer yfinance versions return MultiIndex columns; flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ------------------------------------------------------------ features
def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def make_features(df):
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    f["ret_1d"] = c.pct_change(1)
    f["ret_5d"] = c.pct_change(5)
    f["ret_10d"] = c.pct_change(10)
    f["ret_20d"] = c.pct_change(20)
    f["sma10_ratio"] = c / c.rolling(10).mean() - 1
    f["sma50_ratio"] = c / c.rolling(50).mean() - 1
    f["sma200_ratio"] = c / c.rolling(200).mean() - 1
    f["vol_10d"] = c.pct_change().rolling(10).std()
    f["vol_30d"] = c.pct_change().rolling(30).std()
    f["rsi_14"] = rsi(c, 14)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c   # normalized by price
    f["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    f["hl_range"] = (df["High"] - df["Low"]) / c
    return f


def make_labels(df, horizon=HORIZON):
    """1 = BUY (price higher in `horizon` days), 0 = SELL."""
    future_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    return (future_ret > 0).astype(int), future_ret


def build_dataset(df):
    X = make_features(df)
    y, future_ret = make_labels(df)
    data = X.copy()
    data["y"] = y
    data["future_ret"] = future_ret
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    # the last HORIZON rows have no label (future unknown) and were dropped by dropna
    return data.drop(columns=["y", "future_ret"]), data["y"], data["future_ret"]


# ------------------------------------------------------------- modeling
def get_models():
    return {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Perceptron": make_pipeline(
            StandardScaler(), Perceptron(max_iter=1000, tol=1e-3, random_state=42)
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=1e-3,                 # L2 regularization
                early_stopping=True,        # holds out part of train set internally
                max_iter=500,
                random_state=42,
            ),
        ),
    }


def time_split(X, y, test_fraction=TEST_FRACTION, gap=HORIZON):
    """Chronological split. `gap` rows are skipped between train and test so
    that training labels (which look `gap` days ahead) don't overlap the test period."""
    n = len(X)
    split = int(n * (1 - test_fraction))
    return (X.iloc[:split], X.iloc[split + gap:],
            y.iloc[:split], y.iloc[split + gap:])


def main():
    df = load_prices()
    print(f"Loaded {len(df)} rows of {TICKER}: {df.index[0].date()} -> {df.index[-1].date()}")

    X, y, future_ret = build_dataset(df)
    X_train, X_test, y_train, y_test = time_split(X, y)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows "
          f"({X_test.index[0].date()} -> {X_test.index[-1].date()})")
    print(f"Fraction BUY in train: {y_train.mean():.2f} | in test: {y_test.mean():.2f}\n")

    results = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = pred
        print(f"=== {name} ===")
        print(f"Accuracy: {accuracy_score(y_test, pred):.3f} | "
              f"F1: {f1_score(y_test, pred, zero_division=0):.3f}")
        print(confusion_matrix(y_test, pred))
        print()

    # Crude sanity check: would following the signals have made money?
    # (ignores transaction costs, slippage, taxes -- do not trust this blindly)
    test_ret = future_ret.loc[y_test.index]
    print("=== Avg 5-day return when model says BUY vs. always-buy ===")
    print(f"Always buy: {test_ret.mean():.4f}")
    for name, pred in results.items():
        if name.startswith("Baseline"):
            continue
        buys = test_ret[pred == 1]
        print(f"{name}: {buys.mean() if len(buys) else float('nan'):.4f} "
              f"({len(buys)} buy signals)")


if __name__ == "__main__":
    main()
