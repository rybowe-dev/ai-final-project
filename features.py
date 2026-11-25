"""
features.py - data loading, feature engineering, labels and dataset construction.

Leakage rules followed here
---------------------------
* Every feature at date t uses only prices/volume up to and including day t.
* The label at date t looks `horizon` trading days into the future, so the last
  `horizon` rows have no label and are never used for training or evaluation.
* The "most recent" feature row (whose label is still unknown) is kept separately
  and is only ever used for the final live prediction.
* The volatility regime used for analysis is defined with an *expanding* median,
  so the threshold at date t depends only on data available at date t.
"""

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_LABELED_ROWS = 500

FEATURE_DESCRIPTIONS = {
    "ret_1d": "1-day return: yesterday-to-today price change (very short-term momentum or reversal).",
    "ret_5d": "5-day return: about one trading week of momentum.",
    "ret_10d": "10-day return: about two weeks of momentum.",
    "ret_20d": "20-day return: about one month of momentum.",
    "sma10_ratio": "Close relative to its 10-day average. Positive means price is above its short-term trend.",
    "sma50_ratio": "Close relative to its 50-day average (medium-term trend).",
    "sma200_ratio": "Close relative to its 200-day average (long-term trend).",
    "vol_10d": "10-day standard deviation of daily returns (recent volatility).",
    "vol_30d": "30-day standard deviation of daily returns (slower volatility measure).",
    "rsi_14": "Relative Strength Index (Wilder, 14 days). Above ~70 is often read as overbought, below ~30 as oversold.",
    "macd_hist": "MACD histogram (MACD minus its signal line), scaled by price. Measures trend acceleration.",
    "volume_ratio": "Today's volume divided by its 20-day average. High values signal unusual activity.",
    "hl_range": "Intraday high-low range as a fraction of the close (a measure of daily volatility).",
}


# ------------------------------------------------------------------ data
def load_prices(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Download split/dividend-adjusted daily OHLCV data from Yahoo Finance."""
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No price data returned for '{ticker}'. Check the ticker symbol.")
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns MultiIndex columns
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated()].sort_index()
    return drop_incomplete_bar(df.replace([np.inf, -np.inf], np.nan).dropna())


def drop_incomplete_bar(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove today's bar if the US market has not finished trading.

    Yahoo returns a partial intraday bar for the current session; its close and
    volume are not final, so it must not be used as a feature row or label.
    The cutoff is 16:15 New York time (the 16:00 close plus a settling margin).
    """
    now = pd.Timestamp.now(tz="America/New_York") if now is None else now
    if now.tzinfo is None:
        now = now.tz_localize("America/New_York")
    now = now.tz_convert("America/New_York")
    if len(df) and df.index[-1].date() == now.date() and now.time() < dt.time(16, 15):
        return df.iloc[:-1]
    return df


# ------------------------------------------------------------- indicators
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (exponential smoothing with alpha = 1/n).

    Edge cases: no losses in the window -> 100; no movement at all -> 50.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    out = 100 - 100 / (1 + avg_gain / avg_loss)
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return out.where(avg_gain.notna() & avg_loss.notna())


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Technical-indicator features. Infinite values become NaN (dropped later)."""
    c = df["Close"]
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)
    f["ret_1d"] = r
    for n in (5, 10, 20):
        f[f"ret_{n}d"] = c.pct_change(n)
    for n in (10, 50, 200):
        f[f"sma{n}_ratio"] = c / c.rolling(n).mean() - 1
    f["vol_10d"] = r.rolling(10).std()
    f["vol_30d"] = r.rolling(30).std()
    f["rsi_14"] = rsi(c, 14)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    f["hl_range"] = (df["High"] - df["Low"]) / c
    return f.replace([np.inf, -np.inf], np.nan)


def volatility_regime(feats: pd.DataFrame, min_periods: int = 252) -> pd.Series:
    """'High'/'Low' volatility using only past data: is today's 30-day volatility
    above the median of all 30-day volatilities observed up to today?"""
    vol = feats["vol_30d"]
    threshold = vol.expanding(min_periods=min_periods).median()
    regime = pd.Series(np.where(vol > threshold, "High volatility", "Low volatility"), index=vol.index)
    return regime.where(vol.notna() & threshold.notna())


# ----------------------------------------------------------------- labels
def make_labels(close: pd.Series, horizon: int):
    """Label 1 if the close is strictly higher `horizon` trading days later, else 0.
    The last `horizon` rows are NaN because their future is unknown."""
    future_return = close.shift(-horizon) / close - 1
    y = (future_return > 0).astype(float).where(future_return.notna())
    return y, future_return


@dataclass
class Dataset:
    X: pd.DataFrame            # labeled feature rows (chronological)
    y: pd.Series               # 1 = price up after `horizon` days, 0 = not up
    future_return: pd.Series   # realized forward return (used for diagnostics only)
    regime: pd.Series          # volatility regime at each labeled date
    X_latest: pd.DataFrame     # newest feature row (label not yet known)
    horizon: int


def build_dataset(df: pd.DataFrame, horizon: int) -> Dataset:
    feats = make_features(df)
    valid = feats.dropna()
    y, future_return = make_labels(df["Close"], horizon)

    labeled = valid.join(y.rename("_y")).join(future_return.rename("_fut"))
    labeled = labeled.dropna(subset=["_y", "_fut"])
    if len(labeled) < MIN_LABELED_ROWS:
        raise ValueError(
            f"Only {len(labeled)} usable rows; need at least {MIN_LABELED_ROWS}. "
            "Use an earlier start date or a shorter horizon."
        )
    X = labeled[feats.columns]
    return Dataset(
        X=X,
        y=labeled["_y"].astype(int),
        future_return=labeled["_fut"],
        regime=volatility_regime(feats).reindex(X.index),
        X_latest=valid.iloc[[-1]],
        horizon=horizon,
    )