"""
Streamlit dashboard for the buy/sell classifiers.

Setup:  pip install streamlit yfinance pandas numpy scikit-learn
Run:    streamlit run app.py
(Keep this file in the same folder as predictor.py)
"""

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import predictor as core

st.set_page_config(page_title="Buy/Sell Classifier", layout="wide")
st.title("📈 Buy / Sell Classifier")
st.caption("Naive Bayes vs. Perceptron vs. MLP. Educational project, not financial advice.")

# ------------------------------------------------------------ sidebar
with st.sidebar:
    ticker = st.text_input("Ticker", "NVDA").upper().strip()
    start = st.date_input("History start", pd.Timestamp("2010-01-01")).isoformat()
    horizon = st.slider("Prediction horizon (trading days)", 1, 30, 5)
    test_frac = st.slider("Test set fraction", 0.1, 0.4, 0.2, 0.05)
    run = st.button("Train & evaluate", type="primary")


# -------------------------------------------------------------- cached
@st.cache_data(show_spinner="Downloading prices...")
def get_prices(ticker, start):
    return core.load_prices(ticker, start)


@st.cache_resource(show_spinner="Training models...")
def train_all(ticker, start, horizon, test_frac):
    df = get_prices(ticker, start)
    X_all = core.make_features(df)
    y_all, fut = core.make_labels(df, horizon)
    data = X_all.assign(y=y_all, fut=fut).replace([float("inf"), float("-inf")], pd.NA).dropna()
    X, y, fut = data[X_all.columns], data["y"].astype(int), data["fut"]

    X_tr, X_te, y_tr, y_te = core.time_split(X, y, test_frac, gap=horizon)
    models = core.get_models()
    preds = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        preds[name] = pd.Series(m.predict(X_te), index=X_te.index)

    # most recent row: features exist, but its label is still in the future
    latest = X_all.dropna().iloc[[-1]]
    latest_signals = {n: int(m.predict(latest)[0]) for n, m in models.items()}
    return df, y_te, fut.loc[y_te.index], preds, latest, latest_signals


if not run and "ran" not in st.session_state:
    st.info("Pick settings in the sidebar and click **Train & evaluate**.")
    st.stop()
st.session_state["ran"] = True

try:
    df, y_te, fut_te, preds, latest, signals = train_all(ticker, start, horizon, test_frac)
except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.stop()

# -------------------------------------------------------- price chart
st.subheader(f"{ticker} closing price")
st.line_chart(df["Close"])

# ---------------------------------------------------- latest signals
st.subheader(f"Latest signal ({latest.index[0].date()})")
cols = st.columns(len(signals) - 1)
for col, (name, s) in zip(cols, [(n, s) for n, s in signals.items() if not n.startswith("Baseline")]):
    col.metric(name, "BUY" if s == 1 else "SELL")

# ------------------------------------------------------------ metrics
st.subheader(f"Test-set performance ({y_te.index[0].date()} → {y_te.index[-1].date()})")
rows = []
for name, p in preds.items():
    buys = fut_te[p == 1]
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_te, p),
        "F1 (BUY)": f1_score(y_te, p, zero_division=0),
        f"Avg {horizon}d return on BUY signals": buys.mean() if len(buys) else float("nan"),
        "# BUY signals": len(buys),
    })
metrics = pd.DataFrame(rows).set_index("Model")
st.dataframe(metrics.style.format({
    "Accuracy": "{:.3f}", "F1 (BUY)": "{:.3f}",
    f"Avg {horizon}d return on BUY signals": "{:.2%}",
}), width="stretch")
st.caption(f"Always-buy average {horizon}-day return on the test period: {fut_te.mean():.2%}. "
           "Real trading would also face costs and slippage.")

# --------------------------------------------------- confusion matrices
st.subheader("Confusion matrices")
model_names = [n for n in preds if not n.startswith("Baseline")]
for col, name in zip(st.columns(len(model_names)), model_names):
    cm = pd.DataFrame(confusion_matrix(y_te, preds[name], labels=[0, 1]),
                      index=["Actual SELL", "Actual BUY"],
                      columns=["Pred SELL", "Pred BUY"])
    col.markdown(f"**{name}**")
    col.dataframe(cm)

# -------------------------------------------------- predictions over time
st.subheader("Model predictions on test set (1 = BUY)")
st.line_chart(pd.DataFrame({n: preds[n] for n in model_names}))