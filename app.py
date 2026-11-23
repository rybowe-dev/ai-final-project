"""
app.py - Streamlit dashboard.

Run:  streamlit run app.py

Models are trained from the console (python train.py). This dashboard only loads the
saved results from artifacts/, so it opens instantly and never retrains.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import evaluation
import features
import predictor
from predictor import BASELINE_NAME, MODEL_NAMES

UP, DOWN = "#16a34a", "#dc2626"
COLORS = {"Naive Bayes": "#2563eb", "Perceptron": "#f59e0b", "MLP": "#8b5cf6",
          BASELINE_NAME: "#9ca3af"}
METRIC_COLS = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1"]

st.set_page_config(page_title="Stock Direction Predictor", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); border-radius: 12px;
                             padding: 14px 18px; border: 1px solid rgba(128,128,128,0.2);}
</style>
""", unsafe_allow_html=True)


def style(fig, height=420):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(size=13), legend=dict(orientation="h", y=1.12, x=0))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


ARTIFACT_DIR = Path("artifacts")


@st.cache_resource(show_spinner="Loading saved results...")
def load_run(path: str, mtime: float):
    """Load a run saved by train.py. mtime is part of the cache key so re-training refreshes it."""
    return joblib.load(path)


# ----------------------------------------------------------------- header
st.title("📈 Stock Direction Predictor")
st.caption("Compares Gaussian Naive Bayes, a Perceptron and an MLP at forecasting whether a stock's "
           "price will be higher after a chosen number of trading days.")

files = sorted(ARTIFACT_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
if not files:
    st.warning("No trained results found. Train from the console first, then reload this page:")
    st.code("python train.py --ticker NVDA --horizon 5", language="bash")
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📁 Saved runs")
    chosen = st.selectbox("Run", files, format_func=lambda p: p.stem)
    run = load_run(str(chosen), chosen.stat().st_mtime)
    st.caption(f"Trained {run['created']}")
    st.caption("Create or refresh a run from the console:")
    st.code("python train.py --ticker NVDA --horizon 5", language="bash")
    st.caption("The dashboard only loads saved results; it never trains.")

res: evaluation.ExperimentResult = run["result"]
prices: pd.DataFrame = run["prices"]
tk, H = run["ticker"], res.horizon
st.markdown(
    f"**{tk}** · horizon **{H}** trading days · {res.n_labeled:,} labeled days · "
    f"train {res.train_range[0]:%Y-%m-%d} → {res.train_range[1]:%Y-%m-%d} · "
    f"test {res.test_range[0]:%Y-%m-%d} → {res.test_range[1]:%Y-%m-%d} · "
    f"{'tuned' if res.tuned else 'default hyperparameters'}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Comparison", "Feature Analysis", "Market Conditions"])

# =================================================================== TAB 1
with tab1:
    st.subheader(f"Latest predicted direction ({res.latest_date:%Y-%m-%d})")
    cols = st.columns(len(MODEL_NAMES) + 1)
    fp = res.final_predictions
    for col, name in zip(cols, MODEL_NAMES):
        up = fp.loc[name, "Predicted direction"] == 1
        p = fp.loc[name, "P(up)"]
        col.metric(name, "▲ Up" if up else "▼ Down",
                   delta=f"P(up) = {p:.0%}" if not np.isnan(p) else "no probability output",
                   delta_color="off")
    votes = int(fp["Predicted direction"].sum())
    cols[-1].metric("Model agreement", f"{votes} of {len(MODEL_NAMES)} say Up",
                    delta=f"over next {H} trading days", delta_color="off")
    st.caption("These are model forecasts of price **direction**, not BUY/SELL recommendations. "
               "Final models are trained on all labeled history; the evaluation shown in the other tabs "
               "comes from separate models that never saw the test period.")

    st.subheader("Price and moving averages")
    c1, c2 = st.columns([2, 3])
    smas = c1.multiselect("Moving averages", [20, 50, 100, 200], default=[50, 200])
    span = c2.radio("Range", ["6M", "1Y", "3Y", "5Y", "Max"], index=2, horizontal=True)
    days = {"6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "Max": len(prices)}[span]
    view = prices.iloc[-days:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="Close", line=dict(color="#111827", width=2)), 1, 1)
    palette = ["#2563eb", "#f59e0b", "#8b5cf6", "#ef4444"]
    for color, n in zip(palette, smas):
        fig.add_trace(go.Scatter(x=view.index, y=prices["Close"].rolling(n).mean().loc[view.index],
                                 name=f"SMA {n}", line=dict(color=color, width=1.4)), 1, 1)
    fig.add_trace(go.Bar(x=view.index, y=view["Volume"], name="Volume", marker_color="#cbd5e1"), 2, 1)
    fig.update_yaxes(title_text="Price (adj.)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    show(style(fig, 520))

    st.subheader("Technical indicators")
    feats_all = features.make_features(prices)
    left, right = st.columns([3, 2])
    with left:
        rs = feats_all["rsi_14"].loc[view.index]
        rf = go.Figure(go.Scatter(x=rs.index, y=rs, line=dict(color="#2563eb"), name="RSI (14)"))
        rf.add_hrect(y0=70, y1=100, fillcolor=DOWN, opacity=0.08, line_width=0)
        rf.add_hrect(y0=0, y1=30, fillcolor=UP, opacity=0.08, line_width=0)
        rf.update_yaxes(range=[0, 100], title_text="RSI")
        show(style(rf, 300))
    with right:
        latest = res.latest_features.iloc[0]
        table = pd.DataFrame({"Value": latest,
                              "Meaning": [features.FEATURE_DESCRIPTIONS[f] for f in latest.index]})
        table["Value"] = table["Value"].map(lambda v: f"{v:.4f}")
        st.dataframe(table, height=300, width="stretch")

# =================================================================== TAB 2
with tab2:
    st.subheader("Hold-out test performance")
    st.caption(f"Test class balance: {res.test_up_share:.0%} of days were followed by a price increase. "
               "The majority-class baseline always predicts the most common training class, so a model "
               "must beat it (and 0.5 balanced accuracy) to show any skill.")
    mt = res.metrics.loc[[BASELINE_NAME] + MODEL_NAMES, METRIC_COLS]
    st.dataframe(mt.style.format("{:.3f}").highlight_max(axis=0, color="#dcfce7"), width="stretch")

    long = mt.reset_index(names="Model").melt("Model", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_map=COLORS, range_y=[0, 1])
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    show(style(fig, 400))

    st.subheader("Walk-forward cross-validation (training period)")
    cvm = res.cv_scores.copy()
    summary = pd.DataFrame({"Mean": cvm.mean(), "Std": cvm.std()}).T
    fig = go.Figure()
    for name in [BASELINE_NAME] + MODEL_NAMES:
        fig.add_trace(go.Scatter(x=cvm.index, y=cvm[name], name=name, mode="lines+markers",
                                 line=dict(color=COLORS[name])))
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    fig.update_yaxes(title_text="Balanced accuracy")
    show(style(fig, 340))
    c1, c2 = st.columns(2)
    c1.dataframe(summary.style.format("{:.3f}"), width="stretch")
    c2.dataframe(pd.DataFrame({"Selected hyperparameters": {n: predictor.format_params(p)
                                                            for n, p in res.best_params.items()}}),
                 width="stretch")
    st.caption("Each fold trains on the past and validates on the next block (expanding window), with a "
               f"{H}-day gap so labels never overlap. Scores are used to choose hyperparameters, so they are "
               "slightly optimistic; the hold-out test above is the unbiased estimate.")

    st.subheader("Confusion matrices (test set)")
    for col, name in zip(st.columns(len(MODEL_NAMES)), MODEL_NAMES):
        cm = pd.DataFrame(res.confusion[name], index=["Actual Down", "Actual Up"],
                          columns=["Pred Down", "Pred Up"])
        f = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="equal", title=name)
        f.update_coloraxes(showscale=False)
        with col:
            show(style(f, 330))

    st.subheader("Actual vs. predicted direction")
    rows = ["Actual"] + MODEL_NAMES
    z = res.predictions[rows].T.to_numpy()
    hm = go.Figure(go.Heatmap(z=z, x=res.predictions.index, y=rows, zmin=0, zmax=1, showscale=False,
                              colorscale=[[0, DOWN], [0.5, DOWN], [0.5, UP], [1, UP]], ygap=3,
                              text=np.where(z == 1, "Up", "Down"),
                              hovertemplate="%{y}<br>%{x|%Y-%m-%d}: %{text}<extra></extra>"))
    hm.update_yaxes(autorange="reversed")
    show(style(hm, 300))
    st.caption("🟩 price higher after the horizon · 🟥 not higher. Where a model's row matches the Actual row, it was right.")

# =================================================================== TAB 3
with tab3:
    st.subheader("Permutation feature importance")
    st.caption("Computed on the hold-out test set: each feature is shuffled in turn and we measure how much "
               "balanced accuracy drops. A larger drop means the model relies on that feature more.")
    model = st.radio("Model", MODEL_NAMES, horizontal=True)
    imp = res.importance[model].sort_values("Importance")
    fig = go.Figure(go.Bar(x=imp["Importance"], y=imp.index, orientation="h", marker_color=COLORS[model],
                           error_x=dict(type="data", array=imp["Std"], color="#6b7280")))
    fig.update_xaxes(title_text="Drop in balanced accuracy when shuffled")
    show(style(fig, 460))

    top = res.importance[model].head(3)
    st.markdown("**Most influential indicators for this model**")
    for f in top.index:
        st.markdown(f"- `{f}` (importance {top.loc[f, 'Importance']:.3f}): {features.FEATURE_DESCRIPTIONS[f]}")

    st.subheader("Importance across all models")
    allimp = pd.DataFrame({m: res.importance[m]["Importance"] for m in MODEL_NAMES})
    allimp = allimp.loc[allimp.mean(axis=1).sort_values(ascending=False).index]
    show(style(px.imshow(allimp, text_auto=".3f", aspect="auto", color_continuous_scale="Blues"), 480))
    with st.expander("How to interpret this"):
        st.markdown(
            "- Values near **zero** mean shuffling the feature barely changes performance: the model does not use it.\n"
            "- **Negative** values mean the shuffled feature happened to score slightly better, which is noise.\n"
            "- Several features here are correlated (e.g. the moving-average ratios and multi-day returns), so "
            "importance can be shared between them and each looks smaller than the group really is.\n"
            "- Importance describes what the *model* uses, not what causes price moves.")

# =================================================================== TAB 4
with tab4:
    st.subheader("Performance by volatility regime")
    st.caption("A test day is *High volatility* when its 30-day return volatility is above the median of all "
               "30-day volatilities seen up to that day (computed with past data only).")
    rm = res.regime_metrics
    if rm.empty:
        st.warning("Not enough data to split the test period into volatility regimes.")
    else:
        fig = px.bar(rm, x="Model", y="Balanced Accuracy", color="Regime", barmode="group",
                     color_discrete_map={"Low volatility": "#38bdf8", "High volatility": "#f97316"},
                     category_orders={"Model": [BASELINE_NAME] + MODEL_NAMES}, range_y=[0, 1])
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
        show(style(fig, 400))
        counts = rm.drop_duplicates("Regime")[["Regime", "Samples", "Up share"]].set_index("Regime")
        st.dataframe(counts.style.format({"Up share": "{:.1%}"}), width="stretch")
        st.caption("Regime subsets are smaller than the full test set, so differences can easily be noise.")

    st.subheader("Effect of the prediction horizon")
    hm_ = res.horizon_metrics
    fig = px.line(hm_, x="Horizon", y="Balanced Accuracy", color="Model", markers=True,
                  color_discrete_map=COLORS)
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    fig.update_xaxes(title_text="Horizon (trading days)", type="category")
    show(style(fig, 400))
    piv = hm_.pivot(index="Horizon", columns="Model", values="Accuracy")[[BASELINE_NAME] + MODEL_NAMES]
    st.markdown("**Accuracy by horizon**")
    st.dataframe(piv.style.format("{:.3f}"), width="stretch")
    st.caption(f"Hyperparameters tuned at the {H}-day horizon are re-used for every horizon (no re-tuning). "
               "Longer horizons make neighbouring labels overlap heavily, so the effective number of independent "
               "test observations is much smaller than the row count.")

st.divider()
st.caption("Research and education only. Directional accuracy on historical data is not evidence of "
           "profitability: no transaction costs, slippage or position sizing are modelled, and forward "
           "returns over overlapping windows are deliberately not presented as trading profits.")
