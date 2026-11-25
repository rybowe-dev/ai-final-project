"""
experiments_view.py - Streamlit views for the saved experiments in runs/ (made by experiment.py).

Only reads the saved CSV/JSON files through run_results.py; it never trains or re-runs anything.
Called from app.py via render().
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import run_results as rr
from predictor import BASELINE_NAME, MODEL_NAMES

COLORS = {"Naive Bayes": "#2563eb", "Perceptron": "#f59e0b", "MLP": "#8b5cf6", BASELINE_NAME: "#9ca3af"}
RUN_PALETTE = ["#2563eb", "#ef4444", "#16a34a", "#f59e0b", "#8b5cf6", "#0891b2", "#db2777", "#65a30d"]
REGIME_COLORS = {"Low volatility": "#38bdf8", "High volatility": "#f97316"}
# Plotly's camera button saves a high-resolution PNG, suitable for a paper
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3}}
SMALL_REGIME_ROWS = 250  # about one trading year of test days
WINDOW_METRICS = ["Accuracy", "Balanced Accuracy", "MCC", "AUC"]
RUN_COMMAND = "python experiment.py run --name <new_name> --horizon 5 --n-jobs 8"


# ---------------------------------------------------------------- helpers
def style(fig, height=420):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(size=13), legend=dict(orientation="h", y=1.12, x=0))
    return fig


def show(fig, key):
    st.plotly_chart(fig, width="stretch", key=key, config=PLOT_CONFIG)


def download(df: pd.DataFrame, filename: str, key: str, index: bool = True):
    st.download_button("⬇ Download CSV", df.to_csv(index=index).encode(), file_name=filename,
                       mime="text/csv", key=key)


def fmt_table(df: pd.DataFrame, signed=()):
    """Style numeric columns to 3 decimals (signed ones with +/-); missing values become 'n/a'."""
    formats = {c: ("{:+.3f}" if c in signed else "{:.3f}") for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c])}
    return df.style.format(formats, na_rep="n/a")


def ci_text(v, lo, hi, signed=False) -> str:
    if pd.isna(v):
        return "n/a"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def banner(level: str, text: str):
    {"ok": st.success, "info": st.info, "warning": st.warning, "error": st.error}[level](text)


def run_title(run: rr.RunData) -> str:
    c = run.config
    return f"{run.name} · {c.get('ticker')} · {c.get('horizon')}-day horizon"


@st.cache_data(show_spinner="Loading saved experiments...")
def load_runs(runs_dir: str, signature: tuple):
    """Cached by directory contents: re-running experiment.py changes the signature and refreshes it."""
    return rr.discover_runs(Path(runs_dir))


def forest(tables: dict[str, pd.DataFrame], xtitle: str, key: str, ref: float | None = None,
           ref_label: str = "", height: int = 340, exclude=()):
    """Point estimates with horizontal CI bars, one row per model (one marker per run if several)."""
    models = [m for m in rr.ALL_NAMES if m not in exclude and any(m in t.index for t in tables.values())]
    n, fig = len(tables), go.Figure()
    for i, (name, t) in enumerate(tables.items()):
        rows = [m for m in models if m in t.index and pd.notna(t.loc[m, "Value"])]
        if not rows:
            continue
        v = t.loc[rows, "Value"].to_numpy(float)
        lo, hi = t.loc[rows, "CI low"].to_numpy(float), t.loc[rows, "CI high"].to_numpy(float)
        color = RUN_PALETTE[i % len(RUN_PALETTE)] if n > 1 else None
        fig.add_trace(go.Scatter(
            x=v, y=[models.index(m) + (i - (n - 1) / 2) * 0.25 for m in rows], mode="markers",
            name=name if n > 1 else "Estimate",
            marker=dict(size=11, color=color or [COLORS[m] for m in rows], line=dict(width=1, color="white")),
            error_x=dict(type="data", symmetric=False, array=np.nan_to_num(hi - v),
                         arrayminus=np.nan_to_num(v - lo), color=color or "#9ca3af", thickness=2, width=6),
            customdata=np.c_[lo, hi, rows],
            hovertemplate="%{customdata[2]}<br>%{x:.3f} (95% CI %{customdata[0]:.3f} to %{customdata[1]:.3f})"
                          "<extra>" + name + "</extra>"))
    if ref is not None:
        fig.add_vline(x=ref, line_dash="dot", line_color="gray", annotation_text=ref_label,
                      annotation_position="top")
    fig.update_yaxes(tickvals=list(range(len(models))), ticktext=models, autorange="reversed")
    fig.update_xaxes(title_text=xtitle, zeroline=False)
    fig.update_layout(showlegend=n > 1)
    show(style(fig, height), key)


# ----------------------------------------------------------------- render
def render(runs_dir: Path = rr.RUNS_DIR):
    st.subheader("Experiments: walk-forward evaluation")
    st.caption("Results saved by `experiment.py` under `runs/`. This page only reads those files; "
               "changing a selection never trains or re-runs anything.")

    runs, problems = load_runs(str(runs_dir), rr.runs_signature(runs_dir))
    with st.sidebar:
        st.header("🧪 Experiments")
        if runs:
            by_name = {r.name: r for r in runs}
            chosen = st.selectbox("Experiment", list(by_name), format_func=lambda n: run_title(by_name[n]))
            st.caption(f"{len(runs)} experiment(s) found in `{runs_dir}/`")
        st.caption("Create a new experiment from the console:")
        st.code(RUN_COMMAND, language="bash")
        if problems:
            with st.expander(f"⚠ {len(problems)} skipped folder(s)"):
                for name, why in problems.items():
                    st.markdown(f"- `{name}`: {why}")

    if not runs:
        st.warning(f"No completed experiments found in `{runs_dir}/`. Run one from the console, then reload "
                   "this page:")
        st.code(RUN_COMMAND, language="bash")
        return
    run = by_name[chosen]

    tabs = st.tabs(["Summary", "Model performance", "Walk-forward windows", "Volatility regimes",
                    "Compare experiments"])
    with tabs[0]:
        summary_tab(run)
    with tabs[1]:
        performance_tab(run)
    with tabs[2]:
        walk_forward_tab(run)
    with tabs[3]:
        regime_tab(run)
    with tabs[4]:
        compare_tab(runs)
    st.caption("Research and education only. Direction accuracy on historical data is not evidence of "
               "profitability; no trading returns, costs or slippage are modelled.")


# ------------------------------------------------------------------ tab 1
def summary_tab(run: rr.RunData):
    c, m = run.config, run.manifest
    data = m["data"]
    st.markdown(f"### {run_title(run)}")
    cols = st.columns(5)
    cols[0].metric("Ticker", c["ticker"])
    cols[1].metric("Horizon", f"{c['horizon']} days")
    cols[2].metric("Windows", c.get("n_windows", len(m["windows"])))
    cols[3].metric("Seeds", c.get("n_seeds", "n/a"))
    cols[4].metric("Tuning", "On" if c.get("tune") else "Off")
    st.markdown(f"**Data:** {data.get('first')} → {data.get('last')} ({data.get('rows', 'n/a'):,} daily rows) · "
                f"**Created:** {m.get('created', 'n/a')} · **Runtime:** {m.get('runtime_seconds', 'n/a')} s")
    for note in run.notes:
        st.caption(f"ℹ {note}")

    st.markdown("#### Headline results")
    st.caption("Pooled out-of-sample predictions from every test window; Perceptron and MLP are averaged over "
               "seeds. The 95% interval is a moving-block bootstrap (see *Model performance*).")
    s = run.summary
    table = pd.DataFrame(index=[n for n in rr.ALL_NAMES if n in s.index])
    if run.has_ci:
        table["Accuracy [95% CI]"] = [ci_text(*s.loc[n, ["Accuracy", "Acc CI low", "Acc CI high"]]) for n in table.index]
        table["Balanced accuracy [95% CI]"] = [ci_text(*s.loc[n, ["Balanced Accuracy", "BA CI low", "BA CI high"]])
                                               for n in table.index]
        table["Accuracy − baseline [95% CI]"] = [
            ci_text(*s.loc[n, ["Acc - baseline", "Diff CI low", "Diff CI high"]], signed=True) for n in table.index]
    else:
        for col in ["Accuracy", "Balanced Accuracy", "Acc - baseline"]:
            if col in s.columns:
                table[col] = s.loc[table.index, col].map("{:.3f}".format)
    if "Windows beating baseline" in s.columns and "Windows" in s.columns:
        table["Windows beating baseline"] = [f"{int(s.loc[n, 'Windows beating baseline'])}/{int(s.loc[n, 'Windows'])}"
                                             for n in table.index]
    if run.has_ci:
        table = table.join(rr.verdicts(s)).fillna("— (reference)")
    st.dataframe(table, width="stretch")
    download(table, f"{run.name}_headline.csv", f"dl-headline-{run.name}")

    st.markdown("#### Test windows")
    windows = pd.DataFrame([w[:3] for w in m["windows"] if len(w) == 3], columns=["Window", "Test start", "Test end"])
    info = rr.window_info(run.metrics)
    if not windows.empty and not info.empty:
        windows = windows.join(info[[x for x in ["Train rows", "Test rows", "Test up share"] if x in info]],
                               on="Window")
    st.dataframe(fmt_table(windows), hide_index=True, width="stretch")
    st.caption("The first part of the history is training-only. Each window trains on everything before it "
               "(minus a horizon-length gap so no label reaches into the test block) and is tested on the next "
               "chronological block.")

    left, right = st.columns(2)
    with left, st.expander("Configuration (config.json)"):
        st.json(run.config)
    with right, st.expander("Reproducibility (manifest.json)"):
        git = m.get("git", {})
        st.json({"data snapshot": data.get("snapshot"), "data sha256": data.get("sha256"),
                 "config digest": m.get("config_digest"),
                 "git commit": f"{(git.get('commit') or 'n/a')[:10]}{' (uncommitted changes)' if git.get('dirty') else ''}",
                 "environment": m.get("environment")})
    if run.best_params is not None and {"Window", "Model", "params"} <= set(run.best_params.columns):
        with st.expander("Selected hyperparameters per window"):
            bp = run.best_params.pivot(index="Window", columns="Model", values="params")
            st.dataframe(bp, width="stretch")
            download(bp, f"{run.name}_best_params.csv", f"dl-bp-{run.name}")
    if run.report:
        with st.expander("Full text report (report.txt)"):
            st.code(run.report, language=None)


# ------------------------------------------------------------------ tab 2
def performance_tab(run: rr.RunData):
    s = run.summary
    st.markdown("### Measured performance")
    st.caption("These are the scores actually observed on the out-of-sample test windows. "
               "They say nothing yet about how much they could change by chance (see the next section).")

    pooled_cols = [c for c in ["Accuracy", "Balanced Accuracy", "MCC", "AUC (window mean)", "Predicted up share"]
                   if c in s.columns]
    st.markdown("**Pooled over all test days** (one set of predictions; stochastic models averaged over seeds)")
    pooled = s.loc[[n for n in rr.ALL_NAMES if n in s.index], pooled_cols]
    st.dataframe(fmt_table(pooled, signed={"MCC"}), width="stretch")
    download(pooled, f"{run.name}_pooled_metrics.csv", f"dl-pooled-{run.name}")

    fit = rr.per_fit_metrics(run.metrics)
    if fit.empty:
        st.info("metrics.csv is unavailable, so precision, recall and F1 cannot be shown.")
    else:
        st.markdown("**Average of the per-window scores** (includes precision, recall and F1)")
        st.dataframe(fmt_table(fit, signed={"MCC"}).highlight_max(axis=0, color="#dcfce7"), width="stretch")
        st.caption("Each score is computed inside a window (seed-averaged) and then averaged across windows, so "
                   "it can differ slightly from the pooled table. Precision and recall refer to predicting "
                   "*Up*; the baseline always predicts Up, so its recall is 1. AUC is the mean over windows "
                   "(0.5 = chance). Highlighted = best in column.")
        download(fit, f"{run.name}_per_window_average_metrics.csv", f"dl-fit-{run.name}")

        metric = st.selectbox("Metric to plot", list(fit.columns),
                              index=list(fit.columns).index("Balanced Accuracy") if "Balanced Accuracy" in fit else 0,
                              key=f"perf-metric-{run.name}")
        ref = {"Balanced Accuracy": 0.5, "AUC": 0.5, "MCC": 0.0}.get(metric)
        vals = fit[metric].dropna()
        fig = px.bar(vals.rename("Score").reset_index(), x="Model", y="Score", color="Model", text_auto=".3f",
                     color_discrete_map=COLORS, title=f"{metric}: average of per-window scores")
        if ref is not None:
            fig.add_hline(y=ref, line_dash="dot", line_color="gray",
                          annotation_text="chance" if metric != "MCC" else "no association")
        fig.update_layout(showlegend=False)
        show(style(fig, 380), f"perf-bar-{run.name}-{metric}")

    st.markdown("### Statistical uncertainty")
    if not run.has_ci:
        st.warning("summary.csv lacks the bootstrap confidence-interval columns, so no uncertainty can be shown.")
        return
    c = run.config
    st.caption(f"Bars show **95% moving-block bootstrap intervals** ({c.get('n_bootstrap', 'n/a')} resamples). "
               "Resampling test days in blocks keeps the autocorrelation created by overlapping labels. "
               "The interval describes how much the pooled score would vary if this test period were "
               "re-drawn; it does **not** cover future market behaviour, and no correction is made for "
               "comparing several models.")
    quantity = st.radio("Quantity", ["Balanced Accuracy", "Accuracy"], horizontal=True, key=f"ci-q-{run.name}")
    ref = 0.5 if quantity == "Balanced Accuracy" else float(s.loc[BASELINE_NAME, "Accuracy"]) \
        if BASELINE_NAME in s.index else None
    forest({run.name: rr.ci_table(run, quantity)}, quantity, f"forest-{run.name}-{quantity}", ref=ref,
           ref_label="chance (0.5)" if quantity == "Balanced Accuracy" else "majority baseline")
    st.caption("If an interval contains the dotted line, that model is not distinguishable from "
               + ("a coin flip." if quantity == "Balanced Accuracy" else "always predicting the majority class."))

    st.markdown("**Accuracy minus the majority-class baseline**")
    forest({run.name: rr.ci_table(run, "Accuracy - baseline")}, "Accuracy − baseline", f"forest-diff-{run.name}",
           ref=0.0, ref_label="no difference", exclude=(BASELINE_NAME,), height=280)
    st.caption("Positive = better than always predicting the majority class. An interval that excludes 0 marks "
               "a difference unlikely to be bootstrap noise.")
    verdict = rr.verdicts(s)
    st.dataframe(verdict, width="stretch")
    download(verdict, f"{run.name}_ci_reading.csv", f"dl-verdict-{run.name}")

    ci = s.loc[[n for n in rr.ALL_NAMES if n in s.index],
               [col for cols in rr.CI_COLUMNS.values() for col in cols]]
    with st.expander("Interval values"):
        st.dataframe(fmt_table(ci, signed={"Acc - baseline", "Diff CI low", "Diff CI high"}), width="stretch")
        download(ci, f"{run.name}_confidence_intervals.csv", f"dl-ci-{run.name}")


# ------------------------------------------------------------------ tab 3
def _window_figure(table: pd.DataFrame, labels: dict, ylabel: str, ref: float | None, key: str):
    x = [labels.get(int(w), f"W{w}") for w in table.index]
    fig = go.Figure()
    for name in table.columns:
        fig.add_trace(go.Scatter(x=x, y=table[name], name=name, mode="lines+markers",
                                 line=dict(color=COLORS[name])))
    if ref is not None:
        fig.add_hline(y=ref, line_dash="dot", line_color="gray")
    fig.update_yaxes(title_text=ylabel)
    show(style(fig, 380), key)


def walk_forward_tab(run: rr.RunData):
    st.markdown("### Performance across chronological test windows")
    st.caption("Each window is a later period of the history. A model with real skill should beat the baseline "
               "in most windows, not just in one lucky period.")
    options = [m for m in WINDOW_METRICS if not rr.window_metric(run, m).empty]
    if not options:
        st.info("window_summary.csv is unavailable or has no usable columns.")
        return
    choices = options + (["Accuracy − baseline"] if "Accuracy" in options else [])
    metric = st.radio("Metric", choices, horizontal=True, key=f"wf-metric-{run.name}")
    if metric == "Accuracy − baseline":
        acc = rr.window_metric(run, "Accuracy")
        table = acc.drop(columns=BASELINE_NAME, errors="ignore").sub(acc[BASELINE_NAME], axis=0) \
            if BASELINE_NAME in acc else pd.DataFrame()
        ref = 0.0
    else:
        table, ref = rr.window_metric(run, metric), {"Balanced Accuracy": 0.5, "AUC": 0.5, "MCC": 0.0}.get(metric)
    if table.empty:
        st.info("The baseline is missing from window_summary.csv, so differences cannot be computed.")
        return
    labels = rr.window_labels(run.manifest)
    _window_figure(table, labels, metric, ref, f"wf-line-{run.name}-{metric}")

    heat = table.copy()
    heat.index = [labels.get(int(w), f"W{w}") for w in heat.index]
    fig = px.imshow(heat.T, text_auto=".3f", aspect="auto", color_continuous_scale="RdBu",
                    color_continuous_midpoint=ref if ref is not None else heat.stack().median(),
                    title=f"{metric} by window and model")
    fig.update_xaxes(side="top")
    fig.update_coloraxes(showscale=False)
    show(style(fig, 100 + 70 * len(heat.columns)), f"wf-heat-{run.name}-{metric}")
    if ref is not None:
        st.caption(f"Blue = above {'0' if ref == 0 else ref}, red = below. "
                   "Look for colour that stays on one side across windows rather than a single strong cell.")

    st.markdown("**Consistency across windows**")
    cons = pd.DataFrame({"Mean": table.mean(), "SD": table.std(ddof=1), "Worst window": table.min(),
                         "Best window": table.max()})
    if ref is not None:
        cons[f"Windows above {'0' if ref == 0 else ref}"] = (table > ref).sum().astype(int).astype(str) + f"/{len(table)}"
    st.dataframe(fmt_table(cons, signed={"Mean", "Worst window", "Best window"} if ref == 0 else ()),
                 width="stretch")
    download(table.rename(index=labels), f"{run.name}_windows_{metric.replace(' ', '_')}.csv", f"dl-win-{run.name}-{metric}")
    st.caption("A large SD or a best window far from the others means the average is carried by particular periods.")

    info = rr.window_info(run.metrics)
    if not info.empty:
        with st.expander("Window sizes and class balance"):
            st.dataframe(fmt_table(info.rename(index=labels)), width="stretch")
            st.caption("*Test up share* is the fraction of days followed by a price increase; it is the accuracy "
                       "of the majority baseline when that is the training majority.")

    seed_section(run)

    if run.grid is not None:
        with st.expander("Hyperparameter search results (grid_results.csv)"):
            st.dataframe(run.grid, hide_index=True, width="stretch")
            download(run.grid, f"{run.name}_grid_results.csv", f"dl-grid-{run.name}", index=False)


def seed_section(run: rr.RunData):
    st.markdown("### Variation across random seeds")
    metrics = [m for m in ["Balanced Accuracy", "Accuracy", "MCC", "AUC"]
               if not rr.seed_metrics(run.metrics, m).empty]
    if not metrics:
        st.info("Seed variation is unavailable: the run has a single seed or no metrics.csv.")
        return
    metric = st.radio("Metric", metrics, horizontal=True, key=f"seed-metric-{run.name}")
    seeds, labels = rr.seed_metrics(run.metrics, metric), rr.window_labels(run.manifest)
    seeds = seeds.assign(Period=seeds["Window"].map(lambda w: labels.get(int(w), f"W{w}"))).sort_values("Window")
    fig = px.box(seeds, x="Period", y=metric, color="Model", points="all", color_discrete_map=COLORS)
    fig.update_layout(boxmode="group")
    if metric != "Accuracy":
        fig.add_hline(y=0.0 if metric == "MCC" else 0.5, line_dash="dot", line_color="gray")
    show(style(fig, 400), f"seed-box-{run.name}-{metric}")
    st.caption(f"Each dot is one seed ({seeds['Seed'].nunique()} seeds). Naive Bayes and the baseline are "
               "deterministic and therefore have no spread. A box that straddles the dotted line means "
               "the result depends on the seed as much as on the model.")
    spread = rr.seed_spread(run.metrics, metric)
    st.dataframe(fmt_table(spread.rename(index=labels, level="Window")), width="stretch")
    download(spread.reset_index(), f"{run.name}_seed_spread_{metric.replace(' ', '_')}.csv",
             f"dl-seed-{run.name}-{metric}", index=False)
    conv = rr.convergence(run.metrics)
    if not conv.empty:
        with st.expander("Training convergence"):
            st.dataframe(conv.style.format({"Fits converged": "{:.0%}", "Fits": "{:.0f}"}), width="stretch")
            st.caption("Share of fits that stopped before reaching `max_iter`. Low values mean the optimiser "
                       "did not fully converge, which can add to seed-to-seed variation.")


# ------------------------------------------------------------------ tab 4
def regime_tab(run: rr.RunData):
    st.markdown("### Performance in high- and low-volatility periods")
    st.caption("A test day is *High volatility* when its 30-day return volatility is above the median of all "
               "values seen up to that day (past data only). Scores are pooled over the test windows.")
    reg = run.regime_summary
    needed = {"Regime", "Model", "Rows", "Up share", "Accuracy", "Balanced Accuracy"}
    if reg is None or reg.empty or not needed <= set(reg.columns):
        st.info("No volatility-regime results were saved for this experiment (a regime needs enough test days "
                "of both classes).")
        return
    metric = st.radio("Metric", ["Balanced Accuracy", "Accuracy"], horizontal=True, key=f"reg-metric-{run.name}")
    fig = px.bar(reg, x="Model", y=metric, color="Regime", barmode="group", text_auto=".3f",
                 color_discrete_map=REGIME_COLORS,
                 category_orders={"Model": rr.ALL_NAMES, "Regime": list(REGIME_COLORS)}, range_y=[0, 1])
    if metric == "Balanced Accuracy":
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="chance")
    show(style(fig, 420), f"reg-bar-{run.name}-{metric}")

    st.markdown("**Sample sizes**")
    sizes = reg.drop_duplicates("Regime").set_index("Regime")[["Rows", "Up share"]]
    sizes["Reliability"] = np.where(sizes["Rows"] < SMALL_REGIME_ROWS,
                                    f"Small (< {SMALL_REGIME_ROWS} days)", "Moderate")
    st.dataframe(sizes.style.format({"Up share": "{:.1%}", "Rows": "{:,.0f}"}), width="stretch")
    st.caption("*Rows* counts test days, but neighbouring days share most of their forward window, so the "
               "number of independent observations is smaller than shown. In the volatility regime with fewer "
               "days, differences between models are especially easy to produce by chance.")

    table = reg.pivot(index="Model", columns="Regime", values=metric).reindex(rr.ALL_NAMES).dropna(how="all")
    if {"Low volatility", "High volatility"} <= set(table.columns):
        table["High − Low"] = table["High volatility"] - table["Low volatility"]
    st.markdown(f"**{metric} by regime**")
    st.dataframe(fmt_table(table, signed={"High − Low"}), width="stretch")
    st.caption("The regime results carry no bootstrap intervals, so a difference between regimes or models here "
               "should be read as descriptive, not as evidence of a real effect.")
    download(reg, f"{run.name}_regime_summary.csv", f"dl-regime-{run.name}", index=False)


# ------------------------------------------------------------------ tab 5
def compare_tab(runs: list[rr.RunData]):
    st.markdown("### Compare experiments")
    if len(runs) < 2:
        st.info("Only one experiment exists. Run another (for example a different horizon or ticker) to compare.")
        return
    by_name = {r.name: r for r in runs}
    picked = st.multiselect("Experiments to compare", list(by_name), default=list(by_name)[:2],
                            format_func=lambda n: run_title(by_name[n]), key="compare-pick")
    if len(picked) < 2:
        st.info("Select at least two experiments.")
        return
    chosen = [by_name[n] for n in picked]

    for level, text in rr.comparability(chosen):
        banner(level, text)
    diffs = rr.config_differences(chosen)
    st.markdown("**Settings that differ**")
    if diffs.empty:
        st.caption("None: configuration, data and test windows are identical.")
    else:
        st.dataframe(diffs, width="stretch")
        download(diffs, "experiment_differences.csv", "dl-diffs")

    usable = [r for r in chosen if r.has_ci]
    for r in (r for r in chosen if not r.has_ci):
        st.warning(f"`{r.name}` lacks confidence-interval columns and is left out of the charts.")
    if len(usable) >= 2:
        quantity = st.radio("Quantity", ["Balanced Accuracy", "Accuracy - baseline", "Accuracy"], horizontal=True,
                            key="compare-q")
        refs = {"Balanced Accuracy": (0.5, "chance (0.5)"), "Accuracy - baseline": (0.0, "no difference"),
                "Accuracy": (None, "")}
        ref, label = refs[quantity]
        forest({r.name: rr.ci_table(r, quantity) for r in usable}, quantity.replace(" - ", " − "),
               f"compare-forest-{'-'.join(picked)}-{quantity}", ref=ref, ref_label=label,
               exclude=(BASELINE_NAME,) if quantity == "Accuracy - baseline" else (),
               height=260 + 90 * len(usable))
        st.caption("Markers are pooled scores with 95% bootstrap intervals. Overlapping intervals mean the "
                   "difference between experiments is not reliably distinguishable from resampling noise; "
                   "the intervals come from different resamples, so a formal difference test would need a "
                   "paired analysis.")

    rows = []
    for r in chosen:
        s = r.summary
        for model in [n for n in rr.ALL_NAMES if n in s.index]:
            row = s.loc[model]
            entry = {"Experiment": r.name, "Model": model}
            for col in ["Accuracy", "Balanced Accuracy", "Acc - baseline", "MCC", "AUC (window mean)"]:
                entry[col] = row.get(col, np.nan)
            rows.append(entry)
    table = pd.DataFrame(rows).set_index(["Model", "Experiment"]).sort_index(
        key=lambda idx: idx.map(rr.ALL_NAMES.index) if idx.name == "Model" else idx)
    st.markdown("**Pooled metrics**")
    st.dataframe(fmt_table(table, signed={"Acc - baseline", "MCC"}), width="stretch")
    download(table.reset_index(), "experiment_comparison.csv", "dl-compare", index=False)
