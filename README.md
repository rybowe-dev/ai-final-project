# Stock Direction Predictor

Compares three machine learning models (**Gaussian Naive Bayes**, **Perceptron**, **MLP**) at forecasting whether a stock's price will be higher after a chosen number of trading days. Starting with NVIDIA (`NVDA`), with an interactive Streamlit dashboard.

> Educational project only. Predictions are forecasts of price **direction**, not BUY/SELL recommendations, and are not financial advice.

## Project structure

| File | Purpose |
|------|---------|
| `features.py` | Data loading, technical-indicator features (Wilder RSI, MACD, SMA ratios, volatility, volume), labels, volatility regimes, dataset construction |
| `predictor.py` | Model definitions, `GridSearchCV` tuning with walk-forward CV, final-model training, latest prediction |
| `evaluation.py` | Chronological split, metrics, permutation importance, regime and horizon analysis, end-to-end `run_pipeline()` |
| `train.py` | Console script: trains and evaluates everything, saves results to `artifacts/` |
| `experiment.py` | Saved, repeatable experiment runs: walk-forward windows, seeds, bootstrap CIs, `repeat` and `compare` commands |
| `run_results.py` | Read-only loading and analysis of the saved runs in `runs/` (no training) |
| `experiments_view.py` | Streamlit views for the saved experiments (summary, performance, windows, regimes, comparison) |
| `tests/` | `pytest` tests for leakage safeguards, the experiment framework and the dashboard's run loading/views (`pip install -r requirements-dev.txt`, then `pytest`) |
| `app.py` | Streamlit dashboard (Plotly charts) that loads the saved results from `artifacts/` and `runs/` |
| `requirements.txt` | Python dependencies |

## Methodology

- **Label:** 1 if the close is higher `horizon` trading days later (default 5), otherwise 0.
- **Features:** use only data available on the prediction day. The newest row (label unknown) is kept aside for live prediction.
- **Split:** chronological hold-out. The last `horizon` rows of the training set are removed so no training label reaches into the test period.
- **Tuning:** `GridSearchCV` with `TimeSeriesSplit(gap=horizon)` on the training period only (MLP architecture and regularisation, Perceptron alpha, Naive Bayes smoothing).
- **MLP early stopping is disabled** because scikit-learn's version uses a random validation split, which leaks future data in time series. Regularisation is used instead.
- **Scaling** happens inside each pipeline, so statistics come from training data only.
- **Evaluation vs. inference:** evaluation models are trained on the training period only. Separate final models are trained on all labeled data for the latest prediction.
- **Metrics:** accuracy, balanced accuracy, precision, recall, F1 and confusion matrices, always compared with a majority-class baseline.
- **Analysis:** permutation feature importance (hold-out set), performance in high vs. low volatility regimes (expanding-median threshold, past data only), and comparison across horizons.
- **No trading profits are reported.** Forward returns over overlapping windows are not independent, and costs/slippage are not modelled.

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Usage

**1. Train from the console** (saves results to `artifacts/`):

```bash
python train.py                              # NVDA, 5-day horizon, with tuning
python train.py --ticker AAPL --horizon 10   # another stock / horizon
python train.py --no-tune                    # faster, default hyperparameters
```

The metrics, selected hyperparameters and latest predictions are also printed to the terminal.
Run `python train.py --help` for all options.

**2. Open the dashboard** (loads the saved results instantly, never trains):

```bash
streamlit run app.py
```

Use **View** in the sidebar to switch between two sets of saved results:

- **Single hold-out run (train.py):** pick a run in the sidebar. Tabs: **Overview**, **Model Comparison**, **Feature Analysis**, **Market Conditions**. Re-run `train.py` for fresher data or different settings.
- **Experiments (experiment.py):** pick an experiment from `runs/`. Tabs: **Summary** (metadata, configuration, headline results), **Model performance** (measured metrics, then bootstrap confidence intervals and differences from the baseline), **Walk-forward windows** (per-window results, seed variation), **Volatility regimes** (with sample sizes) and **Compare experiments** (flags differing data, windows and settings). Tables have CSV downloads, and each chart's camera button saves a high-resolution PNG.

The dashboard never trains or re-runs experiments; new runs appear after a page reload.

## Experiments (saved, repeatable, comparable runs)

`experiment.py` evaluates the three models and the majority baseline over several chronological walk-forward windows and saves everything to `runs/<name>/`.

```bash
python experiment.py run --name nvda_h5 --horizon 5 --n-jobs 8      # downloads + snapshots data, ~2-3 min on 8 workers
python experiment.py run --name nvda_h10 --horizon 10 \
       --snapshot data/snapshots/<file>.csv                         # same data -> comparable
python experiment.py repeat nvda_h5                                 # re-run from saved config + data, verify identical metrics
python experiment.py compare nvda_h5 nvda_h10                       # side-by-side table, flags config/data differences
python experiment.py list
```

- **Data:** prices are saved to `data/snapshots/` (named by SHA-256) and every run reads them back from there. Today's incomplete bar is dropped. Use the same `--snapshot` for runs you want to compare.
- **Windows:** the first half of the history is training only; the rest is split into `--windows` test blocks (expanding training set, `horizon`-row gap). Window dates are the same for every horizon.
- **Tuning and seeds:** `GridSearchCV` (walk-forward CV) runs inside each window on its training data only. Perceptron and MLP are then refit with `--seeds` random seeds to show seed-to-seed variance.
- **Statistics:** pooled out-of-sample accuracy, balanced accuracy and accuracy-minus-baseline with moving-block-bootstrap 95% CIs, plus MCC, AUC, per-window and per-volatility-regime results, MLP/Perceptron convergence, and the score of every hyperparameter-grid point (`grid_results.csv`).
- **Saved per run:** `config.json`, `manifest.json` (data hash, library versions, git commit, windows), `metrics.csv`, `predictions.csv.gz`, `grid_results.csv`, `best_params.csv`, `summary.csv`, `window_summary.csv`, `regime_summary.csv`, `report.txt`.
- Use `--no-tune --seeds 3 --windows 3` for a quick smoke run. `--n-jobs` parallelises the grid search and the seed refits without changing results; run one experiment at a time, since several at once oversubscribe the CPU.

## Limitations

- Daily price direction is very noisy; accuracy near the baseline is expected.
- Results come from one hold-out period and can be luck. Regime subsets are small.
- Horizon comparison re-uses hyperparameters tuned at the main horizon.

## Team

- Rylan Bowe
- Colin Grant
- Leo Yin