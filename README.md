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
| `app.py` | Streamlit dashboard (4 tabs, Plotly charts) that loads the saved results |
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

Pick a saved run in the sidebar. Tabs: **Overview**, **Model Comparison**, **Feature Analysis**, **Market Conditions**.
Re-run `train.py` whenever you want fresher data or different settings.

## Limitations

- Daily price direction is very noisy; accuracy near the baseline is expected.
- Results come from one hold-out period and can be luck. Regime subsets are small.
- Horizon comparison re-uses hyperparameters tuned at the main horizon.

## Team

- Rylan Bowe
- Colin Grant
- Leo Yin