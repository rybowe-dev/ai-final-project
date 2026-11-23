# Stock Buy/Sell Classifier

Compares three machine learning models (Naive Bayes, Perceptron, and MLP) on predicting whether a stock is a **buy** or **sell** at a given point in time. Starting with NVIDIA (`NVDA`), with a Streamlit dashboard for exploring results.

> Educational project only. This is not financial advice.

## How it works

- **Data:** daily prices from Yahoo Finance via `yfinance`.
- **Features:** multi-day returns, moving-average ratios, volatility, RSI, MACD, and volume/range indicators.
- **Label:** BUY if the price is higher N trading days later (default 5), otherwise SELL.
- **Evaluation:** chronological train/test split (no shuffling) with a gap to prevent label leakage, compared against a majority-class baseline.

## Project structure

| File | Purpose |
|------|---------|
| `predictor.py` | Data loading, feature engineering, models, and evaluation (runs standalone) |
| `app.py` | Streamlit dashboard built on `predictor.py` |
| `requirements.txt` | Python dependencies |

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Usage

Run the models in the terminal:

```bash
python predictor.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

Then open http://localhost:8501 if it doesn't open automatically.

## Team

- Name 1
- Name 2
- Name 3