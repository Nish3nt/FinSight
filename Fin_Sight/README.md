# 📈 FinSight — Real-Time Stock Intelligence

> A production-grade stock analysis and forecasting application built with **Streamlit**, **LSTM neural networks**, and **industry-standard evaluation metrics** used by quantitative finance teams.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🌐 Live Demo

🔗 [finsight-94srxzppan2fvumqpctyzb.streamlit.app](https://finsight-94srxzppan2fvumqpctyzb.streamlit.app)

---


---

## 🔭 Overview

**FinSight** is a full-stack financial intelligence dashboard that combines:

- **Real-time stock data** fetched from Yahoo Finance
- **Multi-feature LSTM** neural network for price forecasting
- **Industry-grade model evaluation** (Walk-Forward R², Directional Accuracy, MAPE, Naïve Baseline comparison)
- **Sentiment analysis** on live financial news via Finnhub API
- **Portfolio analytics** including Sharpe Ratio and correlation heatmaps

The project was designed with one goal: build a forecasting system that is evaluated the way **real quantitative finance teams** evaluate models — not just by raw R², but by whether the model genuinely adds signal beyond a simple baseline.

---

## ✨ Features

### 📊 Data & Visualization
- Full S&P 500 ticker support (500+ stocks)
- Adjustable date range from 2010 to present
- Interactive price charts with Plotly
- One-click CSV export

### 🤖 Predictions (LSTM Forecasting)
- **Multi-feature LSTM** trained on 11 technical indicators
- Predicts **log returns** (stationary target — no artificial R² inflation)
- Configurable: forecast horizon, lookback window, epochs, batch size
- **Animated forecast chart** with growing 95% confidence intervals
- Per-day prediction table with upper/lower bounds
- Model caching — fast reload after first training run

### 📉 Industry-Grade Evaluation Dashboard
- **Walk-Forward R²** — R² computed over 5 rolling time windows
- **Standard R²** — overall test set performance
- **Directional Accuracy %** — did the model correctly predict up/down?
- **MAPE** — Mean Absolute Percentage Error
- **RMSE** — average dollar error per trading day
- **Naïve Baseline Comparison** — explicitly shows whether the LSTM beats "predict tomorrow = today"
- **Residuals histogram** — model calibration check
- Colour-coded metrics (green / amber / red) against industry targets

### 💬 Sentiment Analysis
- Live financial news via **Finnhub API** (last 30 days)
- **VADER sentiment scoring** on each headline
- Positive / Negative / Neutral breakdown

### ⚖️ Stock Comparison
- Normalised performance chart (%) for any two tickers
- Annualised return and volatility metrics side by side

### 💼 Portfolio Analyzer
- Multi-ticker portfolio with adjustable weights
- **Expected annual return**, **portfolio volatility**, **Sharpe Ratio**
- **Correlation heatmap** across selected holdings

### 📰 Live News Ticker
- Scrolling headline ticker at the bottom of every page
- Auto-refreshes every 5 minutes

---

## 🧠 Model Architecture

```
Input: (batch, time_step=90, n_features=11)
         │
    ┌────▼────┐
    │  LSTM   │  128 units, return_sequences=True
    └────┬────┘
    Dropout (0.20)
    ┌────▼────┐
    │  LSTM   │  64 units
    └────┬────┘
    Dropout (0.15)
    ┌────▼────┐
    │  Dense  │  32 units, ReLU
    └────┬────┘
    ┌────▼────┐
    │  Dense  │  1 unit  →  Predicted Log Return
    └─────────┘
```

| Parameter        | Value              |
|------------------|--------------------|
| Optimizer        | Adam (lr = 0.001)  |
| Loss Function    | MSE                |
| Early Stopping   | patience = 12      |
| LR Scheduler     | ReduceLROnPlateau (factor=0.5, patience=6) |
| Validation Split | 10%                |
| Train/Test Split | 80% / 20%          |
| Default Epochs   | 80                 |
| Default Batch    | 32                 |

---

## 🔧 Feature Engineering

The model uses **11 technical features** computed from raw OHLCV data:

| Feature        | Category   | Description                                      |
|----------------|------------|--------------------------------------------------|
| `LogReturn`    | **Target** | `log(P_t / P_{t-1})` — stationary return series |
| `LogVolume`    | Volume     | `log(1 + Volume)` — removes skew                |
| `SMA20`        | Trend      | 20-day Simple Moving Average                     |
| `SMA50`        | Trend      | 50-day Simple Moving Average                     |
| `EMA12`        | Trend      | 12-day Exponential Moving Average                |
| `EMA26`        | Trend      | 26-day Exponential Moving Average                |
| `MACD`         | Momentum   | EMA12 − EMA26                                    |
| `MACD_Signal`  | Momentum   | 9-day EMA of MACD                                |
| `RSI`          | Momentum   | 14-day Relative Strength Index                   |
| `BB_Width`     | Volatility | Normalised Bollinger Band width                  |
| `ATR`          | Volatility | 14-day Average True Range                        |

> **Why log returns instead of raw prices?**
> Raw prices produce artificially high R² values because any model that learns "tomorrow ≈ today" scores well. Log returns are stationary and force the model to genuinely learn directional patterns. Predictions are converted back to prices via `P_t = P_{t-1} × exp(LogReturn)`.

---

## 📐 Evaluation Methodology

### Why not just R²?

Standard R² on stock prices is misleading. A naïve model that predicts "tomorrow's price = today's price" achieves R² > 0.97 on most stocks — with zero forecasting skill. FinSight uses the following industry-aligned approach:

### Walk-Forward R² (Primary Metric)
The test set is split into **5 sequential folds**. R² is computed independently on each fold, then averaged. This mirrors how production trading systems are evaluated — on unseen, time-ordered data.

```
Training Data (80%)  │  Test Data (20%)
─────────────────────┼──────────────────────────────────
                     │  Fold1 │ Fold2 │ Fold3 │ Fold4 │ Fold5
                          ↓       ↓       ↓       ↓       ↓
                         R²₁     R²₂     R²₃     R²₄     R²₅
                                  └──── Mean = Walk-Forward R² ────┘
```

### Naïve Baseline Comparison
Every metric (R², MAPE, RMSE) is compared against the **naïve persistence model**. A model is only considered useful if it beats this baseline.

### Directional Accuracy
Measures what percentage of days the model correctly predicted whether the price would go **up or down**. Random guessing = 50%. Industry threshold for a useful signal: > 55%.

### Metric Targets (Colour Coding)

| Metric              | 🟢 Good     | 🟡 Acceptable | 🔴 Poor  |
|---------------------|-------------|---------------|----------|
| Walk-Forward R²     | ≥ 0.80      | ≥ 0.65        | < 0.65   |
| Directional Acc     | ≥ 58%       | ≥ 52%         | < 52%    |
| MAPE                | ≤ 3%        | ≤ 5%          | > 5%     |

---

## 🛠️ Tech Stack

| Layer           | Technology                         |
|-----------------|------------------------------------|
| Frontend        | Streamlit                          |
| Charts          | Plotly                             |
| Deep Learning   | TensorFlow / Keras                 |
| Data Source     | yFinance (Yahoo Finance)           |
| News API        | Finnhub                            |
| NLP / Sentiment | NLTK VADER                         |
| ML Utilities    | scikit-learn (MinMaxScaler, metrics)|
| Data Processing | Pandas, NumPy                      |
| Deployment      | Streamlit Cloud                    |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/finsight.git
cd finsight
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

---


---

## 🚀 Usage



### First-Time Setup

1. Select any S&P 500 ticker from the sidebar (default: **AAPL**)
2. Navigate to the **Predictions** tab
3. Tick **⚠️ Force retrain model** on first run
4. Set epochs to **80** for best accuracy
5. Click anywhere to trigger training (~2–3 minutes first run, cached after)



---

## 📁 Project Structure

```
finsight/
│
├── app.py                  # Main application (all logic)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Optional Streamlit theme config
```

---


---

## 💡 Why These Design Choices?

| Decision | Reason |
|---|---|
| **Log returns as target** | Avoids artificially inflated R² from raw price prediction; forces genuine pattern learning |
| **Walk-Forward validation** | Mimics real trading evaluation; prevents data leakage across time |
| **Naïve baseline** | Makes model value explicit — a recruiter or quant can immediately see the model earns its keep |
| **2-layer LSTM** | Deeper networks need 150+ epochs to converge on financial data; 2 layers hit the accuracy/speed sweet spot |
| **EarlyStopping + ReduceLROnPlateau** | Prevents overfitting; adapts training length to data complexity automatically |
| **Caching (`@st.cache_resource`)** | Trained model is cached for 24 hours — avoids expensive retraining on every page refresh |
| **Growing CI band** | Uncertainty grows with forecast horizon (`±z × σ × √step`) — statistically correct |

---

## ⚠️ Limitations

- **Stock markets are partially random** — no model can reliably predict short-term prices. This app is a learning and analysis tool, not financial advice.
- **LSTM does not account for black swan events** — sudden news, earnings surprises, or macro shocks cannot be forecast from technical indicators alone.
- **Recursive forecasting error accumulates** — multi-day forecasts compound prediction error at each step; longer horizons are less reliable.
- **Survivorship bias** — S&P 500 data only includes current constituents, which may introduce historical bias.

---

## 🔮 Future Improvements

- [ ] Add **Transformer / Attention** model for comparison
- [ ] Integrate **earnings calendar** and **macro events** as features
- [ ] Add **Monte Carlo simulation** for probabilistic forecasting
- [ ] Implement **paper trading backtest** with P&L tracking
- [ ] Add **options chain** data and implied volatility features
- [ ] Support **crypto and international** markets
- [ ] Email/SMS **price alert** system
- [ ] Export predictions as **PDF report**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---


---

> ⚠️ **Disclaimer**: FinSight is built for educational and portfolio demonstration purposes only. It is **not financial advice**. Always consult a qualified financial advisor before making investment decisions.
