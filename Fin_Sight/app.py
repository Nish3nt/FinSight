# app.py
# FinSight Pro - Sentiment-Momentum Fusion Portfolio Generator (single-file)
# Robust, production-minded single-file Streamlit app
# Put your pre-trained model file (if any) in the same folder or rely on in-app training per ticker

import os
import time
from datetime import datetime, timedelta
import concurrent.futures

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas_ta as ta  # ensure installed: pip install pandas_ta

st.set_page_config(page_title="FinSight Pro", layout="wide")
st.title("FinSight Pro — Sentiment-Momentum Fusion Portfolio Generator")

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Configuration")
universe_size = st.sidebar.slider("Universe Size (Top N tickers)", 20, 500, 100, step=10)
sentiment_window = st.sidebar.slider("Sentiment Window (days)", 1, 7, 3)
horizon = st.sidebar.selectbox("Prediction Horizon", ["1-day", "7-day"])
top_n = st.sidebar.slider("Select Top N Stocks for Portfolio", 3, 20, 8)
alert_threshold = st.sidebar.slider("Alert Threshold (momentum absolute)", 0.05, 1.0, 0.2, step=0.01)
allocation_type = st.sidebar.selectbox("Allocation Type", ["Equal Weight", "Risk-Adjusted"])

run_btn = st.sidebar.button("Run Analysis")

# A (short) S&P-ish universe — replace with official list if desired
SP500_TICKERS = [
    "AAPL","MSFT","NVDA","GOOG","AMZN","TSLA","META","JPM","V","JNJ","WMT","UNH","PG","MA","HD",
    "BAC","KO","DIS","NFLX","XOM","PFE","CRM","ADBE","CMCSA","INTC","CSCO","ORCL","T","COST","ABT",
    # ...you can extend or load a list of tickers
]

# ----------------------------
# Helper functions (robust)
# ----------------------------
@st.cache_data(ttl=60 * 30)  # cache for 30 minutes
def safe_download(ticker, start, end):
    """Download data and normalize columns to single-level names."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join(map(str, c)).strip() for c in df.columns.values]
        return df
    except Exception:
        return pd.DataFrame()

def select_price_column(df):
    """Return the best price column name or None."""
    possible = ["Adj Close", "Adj_Close", "Close", "close"]
    for c in possible:
        if c in df.columns:
            return c
    # try any numeric-like column excluding Date/Volume
    numeric_cols = [c for c in df.columns if df[c].dtype != object and c.lower() not in ("volume", "date")]
    if numeric_cols:
        return numeric_cols[0]
    return None

@st.cache_data(ttl=60 * 30)
def ingest_universe(tickers, years=5):
    """Fetch historical data for tickers concurrently. Returns dict ticker->df."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=years * 365)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(safe_download, t, start.isoformat(), end.isoformat()): t for t in tickers[:universe_size]}
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                df = fut.result()
                if df.empty:
                    continue
                price_col = select_price_column(df)
                if price_col is None:
                    continue
                # compute next-day return for horizon=1 by default; will be aligned later
                df = df.copy()
                df["__price_col__"] = df[price_col]
                df["Return_1d"] = df["__price_col__"].pct_change().shift(-1)  # next-day
                results[t] = df.dropna().reset_index(drop=True)
            except Exception:
                continue
    return results

def compute_technicals_for_df(df):
    """Compute a set of technical indicators using pandas_ta on selected price column."""
    df = df.copy()
    price = df["__price_col__"]
    # simple indicators
    df["SMA_20"] = ta.sma(price, length=20)
    df["RSI_14"] = ta.rsi(price, length=14)
    macd = ta.macd(price)
    if isinstance(macd, pd.DataFrame) and "MACD_12_26_9" in macd.columns:
        df["MACD"] = macd["MACD_12_26_9"]
    else:
        df["MACD"] = np.nan
    bb = ta.bbands(price)
    if isinstance(bb, pd.DataFrame):
        # pandas_ta returns columns BBL_5_2.0, etc depending on default; normalize by searching
        cols = list(bb.columns)
        if len(cols) >= 3:
            df["BB_upper"] = bb.iloc[:, 0]
            df["BB_middle"] = bb.iloc[:, 1]
            df["BB_lower"] = bb.iloc[:, 2]
    df["Volume"] = df.get("Volume", 0)
    # drop rows with NaNs in indicators
    feature_cols = ["SMA_20", "RSI_14", "MACD", "BB_upper", "BB_lower", "Volume"]
    # ensure columns exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    features = df[feature_cols].dropna()
    returns = df.loc[features.index, "Return_1d"] if "Return_1d" in df.columns else pd.Series(dtype=float)
    return features, returns

def train_predict_for_ticker(df, horizon_days=1):
    """Train a RandomForest on historical features and predict next horizon return.
       Returns predicted return (float) or None."""
    try:
        features, returns = compute_technicals_for_df(df)
        if features.shape[0] < 80 or len(returns) < 80:
            return None
        # Align for horizon (if horizon_days >1 you can adapt label generation)
        X = features[:-horizon_days]
        y = returns[:-horizon_days]
        if len(X) < 50:
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        model.fit(X_train, y_train)
        # predict from last available features
        pred = model.predict(features.tail(1))[0]
        return float(pred)
    except Exception:
        return None

def predict_universe_returns(data_dict, horizon_days=1):
    """Predict returns for all tickers in data_dict (dict ticker->df)."""
    preds = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(train_predict_for_ticker, df, horizon_days): t for t, df in data_dict.items()}
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                p = fut.result()
                if p is not None:
                    preds[t] = p
            except Exception:
                continue
    return preds

def get_newsapi_key():
    """Try Streamlit secrets first, then environment variable."""
    # Streamlit Cloud: add secrets with key NEWSAPI_KEY
    key = None
    try:
        key = st.secrets["d848a496d874401b9e2129a71adb57ba"]
    except Exception:
        key = os.environ.get("d848a496d874401b9e2129a71adb57ba")
    return key

def sentiment_momentum_for_ticker(ticker, window_days=3):
    """Compute sentiment momentum for a ticker using NewsAPI (requires key)."""
    try:
        api_key = get_newsapi_key()
        if not api_key:
            # no API key configured -> return 0 (neutral)
            return 0.0
        base = "https://newsapi.org/v2/everything"
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=window_days)
        params = {
            "q": f"{ticker} stock OR {ticker} earnings OR {ticker} news",
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "pageSize": 30,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": api_key,
        }
        r = requests.get(base, params=params, timeout=10)
        if r.status_code != 200:
            return 0.0
        arts = r.json().get("articles", [])
        if not arts:
            return 0.0
        # compute average compound sentiment for this window
        scores = [sia.polarity_scores(a.get("title", "") + " " + a.get("description", ""))["compound"] for a in arts[:20]]
        avg_now = np.mean(scores) if scores else 0.0

        # previous window for momentum (same length immediately before)
        prev_to = from_date - timedelta(days=1)
        prev_from = prev_to - timedelta(days=window_days)
        prev_params = params.copy()
        prev_params["from"] = prev_from.isoformat()
        prev_params["to"] = prev_to.isoformat()
        r2 = requests.get(base, params=prev_params, timeout=10)
        prev_avg = 0.0
        if r2.status_code == 200:
            arts2 = r2.json().get("articles", [])
            scores2 = [sia.polarity_scores(a.get("title", "") + " " + a.get("description", ""))["compound"] for a in arts2[:20]]
            prev_avg = np.mean(scores2) if scores2 else 0.0
        momentum = float(avg_now - prev_avg)
        return momentum
    except Exception:
        return 0.0

def compute_sentiment_momentum_for_universe(tickers, window_days=3):
    """Compute sentiment momentum concurrently for tickers."""
    moments = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(sentiment_momentum_for_ticker, t, window_days): t for t in tickers}
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                m = fut.result()
                moments[t] = m
            except Exception:
                moments[t] = 0.0
    return moments

def build_composite_scores(preds, moments):
    """Composite = predicted_return * (1 + momentum). Return dict ticker->score."""
    scores = {}
    for t, p in preds.items():
        m = moments.get(t, 0.0)
        # composite design: take into account sign and bounded momentum
        composite = p * (1.0 + m)
        scores[t] = composite
    return scores

def build_portfolio_from_scores(scores, preds, top_n, allocation_type="Equal Weight"):
    """Rank by score, choose top_n, compute allocation and portfolio metrics."""
    if not scores:
        return {}, pd.DataFrame(), np.nan, np.nan, pd.DataFrame()
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked = ranked[:top_n]
    tickers = [r[0] for r in ranked]
    score_vals = [r[1] for r in ranked]
    # fetch recent price series for risk calc
    price_df = pd.DataFrame()
    for t in tickers:
        try:
            d = yf.download(t, period="6mo", progress=False)
            if "Adj Close" in d.columns:
                price_df[t] = d["Adj Close"]
            elif "Close" in d.columns:
                price_df[t] = d["Close"]
        except Exception:
            continue
    price_df = price_df.dropna(axis=1)
    if price_df.shape[1] == 0:
        # fallback equal weights
        weights = np.ones(len(tickers)) / len(tickers)
        corr = pd.DataFrame()
        exp_ret = np.mean([preds.get(t, 0) for t in tickers])
        risk = 0.0
        allocations = dict(zip(tickers, weights))
        df_rank = pd.DataFrame({"Ticker": tickers, "Score": score_vals})
        return allocations, df_rank, exp_ret, risk, corr
    # compute vol
    rets = price_df.pct_change().dropna()
    vol = rets.std()
    cov = rets.cov()
    corr = rets.corr()
    # allocation logic
    if allocation_type == "Equal Weight":
        w = np.ones(len(price_df.columns)) / len(price_df.columns)
    else:
        # risk-adjusted: weight proportional to predicted return / vol (positive only)
        preds_arr = np.array([max(preds.get(t, 0.0), 0.0) for t in price_df.columns])
        vol_arr = vol[price_df.columns].values
        # avoid division by zero
        ratio = np.where(vol_arr > 0, preds_arr / vol_arr, preds_arr)
        # if all zeros, fallback to equal
        if ratio.sum() <= 0:
            w = np.ones(len(price_df.columns)) / len(price_df.columns)
        else:
            w = ratio / ratio.sum()
    # expected portfolio return (simple dot)
    exp_ret = float(np.dot(w, [preds.get(t, 0.0) for t in price_df.columns]))
    # portfolio risk as sqrt(w^T cov w)
    risk = float(np.sqrt(np.dot(w, np.dot(cov.values, w))))
    allocations = dict(zip(price_df.columns, w))
    df_rank = pd.DataFrame({"Ticker": tickers, "Score": score_vals})
    return allocations, df_rank, exp_ret, risk, corr

def generate_alerts_from_momentum(moments, threshold=0.2):
    return {t: m for t, m in moments.items() if abs(m) >= threshold}

# ----------------------------
# Main flow
# ----------------------------
if run_btn:
    with st.spinner("Ingesting data for universe..."):
        universe = SP500_TICKERS[:universe_size]
        data_dict = ingest_universe(universe, years=5)

    if not data_dict:
        st.error("No valid data was ingested. Try a smaller universe or check yfinance connectivity.")
        st.stop()

    st.info(f"Fetched historical data for {len(data_dict)} tickers.")

    # 1) predict returns
    horizon_days = 1 if horizon == "1-day" else 7
    with st.spinner("Running ML inference across universe (this may take a while)..."):
        preds = predict_universe_returns(data_dict, horizon_days=horizon_days)
    if not preds:
        st.error("No predictions were produced. Possibly not enough historical data for tickers.")
        st.stop()
    st.success(f"Predictions produced for {len(preds)} tickers.")

    # 2) sentiment momentum
    with st.spinner("Computing sentiment momentum (requires NewsAPI key for live news)..."):
        moments = compute_sentiment_momentum_for_universe(list(preds.keys()), window_days=sentiment_window)

    # 3) composite scores and ranking
    scores = build_composite_scores(preds, moments)
    allocations, ranked_df, exp_ret, risk, corr = build_portfolio_from_scores(scores, preds, top_n, allocation_type=allocation_type)

    # 4) alerts
    alerts = generate_alerts_from_momentum(moments, threshold=alert_threshold)

    # ----------------------------
    # UI: Results
    # ----------------------------
    st.header("Top Ranked Stocks by Composite Score")
    if ranked_df.empty:
        st.warning("No ranked stocks to display.")
    else:
        st.dataframe(ranked_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if not np.isnan(exp_ret):
            st.metric("Expected Portfolio Return", f"{exp_ret:.2%}")
        else:
            st.metric("Expected Portfolio Return", "N/A")
    with col2:
        if not np.isnan(risk):
            st.metric("Portfolio Volatility (approx.)", f"{risk:.2%}")
        else:
            st.metric("Portfolio Volatility (approx.)", "N/A")

    # alerts
    st.subheader("Alerts (High Sentiment Momentum)")
    if alerts:
        alert_df = pd.DataFrame(list(alerts.items()), columns=["Ticker", "Momentum"])
        alert_df["Type"] = alert_df["Momentum"].apply(lambda x: "Positive Surge" if x > 0 else "Negative Drop")
        st.dataframe(alert_df, use_container_width=True)
    else:
        st.info("No alerts triggered.")

    # visualizations
    st.header("Portfolio Visualizations")
    if allocations:
        fig_pie = px.pie(values=list(allocations.values()), names=list(allocations.keys()), title="Portfolio Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)

    if isinstance(corr, pd.DataFrame) and not corr.empty:
        fig_heat = ff.create_annotated_heatmap(
            z=corr.values.round(3).tolist(),
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdYlGn",
            showscale=True
        )
        fig_heat.update_layout(title="Correlation Matrix (Top Assets)")
        st.plotly_chart(fig_heat, use_container_width=True)

    # sentiment bar for top_n tickers
    sent_df = pd.DataFrame({"Ticker": list(moments.keys()), "Momentum": list(moments.values())})
    sent_df = sent_df.sort_values("Momentum", ascending=False).head(top_n)
    if not sent_df.empty:
        fig_bar = px.bar(sent_df, x="Ticker", y="Momentum", color="Momentum", color_continuous_scale="RdYlGn", title="Sentiment Momentum (Top N)")
        st.plotly_chart(fig_bar, use_container_width=True)

    # timeline mockup (for each top stock we could show news dates if available)
    st.subheader("News Timeline (mockup based on sentiment window)")
    timeline_items = []
    for t in ranked_df["Ticker"].head(top_n).tolist():
        # fetch recent headlines timestamps if NewsAPI available (here we mock if not)
        dates = pd.date_range(end=datetime.utcnow().date(), periods=min(5, sentiment_window)).tolist()
        for d in dates:
            timeline_items.append(dict(Task=t, Start=d, Finish=d + timedelta(hours=1), Resource="News"))
    if timeline_items:
        fig_tl = px.timeline(timeline_items, x_start="Start", x_end="Finish", y="Task", color="Resource", title="News Timeline (Top Stocks)")
        st.plotly_chart(fig_tl, use_container_width=True)

    st.success("Analysis complete.")

else:
    st.info("Configure parameters in the sidebar and click 'Run Analysis' to start.")
    st.caption("To enable live news-based sentiment momentum, add your NewsAPI key to Streamlit secrets (key: NEWSAPI_KEY) or set NEWSAPI_KEY env var.")
