# =============================================================================
#  FinSight — backend.py
#  Self-installs all dependencies at runtime before any imports
# =============================================================================

import subprocess
import sys

def _install(package):
    """
    Install a package silently.
    Uses run() instead of check_call() so a single failed package
    does not crash the entire app.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package,
         "--quiet", "--no-warn-script-location"],
        capture_output=True, text=True
    )
    return result.returncode == 0

# tensorflow-cpu==2.18.0 is the latest version with Python 3.13 support
# All other packages have wheels for Python 3.13
_PACKAGES = [
    "yfinance==0.2.40",
    "pandas==2.2.2",
    "numpy==1.26.4",
    "plotly==5.22.0",
    "nltk==3.8.1",
    "requests==2.32.3",
    "scikit-learn==1.5.0",
    "streamlit-option-menu==0.3.12",
    "tensorflow-cpu==2.18.0",
    "tf-keras",
]

for _pkg in _PACKAGES:
    _install(_pkg)

# =============================================================================
#  Safe imports after installation
# =============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
import streamlit as st

# TensorFlow import with fallback
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
except ImportError:
    try:
        from tf_keras.models import Sequential
        from tf_keras.layers import LSTM, Dense, Dropout
        from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tf_keras.optimizers import Adam
    except ImportError:
        import keras
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras.optimizers import Adam

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# =============================================================================
#  API KEYS
# =============================================================================
GROQ_API_KEY = "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"
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
FINNHUB_KEY  = "d6qgus9r01qhcrmk4od0d6qgus9r01qhcrmk4odg"

# =============================================================================
#  GROQ LLM
# =============================================================================
def call_groq(prompt, max_tokens=600):
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}",
                   "Content-Type": "application/json"}
        payload = {"model": GROQ_MODEL,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0.4}
        r = requests.post(GROQ_URL, headers=headers,
                          json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content'].strip()
        return f"__ERROR__: {r.status_code} — {r.text}"
    except Exception as e:
        return f"__ERROR__: {e}"

# =============================================================================
#  NEWS
# =============================================================================
@st.cache_data(ttl=300)
def get_news(ticker):
    try:
        to_d   = datetime.now().strftime('%Y-%m-%d')
        from_d = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        url    = (f"https://finnhub.io/api/v1/company-news?symbol={ticker}"
                  f"&from={from_d}&to={to_d}&token={FINNHUB_KEY}")
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            headlines, posts, links = [], [], []
            for art in r.json()[:10]:
                t_ = art.get('headline', '').strip()
                s_ = art.get('source', 'Finnhub')
                u_ = art.get('url', '#')
                if t_:
                    headlines.append(f"**{t_}** - {s_}")
                    posts.append(f"{t_} - {s_}")
                    links.append(u_)
            return (headlines or ["No recent news."]), posts, links
        return [f"Finnhub error {r.status_code}"], [], []
    except Exception as e:
        return [f"Error: {e}"], [], []

# =============================================================================
#  SENTIMENT
# =============================================================================
@st.cache_data(ttl=300)
def compute_vader(posts):
    return [sia.polarity_scores(p)['compound'] for p in posts]

# =============================================================================
#  STOCK DATA
# =============================================================================
@st.cache_data(ttl=600)
def fetch_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        req = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for c in req:
            if c not in df.columns:
                df[c] = np.nan
        df = df[req].dropna(subset=['Adj Close'])
        return df if len(df) >= 120 else None
    except:
        return None

# =============================================================================
#  FEATURE ENGINEERING — 11 features
# =============================================================================
def compute_features(raw):
    df = raw.copy()
    df['LogReturn']   = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['SMA20']       = df['Adj Close'].rolling(20).mean()
    df['SMA50']       = df['Adj Close'].rolling(50).mean()
    df['EMA12']       = df['Adj Close'].ewm(span=12, adjust=False).mean()
    df['EMA26']       = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta             = df['Adj Close'].diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    df['RSI']      = 100 - 100 / (
        1 + up.rolling(14).mean() / (dn.rolling(14).mean() + 1e-9))
    rm = df['Adj Close'].rolling(20).mean()
    rs = df['Adj Close'].rolling(20).std()
    df['BB_Width'] = (2 * rs) / (rm + 1e-9)
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Adj Close'].shift()).abs()
    lc = (df['Low']  - df['Adj Close'].shift()).abs()
    df['ATR']      = (pd.concat([hl, hc, lc], axis=1)
                      .max(axis=1).rolling(14).mean())
    df['LogVolume'] = np.log1p(df['Volume'])
    cols = ['LogReturn', 'LogVolume', 'SMA20', 'SMA50',
            'EMA12', 'EMA26', 'MACD', 'MACD_Signal',
            'RSI', 'BB_Width', 'ATR']
    return df[cols].dropna(), df['Adj Close']

# =============================================================================
#  DRAWDOWN
# =============================================================================
def compute_drawdown(series):
    return (series - series.cummax()) / series.cummax() * 100

# =============================================================================
#  LSTM TRAINING
# =============================================================================
@st.cache_resource(ttl=24 * 3600)
def train_model(ticker, start_str, end_str,
                time_step, epochs, batch_size, retrain_flag):
    t0 = time.time()
    training_time = datetime.now()

    raw = yf.download(ticker, start=start_str, end=end_str, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    if 'Adj Close' not in raw.columns and 'Close' in raw.columns:
        raw['Adj Close'] = raw['Close']

    df_feat, price_s = compute_features(raw)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(df_feat.values)

    X, y = [], []
    for i in range(len(scaled) - time_step):
        X.append(scaled[i:i + time_step, :])
        y.append(scaled[i + time_step, 0])
    X = np.array(X); y = np.array(y)

    n = X.shape[0]; train_n = int(n * 0.80)
    X_tr, y_tr = X[:train_n], y[:train_n]
    X_te, y_te = X[train_n:], y[train_n:]
    n_feat = X.shape[2]

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(time_step, n_feat)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.15),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')

    cbs, val_split = [], 0.0
    if len(X_tr) > 20:
        cbs = [
            EarlyStopping(monitor='val_loss', patience=12,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=6, min_lr=1e-6, verbose=0)
        ]
        val_split = 0.1

    history = model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                        validation_split=val_split, callbacks=cbs, verbose=0)

    bt_pp, bt_ap, bt_pr, bt_ar = [], [], [], []
    dummy_row = np.zeros((1, n_feat))
    for i in range(len(X_te)):
        gi = time_step + train_n + i
        ps = float(model.predict(X_te[i:i+1], verbose=0)[0, 0])
        dummy_row[0, 0] = ps
        plr = float(scaler.inverse_transform(dummy_row)[0, 0])
        alr = float(df_feat['LogReturn'].iloc[gi])
        pp  = float(price_s.iloc[gi - 1]) * np.exp(plr)
        ap  = float(price_s.iloc[gi])
        bt_pp.append(pp); bt_ap.append(ap)
        bt_pr.append(plr); bt_ar.append(alr)

    bt_pp = np.array(bt_pp); bt_ap = np.array(bt_ap)
    bt_pr = np.array(bt_pr); bt_ar = np.array(bt_ar)

    wf, fs = [], max(10, len(bt_pp) // 5)
    for f in range(5):
        s = f * fs; e = min(s + fs, len(bt_pp))
        if e - s < 5: break
        wf.append(float(r2_score(bt_ap[s:e], bt_pp[s:e])))
    wf_r2  = float(np.mean(wf)) if wf else 0.0
    mse_v  = float(mean_squared_error(bt_ap, bt_pp))
    r2_v   = float(r2_score(bt_ap, bt_pp))
    rmse_v = float(np.sqrt(mse_v))
    mape_v = float(np.mean(np.abs((bt_ap-bt_pp)/(np.abs(bt_ap)+1e-9)))*100)
    da_v   = float(np.mean(np.sign(bt_pr)==np.sign(bt_ar))*100)
    np_ = bt_ap[:-1]; na_ = bt_ap[1:]
    n_r2   = float(r2_score(na_, np_))
    n_mape = float(np.mean(np.abs((na_-np_)/(np.abs(na_)+1e-9)))*100)
    n_rmse = float(np.sqrt(mean_squared_error(na_, np_)))
    rs_v   = float(np.std(bt_ap - bt_pp))

    return dict(
        model=model, scaler=scaler,
        df_feat=df_feat, price_series=price_s,
        time_step=time_step, train_n=train_n, n_feat=n_feat,
        bt_pp=bt_pp, bt_ap=bt_ap, bt_pr=bt_pr, bt_ar=bt_ar,
        history=history.history, training_time=training_time,
        training_secs=time.time()-t0, epochs=epochs, batch_size=batch_size,
        mse=mse_v, r2=r2_v, rmse=rmse_v, mape=mape_v, da=da_v,
        wf_r2=wf_r2, wf_list=wf,
        n_r2=n_r2, n_mape=n_mape, n_rmse=n_rmse, resid_std=rs_v
    )
