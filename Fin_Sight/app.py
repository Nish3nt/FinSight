# =============================================================================
#  FinSight — app.py  (STABLE FINAL)
#  Model  : 3-model LSTM Ensemble | 16 Features | No Lambda layers
#  Portfolio : 7 features + Groq AI
#  Fixes  : NaN forecast, Monte Carlo scope, attention replaced with stacked LSTM
# =============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
from streamlit_option_menu import option_menu

nltk.download('vader_lexicon', quiet=True)
sia          = SentimentIntensityAnalyzer()
current_date = datetime.now().date()
st.set_page_config(page_title="FinSight", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stSidebar"]>div:first-child{background:#0b1220;padding:16px 12px}
.block-container{padding-top:.5rem;padding-bottom:.4rem}
.kpi-card{background:#0f172a;border-radius:10px;padding:16px 12px;
          text-align:center;border:1px solid #1e293b;margin-bottom:4px}
.kpi-label{font-size:11px;color:#64748b;text-transform:uppercase;
           letter-spacing:.06em;margin-bottom:6px}
.kpi-value{font-size:24px;font-weight:700;color:#e2e8f0;line-height:1}
.kpi-sub{font-size:11px;color:#475569;margin-top:4px}
.good{color:#22c55e!important}
.warn{color:#f59e0b!important}
.bad{color:#ef4444!important}
.skel{background:linear-gradient(90deg,#111827 25%,#0b1220 50%,#111827 75%);
      background-size:200% 100%;animation:sh 1.4s linear infinite;
      height:110px;border-radius:10px;margin-bottom:10px}
@keyframes sh{0%{background-position:200% 0}100%{background-position:-200% 0}}
.bsbox{background:#0f172a;border:1px solid #1e293b;border-radius:8px;
       padding:12px;margin-bottom:4px}
.ins-card{background:#0f172a;border-left:3px solid #6366f1;
          border-radius:0 8px 8px 0;padding:12px 16px;font-size:13px;
          color:#cbd5e1;margin-bottom:8px;line-height:1.6}
.ai-card{background:linear-gradient(135deg,#0f172a,#1e1b4b);
         border:1px solid #4338ca;border-radius:10px;padding:16px;
         margin-bottom:10px;font-size:13px;color:#c7d2fe;line-height:1.7}
.ai-title{font-size:11px;font-weight:700;color:#818cf8;
          text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.comp-box{background:#0f172a;border:1px solid #1e3a5f;border-radius:12px;
          padding:18px 22px;margin-bottom:16px}
.comp-box p{margin:0;font-size:14px;color:#cbd5e1;line-height:1.8}
</style>
""", unsafe_allow_html=True)

st.title("**FinSight**: Real-Time Stock Intelligence")

# ── Tickers ───────────────────────────────────────────────────────────────────
tickers = sorted(set([
    'A','AAPL','ABBV','ABNB','ABT','ACGL','ACN','ADBE','ADI','ADM','ADP','ADSK',
    'AEE','AEP','AES','AFL','AIG','AIZ','AJG','AKAM','ALB','ALGN','ALL','ALLE',
    'AMAT','AMD','AME','AMGN','AMP','AMT','AMZN','ANET','ANSS','AON','AOS','APA',
    'APD','APH','APTV','ARE','ATO','AVB','AVGO','AVY','AWK','AXON','AXP','AZO',
    'BA','BAC','BALL','BAX','BBWI','BBY','BDX','BEN','BF.B','BG','BIIB','BIO',
    'BK','BKNG','BKR','BLDR','BLK','BMY','BR','BRK.B','BRO','BSX','BWA','BX',
    'BXP','C','CAG','CAH','CARR','CAT','CB','CBOE','CBRE','CCI','CCL','CDNS',
    'CDW','CE','CEG','CFG','CHD','CHRW','CHTR','CI','CINF','CL','CLX','CMA',
    'CMCSA','CME','CMG','CMI','CMS','CNC','CNP','COF','COO','COP','COR','COST',
    'CPAY','CPB','CPRT','CRL','CRM','CSCO','CSGP','CSX','CTAS','CTRA','CTSH',
    'CVS','CVX','D','DAL','DASH','DD','DE','DECK','DELL','DFS','DG','DGX','DHI',
    'DHR','DIS','DLR','DLTR','DOC','DOV','DOW','DPZ','DRI','DTE','DUK','DVA',
    'DVN','DXCM','EA','EBAY','ED','EFX','EG','EIX','EL','ELV','EMN','EMR',
    'ENPH','EOG','EPAM','EQIX','EQR','ES','ESS','ETN','ETR','EVRG','EW','EXC',
    'EXPD','EXPE','F','FANG','FAST','FDS','FDX','FE','FFIV','FI','FICO','FIS',
    'FITB','FOX','FOXA','FRT','FSLR','FTNT','FTV','GD','GE','GEHC','GEN','GEV',
    'GILD','GIS','GL','GLW','GM','GNRC','GOOG','GOOGL','GPC','GPN','GRMN','GS',
    'GWW','HAL','HAS','HBAN','HCA','HD','HES','HIG','HII','HLT','HOLX','HON',
    'HPE','HPQ','HRL','HSIC','HST','HSY','HUBB','HUM','HWM','IBM','ICE','IDXX',
    'IEX','IFF','ILMN','INCY','INTC','INTU','INVH','IP','IPG','IQV','IR','IRM',
    'ISRG','IT','ITW','IVZ','J','JBHT','JBL','JCI','JKHY','JNJ','JPM','K',
    'KDP','KEY','KEYS','KHC','KIM','KLAC','KMB','KMI','KO','KR','KVUE','L',
    'LDOS','LEN','LH','LHX','LIN','LKQ','LLY','LMT','LNT','LOW','LRCX','LULU',
    'LUV','LVS','LW','LYB','LYV','MAA','MAR','MAS','MCD','MCHP','MCK','MCO',
    'MDLZ','MDT','MET','META','MGM','MHK','MKC','MLM','MMC','MMM','MNST','MO',
    'MOH','MOS','MPC','MPWR','MRK','MRNA','MS','MSCI','MSFT','MSI','MTB','MTCH',
    'MTD','MU','NCLH','NDAQ','NDSN','NEE','NEM','NFLX','NI','NKE','NOC','NOW',
    'NRG','NSC','NTAP','NTRS','NVDA','NVR','NWSA','NWS','NXPI','O','ODFL','OKE',
    'OMC','ON','ORCL','ORLY','OTIS','OXY','PANW','PAYC','PAYX','PCAR','PCG',
    'PEG','PEP','PFE','PFG','PG','PGR','PH','PHM','PKG','PLD','PLTR','PM',
    'PNC','PNR','PNW','PODD','POOL','PPL','PRU','PSX','PTC','PWR','PYPL','QCOM',
    'REG','REGN','RF','RJF','RL','RMD','ROK','ROL','ROP','ROST','RSG','RTX',
    'RVTY','SBAC','SBUX','SCHW','SHW','SJM','SLB','SMCI','SNA','SNPS','SO',
    'SOLV','SPG','SPGI','SRE','STE','STLD','STT','STX','STZ','SW','SWK','SWKS',
    'SYF','SYK','SYY','T','TAP','TDG','TDY','TECH','TEL','TER','TSLA','TFC',
    'TFX','TGT','TJX','TKO','TMO','TMUS','TPR','TRGP','TRMB','TROW','TRV',
    'TSCO','TSN','TT','TTWO','TXN','TXT','TYL','UAL','UBER','UDR','UHS','ULTA',
    'UNH','UNP','UPS','URI','USB','V','VFC','VICI','VLO','VLTO','VMC','VRSK',
    'VRSN','VRTX','VST','VTR','VZ','WAB','WAT','WBA','WBD','WDC','WEC','WELL',
    'WFC','WM','WMB','WMT','WRB','WST','WTW','WY','WYNN','XEL','XOM','XYL',
    'YUM','ZBH','ZBRA','ZTS'
]))

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Main Stock",   tickers,
                                        index=tickers.index('AAPL'))
compare_ticker  = st.sidebar.selectbox("Compare With", tickers,
                                        index=tickers.index('MSFT'))
start_date = st.sidebar.date_input("Start Date",
                                    pd.to_datetime('2010-01-01').date())
end_date   = st.sidebar.date_input("End Date", current_date)
st.sidebar.markdown("---")
st.sidebar.markdown("**Prediction Settings**")
days       = st.sidebar.slider("Forecast Days", 1, 30, 7)
time_step  = st.sidebar.slider("Lookback Window", 60, 120, 90, step=10)
epochs     = st.sidebar.slider("Training Epochs", 40, 120, 80, step=5)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
retrain    = st.sidebar.checkbox("Force Retrain", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("**AI Insights (Portfolio Tab)**")
groq_key = st.sidebar.text_input("Groq API Key", type="password",
                                   placeholder="gsk_...")
st.sidebar.caption("Free key at console.groq.com")

if start_date > end_date:
    st.error("Start date must be before end date."); st.stop()
if end_date > current_date:
    end_date = current_date

# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_news(ticker):
    api_key = "d6qgus9r01qhcrmk4od0d6qgus9r01qhcrmk4odg"
    try:
        to_d   = datetime.now().strftime('%Y-%m-%d')
        from_d = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        url    = (f"https://finnhub.io/api/v1/company-news?symbol={ticker}"
                  f"&from={from_d}&to={to_d}&token={api_key}")
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

news_headlines, news_posts, news_links = get_news(selected_ticker)

@st.cache_data(ttl=300)
def compute_vader(posts):
    return [sia.polarity_scores(p)['compound'] for p in posts]
vader_scores = compute_vader(news_posts)

@st.cache_data(ttl=600)
def fetch_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        req = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for c in req:
            if c not in df.columns: df[c] = np.nan
        df = df[req].dropna(subset=['Adj Close'])
        return df if len(df) >= 120 else None
    except:
        return None

data_main    = fetch_stock_data(selected_ticker, start_date, end_date)
data_compare = fetch_stock_data(compare_ticker,  start_date, end_date)

# ── Feature Engineering — 16 features, NaN-safe ──────────────────────────────
def compute_features(raw):
    df  = raw.copy()
    ac  = df['Adj Close']
    # Returns
    df['LogReturn']   = np.log(ac / ac.shift(1))
    df['Lag1Return']  = df['LogReturn'].shift(1)
    # Trend
    df['SMA20']       = ac.rolling(20).mean()
    df['SMA50']       = ac.rolling(50).mean()
    df['EMA12']       = ac.ewm(span=12, adjust=False).mean()
    df['EMA26']       = ac.ewm(span=26, adjust=False).mean()
    # Momentum
    df['MACD']        = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = ac.diff()
    up = delta.clip(lower=0); dn = -delta.clip(upper=0)
    df['RSI']         = 100 - 100 / (1 + up.rolling(14).mean() /
                                     (dn.rolling(14).mean() + 1e-9))
    df['ROC10']       = ac.pct_change(10) * 100
    high14 = df['High'].rolling(14).max()
    low14  = df['Low'].rolling(14).min()
    df['WilliamsR']   = -100 * (high14 - ac) / (high14 - low14 + 1e-9)
    df['StochK']      = 100  * (ac - low14)  / (high14 - low14 + 1e-9)
    # Volatility
    rm = ac.rolling(20).mean(); rs = ac.rolling(20).std()
    df['BB_Width']    = (2 * rs) / (rm + 1e-9)
    hl = df['High'] - df['Low']
    hc = (df['High'] - ac.shift()).abs()
    lc = (df['Low']  - ac.shift()).abs()
    df['ATR']         = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    # Volume
    df['LogVolume']   = np.log1p(df['Volume'])
    obv_raw           = (np.sign(df['LogReturn']) * df['Volume']).cumsum()
    obv_mean          = obv_raw.rolling(20).mean()
    obv_std           = obv_raw.rolling(20).std()
    df['OBV_norm']    = (obv_raw - obv_mean) / (obv_std + 1e-9)

    cols = ['LogReturn', 'Lag1Return', 'LogVolume', 'OBV_norm',
            'SMA20', 'SMA50', 'EMA12', 'EMA26',
            'MACD', 'MACD_Signal', 'RSI', 'ROC10',
            'WilliamsR', 'StochK', 'BB_Width', 'ATR']
    out = df[cols].copy()
    # NaN guard — ffill then bfill then zero
    out = out.ffill().bfill().fillna(0)
    return out.dropna(), ac

# ── Build one LSTM model (no Lambda, no custom layers) ────────────────────────
def build_lstm(time_step, n_feat, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=(time_step, n_feat)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.15),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.10),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# ── Groq AI helper ────────────────────────────────────────────────────────────
def call_groq(api_key, prompt):
    try:
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        payload = {"model": "llama-3.1-70b-versatile",
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 1024, "temperature": 0.4}
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content']
        return f"Groq error {r.status_code}: {r.text}"
    except Exception as e:
        return f"Error calling Groq: {e}"

def compute_drawdown(series):
    roll_max = series.cummax()
    return (series - roll_max) / roll_max * 100

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab = option_menu(None,
    ["Data & Viz", "Predictions", "Sentiment",
     "Comparison", "Portfolio Analyzer"],
    icons=["table", "graph-up", "chat-dots",
           "arrow-left-right", "pie-chart"],
    orientation="horizontal")

# ==============================================================================
#  TAB 1 — DATA & VIZ
# ==============================================================================
if tab == "Data & Viz":
    st.subheader(f"**{selected_ticker}** — Price History")
    if data_main is not None:
        st.dataframe(data_main.tail(100), use_container_width=True)
        st.download_button("Download CSV",
                           data_main.to_csv().encode(),
                           f"{selected_ticker}.csv")
        fig = px.line(data_main, x=data_main.index, y='Adj Close',
                      title=f"{selected_ticker} — Adjusted Close Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data available. Try expanding the date range.")


# ==============================================================================
#  TAB 2 — PREDICTIONS
# ==============================================================================
elif tab == "Predictions":

    if data_main is None:
        st.error("Not enough data. Expand date range."); st.stop()

    df_features, price_series = compute_features(data_main)
    if len(df_features) < time_step + 60:
        st.error(f"Need at least {time_step+60} rows. Got {len(df_features)}.")
        st.stop()

    @st.cache_resource(ttl=24 * 3600)
    def train_ensemble(ticker, start_str, end_str, time_step,
                       epochs, batch_size, retrain_flag, _n_rows):
        t0 = time.time()
        training_time = datetime.now()

        raw = yf.download(ticker, start=start_str, end=end_str, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if 'Adj Close' not in raw.columns and 'Close' in raw.columns:
            raw['Adj Close'] = raw['Close']

        df_feat, price_s = compute_features(raw)
        n_feat = df_feat.shape[1]  # 16

        # Exponential sample weights — recent data matters more
        n_rows  = len(df_feat)
        sw_full = np.exp(np.linspace(-2.0, 0, n_rows))

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(df_feat.values)
        # Extra NaN safety on scaled array
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Build sequences
        X, y = [], []
        for i in range(len(scaled) - time_step):
            X.append(scaled[i:i + time_step, :])
            y.append(scaled[i + time_step, 0])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        n       = X.shape[0]
        train_n = int(n * 0.80)
        X_tr, y_tr = X[:train_n], y[:train_n]
        X_te, y_te = X[train_n:], y[train_n:]
        sw_tr = sw_full[time_step: time_step + train_n]
        sw_tr = sw_tr / sw_tr.sum() * len(sw_tr)

        cbs = [
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=1e-6, verbose=0)
        ]

        # Train 3 models — different seeds for diversity
        models, train_histories = [], []
        for seed in [42, 7, 99]:
            m = build_lstm(time_step, n_feat, seed=seed)
            h = m.fit(X_tr, y_tr,
                      epochs=epochs, batch_size=batch_size,
                      validation_split=0.1,
                      callbacks=cbs,
                      sample_weight=sw_tr,
                      verbose=0)
            models.append(m)
            train_histories.append(h.history)

        # Ensemble predict — batch, then average
        all_preds = []
        for m in models:
            p = m.predict(X_te, verbose=0).flatten()
            all_preds.append(p)
        ens_sc = np.mean(all_preds, axis=0)  # shape (n_test,)

        # Inverse transform
        dummy        = np.zeros((len(ens_sc), n_feat), dtype=np.float32)
        dummy[:, 0]  = ens_sc
        all_lr       = scaler.inverse_transform(dummy)[:, 0]
        all_lr       = np.nan_to_num(all_lr, nan=0.0)

        # Reconstruct prices
        bt_pp, bt_ap, bt_pr, bt_ar = [], [], [], []
        for i in range(len(all_lr)):
            gi  = time_step + train_n + i
            plr = float(all_lr[i])
            alr = float(df_feat['LogReturn'].iloc[gi])
            pp  = float(price_s.iloc[gi - 1]) * np.exp(plr)
            ap  = float(price_s.iloc[gi])
            bt_pp.append(pp); bt_ap.append(ap)
            bt_pr.append(plr); bt_ar.append(alr)

        bt_pp = np.array(bt_pp); bt_ap = np.array(bt_ap)
        bt_pr = np.array(bt_pr); bt_ar = np.array(bt_ar)

        # Metrics
        wf, fs = [], max(10, len(bt_pp) // 5)
        for f in range(5):
            s = f * fs; e = min(s + fs, len(bt_pp))
            if e - s < 5: break
            wf.append(float(r2_score(bt_ap[s:e], bt_pp[s:e])))
        wf_r2  = float(np.mean(wf)) if wf else 0.0
        r2_v   = float(r2_score(bt_ap, bt_pp))
        mse_v  = float(mean_squared_error(bt_ap, bt_pp))
        rmse_v = float(np.sqrt(mse_v))
        mape_v = float(np.mean(
            np.abs((bt_ap - bt_pp) / (np.abs(bt_ap) + 1e-9))) * 100)
        da_v   = float(
            np.mean(np.sign(bt_pr) == np.sign(bt_ar)) * 100)

        np_ = bt_ap[:-1]; na_ = bt_ap[1:]
        n_r2   = float(r2_score(na_, np_))
        n_mape = float(np.mean(
            np.abs((na_ - np_) / (np.abs(na_) + 1e-9))) * 100)
        n_rmse = float(np.sqrt(mean_squared_error(na_, np_)))
        rs_v   = float(np.std(bt_ap - bt_pp))

        return dict(
            models=models, scaler=scaler,
            df_feat=df_feat, price_series=price_s,
            time_step=time_step, train_n=train_n, n_feat=n_feat,
            bt_pp=bt_pp, bt_ap=bt_ap, bt_pr=bt_pr, bt_ar=bt_ar,
            history=train_histories[0],
            training_time=training_time,
            training_secs=time.time() - t0,
            r2=r2_v, wf_r2=wf_r2, wf_list=wf,
            rmse=rmse_v, mape=mape_v, da=da_v,
            n_r2=n_r2, n_mape=n_mape, n_rmse=n_rmse,
            resid_std=rs_v
        )

    ph = st.empty()
    with ph.container():
        st.markdown('<div class="skel"></div>', unsafe_allow_html=True)
        st.markdown('<div class="skel"></div>', unsafe_allow_html=True)
    try:
        art = train_ensemble(
            selected_ticker, str(start_date), str(end_date),
            time_step, epochs, batch_size, retrain,
            _n_rows=len(df_features))
    finally:
        ph.empty()

    # Unpack
    models       = art['models']
    scaler       = art['scaler']
    df_used      = art['df_feat']
    price_s      = art['price_series']
    train_n      = art['train_n']
    bt_preds     = art['bt_pp'];   bt_actuals  = art['bt_ap']
    bt_pred_ret  = art['bt_pr'];   bt_act_ret  = art['bt_ar']
    history      = art['history']
    n_feat       = art['n_feat']
    resid_std    = art['resid_std']
    last_price   = float(price_s.iloc[-1])
    beat_r2   = art['r2']   > art['n_r2']
    beat_mape = art['mape'] < art['n_mape']
    beat_rmse = art['rmse'] < art['n_rmse']
    wins      = sum([beat_r2, beat_mape, beat_rmse])

    def ccls(v, g, w, hi=True):
        if hi: return "good" if v >= g else "warn" if v >= w else "bad"
        return "good" if v <= g else "warn" if v <= w else "bad"

    # Confidence score
    r2n  = max(0, min(100, art['r2'] * 100))
    dan  = max(0, min(100, (art['da'] - 50) * 5))
    mpn  = max(0, min(100, (10 - art['mape']) * 10))
    conf = int(0.30 * r2n + 0.40 * dan + 0.20 * mpn + 0.10 * wins * 33.3)
    conf = max(0, min(100, conf))
    clbl = "High" if conf >= 70 else "Medium" if conf >= 45 else "Low"
    ccol = "#22c55e" if conf >= 70 else "#f59e0b" if conf >= 45 else "#ef4444"

    # ── FORECAST — using compute_features on extended price series ────────────
    # This approach is NaN-safe: extend actual dataframe, recompute features,
    # then scale and predict. No incremental update needed.
    extended_raw  = data_main.copy()
    fp_prices     = []
    chain_price   = last_price

    for step in range(days):
        # Build the feature window from the last time_step rows of extended_raw
        feat_ext, _ = compute_features(extended_raw)
        if len(feat_ext) < time_step:
            fp_prices.append(chain_price)
            continue

        window = feat_ext.values[-time_step:]
        window_scaled = scaler.transform(window).astype(np.float32)
        window_scaled = np.nan_to_num(window_scaled, nan=0.0)
        inp = window_scaled.reshape(1, time_step, n_feat)

        # Ensemble predict
        step_preds = []
        for m in models:
            p = float(m.predict(inp, verbose=0)[0, 0])
            step_preds.append(p)
        pred_sc = float(np.mean(step_preds))

        # Inverse transform
        dummy_row        = np.zeros((1, n_feat), dtype=np.float32)
        dummy_row[0, 0]  = pred_sc
        pred_lr          = float(scaler.inverse_transform(dummy_row)[0, 0])
        pred_lr          = np.clip(pred_lr, -0.15, 0.15)  # cap at ±15%
        pred_price       = chain_price * np.exp(pred_lr)
        pred_price       = float(np.nan_to_num(pred_price, nan=chain_price))
        fp_prices.append(pred_price)
        chain_price = pred_price

        # Append synthetic row to extended_raw for next step
        last_row = extended_raw.iloc[-1].copy()
        new_row  = pd.Series({
            'Open':      pred_price,
            'High':      pred_price * 1.005,
            'Low':       pred_price * 0.995,
            'Close':     pred_price,
            'Adj Close': pred_price,
            'Volume':    float(extended_raw['Volume'].iloc[-5:].mean())
        }, name=extended_raw.index[-1] + pd.Timedelta(days=1))
        extended_raw = pd.concat([extended_raw,
                                   new_row.to_frame().T])

    # Confidence intervals
    floor  = last_price * 0.50
    z      = 1.96
    uppers, lowers = [], []
    for i, p in enumerate(fp_prices):
        hw = min(z * resid_std * np.sqrt(i + 1), 0.15 * p)
        uppers.append(p + hw)
        lowers.append(max(p - hw, floor))

    future_dates = pd.date_range(
        start=data_main.index[-1] + pd.Timedelta(days=1),
        periods=days, freq='B')

    future_df = pd.DataFrame({
        'Date':      future_dates,
        'Predicted': [round(p, 2) for p in fp_prices],
        'Upper':     [round(u, 2) for u in uppers],
        'Lower':     [round(l, 2) for l in lowers],
        'Change %':  [f"{(p - last_price) / last_price * 100:+.2f}%"
                      for p in fp_prices]
    })

    # ── SECTION 1: Forecast chart ─────────────────────────────────────────────
    st.markdown(f"### {selected_ticker} — {days}-Day Price Forecast")

    age     = datetime.now() - art['training_time']
    age_str = f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m ago"
    info_cols = st.columns(6)
    for col, (lbl, val) in zip(info_cols, [
        ("Model",    "3-LSTM Ensemble"),
        ("Features", "16 signals"),
        ("Lookback", f"{time_step}d"),
        ("Train Time", f"{art['training_secs']:.0f}s"),
        ("Cached",   age_str),
        ("Confidence", f"{conf}/100 ({clbl})")
    ]):
        col.markdown(
            f"<div style='font-size:10px;color:#475569;text-transform:uppercase;"
            f"letter-spacing:.06em'>{lbl}</div>"
            f"<div style='font-size:13px;color:#94a3b8;font-weight:600'>{val}</div>",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    hx = price_s.index; hy = price_s.values
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(
        x=hx, y=hy, name='Historical',
        line=dict(color='#3b82f6', width=1.5)))
    fig_f.add_trace(go.Scatter(
        x=future_dates, y=fp_prices,
        name='Forecast', mode='lines+markers',
        line=dict(color='#f59e0b', width=2.5),
        marker=dict(size=6)))
    xb = list(future_dates) + list(future_dates[::-1])
    yb = list(uppers) + list(lowers[::-1])
    fig_f.add_trace(go.Scatter(
        x=xb, y=yb, fill='toself',
        fillcolor='rgba(245,158,11,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% CI'))
    fig_f.add_trace(go.Scatter(
        x=[price_s.index[-1]], y=[last_price],
        mode='markers',
        marker=dict(size=10, color='#ffffff', symbol='circle'),
        name=f'Last Close ${last_price:.2f}'))
    fig_f.update_layout(
        height=480,
        plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b', title='Price (USD)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h',
                    yanchor='bottom', y=1.01),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified')
    st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("##### Day-by-Day Forecast")
    st.dataframe(
        future_df.style.format(
            {"Predicted": "${:.2f}", "Upper": "${:.2f}", "Lower": "${:.2f}"}),
        use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 2: KPI metrics ────────────────────────────────────────────────
    st.markdown("### Model Performance Metrics")
    st.caption("Measured on the 20% test set the model never saw during training.")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    for col, lbl, val, sub, color in [
        (k1, "Confidence",         f"{conf}/100",
         clbl,             ccol),
        (k2, "Directional Accuracy", f"{art['da']:.1f}%",
         "Random = 50%",
         "#22c55e" if art['da'] >= 60 else
         "#f59e0b" if art['da'] >= 53 else "#ef4444"),
        (k3, "Walk-Forward R²",    f"{art['wf_r2']:.3f}",
         "5 time windows",
         "#22c55e" if art['wf_r2'] >= 0.80 else
         "#f59e0b" if art['wf_r2'] >= 0.65 else "#ef4444"),
        (k4, "MAPE",               f"{art['mape']:.2f}%",
         "Avg % price error",
         "#22c55e" if art['mape'] <= 2 else
         "#f59e0b" if art['mape'] <= 4 else "#ef4444"),
        (k5, "RMSE",               f"${art['rmse']:.2f}",
         "Avg $ error/day",  "#94a3b8"),
        (k6, "Ensemble Models",    "3",
         "LSTM + BatchNorm", "#818cf8"),
    ]:
        col.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{lbl}</div>
          <div class="kpi-value" style="color:{color}">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 3: Naïve baseline ─────────────────────────────────────────────
    st.markdown("### vs Naïve Baseline")
    st.caption("Naïve = tomorrow's price equals today's. A useful model must beat this.")
    vc_  = "good" if wins == 3 else "warn" if wins >= 2 else "bad"
    vt_  = ("Beats baseline on all 3 metrics" if wins == 3
            else f"Beats baseline on {wins}/3" if wins >= 2
            else "Does not beat baseline")
    b1, b2, b3, b4 = st.columns(4)
    r2c_ = "good" if beat_r2   else "bad"
    mpc_ = "good" if beat_mape else "bad"
    rmc_ = "good" if beat_rmse else "bad"
    for col, title, vals in [
        (b1, "METRIC",    [("R² (higher better)", "#64748b"),
                           ("MAPE (lower better)", "#64748b"),
                           ("RMSE (lower better)", "#64748b")]),
        (b2, "OUR ENSEMBLE",
         [(f"{art['r2']:.3f}",    r2c_),
          (f"{art['mape']:.2f}%", mpc_),
          (f"${art['rmse']:.2f}", rmc_)]),
        (b3, "NAÏVE BASELINE",
         [(f"{art['n_r2']:.3f}",    "#94a3b8"),
          (f"{art['n_mape']:.2f}%", "#94a3b8"),
          (f"${art['n_rmse']:.2f}", "#94a3b8")]),
        (b4, "VERDICT",
         [(vt_, vc_), ("", "#475569"), ("", "#475569")]),
    ]:
        rows = "".join(
            f"<div class='{c}' style='margin-top:8px;font-size:13px'>{v}</div>"
            for v, c in vals)
        col.markdown(
            f'<div class="bsbox"><div style="font-size:11px;color:#475569;'
            f'margin-bottom:6px">{title}</div>{rows}</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 4: Charts 2x2 ────────────────────────────────────────────────
    st.markdown("### Backtest Analysis")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Actual vs Predicted Price**")
        bt_start = art['time_step'] + train_n
        bt_idx   = df_used.index[bt_start: bt_start + len(bt_preds)]
        fig_bt   = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_idx, y=bt_actuals,
                                    name='Actual',
                                    line=dict(color='#3b82f6', width=1.5)))
        fig_bt.add_trace(go.Scatter(x=bt_idx, y=bt_preds,
                                    name='Predicted',
                                    line=dict(color='#f59e0b', width=1.5)))
        fig_bt.update_layout(
            height=280, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_bt, use_container_width=True)

    with ch2:
        st.markdown("**Training Loss**")
        ep_r  = list(range(1, len(history.get('loss', [])) + 1))
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=ep_r, y=history.get('loss', []),
                                   mode='lines', name='Train',
                                   line=dict(color='#3b82f6')))
        if 'val_loss' in history:
            fig_l.add_trace(go.Scatter(x=ep_r, y=history['val_loss'],
                                       mode='lines', name='Validation',
                                       line=dict(color='#f59e0b')))
        fig_l.update_layout(
            height=280, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e293b', title='Epoch'),
            yaxis=dict(gridcolor='#1e293b', title='Loss'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_l, use_container_width=True)

    ch3, ch4 = st.columns(2)

    with ch3:
        st.markdown("**Directional Accuracy**")
        st.caption("Green = correct UP/DOWN call. Any score > 50% beats random.")
        dir_c  = (np.sign(bt_pred_ret) == np.sign(bt_act_ret)).astype(int)
        bt_i2  = df_used.index[art['time_step'] + train_n:
                                art['time_step'] + train_n + len(bt_preds)]
        bclrs  = ['#22c55e' if c == 1 else '#ef4444' for c in dir_c]
        fig_d  = go.Figure()
        fig_d.add_trace(go.Bar(x=bt_i2, y=dir_c,
                               marker_color=bclrs, showlegend=False))
        fig_d.add_hline(y=0.5, line_dash='dash', line_color='#475569',
                        annotation_text="50% Random")
        fig_d.update_layout(
            height=250, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b'),
            margin=dict(l=0, r=0, t=10, b=0),
            title=dict(
                text=f"Correct {art['da']:.1f}%"
                     f" ({int(dir_c.sum())}/{len(dir_c)})",
                font=dict(size=12, color='#94a3b8')))
        st.plotly_chart(fig_d, use_container_width=True)

    with ch4:
        st.markdown("**Error Distribution**")
        st.caption("Tight bell near $0 = model errors are small and balanced.")
        residuals = bt_actuals - bt_preds
        fig_r = go.Figure()
        fig_r.add_trace(go.Histogram(x=residuals, nbinsx=40,
                                     marker_color='#6366f1', opacity=0.8))
        fig_r.add_vline(x=0, line_color='#94a3b8', line_dash='dash')
        fig_r.update_layout(
            height=250, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e293b', title='Error ($)'),
            yaxis=dict(gridcolor='#1e293b'),
            margin=dict(l=0, r=0, t=10, b=0),
            title=dict(
                text=f"Mean ${residuals.mean():.2f}"
                     f"  Std ${residuals.std():.2f}",
                font=dict(size=12, color='#94a3b8')))
        st.plotly_chart(fig_r, use_container_width=True)

    if art['wf_list']:
        st.markdown("**Walk-Forward R² Across Time Windows**")
        st.caption("Each bar = model accuracy in a different time period. Consistent = works across all market conditions.")
        wfc    = ['#22c55e' if v >= 0.80 else
                  '#f59e0b' if v >= 0.65 else '#ef4444'
                  for v in art['wf_list']]
        fig_wf = go.Figure()
        for fn, fv, fc in zip(
                [f"Window {i+1}" for i in range(len(art['wf_list']))],
                art['wf_list'], wfc):
            fig_wf.add_trace(go.Bar(
                x=[fn], y=[fv], marker_color=fc,
                text=[f"{fv:.3f}"], textposition='outside',
                showlegend=False))
        fig_wf.add_hline(y=0.80, line_dash='dash', line_color='#22c55e',
                         annotation_text="Target 0.80")
        fig_wf.add_hline(y=0.65, line_dash='dot', line_color='#f59e0b',
                         annotation_text="Acceptable 0.65")
        ymin = min(min(art['wf_list']) - 0.05, -0.05)
        fig_wf.update_layout(
            height=260, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', range=[ymin, 1.05]),
            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fs1, fs2, fs3, fs4 = st.columns(4)
    fs1.metric("Confidence",     f"{conf}/100 ({clbl})")
    fs2.metric("Directional Acc", f"{art['da']:.1f}%")
    fs3.metric("Avg Price Error", f"{art['mape']:.2f}%")
    fs4.metric("Beats Naïve",
               "Yes (all 3)" if wins == 3
               else f"Partial ({wins}/3)" if wins >= 2 else "No")


# ==============================================================================
#  TAB 3 — SENTIMENT
# ==============================================================================
elif tab == "Sentiment":
    st.subheader("News Sentiment Analysis")
    if news_posts:
        df_s = pd.DataFrame({'News': news_posts,
                             'Link': news_links,
                             'Score': vader_scores})
        def color_score(val):
            c = '#22c55e' if val > 0.1 else '#ef4444' if val < -0.1 else '#94a3b8'
            return f"color: {c}"
        st.dataframe(
            df_s.style.map(color_score, subset=['Score'])
                      .format({'Score': '{:.3f}'}),
            use_container_width=True)
        pos = sum(1 for s in vader_scores if s >  0.1)
        neg = sum(1 for s in vader_scores if s < -0.1)
        neu = len(vader_scores) - pos - neg
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", pos)
        c2.metric("Negative", neg)
        c3.metric("Neutral",  neu)
        avg = np.mean(vader_scores) if vader_scores else 0
        overall = ("Bullish" if avg > 0.05
                   else "Bearish" if avg < -0.05 else "Neutral")
        st.info(f"Overall sentiment for **{selected_ticker}**: "
                f"**{overall}** (avg {avg:.3f})")
    else:
        st.info("No recent news found for this ticker.")

# ==============================================================================
#  TAB 4 — COMPARISON
# ==============================================================================
elif tab == "Comparison":
    st.subheader(f"{selected_ticker} vs {compare_ticker} — Head to Head")
    if data_main is None or data_compare is None:
        st.error("Not enough data for one or both tickers."); st.stop()

    bm  = data_main['Adj Close'].iloc[0]
    bc  = data_compare['Adj Close'].iloc[0]
    dm  = (data_main['Adj Close'] / bm - 1) * 100
    dc  = (data_compare['Adj Close'] / bc - 1) * 100
    rm  = float(dm.iloc[-1]); rc  = float(dc.iloc[-1])
    vm  = float(data_main['Adj Close'].pct_change().std() * np.sqrt(252) * 100)
    vc2 = float(data_compare['Adj Close'].pct_change().std() * np.sqrt(252) * 100)
    ra_m = rm / vm if vm > 0 else 0
    ra_c = rc / vc2 if vc2 > 0 else 0
    dd_m = compute_drawdown(data_main['Adj Close'])
    dd_c = compute_drawdown(data_compare['Adj Close'])
    mdd_m = float(dd_m.min()); mdd_c = float(dd_c.min())
    better_ret = selected_ticker if rm > rc else compare_ticker
    ra_txt = ("both performed similarly on risk-adjusted basis"
              if abs(ra_m - ra_c) < 0.5
              else (f"{selected_ticker} had better risk-adjusted returns"
                    if ra_m > ra_c
                    else f"{compare_ticker} had better risk-adjusted returns"))

    st.markdown(f"""<div class="comp-box"><p>
    Since <b>{start_date}</b>, <b>{selected_ticker}</b> returned
    <b>{rm:+.1f}%</b> vs <b>{compare_ticker}</b> at <b>{rc:+.1f}%</b>
    — a gap of <b>{abs(rm-rc):.1f}pp</b>.
    <b>{better_ret}</b> was the stronger raw performer.
    On risk-adjusted terms, {ra_txt}.
    </p></div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{selected_ticker} Return",     f"{rm:+.2f}%")
    c2.metric(f"{compare_ticker} Return",      f"{rc:+.2f}%")
    c3.metric(f"{selected_ticker} Volatility", f"{vm:.1f}%")
    c4.metric(f"{compare_ticker} Volatility",  f"{vc2:.1f}%")

    # Chart 1 — normalised return
    st.markdown("#### Cumulative Return")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data_main.index, y=dm,
                              name=selected_ticker,
                              line=dict(color='#26A69A')))
    fig1.add_trace(go.Scatter(x=data_compare.index, y=dc,
                              name=compare_ticker,
                              line=dict(color='#AB47BC')))
    fig1.add_hline(y=0, line_dash='dot', line_color='#475569')
    fig1.update_layout(
        height=380, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b', title='Return (%)'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 — rolling 12-month
    st.markdown("#### Rolling 12-Month Return")
    roll_m = data_main['Adj Close'].pct_change(252) * 100
    roll_c = data_compare['Adj Close'].pct_change(252) * 100
    fig2   = go.Figure()
    fig2.add_trace(go.Scatter(x=data_main.index, y=roll_m,
                              name=selected_ticker,
                              line=dict(color='#26A69A'), fill='tozeroy',
                              fillcolor='rgba(38,166,154,0.08)'))
    fig2.add_trace(go.Scatter(x=data_compare.index, y=roll_c,
                              name=compare_ticker,
                              line=dict(color='#AB47BC'), fill='tozeroy',
                              fillcolor='rgba(171,71,188,0.08)'))
    fig2.add_hline(y=0, line_color='#475569', line_dash='dash')
    fig2.update_layout(
        height=320, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b', title='12M Return (%)'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3+4 — drawdown + risk-return
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Drawdown from Peak**")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=data_main.index, y=dd_m,
                                  name=selected_ticker,
                                  line=dict(color='#26A69A'), fill='tozeroy',
                                  fillcolor='rgba(38,166,154,0.12)'))
        fig3.add_trace(go.Scatter(x=data_compare.index, y=dd_c,
                                  name=compare_ticker,
                                  line=dict(color='#AB47BC'), fill='tozeroy',
                                  fillcolor='rgba(171,71,188,0.12)'))
        fig3.update_layout(
            height=300, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', title='Drawdown (%)'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with cc2:
        st.markdown("**Risk vs Return**")
        st.caption("Top-left = ideal. Bottom-right = worst.")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=[vm, vc2], y=[rm, rc], mode='markers+text',
            text=[selected_ticker, compare_ticker],
            textposition='top center',
            marker=dict(size=22,
                        color=['#26A69A', '#AB47BC'],
                        symbol='diamond')))
        fig4.update_layout(
            height=300, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e293b', title='Volatility (Risk) %'),
            yaxis=dict(gridcolor='#1e293b', title='Total Return %'),
            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Summary Table")
    summary_df = pd.DataFrame({
        'Metric':        ['Total Return', 'Volatility',
                          'Max Drawdown', 'Risk-Adj Return'],
        selected_ticker: [f"{rm:+.2f}%", f"{vm:.1f}%",
                          f"{mdd_m:.1f}%", f"{ra_m:.2f}x"],
        compare_ticker:  [f"{rc:+.2f}%", f"{vc2:.1f}%",
                          f"{mdd_c:.1f}%", f"{ra_c:.2f}x"],
        'Winner':        [
            selected_ticker if rm  > rc   else compare_ticker,
            selected_ticker if vm  < vc2  else compare_ticker,
            selected_ticker if mdd_m > mdd_c else compare_ticker,
            selected_ticker if ra_m > ra_c   else compare_ticker
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ==============================================================================
#  TAB 5 — PORTFOLIO ANALYZER (all 7 features, fixed scopes)
# ==============================================================================
elif tab == "Portfolio Analyzer":
    st.subheader("Portfolio Analyzer — Professional Suite")

    port_tickers = st.multiselect(
        "Select Portfolio Stocks", tickers,
        default=[selected_ticker, compare_ticker])
    if len(port_tickers) < 2:
        st.warning("Select at least 2 tickers."); st.stop()

    # ── Feature 1: Portfolio Creation ─────────────────────────────────────────
    st.markdown("#### Portfolio Weights")
    weights, total_w = [], 0.0
    wcols = st.columns(len(port_tickers))
    for i, tick in enumerate(port_tickers):
        w = wcols[i].number_input(
            f"{tick} (%)", 0.0, 100.0,
            round(100.0 / len(port_tickers), 2),
            key=f"w_{tick}")
        weights.append(w / 100); total_w += w

    bar_col = '#22c55e' if abs(total_w - 100) < 0.1 else '#ef4444'
    st.markdown(
        f'<div style="background:#1e293b;border-radius:6px;height:10px;'
        f'margin-bottom:4px"><div style="width:{min(total_w,100):.1f}%;'
        f'height:10px;border-radius:6px;background:{bar_col}"></div></div>'
        f'<div style="font-size:12px;color:{bar_col};margin-bottom:12px">'
        f'Total: {total_w:.1f}%</div>',
        unsafe_allow_html=True)

    if abs(total_w - 100) > 0.1:
        st.warning(f"Weights must sum to 100%. Currently {total_w:.1f}%")
        st.stop()

    # Fetch data
    data_dict = {}
    for tick in port_tickers:
        d = fetch_stock_data(tick, start_date, end_date)
        if d is None:
            st.error(f"Data missing for {tick}."); st.stop()
        data_dict[tick] = d['Adj Close']

    port_df  = pd.DataFrame(data_dict).dropna()
    rets     = port_df.pct_change().dropna()
    m_ret    = rets.mean() * 252
    cov_mat  = rets.cov() * 252
    w_np     = np.array(weights)
    n_assets = len(port_tickers)
    p_ret    = float(np.dot(m_ret, w_np))
    p_vol    = float(np.sqrt(np.dot(w_np.T, np.dot(cov_mat, w_np))))
    sharpe   = (p_ret - 0.03) / p_vol if p_vol > 0 else 0.0

    port_daily = (rets * w_np).sum(axis=1)
    downside   = port_daily[port_daily < 0].std() * np.sqrt(252)
    sortino    = (p_ret - 0.03) / downside if downside > 0 else 0.0

    # Beta vs SPY
    beta   = 1.0
    spy_d  = fetch_stock_data('SPY', start_date, end_date)
    if spy_d is not None:
        spy_ret_s = spy_d['Adj Close'].pct_change().dropna()
        pa  = port_daily.reindex(spy_ret_s.index).dropna()
        sa  = spy_ret_s.reindex(pa.index).dropna()
        pa  = pa.reindex(sa.index).dropna()
        if len(pa) > 30:
            cov_ps  = np.cov(pa, sa)[0, 1]
            var_spy = float(np.var(sa))
            beta    = float(cov_ps / var_spy) if var_spy > 0 else 1.0

    port_idx   = (1 + port_daily).cumprod()
    roll_max_p = port_idx.cummax()
    dd_port    = (port_idx - roll_max_p) / roll_max_p * 100
    max_dd_p   = float(dd_port.min())
    port_cum   = port_idx * 100 - 100
    var_95     = float(np.percentile(port_daily, 5) * 100)
    cvar_95    = float(
        port_daily[port_daily <= np.percentile(port_daily, 5)].mean() * 100)

    # ── Feature 2: Performance Analytics ──────────────────────────────────────
    st.markdown("#### Performance Analytics")
    pk1,pk2,pk3,pk4,pk5,pk6,pk7 = st.columns(7)
    for col, lbl, val, color in [
        (pk1, "Annual Return",    f"{p_ret*100:.2f}%",
         "#22c55e" if p_ret > 0.08 else "#f59e0b"),
        (pk2, "Annual Volatility", f"{p_vol*100:.2f}%", "#94a3b8"),
        (pk3, "Sharpe Ratio",     f"{sharpe:.2f}",
         "#22c55e" if sharpe > 1 else "#f59e0b" if sharpe > 0.5 else "#ef4444"),
        (pk4, "Sortino Ratio",    f"{sortino:.2f}",
         "#22c55e" if sortino > 1.5 else "#f59e0b" if sortino > 0.8 else "#ef4444"),
        (pk5, "Beta vs S&P",      f"{beta:.2f}",
         "#22c55e" if 0.8 <= beta <= 1.2 else "#f59e0b"),
        (pk6, "Max Drawdown",     f"{max_dd_p:.1f}%",
         "#ef4444" if max_dd_p < -30 else "#f59e0b"),
        (pk7, "VaR 95% Daily",    f"{var_95:.2f}%", "#ef4444"),
    ]:
        col.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{lbl}</div>
          <div class="kpi-value" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Portfolio vs SPY
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(
        x=port_cum.index, y=port_cum.values,
        name='Your Portfolio', line=dict(color='#6366f1', width=2)))
    if spy_d is not None:
        spy_ret2 = spy_d['Adj Close'].pct_change().dropna()
        spy_cum2 = (1 + spy_ret2).cumprod() * 100 - 100
        spy_cum2 = spy_cum2.reindex(port_cum.index, method='ffill').dropna()
        fig_b.add_trace(go.Scatter(
            x=spy_cum2.index, y=spy_cum2.values,
            name='S&P 500 (SPY)',
            line=dict(color='#94a3b8', width=1.5, dash='dash')))
        pf = float(port_cum.iloc[-1]); sf = float(spy_cum2.iloc[-1])
        beat = pf > sf
        st.markdown(
            f"<div class='ins-card'>"
            f"{'✅ Portfolio outperformed' if beat else '📉 Portfolio underperformed'}"
            f" S&P 500 by <b>{abs(pf-sf):.1f}pp</b> over the period.</div>",
            unsafe_allow_html=True)
    fig_b.add_hline(y=0, line_dash='dot', line_color='#475569')
    fig_b.update_layout(
        height=380, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b', title='Cumulative Return (%)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h',
                    y=1.02, x=0, yanchor='bottom'),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Portfolio vs S&P 500 Benchmark",
                   font=dict(color='#94a3b8', size=13)))
    st.plotly_chart(fig_b, use_container_width=True)

    # ── Feature 3: Risk Metrics ────────────────────────────────────────────────
    st.markdown("#### Risk Metrics")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Portfolio Drawdown**")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_port.index, y=dd_port.values,
            name='Drawdown', fill='tozeroy',
            fillcolor='rgba(239,68,68,0.12)',
            line=dict(color='#ef4444')))
        fig_dd.update_layout(
            height=280, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', title='Drawdown (%)'),
            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_dd, use_container_width=True)

    with rc2:
        st.markdown("**Individual Stock Contributions**")
        contrib_ret  = [float(m_ret[t] * w_np[i] * 100)
                        for i, t in enumerate(port_tickers)]
        contrib_risk = [
            float(np.dot(cov_mat.iloc[i].values, w_np) / p_vol * w_np[i] * 100)
            for i in range(n_assets)]
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(name='Return Contrib.',
                               x=port_tickers, y=contrib_ret,
                               marker_color='#22c55e'))
        fig_c.add_trace(go.Bar(name='Risk Contrib.',
                               x=port_tickers, y=contrib_risk,
                               marker_color='#ef4444'))
        fig_c.update_layout(
            barmode='group', height=280,
            plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'),
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b', title='%'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_c, use_container_width=True)

    # ── Feature 4: Correlation Matrix ─────────────────────────────────────────
    st.markdown("#### Correlation Matrix")
    st.caption("+1 = always move together. 0 = independent. -1 = opposite (hedge).")
    fig_h = px.imshow(rets.corr(), text_auto=True, aspect='auto',
                      color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig_h.update_layout(
        height=320, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
        font=dict(color='#94a3b8'), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_h, use_container_width=True)

    corr_matrix = rets.corr()
    max_pair = ('', '', 0.0); min_pair = ('', '', 1.0)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            v = float(corr_matrix.iloc[i, j])
            if v > max_pair[2]: max_pair = (port_tickers[i], port_tickers[j], v)
            if v < min_pair[2]: min_pair = (port_tickers[i], port_tickers[j], v)
    st.markdown(
        f"<div class='ins-card'>Most correlated: "
        f"<b>{max_pair[0]} & {max_pair[1]}</b> ({max_pair[2]:.2f}) — "
        f"less diversification benefit. "
        f"Least correlated: <b>{min_pair[0]} & {min_pair[1]}</b> "
        f"({min_pair[2]:.2f}) — best diversifier pair.</div>",
        unsafe_allow_html=True)

    # ── Feature 5: Efficient Frontier ─────────────────────────────────────────
    st.markdown("#### Efficient Frontier")
    st.caption("3,000 random weight combinations. Colour = Sharpe Ratio. Red = your portfolio. Gold = optimal.")
    n_sim  = 3000
    s_rets, s_vols, s_sharpes, s_wts = [], [], [], []
    for _ in range(n_sim):
        w_   = np.random.dirichlet(np.ones(n_assets))
        r_   = float(np.dot(m_ret, w_))
        v_   = float(np.sqrt(np.dot(w_.T, np.dot(cov_mat, w_))))
        s_   = (r_ - 0.03) / v_ if v_ > 0 else 0
        s_rets.append(r_ * 100); s_vols.append(v_ * 100)
        s_sharpes.append(s_); s_wts.append(w_)

    hover_ef = [" | ".join([f"{port_tickers[j]}: {s_wts[i][j]*100:.1f}%"
                            for j in range(n_assets)]) for i in range(n_sim)]
    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=s_vols, y=s_rets, mode='markers',
        marker=dict(size=4, color=s_sharpes, colorscale='Viridis',
                    showscale=True, colorbar=dict(title='Sharpe')),
        text=hover_ef,
        hovertemplate='Return:%{y:.1f}%<br>Risk:%{x:.1f}%<br>%{text}<extra></extra>',
        name='Simulated'))
    fig_ef.add_trace(go.Scatter(
        x=[p_vol * 100], y=[p_ret * 100], mode='markers+text',
        marker=dict(size=20, color='red', symbol='star'),
        text=['Your Portfolio'], textposition='top center',
        name='Your Portfolio'))
    best_idx = int(np.argmax(s_sharpes))
    fig_ef.add_trace(go.Scatter(
        x=[s_vols[best_idx]], y=[s_rets[best_idx]], mode='markers+text',
        marker=dict(size=20, color='#fbbf24', symbol='star'),
        text=['Optimal'], textposition='top center',
        name='Max Sharpe'))
    fig_ef.update_layout(
        height=500, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='#1e293b', title='Risk (Volatility %)'),
        yaxis=dict(gridcolor='#1e293b', title='Expected Return (%)'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_ef, use_container_width=True)

    best_w   = s_wts[best_idx]
    opt_hint = " | ".join([f"{port_tickers[j]}: {best_w[j]*100:.1f}%"
                           for j in range(n_assets)])
    st.markdown(
        f"<div class='ins-card'>Optimal weights (Sharpe "
        f"{s_sharpes[best_idx]:.2f}): <b>{opt_hint}</b>. "
        f"Your current Sharpe: <b>{sharpe:.2f}</b>.</div>",
        unsafe_allow_html=True)

    # ── Feature 6: Monte Carlo Simulation ────────────────────────────────────
    st.markdown("#### Monte Carlo Simulation — 1-Year Outlook")
    st.caption("500 simulated future portfolio paths. Starting value = $100 invested today.")

    n_mc        = 500
    n_days_mc   = 252
    daily_mean  = float(port_daily.mean())
    daily_std   = float(port_daily.std())
    start_val   = 100.0   # normalised $100 — no dependency on prediction tab

    mc_paths = np.zeros((n_mc, n_days_mc))
    for i in range(n_mc):
        sim_r         = np.random.normal(daily_mean, daily_std, n_days_mc)
        mc_paths[i]   = start_val * (1 + sim_r).cumprod()

    mc_end = mc_paths[:, -1]
    p5_    = float(np.percentile(mc_end, 5))
    p50_   = float(np.percentile(mc_end, 50))
    p95_   = float(np.percentile(mc_end, 95))

    fig_mc = go.Figure()
    for i in range(0, n_mc, 5):
        fig_mc.add_trace(go.Scatter(
            x=list(range(n_days_mc)), y=mc_paths[i],
            mode='lines',
            line=dict(color='rgba(99,102,241,0.06)', width=1),
            showlegend=False))
    for path, name, color in [
        (np.percentile(mc_paths, 50, axis=0), 'Median',    '#6366f1'),
        (np.percentile(mc_paths, 95, axis=0), 'Best 5%',   '#22c55e'),
        (np.percentile(mc_paths, 5,  axis=0), 'Worst 5%',  '#ef4444'),
    ]:
        fig_mc.add_trace(go.Scatter(
            x=list(range(n_days_mc)), y=path,
            name=name, line=dict(color=color, width=2)))
    fig_mc.update_layout(
        height=400, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='#1e293b', title='Trading Days'),
        yaxis=dict(gridcolor='#1e293b', title='Portfolio Value ($)'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_mc, use_container_width=True)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Median Outcome (1Y)", f"${p50_:.2f}",
               f"{(p50_-start_val)/start_val*100:+.1f}%")
    mc2.metric("Best Case (95th %)",  f"${p95_:.2f}",
               f"{(p95_-start_val)/start_val*100:+.1f}%")
    mc3.metric("Worst Case (5th %)",  f"${p5_:.2f}",
               f"{(p5_-start_val)/start_val*100:+.1f}%")

    st.markdown(
        f"<div class='ins-card'>Starting from <b>$100</b>, the median"
        f" 1-year outcome is <b>${p50_:.2f}</b>. In the worst 5% of"
        f" scenarios the portfolio falls to <b>${p5_:.2f}</b>."
        f" In the best 5% it reaches <b>${p95_:.2f}</b>.</div>",
        unsafe_allow_html=True)

    # Full analytics table
    st.markdown("#### Individual Stock Analytics")
    indiv_df = pd.DataFrame({
        'Ticker':          port_tickers,
        'Weight':          [f"{w*100:.1f}%" for w in weights],
        'Ann. Return':     [f"{float(m_ret[t])*100:.2f}%" for t in port_tickers],
        'Ann. Volatility': [f"{float(np.sqrt(cov_mat.loc[t,t]))*100:.2f}%"
                            for t in port_tickers],
        'Return Contrib.': [f"{r:.2f}%" for r in contrib_ret],
        'Risk Contrib.':   [f"{r:.2f}%" for r in contrib_risk],
    })
    st.dataframe(indiv_df, use_container_width=True, hide_index=True)

    # ── Feature 7: Groq AI Insights ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 AI Insights Dashboard")
    st.caption("Powered by Llama 3.1 70B via Groq. Add your free Groq API key in the sidebar.")

    if not groq_key:
        st.info(
            "**To enable AI Insights:** Get your free API key at "
            "[console.groq.com](https://console.groq.com) "
            "(takes 2 minutes, no credit card). "
            "Then paste it into the **Groq API Key** field in the sidebar.")
    else:
        if st.button("Generate AI Portfolio Insights", type="primary"):
            prompt = f"""You are a senior portfolio manager with 20 years of experience.
Analyze this portfolio and provide exactly 5 specific, data-driven insights.
Each insight must cite the actual numbers. Be direct and actionable.

Portfolio:
- Stocks: {', '.join([f"{port_tickers[i]} ({weights[i]*100:.1f}%)" for i in range(n_assets)])}
- Expected Annual Return: {p_ret*100:.2f}%
- Annual Volatility: {p_vol*100:.2f}%
- Sharpe Ratio: {sharpe:.2f}
- Sortino Ratio: {sortino:.2f}
- Beta vs S&P 500: {beta:.2f}
- Max Drawdown: {max_dd_p:.1f}%
- Daily VaR (95%): {var_95:.2f}%
- CVaR (95%): {cvar_95:.2f}%
- Most correlated pair: {max_pair[0]} & {max_pair[1]} (r={max_pair[2]:.2f})
- Least correlated pair: {min_pair[0]} & {min_pair[1]} (r={min_pair[2]:.2f})
- Monte Carlo median 1Y: ${p50_:.2f} from $100
- Monte Carlo worst 5%: ${p5_:.2f}
- Current Sharpe: {sharpe:.2f} vs Optimal: {s_sharpes[best_idx]:.2f}
- Optimal weights: {opt_hint}

Format your response as exactly 5 insights:
INSIGHT 1: [Short Title]
[2-3 sentences with numbers and recommendation]

INSIGHT 2: [Short Title]
[2-3 sentences with numbers and recommendation]

Continue for insights 3, 4, 5."""

            with st.spinner("Generating insights with Llama 3.1 70B..."):
                response = call_groq(groq_key, prompt)

            if response.startswith("Error") or "error" in response.lower()[:20]:
                st.error(f"Groq API error: {response}")
            else:
                parts = [p.strip() for p in response.split('\n\n') if p.strip()]
                for part in parts:
                    lines  = part.split('\n', 1)
                    title  = lines[0].replace('INSIGHT', '').strip(' 12345:')
                    body   = lines[1].strip() if len(lines) > 1 else part
                    st.markdown(f"""<div class="ai-card">
                      <div class="ai-title">AI Insight — {title}</div>
                      {body}
                    </div>""", unsafe_allow_html=True)

# ==============================================================================
#  NEWS TICKER — bottom of every page
# ==============================================================================
st.markdown("---")
st.markdown("### Latest Headlines")
all_h    = news_headlines + news_headlines
anim_dur = max(15, len(news_headlines) * 3)
st.markdown(f"""
<style>
.ticker-container{{height:160px;overflow:hidden;background:#0f172a;padding:14px;
  border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.3);color:#fff;
  font-family:'Segoe UI',sans-serif;position:relative}}
.ticker-wrapper{{animation:scroll-up {anim_dur}s linear infinite;will-change:transform}}
@keyframes scroll-up{{0%{{transform:translateY(0)}}100%{{transform:translateY(-50%)}}}}
.ticker-item{{padding:10px 0;font-size:14px;line-height:1.5;min-height:38px;
  overflow:hidden;word-wrap:break-word;border-bottom:1px solid #1e293b}}
</style>""", unsafe_allow_html=True)
html_c = '<div class="ticker-container"><div class="ticker-wrapper">'
for h in all_h:
    html_c += f'<div class="ticker-item">{h}</div>'
html_c += '</div></div>'
st.markdown(html_c, unsafe_allow_html=True)
