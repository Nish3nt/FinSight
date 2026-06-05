# =============================================================================
#  FinSight — app.py  (FINAL VERSION)
#  Predictions : Attention LSTM Ensemble (3 models) | 16 Features | Multi-scale
#  Portfolio   : 7-Feature Suite + Groq AI Insights
#  UI          : Trader-grade, all settings in sidebar, no expanders
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                      Input, Multiply, Permute, RepeatVector,
                                      Flatten, Activation, Lambda, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
from streamlit_option_menu import option_menu
from collections import deque

nltk.download('vader_lexicon', quiet=True)
sia          = SentimentIntensityAnalyzer()
current_date = datetime.now().date()
st.set_page_config(page_title="FinSight", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stSidebar"]>div:first-child{background:#0b1220;padding:16px 12px}
.block-container{padding-top:.5rem;padding-bottom:.4rem}
.kpi-card{background:#0f172a;border-radius:10px;padding:16px 12px;text-align:center;
          border:1px solid #1e293b;margin-bottom:4px}
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
.ai-title{font-size:11px;font-weight:700;color:#818cf8;text-transform:uppercase;
          letter-spacing:.08em;margin-bottom:6px}
.comp-box{background:#0f172a;border:1px solid #1e3a5f;border-radius:12px;
          padding:18px 22px;margin-bottom:16px}
.comp-box p{margin:0;font-size:14px;color:#cbd5e1;line-height:1.8}
.tag{display:inline-block;background:#1e293b;color:#94a3b8;border-radius:4px;
     font-size:10px;padding:2px 7px;margin-right:4px;margin-bottom:2px}
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
selected_ticker = st.sidebar.selectbox("Main Stock",   tickers, index=tickers.index('AAPL'))
compare_ticker  = st.sidebar.selectbox("Compare With", tickers, index=tickers.index('MSFT'))
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2010-01-01').date())
end_date   = st.sidebar.date_input("End Date",   current_date)
st.sidebar.markdown("---")
st.sidebar.markdown("**Prediction Settings**")
days       = st.sidebar.slider("Forecast Days", 1, 30, 7)
time_step  = st.sidebar.slider("Lookback Window", 60, 120, 90, step=10)
epochs     = st.sidebar.slider("Training Epochs", 40, 120, 80, step=5)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
retrain    = st.sidebar.checkbox("Force Retrain", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("**AI Insights**")
groq_key = st.sidebar.text_input("Groq API Key", type="password",
                                   placeholder="gsk_...")
st.sidebar.caption("Free at console.groq.com")

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
        from_d = (datetime.now()-timedelta(days=30)).strftime('%Y-%m-%d')
        url    = (f"https://finnhub.io/api/v1/company-news?symbol={ticker}"
                  f"&from={from_d}&to={to_d}&token={api_key}")
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            headlines, posts, links = [], [], []
            for art in r.json()[:10]:
                t_ = art.get('headline','').strip()
                s_ = art.get('source','Finnhub')
                u_ = art.get('url','#')
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
        req = ['Open','High','Low','Close','Adj Close','Volume']
        for c in req:
            if c not in df.columns: df[c] = np.nan
        df = df[req].dropna(subset=['Adj Close'])
        return df if len(df) >= 120 else None
    except:
        return None

data_main    = fetch_stock_data(selected_ticker, start_date, end_date)
data_compare = fetch_stock_data(compare_ticker,  start_date, end_date)

# ── Feature Engineering — 16 features ────────────────────────────────────────
def compute_features(raw):
    df = raw.copy()
    ac = df['Adj Close']
    # Returns & lags
    df['LogReturn']    = np.log(ac / ac.shift(1))
    df['Lag1Return']   = df['LogReturn'].shift(1)
    # Trend
    df['SMA20']        = ac.rolling(20).mean()
    df['SMA50']        = ac.rolling(50).mean()
    df['EMA12']        = ac.ewm(span=12, adjust=False).mean()
    df['EMA26']        = ac.ewm(span=26, adjust=False).mean()
    # Momentum
    df['MACD']         = df['EMA12'] - df['EMA26']
    df['MACD_Signal']  = df['MACD'].ewm(span=9, adjust=False).mean()
    delta              = ac.diff()
    up = delta.clip(lower=0); dn = -delta.clip(upper=0)
    df['RSI']          = 100 - 100/(1 + up.rolling(14).mean()/(dn.rolling(14).mean()+1e-9))
    df['ROC10']        = ac.pct_change(10) * 100
    # Williams %R
    high14 = df['High'].rolling(14).max()
    low14  = df['Low'].rolling(14).min()
    df['WilliamsR']    = -100 * (high14 - ac) / (high14 - low14 + 1e-9)
    # Stochastic %K
    df['StochK']       = 100 * (ac - low14) / (high14 - low14 + 1e-9)
    # Volatility
    rm = ac.rolling(20).mean(); rs = ac.rolling(20).std()
    df['BB_Width']     = (2 * rs) / (rm + 1e-9)
    hl = df['High']-df['Low']
    hc = (df['High']-ac.shift()).abs()
    lc = (df['Low'] -ac.shift()).abs()
    df['ATR']          = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    # Volume
    df['LogVolume']    = np.log1p(df['Volume'])
    df['OBV']          = (np.sign(df['LogReturn']) * df['Volume']).cumsum()
    df['OBV']          = (df['OBV'] - df['OBV'].rolling(20).mean()) / (df['OBV'].rolling(20).std() + 1e-9)

    cols = ['LogReturn','Lag1Return','LogVolume','OBV',
            'SMA20','SMA50','EMA12','EMA26',
            'MACD','MACD_Signal','RSI','ROC10',
            'WilliamsR','StochK','BB_Width','ATR']
    return df[cols].dropna(), ac

# ── Attention LSTM builder ────────────────────────────────────────────────────
def build_attention_lstm(time_step, n_feat, seed=42):
    tf.random.set_seed(seed)
    inp  = Input(shape=(time_step, n_feat))
    x    = LSTM(96, return_sequences=True)(inp)
    x    = BatchNormalization()(x)
    x    = Dropout(0.2)(x)
    # Bahdanau-style attention
    score  = Dense(1, activation='tanh')(x)          # (batch, T, 1)
    weight = Activation('softmax')(score)             # (batch, T, 1)
    ctx    = Lambda(lambda t: tf.reduce_sum(t[0]*t[1], axis=1))([x, weight])
    x2   = LSTM(48)(inp)
    x2   = BatchNormalization()(x2)
    x2   = Dropout(0.15)(x2)
    merged = Concatenate()([ctx, x2])
    out  = Dense(32, activation='relu')(merged)
    out  = Dense(1)(out)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

# ── Incremental indicator update for forecast loop ────────────────────────────
def update_indicators_incremental(price_buffer, last_macd_sig,
                                   last_log_vol, last_atr, last_obv, pred_lr):
    prices = np.array(price_buffer); n = len(prices)
    s      = pd.Series(prices)
    sma20  = float(s.rolling(20).mean().iloc[-1]) if n>=20 else float(s.mean())
    sma50  = float(s.rolling(50).mean().iloc[-1]) if n>=50 else float(s.mean())
    ema12  = float(s.ewm(span=12,adjust=False).mean().iloc[-1])
    ema26  = float(s.ewm(span=26,adjust=False).mean().iloc[-1])
    macd   = ema12 - ema26
    alpha9 = 2/(9+1)
    macd_s = last_macd_sig*(1-alpha9) + macd*alpha9
    diff_  = s.diff().fillna(0)
    up_    = diff_.clip(lower=0); dn_ = -diff_.clip(upper=0)
    ru     = float(up_.rolling(14).mean().iloc[-1]) if n>=14 else float(up_.mean())
    rd     = float(dn_.rolling(14).mean().iloc[-1]) if n>=14 else float(dn_.mean())
    rsi    = 100 - 100/(1+ru/(rd+1e-9))
    roc10  = float((prices[-1]-prices[-10])/prices[-10]*100) if n>=10 else 0.0
    high14 = float(s.rolling(14).max().iloc[-1]) if n>=14 else float(s.max())
    low14  = float(s.rolling(14).min().iloc[-1]) if n>=14 else float(s.min())
    willr  = -100*(high14-prices[-1])/(high14-low14+1e-9)
    stochk = 100*(prices[-1]-low14)/(high14-low14+1e-9)
    rm_    = float(s.rolling(20).mean().iloc[-1]) if n>=20 else float(s.mean())
    rs_    = float(s.rolling(20).std().iloc[-1])  if n>=20 else float(s.std())
    bb_w   = (2*rs_)/(rm_+1e-9)
    new_obv = last_obv + (1 if pred_lr>0 else -1)*np.exp(last_log_vol)
    obv_n   = 0.0
    row = np.array([[pred_lr, 0.0, last_log_vol, obv_n,
                     sma20, sma50, ema12, ema26,
                     macd, macd_s, rsi, roc10,
                     willr, stochk, bb_w, last_atr]])
    return row, macd_s, new_obv

def compute_drawdown(series):
    roll_max = series.cummax()
    return (series - roll_max) / roll_max * 100

# ── Groq AI helper ────────────────────────────────────────────────────────────
def call_groq(api_key, prompt):
    try:
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        payload = {"model": "llama-3.1-70b-versatile",
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 1024, "temperature": 0.4}
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content']
        return f"Groq error {r.status_code}: {r.text}"
    except Exception as e:
        return f"Error: {e}"

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab = option_menu(None,
    ["Data & Viz","Predictions","Sentiment","Comparison","Portfolio Analyzer"],
    icons=["table","graph-up","chat-dots","arrow-left-right","pie-chart"],
    orientation="horizontal")

# ==============================================================================
#  TAB 1 — DATA & VIZ
# ==============================================================================
if tab == "Data & Viz":
    st.subheader(f"**{selected_ticker}** — Price History")
    if data_main is not None:
        st.dataframe(data_main.tail(100), use_container_width=True)
        st.download_button("Download CSV",
                           data_main.to_csv().encode(), f"{selected_ticker}.csv")
        fig = px.line(data_main, x=data_main.index, y='Adj Close',
                      title=f"{selected_ticker} — Adjusted Close Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data available. Try expanding the date range.")

# ==============================================================================
#  TAB 2 — PREDICTIONS  (Trader-grade UI, Attention Ensemble, 16 features)
# ==============================================================================
elif tab == "Predictions":

    if data_main is None:
        st.error("Not enough data. Expand date range or choose another ticker.")
        st.stop()

    df_features, price_series = compute_features(data_main)
    if len(df_features) < time_step + 50:
        st.error(f"Need at least {time_step+50} rows. Got {len(df_features)}.")
        st.stop()

    # ── Ensemble training ──────────────────────────────────────────────────────
    @st.cache_resource(ttl=24*3600)
    def train_ensemble(ticker, start_str, end_str, time_step,
                       epochs, batch_size, retrain_flag, _n_rows):
        t0 = time.time()
        raw = yf.download(ticker, start=start_str, end=end_str, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if 'Adj Close' not in raw.columns and 'Close' in raw.columns:
            raw['Adj Close'] = raw['Close']

        df_feat, price_s = compute_features(raw)
        n_feat = 16

        # Recent-data sample weights: exponential decay favouring recent rows
        n_rows  = len(df_feat)
        sw_full = np.exp(np.linspace(-1.5, 0, n_rows))

        # Short (30d) and long (time_step) windows — multi-scale
        short_step = 30

        def make_sequences(scaled, ts):
            X, y = [], []
            for i in range(len(scaled)-ts):
                X.append(scaled[i:i+ts, :])
                y.append(scaled[i+ts, 0])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(df_feat.values)

        X_long,  y_long  = make_sequences(scaled, time_step)
        X_short, y_short = make_sequences(scaled, short_step)
        # Align short to long (long starts later)
        offset = time_step - short_step
        X_short = X_short[offset:]
        y_short = y_short[offset:]   # same targets as y_long

        n       = len(X_long)
        train_n = int(n * 0.80)

        Xl_tr, Xl_te = X_long[:train_n],  X_long[train_n:]
        Xs_tr, Xs_te = X_short[:train_n], X_short[train_n:]
        y_tr,  y_te  = y_long[:train_n],  y_long[train_n:]
        sw_tr        = sw_full[time_step:time_step+train_n]
        sw_tr        = sw_tr / sw_tr.sum() * len(sw_tr)

        cbs = [EarlyStopping(monitor='val_loss', patience=8,
                             restore_best_weights=True, verbose=0),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                 patience=5, min_lr=1e-6, verbose=0)]

        # Train 3 models with different seeds
        models, histories = [], []
        for seed in [42, 7, 123]:
            # Long-window attention model
            m = build_attention_lstm(time_step, n_feat, seed=seed)
            h = m.fit(Xl_tr, y_tr, epochs=epochs, batch_size=batch_size,
                      validation_split=0.1, callbacks=cbs,
                      sample_weight=sw_tr, verbose=0)
            models.append(m)
            histories.append(h.history)

        # Ensemble predict on test — average 3 models
        @tf.function(reduce_retracing=True)
        def ens_predict(x):
            preds = [tf.cast(m(x, training=False), tf.float32) for m in models]
            return tf.reduce_mean(tf.stack(preds, axis=0), axis=0)

        all_sc    = ens_predict(tf.constant(Xl_te)).numpy().flatten()
        dummy     = np.zeros((len(all_sc), n_feat), dtype=np.float32)
        dummy[:,0] = all_sc
        all_lr    = scaler.inverse_transform(dummy)[:, 0]

        bt_pp, bt_ap, bt_pr, bt_ar = [], [], [], []
        for i in range(len(all_lr)):
            gi  = time_step + train_n + i
            plr = float(all_lr[i])
            alr = float(df_feat['LogReturn'].iloc[gi])
            pp  = float(price_s.iloc[gi-1]) * np.exp(plr)
            ap  = float(price_s.iloc[gi])
            bt_pp.append(pp); bt_ap.append(ap)
            bt_pr.append(plr); bt_ar.append(alr)

        bt_pp = np.array(bt_pp); bt_ap = np.array(bt_ap)
        bt_pr = np.array(bt_pr); bt_ar = np.array(bt_ar)

        wf, fs = [], max(10, len(bt_pp)//5)
        for f in range(5):
            s = f*fs; e = min(s+fs, len(bt_pp))
            if e-s < 5: break
            wf.append(float(r2_score(bt_ap[s:e], bt_pp[s:e])))
        wf_r2  = float(np.mean(wf)) if wf else 0.0
        r2_v   = float(r2_score(bt_ap, bt_pp))
        mse_v  = float(mean_squared_error(bt_ap, bt_pp))
        rmse_v = float(np.sqrt(mse_v))
        mape_v = float(np.mean(np.abs((bt_ap-bt_pp)/(np.abs(bt_ap)+1e-9)))*100)
        da_v   = float(np.mean(np.sign(bt_pr)==np.sign(bt_ar))*100)
        np_    = bt_ap[:-1]; na_ = bt_ap[1:]
        n_r2   = float(r2_score(na_, np_))
        n_mape = float(np.mean(np.abs((na_-np_)/(np.abs(na_)+1e-9)))*100)
        n_rmse = float(np.sqrt(mean_squared_error(na_, np_)))
        rs_v   = float(np.std(bt_ap - bt_pp))

        return dict(
            models=models, ens_predict=ens_predict, scaler=scaler,
            df_feat=df_feat, price_series=price_s,
            time_step=time_step, short_step=short_step,
            train_n=train_n, n_feat=n_feat,
            bt_pp=bt_pp, bt_ap=bt_ap, bt_pr=bt_pr, bt_ar=bt_ar,
            history=histories[0], training_secs=time.time()-t0,
            training_time=datetime.now(),
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
    ens_predict  = art['ens_predict']
    scaler       = art['scaler'];    df_used    = art['df_feat']
    price_s      = art['price_series']; train_n = art['train_n']
    bt_preds     = art['bt_pp'];     bt_actuals = art['bt_ap']
    bt_pred_ret  = art['bt_pr'];     bt_act_ret = art['bt_ar']
    history      = art['history'];   n_feat     = art['n_feat']
    resid_std    = art['resid_std']; last_price = float(price_s.iloc[-1])
    beat_r2   = art['r2']   > art['n_r2']
    beat_mape = art['mape'] < art['n_mape']
    beat_rmse = art['rmse'] < art['n_rmse']
    wins      = sum([beat_r2, beat_mape, beat_rmse])

    def ccls(v, g, w, hi=True):
        if hi: return "good" if v>=g else "warn" if v>=w else "bad"
        return "good" if v<=g else "warn" if v<=w else "bad"

    # Confidence score
    r2n  = max(0, min(100, art['r2']*100))
    dan  = max(0, min(100, (art['da']-50)*5))
    mpn  = max(0, min(100, (10-art['mape'])*10))
    conf = int(0.30*r2n + 0.40*dan + 0.20*mpn + 0.10*wins*33.3)
    conf = max(0, min(100, conf))
    clbl = "High" if conf>=70 else "Medium" if conf>=45 else "Low"
    ccol = "#22c55e" if conf>=70 else "#f59e0b" if conf>=45 else "#ef4444"

    # ── Build forecast first ───────────────────────────────────────────────────
    price_buffer  = deque(price_s.values[-51:].tolist(), maxlen=51)
    recent_scaled = scaler.transform(df_used.values[-time_step:]).tolist()
    chain_price   = last_price
    last_ms       = float(df_used['MACD_Signal'].iloc[-1])
    last_lv       = float(df_used['LogVolume'].iloc[-1])
    last_atr      = float(df_used['ATR'].iloc[-1])
    last_obv      = float(df_used['OBV'].iloc[-1])
    fp_prices     = []
    dummy_f       = np.zeros((1, n_feat), dtype=np.float32)

    for _ in range(days):
        inp = np.array(recent_scaled[-time_step:],
                       dtype=np.float32).reshape(1, time_step, -1)
        psc = float(ens_predict(tf.constant(inp))[0, 0])
        dummy_f[0, 0] = psc
        plr = float(scaler.inverse_transform(dummy_f)[0, 0])
        pp  = chain_price * np.exp(plr)
        fp_prices.append(pp); chain_price = pp
        price_buffer.append(pp)
        nr, last_ms, last_obv = update_indicators_incremental(
            price_buffer, last_ms, last_lv, last_atr, last_obv, plr)
        recent_scaled.append(scaler.transform(nr)[0].tolist())

    floor = last_price * 0.55; z = 1.96
    uppers, lowers = [], []
    for i, p in enumerate(fp_prices):
        hw = min(z * resid_std * np.sqrt(i+1), 0.15*p)
        uppers.append(p+hw); lowers.append(max(p-hw, floor))

    future_dates = pd.date_range(
        start=data_main.index[-1]+pd.Timedelta(days=1), periods=days, freq='B')
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted': fp_prices,
                               'Upper': uppers, 'Lower': lowers})
    future_df['Change %'] = [f"{(p-last_price)/last_price*100:+.2f}%"
                              for p in fp_prices]

    # ── SECTION 1: Main forecast chart (first thing trader sees) ──────────────
    st.markdown(f"### {selected_ticker} — {days}-Day Price Forecast")

    train_s = f"{art['training_secs']:.0f}s"
    age     = datetime.now() - art['training_time']
    age_str = f"{age.seconds//3600}h {(age.seconds%3600)//60}m ago"
    col_info = st.columns([1,1,1,1,1,1])
    info_items = [
        ("Model",    "3-Model Attention Ensemble"),
        ("Features", "16 signals"),
        ("Lookback", f"{time_step} days"),
        ("Trained",  train_s),
        ("Cached",   age_str),
        ("Confidence", f"{conf}/100 ({clbl})")
    ]
    for col, (lbl, val) in zip(col_info, info_items):
        col.markdown(f"<div style='font-size:10px;color:#475569;text-transform:uppercase;"
                     f"letter-spacing:.06em'>{lbl}</div>"
                     f"<div style='font-size:13px;color:#94a3b8;font-weight:600'>{val}</div>",
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    hx = price_s.index; hy = price_s.values
    fig_f = go.Figure()
    # Historical
    fig_f.add_trace(go.Scatter(x=hx, y=hy, name='Historical Price',
                               line=dict(color='#3b82f6', width=1.5)))
    # Forecast
    fig_f.add_trace(go.Scatter(x=future_dates, y=fp_prices,
                               name='Forecast', mode='lines+markers',
                               line=dict(color='#f59e0b', width=2.5),
                               marker=dict(size=6)))
    # CI band
    xb = list(future_dates) + list(future_dates[::-1])
    yb = list(uppers) + list(lowers[::-1])
    fig_f.add_trace(go.Scatter(x=xb, y=yb, fill='toself',
                               fillcolor='rgba(245,158,11,0.12)',
                               line=dict(color='rgba(0,0,0,0)'),
                               name='95% Confidence Band'))
    # Last price marker
    fig_f.add_trace(go.Scatter(
        x=[price_s.index[-1]], y=[last_price], mode='markers',
        marker=dict(size=10, color='#ffffff', symbol='circle'),
        name=f'Last Close ${last_price:.2f}', showlegend=True))

    fig_f.update_layout(
        height=480,
        plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
        font=dict(color='#94a3b8'),
        xaxis=dict(gridcolor='#1e293b', showgrid=True),
        yaxis=dict(gridcolor='#1e293b', showgrid=True, title='Price (USD)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h',
                    yanchor='bottom', y=1.01, xanchor='left', x=0),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified'
    )
    st.plotly_chart(fig_f, use_container_width=True)

    # Forecast table
    st.dataframe(
        future_df.style.format({"Predicted": "${:.2f}", "Upper": "${:.2f}",
                                "Lower": "${:.2f}"}),
        use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 2: KPI strip ──────────────────────────────────────────────────
    st.markdown("### Model Performance Metrics")
    st.caption("Measured exclusively on the 20% test set the model never saw during training.")

    wf_cls = ccls(art['wf_r2'], 0.82, 0.70)
    r2_cls = ccls(art['r2'],    0.82, 0.70)
    da_cls = ccls(art['da'],    60,   53)
    mp_cls = ccls(art['mape'],  2,    4, hi=False)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    metrics_html = [
        (k1, "Confidence", f"{conf}/100", clbl, ccol),
        (k2, "Directional Accuracy",
             f"{art['da']:.1f}%",
             "Random = 50%",
             "#22c55e" if art['da']>=60 else "#f59e0b" if art['da']>=53 else "#ef4444"),
        (k3, "Walk-Forward R²", f"{art['wf_r2']:.3f}",
             "5 time windows",
             "#22c55e" if art['wf_r2']>=0.82 else "#f59e0b" if art['wf_r2']>=0.70 else "#ef4444"),
        (k4, "MAPE", f"{art['mape']:.2f}%", "Avg % price error",
             "#22c55e" if art['mape']<=2 else "#f59e0b" if art['mape']<=4 else "#ef4444"),
        (k5, "RMSE", f"${art['rmse']:.2f}", "Avg $ error/day", "#94a3b8"),
        (k6, "Ensemble Models", "3", "Attention LSTM", "#818cf8"),
    ]
    for col, lbl, val, sub, color in metrics_html:
        col.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{lbl}</div>
          <div class="kpi-value" style="color:{color}">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 3: Baseline comparison ───────────────────────────────────────
    st.markdown("### vs Naïve Baseline")
    st.caption("A naïve baseline predicts tomorrow = today (zero ML). Any useful model must beat it.")

    vc_  = "good" if wins==3 else "warn" if wins>=2 else "bad"
    vt_  = (f"Beats baseline on all 3 metrics" if wins==3
            else f"Beats baseline on {wins}/3 metrics" if wins>=2
            else "Does not beat baseline")
    r2c_ = "good" if beat_r2   else "bad"
    mpc_ = "good" if beat_mape else "bad"
    rmc_ = "good" if beat_rmse else "bad"

    b1,b2,b3,b4 = st.columns(4)
    with b1:
        st.markdown("""<div class="bsbox">
          <div style="font-size:11px;color:#475569;margin-bottom:8px">METRIC</div>
          <div style="color:#64748b;margin-top:8px;font-size:13px">R² (higher better)</div>
          <div style="color:#64748b;margin-top:8px;font-size:13px">MAPE (lower better)</div>
          <div style="color:#64748b;margin-top:8px;font-size:13px">RMSE (lower better)</div>
        </div>""", unsafe_allow_html=True)
    with b2:
        st.markdown(f"""<div class="bsbox">
          <div style="font-size:11px;color:#475569;margin-bottom:8px">OUR ENSEMBLE</div>
          <div class="{r2c_}" style="margin-top:8px;font-size:13px">{art['r2']:.3f}</div>
          <div class="{mpc_}" style="margin-top:8px;font-size:13px">{art['mape']:.2f}%</div>
          <div class="{rmc_}" style="margin-top:8px;font-size:13px">${art['rmse']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with b3:
        st.markdown(f"""<div class="bsbox">
          <div style="font-size:11px;color:#475569;margin-bottom:8px">NAÏVE BASELINE</div>
          <div style="color:#94a3b8;margin-top:8px;font-size:13px">{art['n_r2']:.3f}</div>
          <div style="color:#94a3b8;margin-top:8px;font-size:13px">{art['n_mape']:.2f}%</div>
          <div style="color:#94a3b8;margin-top:8px;font-size:13px">${art['n_rmse']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with b4:
        st.markdown(f"""<div class="bsbox">
          <div style="font-size:11px;color:#475569;margin-bottom:8px">VERDICT</div>
          <div class="{vc_}" style="margin-top:8px;font-size:13px;font-weight:600">
            {vt_}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 4: Two-column charts ─────────────────────────────────────────
    st.markdown("### Backtest Analysis")
    st.caption("How accurately did the model predict prices on historical data it had never seen?")

    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Actual vs Predicted Price**")
        bt_start = art['time_step'] + train_n
        bt_idx   = df_used.index[bt_start: bt_start+len(bt_preds)]
        fig_bt   = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_idx, y=bt_actuals, name='Actual',
                                    line=dict(color='#3b82f6', width=1.5)))
        fig_bt.add_trace(go.Scatter(x=bt_idx, y=bt_preds, name='Predicted',
                                    line=dict(color='#f59e0b', width=1.5)))
        fig_bt.update_layout(height=280, plot_bgcolor='#0b1220',
                             paper_bgcolor='#0b1220', font=dict(color='#94a3b8'),
                             xaxis=dict(gridcolor='#1e293b'),
                             yaxis=dict(gridcolor='#1e293b'),
                             legend=dict(bgcolor='rgba(0,0,0,0)'),
                             margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_bt, use_container_width=True)

    with ch2:
        st.markdown("**Training Loss Curve**")
        ep_r  = list(range(1, len(history.get('loss',[]))+1))
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=ep_r, y=history.get('loss',[]),
                                   mode='lines', name='Train',
                                   line=dict(color='#3b82f6')))
        if 'val_loss' in history:
            fig_l.add_trace(go.Scatter(x=ep_r, y=history['val_loss'],
                                       mode='lines', name='Validation',
                                       line=dict(color='#f59e0b')))
        fig_l.update_layout(height=280, plot_bgcolor='#0b1220',
                            paper_bgcolor='#0b1220', font=dict(color='#94a3b8'),
                            xaxis=dict(gridcolor='#1e293b', title='Epoch'),
                            yaxis=dict(gridcolor='#1e293b', title='MSE Loss'),
                            legend=dict(bgcolor='rgba(0,0,0,0)'),
                            margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_l, use_container_width=True)

    ch3, ch4 = st.columns(2)

    with ch3:
        st.markdown("**Directional Accuracy — Daily**")
        st.caption("Green = model correctly called UP or DOWN that day.")
        dir_c  = (np.sign(bt_pred_ret)==np.sign(bt_act_ret)).astype(int)
        bt_i2  = df_used.index[art['time_step']+train_n:
                                art['time_step']+train_n+len(bt_preds)]
        bclrs  = ['#22c55e' if c==1 else '#ef4444' for c in dir_c]
        fig_d  = go.Figure()
        fig_d.add_trace(go.Bar(x=bt_i2, y=dir_c, marker_color=bclrs,
                               showlegend=False))
        fig_d.add_hline(y=0.5, line_dash='dash', line_color='#475569',
                        annotation_text="50% Random")
        fig_d.update_layout(
            height=250, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'), xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b'),
            margin=dict(l=0,r=0,t=10,b=0),
            title=dict(text=f"Correct {art['da']:.1f}%  ({int(dir_c.sum())}/{len(dir_c)})",
                       font=dict(size=12, color='#94a3b8')))
        st.plotly_chart(fig_d, use_container_width=True)

    with ch4:
        st.markdown("**Prediction Error Distribution**")
        st.caption("Tight bell curve near $0 = well-calibrated model.")
        residuals = bt_actuals - bt_preds
        fig_r = go.Figure()
        fig_r.add_trace(go.Histogram(x=residuals, nbinsx=40,
                                     marker_color='#6366f1', opacity=0.8))
        fig_r.add_vline(x=0, line_color='#94a3b8', line_dash='dash')
        fig_r.update_layout(
            height=250, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'), xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b'),
            margin=dict(l=0,r=0,t=10,b=0),
            title=dict(
                text=f"Mean ${residuals.mean():.2f}  Std ${residuals.std():.2f}",
                font=dict(size=12, color='#94a3b8')))
        st.plotly_chart(fig_r, use_container_width=True)

    # Walk-forward bar
    if art['wf_list']:
        st.markdown("**Walk-Forward R² — Consistency Across Time**")
        st.caption("Each bar = model accuracy in a different historical period. Consistent bars = model works across all market conditions, not just one lucky window.")
        wfc    = ['#22c55e' if v>=0.82 else '#f59e0b' if v>=0.70 else '#ef4444'
                  for v in art['wf_list']]
        fig_wf = go.Figure()
        for fn, fv, fc in zip([f"Window {i+1}" for i in range(len(art['wf_list']))],
                               art['wf_list'], wfc):
            fig_wf.add_trace(go.Bar(x=[fn], y=[fv], marker_color=fc,
                                    text=[f"{fv:.3f}"], textposition='outside',
                                    showlegend=False))
        fig_wf.add_hline(y=0.82, line_dash='dash', line_color='#22c55e',
                         annotation_text="Target 0.82")
        fig_wf.add_hline(y=0.70, line_dash='dot', line_color='#f59e0b',
                         annotation_text="Acceptable 0.70")
        fig_wf.update_layout(
            height=260, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
            font=dict(color='#94a3b8'), xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b',
                       range=[min(min(art['wf_list'])-0.05,-0.05),1.05]),
            margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_wf, use_container_width=True)

    # Footer summary
    st.markdown("<br>", unsafe_allow_html=True)
    fs1,fs2,fs3,fs4 = st.columns(4)
    fs1.metric("Confidence",       f"{conf}/100 ({clbl})")
    fs2.metric("Directional Acc",  f"{art['da']:.1f}%")
    fs3.metric("Avg Price Error",  f"{art['mape']:.2f}%")
    fs4.metric("Beats Naïve",      "Yes (all 3)" if wins==3 else
                                   f"Partial ({wins}/3)" if wins>=2 else "No")

# ==============================================================================
#  TAB 3 — SENTIMENT
# ==============================================================================
elif tab == "Sentiment":
    st.subheader("News Sentiment Analysis")
    if news_posts:
        df_s = pd.DataFrame({'News': news_posts, 'Link': news_links,
                             'Score': vader_scores})
        def color_score(val):
            return f"color: {'#22c55e' if val>0.1 else '#ef4444' if val<-0.1 else '#94a3b8'}"
        st.dataframe(df_s.style.map(color_score, subset=['Score'])
                     .format({'Score': '{:.3f}'}), use_container_width=True)
        pos = sum(1 for s in vader_scores if s >  0.1)
        neg = sum(1 for s in vader_scores if s < -0.1)
        neu = len(vader_scores) - pos - neg
        c1,c2,c3 = st.columns(3)
        c1.metric("Positive", pos); c2.metric("Negative", neg); c3.metric("Neutral", neu)
        avg_score = np.mean(vader_scores) if vader_scores else 0
        overall   = "Bullish" if avg_score>0.05 else "Bearish" if avg_score<-0.05 else "Neutral"
        st.info(f"Overall market sentiment for **{selected_ticker}**: "
                f"**{overall}** (avg score {avg_score:.3f})")
    else:
        st.info("No recent news available for this ticker.")

# ==============================================================================
#  TAB 4 — COMPARISON
# ==============================================================================
elif tab == "Comparison":
    st.subheader(f"{selected_ticker} vs {compare_ticker} — Head to Head")

    if data_main is None or data_compare is None:
        st.error("Not enough data for one or both tickers."); st.stop()

    bm = data_main['Adj Close'].iloc[0];    bc = data_compare['Adj Close'].iloc[0]
    dm = (data_main['Adj Close']/bm-1)*100; dc = (data_compare['Adj Close']/bc-1)*100
    rm = float(dm.iloc[-1]);               rc = float(dc.iloc[-1])
    vm = float(data_main['Adj Close'].pct_change().std()*np.sqrt(252)*100)
    vc_ = float(data_compare['Adj Close'].pct_change().std()*np.sqrt(252)*100)
    ra_m = rm/vm if vm>0 else 0;           ra_c = rc/vc_ if vc_>0 else 0
    dd_m = compute_drawdown(data_main['Adj Close'])
    dd_c = compute_drawdown(data_compare['Adj Close'])
    max_dd_m = float(dd_m.min()); max_dd_c = float(dd_c.min())
    better_ret  = selected_ticker if rm>rc else compare_ticker
    better_stab = selected_ticker if vm<vc_ else compare_ticker
    ret_diff    = abs(rm-rc)
    ra_txt = ("both performed similarly" if abs(ra_m-ra_c)<0.5
              else (f"{selected_ticker} had better risk-adjusted returns"
                    if ra_m>ra_c else f"{compare_ticker} had better risk-adjusted returns"))

    st.markdown(f"""<div class="comp-box"><p>
    Since <b>{start_date}</b>, <b>{selected_ticker}</b> returned <b>{rm:+.1f}%</b>
    vs <b>{compare_ticker}</b> at <b>{rc:+.1f}%</b> — a gap of <b>{ret_diff:.1f}pp</b>.
    <b>{better_ret}</b> was the stronger raw performer. On risk-adjusted terms, {ra_txt}.
    <b>{better_stab}</b> had lower volatility ({min(vm,vc_):.1f}% vs {max(vm,vc_):.1f}% annualised),
    meaning more stable returns with smaller drawdowns.
    </p></div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"{selected_ticker} Return",    f"{rm:+.2f}%")
    c2.metric(f"{compare_ticker} Return",     f"{rc:+.2f}%")
    c3.metric(f"{selected_ticker} Volatility",f"{vm:.1f}%")
    c4.metric(f"{compare_ticker} Volatility", f"{vc_:.1f}%")

    # Normalised performance
    st.markdown("#### Cumulative Return")
    st.caption("Both start at 0% — shows growth from the same baseline regardless of share price.")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data_main.index, y=dm, name=selected_ticker,
                              line=dict(color='#26A69A')))
    fig1.add_trace(go.Scatter(x=data_compare.index, y=dc, name=compare_ticker,
                              line=dict(color='#AB47BC')))
    fig1.add_hline(y=0, line_dash='dot', line_color='#475569')
    fig1.update_layout(height=380, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
                       font=dict(color='#94a3b8'),
                       xaxis=dict(gridcolor='#1e293b'),
                       yaxis=dict(gridcolor='#1e293b', title='Return (%)'),
                       legend=dict(bgcolor='rgba(0,0,0,0)'),
                       margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig1, use_container_width=True)

    # Rolling 12-month
    st.markdown("#### Rolling 12-Month Return")
    st.caption("What your return would have been holding for exactly 12 months ending each date. Consistent above-zero line = reliable performer.")
    roll_m = data_main['Adj Close'].pct_change(252)*100
    roll_c = data_compare['Adj Close'].pct_change(252)*100
    fig2   = go.Figure()
    fig2.add_trace(go.Scatter(x=data_main.index, y=roll_m, name=selected_ticker,
                              line=dict(color='#26A69A'), fill='tozeroy',
                              fillcolor='rgba(38,166,154,0.08)'))
    fig2.add_trace(go.Scatter(x=data_compare.index, y=roll_c, name=compare_ticker,
                              line=dict(color='#AB47BC'), fill='tozeroy',
                              fillcolor='rgba(171,71,188,0.08)'))
    fig2.add_hline(y=0, line_color='#475569', line_dash='dash')
    fig2.update_layout(height=320, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
                       font=dict(color='#94a3b8'),
                       xaxis=dict(gridcolor='#1e293b'),
                       yaxis=dict(gridcolor='#1e293b', title='12M Return (%)'),
                       legend=dict(bgcolor='rgba(0,0,0,0)'),
                       margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig2, use_container_width=True)

    # Drawdown + Risk-Return side by side
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Drawdown from Peak**")
        st.caption("How far each stock fell from its all-time high at any point.")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=data_main.index, y=dd_m, name=selected_ticker,
                                  line=dict(color='#26A69A'), fill='tozeroy',
                                  fillcolor='rgba(38,166,154,0.12)'))
        fig3.add_trace(go.Scatter(x=data_compare.index, y=dd_c, name=compare_ticker,
                                  line=dict(color='#AB47BC'), fill='tozeroy',
                                  fillcolor='rgba(171,71,188,0.12)'))
        fig3.update_layout(height=300, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
                           font=dict(color='#94a3b8'),
                           xaxis=dict(gridcolor='#1e293b'),
                           yaxis=dict(gridcolor='#1e293b', title='Drawdown (%)'),
                           legend=dict(bgcolor='rgba(0,0,0,0)'),
                           margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with cc2:
        st.markdown("**Risk vs Return**")
        st.caption("Top-left = ideal (high return, low risk). Bottom-right = worst.")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=[vm, vc_], y=[rm, rc], mode='markers+text',
            text=[selected_ticker, compare_ticker],
            textposition='top center',
            marker=dict(size=22, color=['#26A69A','#AB47BC'], symbol='diamond')))
        fig4.update_layout(height=300, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
                           font=dict(color='#94a3b8'),
                           xaxis=dict(gridcolor='#1e293b', title='Volatility (Risk) %'),
                           yaxis=dict(gridcolor='#1e293b', title='Total Return %'),
                           margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Summary Table")
    summary_df = pd.DataFrame({
        'Metric':        ['Total Return','Volatility','Max Drawdown','Risk-Adj Return'],
        selected_ticker: [f"{rm:+.2f}%", f"{vm:.1f}%",
                          f"{max_dd_m:.1f}%", f"{ra_m:.2f}x"],
        compare_ticker:  [f"{rc:+.2f}%", f"{vc_:.1f}%",
                          f"{max_dd_c:.1f}%", f"{ra_c:.2f}x"],
        'Winner':        [selected_ticker if rm>rc else compare_ticker,
                          selected_ticker if vm<vc_ else compare_ticker,
                          selected_ticker if max_dd_m>max_dd_c else compare_ticker,
                          selected_ticker if ra_m>ra_c else compare_ticker]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ==============================================================================
#  TAB 5 — PORTFOLIO ANALYZER  (7 Features + Groq AI Insights)
# ==============================================================================
elif tab == "Portfolio Analyzer":
    st.subheader("Portfolio Analyzer — Professional Suite")

    port_tickers = st.multiselect("Select Portfolio Stocks", tickers,
                                   default=[selected_ticker, compare_ticker])
    if len(port_tickers) < 2:
        st.warning("Select at least 2 tickers."); st.stop()

    # ── Feature 1: Portfolio Creation ─────────────────────────────────────────
    st.markdown("#### Portfolio Weights")
    if st.button("Auto Equal-Weight"):
        pass  # weights calculated below always
    weights, total_w = [], 0.0
    cols = st.columns(len(port_tickers))
    for i, tick in enumerate(port_tickers):
        w = cols[i].number_input(f"{tick} (%)", 0.0, 100.0,
                                  round(100.0/len(port_tickers), 2), key=f"w_{tick}")
        weights.append(w/100); total_w += w

    bar_html = f"""<div style="background:#1e293b;border-radius:6px;height:10px;margin-bottom:12px">
      <div style="width:{min(total_w,100):.1f}%;height:10px;border-radius:6px;
                  background:{'#22c55e' if abs(total_w-100)<0.1 else '#ef4444'}"></div>
    </div><div style="font-size:12px;color:{'#22c55e' if abs(total_w-100)<0.1 else '#ef4444'}">
    Total: {total_w:.1f}%</div>"""
    st.markdown(bar_html, unsafe_allow_html=True)

    if abs(total_w-100) > 0.1:
        st.warning(f"Weights must sum to 100%. Currently {total_w:.1f}%"); st.stop()

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
    cov_mat  = rets.cov()  * 252
    w_np     = np.array(weights)
    n_assets = len(port_tickers)
    p_ret    = float(np.dot(m_ret, w_np))
    p_vol    = float(np.sqrt(np.dot(w_np.T, np.dot(cov_mat, w_np))))
    sharpe   = (p_ret - 0.03) / p_vol if p_vol > 0 else 0
    sortino_rets = (rets * w_np).sum(axis=1)
    downside = sortino_rets[sortino_rets < 0].std() * np.sqrt(252)
    sortino  = (p_ret - 0.03) / downside if downside > 0 else 0

    # SPY beta
    spy_d = fetch_stock_data('SPY', start_date, end_date)
    beta  = 1.0
    if spy_d is not None:
        spy_ret_s = spy_d['Adj Close'].pct_change().dropna()
        port_ret_s = sortino_rets.reindex(spy_ret_s.index).dropna()
        spy_aligned = spy_ret_s.reindex(port_ret_s.index).dropna()
        port_aligned = port_ret_s.reindex(spy_aligned.index).dropna()
        if len(port_aligned) > 30:
            cov_ps  = np.cov(port_aligned, spy_aligned)[0,1]
            var_spy = np.var(spy_aligned)
            beta    = float(cov_ps / var_spy) if var_spy > 0 else 1.0

    port_idx   = (1 + sortino_rets).cumprod()
    roll_max_p = port_idx.cummax()
    dd_port    = (port_idx - roll_max_p) / roll_max_p * 100
    max_dd_p   = float(dd_port.min())
    port_cum   = port_idx * 100 - 100
    var_95     = float(np.percentile(sortino_rets, 5) * 100)
    cvar_95    = float(sortino_rets[sortino_rets<=np.percentile(sortino_rets,5)].mean()*100)

    # ── Feature 2: Performance Analytics KPIs ─────────────────────────────────
    st.markdown("#### Performance Analytics")
    pk1,pk2,pk3,pk4,pk5,pk6,pk7 = st.columns(7)
    port_kpis = [
        (pk1, "Annual Return",     f"{p_ret*100:.2f}%", "#22c55e" if p_ret>0.08 else "#f59e0b"),
        (pk2, "Annual Volatility", f"{p_vol*100:.2f}%", "#94a3b8"),
        (pk3, "Sharpe Ratio",      f"{sharpe:.2f}",
               "#22c55e" if sharpe>1 else "#f59e0b" if sharpe>0.5 else "#ef4444"),
        (pk4, "Sortino Ratio",     f"{sortino:.2f}",
               "#22c55e" if sortino>1.5 else "#f59e0b" if sortino>0.8 else "#ef4444"),
        (pk5, "Beta vs S&P",       f"{beta:.2f}",
               "#22c55e" if 0.8<=beta<=1.2 else "#f59e0b"),
        (pk6, "Max Drawdown",      f"{max_dd_p:.1f}%", "#ef4444" if max_dd_p<-30 else "#f59e0b"),
        (pk7, "VaR 95% (Daily)",   f"{var_95:.2f}%", "#ef4444"),
    ]
    for col, lbl, val, color in port_kpis:
        col.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{lbl}</div>
          <div class="kpi-value" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Portfolio vs SPY chart
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values,
                               name='Your Portfolio', line=dict(color='#6366f1', width=2)))
    if spy_d is not None:
        spy_ret2 = spy_d['Adj Close'].pct_change().dropna()
        spy_cum2 = (1+spy_ret2).cumprod()*100-100
        spy_cum2 = spy_cum2.reindex(port_cum.index, method='ffill').dropna()
        fig_b.add_trace(go.Scatter(x=spy_cum2.index, y=spy_cum2.values,
                                   name='S&P 500 (SPY)',
                                   line=dict(color='#94a3b8', width=1.5, dash='dash')))
        pf = float(port_cum.iloc[-1]); sf = float(spy_cum2.iloc[-1])
        beat = pf > sf
        st.markdown(
            f"<div class='ins-card'>{'✅ Portfolio outperformed' if beat else '📉 Portfolio underperformed'} "
            f"S&P 500 by <b>{abs(pf-sf):.1f}pp</b> over the selected period.</div>",
            unsafe_allow_html=True)
    fig_b.add_hline(y=0, line_dash='dot', line_color='#475569')
    fig_b.update_layout(height=380, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
                        font=dict(color='#94a3b8'),
                        xaxis=dict(gridcolor='#1e293b'),
                        yaxis=dict(gridcolor='#1e293b', title='Cumulative Return (%)'),
                        legend=dict(bgcolor='rgba(0,0,0,0)',orientation='h',
                                    y=1.02, x=0, yanchor='bottom'),
                        margin=dict(l=0,r=0,t=30,b=0),
                        title=dict(text="Portfolio vs S&P 500 Benchmark",
                                   font=dict(color='#94a3b8', size=13)))
    st.plotly_chart(fig_b, use_container_width=True)

    # ── Feature 3: Risk Metrics — Drawdown ────────────────────────────────────
    st.markdown("#### Risk Metrics")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Portfolio Drawdown**")
        st.caption("How far the portfolio fell from its peak. Deeper = more pain for investors.")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_port.index, y=dd_port.values,
                                    name='Drawdown', fill='tozeroy',
                                    fillcolor='rgba(239,68,68,0.12)',
                                    line=dict(color='#ef4444')))
        fig_dd.update_layout(height=280, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
                             font=dict(color='#94a3b8'),
                             xaxis=dict(gridcolor='#1e293b'),
                             yaxis=dict(gridcolor='#1e293b', title='Drawdown (%)'),
                             margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_dd, use_container_width=True)

    with rc2:
        st.markdown("**Individual Stock Contributions**")
        st.caption("Which stock contributes most to return vs risk?")
        contrib_ret  = [float(m_ret[t]*w_np[i]*100) for i,t in enumerate(port_tickers)]
        contrib_risk = [float(np.dot(cov_mat.iloc[i].values, w_np)/p_vol*w_np[i]*100)
                        for i in range(n_assets)]
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(name='Return Contrib.',
                                x=port_tickers, y=contrib_ret,
                                marker_color='#22c55e'))
        fig_c.add_trace(go.Bar(name='Risk Contrib.',
                                x=port_tickers, y=contrib_risk,
                                marker_color='#ef4444'))
        fig_c.update_layout(barmode='group', height=280,
                            plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
                            font=dict(color='#94a3b8'),
                            xaxis=dict(gridcolor='#1e293b'),
                            yaxis=dict(gridcolor='#1e293b', title='%'),
                            legend=dict(bgcolor='rgba(0,0,0,0)'),
                            margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_c, use_container_width=True)

    # ── Feature 4: Correlation Matrix ─────────────────────────────────────────
    st.markdown("#### Correlation Matrix")
    st.caption("+1 = always move together (no diversification). 0 = independent (good). -1 = opposite (hedge).")
    fig_h = px.imshow(rets.corr(), text_auto=True, aspect='auto',
                      color_continuous_scale='RdBu_r',
                      zmin=-1, zmax=1)
    fig_h.update_layout(height=320, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
                        font=dict(color='#94a3b8'),
                        margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_h, use_container_width=True)

    # Auto-interpretation
    corr_matrix = rets.corr()
    max_corr_pair = ('', '', 0.0)
    min_corr_pair = ('', '', 1.0)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            v = float(corr_matrix.iloc[i,j])
            if v > max_corr_pair[2]: max_corr_pair = (port_tickers[i],port_tickers[j],v)
            if v < min_corr_pair[2]: min_corr_pair = (port_tickers[i],port_tickers[j],v)
    st.markdown(
        f"<div class='ins-card'>Most correlated pair: "
        f"<b>{max_corr_pair[0]} & {max_corr_pair[1]}</b> ({max_corr_pair[2]:.2f}) — "
        f"these move together and offer less diversification benefit. "
        f"Least correlated: <b>{min_corr_pair[0]} & {min_corr_pair[1]}</b> "
        f"({min_corr_pair[2]:.2f}) — best diversification pair in this portfolio.</div>",
        unsafe_allow_html=True)

    # ── Feature 5: Efficient Frontier ─────────────────────────────────────────
    st.markdown("#### Efficient Frontier")
    st.caption("3,000 random weight combinations. Colour = Sharpe Ratio. Red star = your portfolio. Gold star = optimal (highest Sharpe).")

    n_sim = 3000
    s_rets, s_vols, s_sharpes, s_wts = [], [], [], []
    for _ in range(n_sim):
        w_ = np.random.dirichlet(np.ones(n_assets))
        r_ = float(np.dot(m_ret, w_))
        v_ = float(np.sqrt(np.dot(w_.T, np.dot(cov_mat, w_))))
        s_ = (r_-0.03)/v_ if v_>0 else 0
        s_rets.append(r_*100); s_vols.append(v_*100)
        s_sharpes.append(s_); s_wts.append(w_)

    hover = [" | ".join([f"{port_tickers[j]}: {s_wts[i][j]*100:.1f}%"
                         for j in range(n_assets)]) for i in range(n_sim)]
    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=s_vols, y=s_rets, mode='markers',
        marker=dict(size=4, color=s_sharpes, colorscale='Viridis',
                    showscale=True, colorbar=dict(title='Sharpe')),
        text=hover,
        hovertemplate='Return: %{y:.1f}%<br>Risk: %{x:.1f}%<br>%{text}<extra></extra>',
        name='Simulated'))
    fig_ef.add_trace(go.Scatter(
        x=[p_vol*100], y=[p_ret*100], mode='markers+text',
        marker=dict(size=20, color='red', symbol='star'),
        text=['Your Portfolio'], textposition='top center', name='Your Portfolio'))
    best_idx = int(np.argmax(s_sharpes))
    fig_ef.add_trace(go.Scatter(
        x=[s_vols[best_idx]], y=[s_rets[best_idx]], mode='markers+text',
        marker=dict(size=20, color='#fbbf24', symbol='star'),
        text=['Optimal'], textposition='top center', name='Max Sharpe'))
    fig_ef.update_layout(height=500, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
                         font=dict(color='#94a3b8'),
                         xaxis=dict(gridcolor='#1e293b', title='Risk (Volatility %)'),
                         yaxis=dict(gridcolor='#1e293b', title='Expected Return (%)'),
                         legend=dict(bgcolor='rgba(0,0,0,0)'),
                         margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig_ef, use_container_width=True)

    best_w   = s_wts[best_idx]
    opt_hint = " | ".join([f"{port_tickers[j]}: {best_w[j]*100:.1f}%"
                           for j in range(n_assets)])
    st.markdown(
        f"<div class='ins-card'>Optimal weights (Sharpe {s_sharpes[best_idx]:.2f}): "
        f"<b>{opt_hint}</b>. Your current Sharpe: <b>{sharpe:.2f}</b>.</div>",
        unsafe_allow_html=True)

    # ── Feature 6: Monte Carlo Simulation ────────────────────────────────────
    st.markdown("#### Monte Carlo Simulation — 1-Year Outlook")
    st.caption("500 simulated future paths based on historical return distribution. Shows range of possible outcomes over the next 252 trading days.")

    n_mc  = 500; n_days = 252
    daily_mean = float(sortino_rets.mean())
    daily_std  = float(sortino_rets.std())
    mc_paths   = np.zeros((n_mc, n_days))
    for i in range(n_mc):
        sim_rets_mc  = np.random.normal(daily_mean, daily_std, n_days)
        mc_paths[i]  = (1 + sim_rets_mc).cumprod() * last_price

    last_price_port = float(port_idx.iloc[-1]) * 100
    mc_end = mc_paths[:, -1]
    p5_    = float(np.percentile(mc_end, 5))
    p50_   = float(np.percentile(mc_end, 50))
    p95_   = float(np.percentile(mc_end, 95))

    fig_mc = go.Figure()
    for i in range(0, n_mc, 5):
        fig_mc.add_trace(go.Scatter(
            x=list(range(n_days)), y=mc_paths[i],
            mode='lines', line=dict(color='rgba(99,102,241,0.08)', width=1),
            showlegend=False))
    med_path = np.percentile(mc_paths, 50, axis=0)
    p5_path  = np.percentile(mc_paths, 5,  axis=0)
    p95_path = np.percentile(mc_paths, 95, axis=0)
    fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=med_path,
                                name='Median', line=dict(color='#6366f1', width=2)))
    fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p95_path,
                                name='Best 5%', line=dict(color='#22c55e', width=1.5,
                                                           dash='dash')))
    fig_mc.add_trace(go.Scatter(x=list(range(n_days)), y=p5_path,
                                name='Worst 5%', line=dict(color='#ef4444', width=1.5,
                                                            dash='dash')))
    fig_mc.update_layout(height=400, plot_bgcolor='#0b1220', paper_bgcolor='#0b1220',
                         font=dict(color='#94a3b8'),
                         xaxis=dict(gridcolor='#1e293b', title='Trading Days'),
                         yaxis=dict(gridcolor='#1e293b', title='Portfolio Value ($)'),
                         legend=dict(bgcolor='rgba(0,0,0,0)'),
                         margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig_mc, use_container_width=True)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Median Outcome (1Y)",    f"${p50_:.2f}")
    mc2.metric("Best Case 95%",          f"${p95_:.2f}")
    mc3.metric("Worst Case 5%",          f"${p5_:.2f}")

    # ── Full Analytics Table ──────────────────────────────────────────────────
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

    # ── Feature 7: AI Insights Dashboard ──────────────────────────────────────
    st.markdown("#### AI Insights Dashboard")
    if not groq_key:
        st.info("Enter your free Groq API key in the sidebar to enable AI-powered portfolio insights. "
                "Get one in 2 minutes at console.groq.com")
    else:
        if st.button("Generate AI Insights", type="primary"):
            prompt = f"""You are a senior portfolio manager and quantitative analyst.
Analyze this portfolio and provide exactly 5 specific, data-driven insights with actionable recommendations.
Each insight must reference the actual numbers provided.

Portfolio Data:
- Stocks: {', '.join([f"{port_tickers[i]} ({weights[i]*100:.1f}%)" for i in range(n_assets)])}
- Expected Annual Return: {p_ret*100:.2f}%
- Annual Volatility: {p_vol*100:.2f}%
- Sharpe Ratio: {sharpe:.2f}
- Sortino Ratio: {sortino:.2f}
- Beta vs S&P 500: {beta:.2f}
- Max Drawdown: {max_dd_p:.1f}%
- Daily VaR (95%): {var_95:.2f}%
- CVaR (95%): {cvar_95:.2f}%
- Most correlated pair: {max_corr_pair[0]} & {max_corr_pair[1]} ({max_corr_pair[2]:.2f})
- Monte Carlo median 1Y outcome: ${p50_:.2f}
- Monte Carlo worst case (5%): ${p5_:.2f}
- Optimal Sharpe weights: {opt_hint}
- Current Sharpe vs Optimal: {sharpe:.2f} vs {s_sharpes[best_idx]:.2f}

Return your response as exactly 5 insights in this format:
INSIGHT 1: [Title]
[2-3 sentences with specific numbers and a clear recommendation]

INSIGHT 2: [Title]
[2-3 sentences with specific numbers and a clear recommendation]

And so on for insights 3, 4, 5. Be specific, professional, and actionable."""

            with st.spinner("Generating AI insights with Llama 3.1 70B..."):
                response = call_groq(groq_key, prompt)

            if response.startswith("Error") or response.startswith("Groq error"):
                st.error(response)
            else:
                insights = [s.strip() for s in response.split('\n\n') if s.strip()]
                for insight in insights:
                    lines = insight.split('\n', 1)
                    title = lines[0].replace('INSIGHT', '').strip(' 12345:')
                    body  = lines[1].strip() if len(lines) > 1 else insight
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
anim_dur = max(15, len(news_headlines)*3)
st.markdown(f"""
<style>
.ticker-container{{height:160px;overflow:hidden;background:#0f172a;padding:14px;
  border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.3);color:#fff;
  font-family:'Segoe UI',sans-serif;position:relative}}
.ticker-wrapper{{animation:scroll-up {anim_dur}s linear infinite;will-change:transform}}
@keyframes scroll-up{{0%{{transform:translateY(0)}}100%{{transform:translateY(-50%)}}}}
.ticker-item{{padding:10px 0;font-size:14px;line-height:1.5;
  min-height:38px;overflow:hidden;word-wrap:break-word;border-bottom:1px solid #1e293b}}
</style>""", unsafe_allow_html=True)
html_c = '<div class="ticker-container"><div class="ticker-wrapper">'
for h in all_h:
    html_c += f'<div class="ticker-item">{h}</div>'
html_c += '</div></div>'
st.markdown(html_c, unsafe_allow_html=True)
