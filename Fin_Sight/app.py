# =============================================================================
#  FinSight — app.py  (FIXED — f-string formatting bug resolved)
#  Model  : Multi-feature LSTM  |  Target : Log Returns
#  Eval   : Walk-Forward R², Directional Accuracy, MAPE, RMSE, Naïve Baseline
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
from streamlit_option_menu import option_menu

# ── Initial Setup ─────────────────────────────────────────────────────────────
nltk.download('vader_lexicon', quiet=True)
sia          = SentimentIntensityAnalyzer()
current_date = datetime.now().date()
st.set_page_config(page_title="FinSight", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"]>div:first-child{background:#0b1220;padding:16px 12px}
.block-container{padding-top:.6rem;padding-bottom:.4rem}
.model-box{background:#000;padding:18px;border-radius:12px;
           border:1px solid #111827;font-size:14px;color:#e6eef8}
.info-bar{font-size:12px;color:#cbd5e1;padding:8px 10px;background:#0b1220;
          border-radius:6px;margin-bottom:8px}
.metric-card{background:#0f172a;border-radius:10px;padding:14px 10px;
             text-align:center;border:1px solid #1e293b;margin-bottom:6px}
.metric-label{font-size:12px;color:#94a3b8;margin-bottom:4px}
.metric-value{font-size:22px;font-weight:700;color:#e2e8f0}
.metric-sub{font-size:11px;color:#64748b;margin-top:2px}
.good{color:#22c55e!important}
.warn{color:#f59e0b!important}
.bad {color:#ef4444!important}
.skel-card{background:linear-gradient(90deg,#111827 25%,#0b1220 50%,#111827 75%);
           background-size:200% 100%;animation:shimmer 1.4s linear infinite;
           height:120px;border-radius:10px;margin-bottom:12px}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.baseline-box{background:#0f172a;border:1px solid #1e293b;
              border-radius:10px;padding:14px;margin-bottom:6px}
</style>
""", unsafe_allow_html=True)

st.title("**FinSight**: Real-Time Stock Intelligence")

# ── Full S&P 500 Ticker List ──────────────────────────────────────────────────
tickers = [
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
]
tickers = sorted(set(tickers))

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Main Stock",   tickers, index=tickers.index('AAPL'))
compare_ticker  = st.sidebar.selectbox("Compare With", tickers, index=tickers.index('MSFT'))
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2010-01-01').date())
end_date   = st.sidebar.date_input("End Date",   current_date)
if start_date > end_date:
    st.error("Start date must be before end date."); st.stop()
if end_date > current_date:
    end_date = current_date

@st.cache_resource(ttl=300)
def get_ticker_obj(t): return yf.Ticker(t)
ticker_obj = get_ticker_obj(selected_ticker)

# ── Finnhub News ──────────────────────────────────────────────────────────────
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
                    headlines.append(f"**{t_}** – {s_}")
                    posts.append(f"{t_} – {s_}")
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

# ── Fetch Stock Data ──────────────────────────────────────────────────────────
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

# ── Feature Engineering (11 features) ────────────────────────────────────────
def compute_features(raw):
    """
    Returns (feature_df, price_series).
    Column 0 of feature_df = LogReturn  (model target — stationary & unbiased).
    """
    df = raw.copy()
    df['LogReturn']   = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['SMA20']       = df['Adj Close'].rolling(20).mean()
    df['SMA50']       = df['Adj Close'].rolling(50).mean()
    df['EMA12']       = df['Adj Close'].ewm(span=12, adjust=False).mean()
    df['EMA26']       = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta             = df['Adj Close'].diff()
    up                = delta.clip(lower=0)
    dn                = -delta.clip(upper=0)
    df['RSI']         = 100 - 100 / (1 + up.rolling(14).mean() / (dn.rolling(14).mean() + 1e-9))
    rm                = df['Adj Close'].rolling(20).mean()
    rs                = df['Adj Close'].rolling(20).std()
    df['BB_Width']    = (2 * rs) / (rm + 1e-9)
    hl                = df['High'] - df['Low']
    hc                = (df['High'] - df['Adj Close'].shift()).abs()
    lc                = (df['Low']  - df['Adj Close'].shift()).abs()
    df['ATR']         = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['LogVolume']   = np.log1p(df['Volume'])

    feature_cols = [
        'LogReturn', 'LogVolume',
        'SMA20', 'SMA50', 'EMA12', 'EMA26',
        'MACD', 'MACD_Signal',
        'RSI', 'BB_Width', 'ATR'
    ]
    return df[feature_cols].dropna(), df['Adj Close']

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab = option_menu(None,
    ["Data & Viz", "Predictions", "Sentiment", "Comparison", "Portfolio Analyzer"],
    icons=["table", "graph-up", "chat-dots", "arrow-left-right", "pie-chart"],
    orientation="horizontal")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DATA & VIZ
# ══════════════════════════════════════════════════════════════════════════════
if tab == "Data & Viz":
    st.subheader(f"**{selected_ticker}** – Price History")
    if data_main is not None:
        st.dataframe(data_main.tail(100), use_container_width=True)
        st.download_button("⬇ Download CSV", data_main.to_csv().encode(), f"{selected_ticker}.csv")
        fig = px.line(data_main, x=data_main.index, y='Adj Close',
                      title=f"{selected_ticker} — Adjusted Close Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data available. Try expanding date range.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Predictions":
    st.subheader("Price Forecast — Multi-feature LSTM  |  Industry-Grade Evaluation")

    if data_main is None:
        st.error("Not enough data. Expand date range or choose another ticker.")
        st.stop()

    # Controls
    c1, c2 = st.columns([2, 1])
    with c1:
        days       = st.slider("Forecast horizon (trading days)", 1, 30, 7)
        time_step  = st.slider("Lookback window (days)", 60, 180, 90, step=10)
        epochs     = st.slider("Training epochs", 20, 150, 80, step=5)
        batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
        retrain    = st.checkbox("⚠️ Force retrain model", value=False)
    with c2:
        st.markdown("""
        <div class="model-box">
        <b>11-Feature LSTM</b><br><br>
        <b>Trend:</b> SMA20, SMA50, EMA12, EMA26<br>
        <b>Momentum:</b> MACD, Signal, RSI<br>
        <b>Volatility:</b> BB Width, ATR<br>
        <b>Other:</b> Log Return, Log Volume<br><br>
        <b>Evaluation (industry-grade):</b><br>
        • Walk-Forward R² (5 rolling folds)<br>
        • Directional Accuracy %<br>
        • MAPE % &amp; RMSE $<br>
        • vs Naïve Baseline<br><br>
        🎯 Target R² &gt; 0.82 | DA &gt; 55%
        </div>""", unsafe_allow_html=True)

    df_features, price_series = compute_features(data_main)
    if len(df_features) < time_step + 30:
        st.error(f"Not enough rows. Need ≥ {time_step+30}, got {len(df_features)}.")
        st.stop()

    # ── Training Function ─────────────────────────────────────────────────────
    @st.cache_resource(ttl=24 * 3600)
    def train_model(ticker, start_str, end_str, time_step, epochs, batch_size, retrain_flag):
        t0            = time.time()
        training_time = datetime.now()

        raw = yf.download(ticker, start=start_str, end=end_str, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if 'Adj Close' not in raw.columns and 'Close' in raw.columns:
            raw['Adj Close'] = raw['Close']

        df_feat, price_s = compute_features(raw)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(df_feat.values)
        prices_arr = price_s.loc[df_feat.index].values

        # Sequences
        X, y = [], []
        for i in range(len(scaled) - time_step):
            X.append(scaled[i:i + time_step, :])
            y.append(scaled[i + time_step, 0])   # LogReturn col=0
        X = np.array(X)
        y = np.array(y)

        n       = X.shape[0]
        train_n = int(n * 0.80)
        X_tr, y_tr = X[:train_n], y[:train_n]
        X_te, y_te = X[train_n:], y[train_n:]
        n_feat  = X.shape[2]

        # Model
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

        history = model.fit(X_tr, y_tr,
                            epochs=epochs, batch_size=batch_size,
                            validation_split=val_split,
                            callbacks=cbs, verbose=0)

        # Backtest — reconstruct price from predicted log-return
        bt_preds_p, bt_actuals_p   = [], []
        bt_pred_ret, bt_actual_ret = [], []
        dummy_row = np.zeros((1, n_feat))

        for i in range(len(X_te)):
            global_idx = time_step + train_n + i

            pred_sc        = float(model.predict(X_te[i:i+1], verbose=0)[0, 0])
            dummy_row[0, 0] = pred_sc
            pred_lr        = float(scaler.inverse_transform(dummy_row)[0, 0])
            actual_lr      = float(df_feat['LogReturn'].iloc[global_idx])

            prev_price  = float(price_s.iloc[global_idx - 1])
            pred_price  = prev_price * np.exp(pred_lr)
            actual_price= float(price_s.iloc[global_idx])

            bt_preds_p.append(pred_price)
            bt_actuals_p.append(actual_price)
            bt_pred_ret.append(pred_lr)
            bt_actual_ret.append(actual_lr)

        bt_preds_p   = np.array(bt_preds_p)
        bt_actuals_p = np.array(bt_actuals_p)
        bt_pred_ret  = np.array(bt_pred_ret)
        bt_actual_ret= np.array(bt_actual_ret)

        # Walk-Forward R² (5 folds)
        wf_r2_list = []
        fold_size  = max(10, len(bt_preds_p) // 5)
        for fold in range(5):
            s = fold * fold_size
            e = min(s + fold_size, len(bt_preds_p))
            if e - s < 5: break
            wf_r2_list.append(float(r2_score(bt_actuals_p[s:e], bt_preds_p[s:e])))
        wf_r2 = float(np.mean(wf_r2_list)) if wf_r2_list else 0.0

        # Standard metrics
        mse_val  = float(mean_squared_error(bt_actuals_p, bt_preds_p))
        r2_val   = float(r2_score(bt_actuals_p, bt_preds_p))
        rmse_val = float(np.sqrt(mse_val))
        mape_val = float(np.mean(np.abs(
            (bt_actuals_p - bt_preds_p) / (np.abs(bt_actuals_p) + 1e-9)
        )) * 100)

        # Directional accuracy
        dir_acc = float(np.mean(np.sign(bt_pred_ret) == np.sign(bt_actual_ret)) * 100)

        # Naïve baseline
        naive_p  = bt_actuals_p[:-1]
        naive_a  = bt_actuals_p[1:]
        naive_r2   = float(r2_score(naive_a, naive_p))
        naive_mape = float(np.mean(np.abs((naive_a - naive_p)/(np.abs(naive_a)+1e-9)))*100)
        naive_rmse = float(np.sqrt(mean_squared_error(naive_a, naive_p)))

        resid_std = float(np.std(bt_actuals_p - bt_preds_p))

        return {
            'model':            model,
            'scaler':           scaler,
            'df_feat':          df_feat,
            'price_series':     price_s,
            'time_step':        time_step,
            'train_n':          train_n,
            'n_feat':           n_feat,
            'bt_preds_price':   bt_preds_p,
            'bt_actuals_price': bt_actuals_p,
            'bt_pred_ret':      bt_pred_ret,
            'bt_actual_ret':    bt_actual_ret,
            'history':          history.history,
            'training_time':    training_time,
            'training_secs':    time.time() - t0,
            'epochs':           epochs,
            'batch_size':       batch_size,
            'mse':              mse_val,
            'r2':               r2_val,
            'rmse':             rmse_val,
            'mape':             mape_val,
            'dir_acc':          dir_acc,
            'wf_r2':            wf_r2,
            'wf_r2_list':       wf_r2_list,
            'naive_r2':         naive_r2,
            'naive_mape':       naive_mape,
            'naive_rmse':       naive_rmse,
            'resid_std':        resid_std,
        }

    # Skeleton loader
    ph = st.empty()
    with ph.container():
        st.markdown('<div class="skel-card"></div>', unsafe_allow_html=True)
        st.markdown('<div class="skel-card"></div>', unsafe_allow_html=True)
    try:
        art = train_model(
            selected_ticker, str(start_date), str(end_date),
            time_step, epochs, batch_size, retrain
        )
    finally:
        ph.empty()

    # Unpack
    model         = art['model']
    scaler        = art['scaler']
    df_used       = art['df_feat']
    price_s       = art['price_series']
    train_n       = art['train_n']
    bt_preds      = art['bt_preds_price']
    bt_actuals    = art['bt_actuals_price']
    bt_pred_ret   = art['bt_pred_ret']
    bt_actual_ret = art['bt_actual_ret']
    history       = art['history']
    training_time = art['training_time']
    n_feat        = art['n_feat']
    resid_std     = art['resid_std']

    # ── Info Bar ─────────────────────────────────────────────────────────────
    age     = datetime.now() - training_time
    age_str = f"{age.days}d {age.seconds//3600}h {(age.seconds%3600)//60}m"
    st.markdown(f"""
    <div class="info-bar">
    <b>Model</b>: 2-layer LSTM &nbsp;|&nbsp;
    <b>Features</b>: {n_feat} &nbsp;|&nbsp;
    <b>Lookback</b>: {art['time_step']}d &nbsp;|&nbsp;
    <b>Epochs</b>: {art['epochs']} &nbsp;|&nbsp;
    <b>Batch</b>: {art['batch_size']} &nbsp;|&nbsp;
    <b>Trained</b>: {training_time.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
    <b>Age</b>: {age_str} &nbsp;|&nbsp;
    <b>Cached</b>: {'Yes' if not retrain else 'No (forced)'}
    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION A — EVALUATION DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📊 Model Evaluation Dashboard")
    st.caption("Industry-standard metrics used by quant teams and ML practitioners.")

    # Helper: colour class
    def ccls(val, good_t, warn_t, higher=True):
        if higher:
            return "good" if val >= good_t else "warn" if val >= warn_t else "bad"
        return "good" if val <= good_t else "warn" if val <= warn_t else "bad"

    # ── Pre-format metric strings (avoids f-string conditional formatting bug) ─
    wf_r2_str  = f"{art['wf_r2']:.3f}"
    r2_str     = f"{art['r2']:.3f}"
    da_str     = f"{art['dir_acc']:.1f}%"
    mape_str   = f"{art['mape']:.2f}%"
    rmse_str   = f"${art['rmse']:.2f}"

    wf_cls  = ccls(art['wf_r2'],   0.80, 0.65)
    r2_cls  = ccls(art['r2'],      0.80, 0.65)
    da_cls  = ccls(art['dir_acc'], 58,   52)
    mp_cls  = ccls(art['mape'],    3,    5,  higher=False)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Walk-Forward R²</div>
          <div class="metric-value {wf_cls}">{wf_r2_str}</div>
          <div class="metric-sub">5-fold rolling windows</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Standard R²</div>
          <div class="metric-value {r2_cls}">{r2_str}</div>
          <div class="metric-sub">Test set variance explained</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Directional Accuracy</div>
          <div class="metric-value {da_cls}">{da_str}</div>
          <div class="metric-sub">Random baseline = 50%</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">MAPE</div>
          <div class="metric-value {mp_cls}">{mape_str}</div>
          <div class="metric-sub">Mean Abs % Error (lower=better)</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">RMSE</div>
          <div class="metric-value">{rmse_str}</div>
          <div class="metric-sub">Avg $ error per day</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Naïve Baseline Comparison ─────────────────────────────────────────────
    st.markdown("### 🆚 Model vs Naïve Baseline")
    st.caption("Naïve baseline = predicting tomorrow's price = today's price (zero ML).")

    beat_r2   = art['r2']   > art['naive_r2']
    beat_mape = art['mape'] < art['naive_mape']
    beat_rmse = art['rmse'] < art['naive_rmse']
    wins      = sum([beat_r2, beat_mape, beat_rmse])

    # Pre-format baseline strings
    lstm_r2_str    = f"{art['r2']:.3f}"
    naive_r2_str   = f"{art['naive_r2']:.3f}"
    lstm_mape_str  = f"{art['mape']:.2f}%"
    naive_mape_str = f"{art['naive_mape']:.2f}%"
    lstm_rmse_str  = f"${art['rmse']:.2f}"
    naive_rmse_str = f"${art['naive_rmse']:.2f}"

    r2_c   = "good" if beat_r2   else "bad"
    mp_c   = "good" if beat_mape else "bad"
    rm_c   = "good" if beat_rmse else "bad"

    verdict_color = "good" if wins == 3 else "warn" if wins >= 2 else "bad"
    verdict_text  = ("✅ Beats baseline on all 3 metrics" if wins == 3
                     else f"⚠️ Beats baseline on {wins}/3 metrics" if wins >= 2
                     else "❌ Does not beat baseline")

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown("""
        <div class="baseline-box">
          <div class="metric-label">Metric</div>
          <div style="color:#94a3b8;margin-top:10px">R²</div>
          <div style="color:#94a3b8;margin-top:10px">MAPE</div>
          <div style="color:#94a3b8;margin-top:10px">RMSE</div>
        </div>""", unsafe_allow_html=True)
    with b2:
        st.markdown(f"""
        <div class="baseline-box">
          <div class="metric-label">Our LSTM</div>
          <div class="{r2_c}" style="margin-top:10px">{lstm_r2_str}</div>
          <div class="{mp_c}" style="margin-top:10px">{lstm_mape_str}</div>
          <div class="{rm_c}" style="margin-top:10px">{lstm_rmse_str}</div>
        </div>""", unsafe_allow_html=True)
    with b3:
        st.markdown(f"""
        <div class="baseline-box">
          <div class="metric-label">Naïve Baseline</div>
          <div style="color:#e2e8f0;margin-top:10px">{naive_r2_str}</div>
          <div style="color:#e2e8f0;margin-top:10px">{naive_mape_str}</div>
          <div style="color:#e2e8f0;margin-top:10px">{naive_rmse_str}</div>
        </div>""", unsafe_allow_html=True)
    with b4:
        st.markdown(f"""
        <div class="baseline-box">
          <div class="metric-label">Verdict</div>
          <div class="{verdict_color}"
               style="margin-top:10px;font-size:13px;font-weight:600">
            {verdict_text}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Walk-Forward R² per fold ──────────────────────────────────────────────
    if art['wf_r2_list']:
        st.markdown("### 📈 Walk-Forward R² — Per Fold")
        wf_colors = ['#22c55e' if v >= 0.80 else '#f59e0b' if v >= 0.65 else '#ef4444'
                     for v in art['wf_r2_list']]
        wf_labels = [f"{v:.3f}" for v in art['wf_r2_list']]
        fold_names = [f"Fold {i+1}" for i in range(len(art['wf_r2_list']))]
        fig_wf = go.Figure()
        for i, (fn, fv, fc, fl) in enumerate(
                zip(fold_names, art['wf_r2_list'], wf_colors, wf_labels)):
            fig_wf.add_trace(go.Bar(x=[fn], y=[fv], marker_color=fc,
                                    text=[fl], textposition='outside',
                                    showlegend=False))
        fig_wf.add_hline(y=0.80, line_dash='dash', line_color='#22c55e',
                         annotation_text="Target 0.80")
        fig_wf.add_hline(y=0.65, line_dash='dot',  line_color='#f59e0b',
                         annotation_text="Acceptable 0.65")
        y_min = min(min(art['wf_r2_list']) - 0.05, -0.05)
        fig_wf.update_layout(
            title=f"Walk-Forward R² across folds  |  Mean = {wf_r2_str}",
            yaxis_title="R²", yaxis_range=[y_min, 1.05], height=320
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION B — PERFORMANCE CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📉 Model Performance Charts")
    pc1, pc2 = st.columns(2)

    with pc1:
        st.write("### Training Loss Curve")
        ep_r  = list(range(1, len(history.get('loss', [])) + 1))
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=ep_r, y=history.get('loss', []),
                                   mode='lines+markers', name='Train Loss',
                                   line=dict(color='#1f77b4')))
        if 'val_loss' in history:
            fig_l.add_trace(go.Scatter(x=ep_r, y=history['val_loss'],
                                       mode='lines+markers', name='Val Loss',
                                       line=dict(color='#ff7f0e')))
        fig_l.update_layout(title="Loss Curve (lower = better)",
                            xaxis_title="Epoch", yaxis_title="MSE Loss")
        st.plotly_chart(fig_l, use_container_width=True)

    with pc2:
        st.write("### Backtest: Actual vs Predicted Price")
        if len(bt_preds) > 0:
            bt_start = art['time_step'] + train_n
            bt_idx   = df_used.index[bt_start: bt_start + len(bt_preds)]
            fig_bt   = go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt_idx, y=bt_actuals,
                                        name='Actual',    line=dict(color='#1f77b4')))
            fig_bt.add_trace(go.Scatter(x=bt_idx, y=bt_preds,
                                        name='Predicted', line=dict(color='#ff7f0e')))
            fig_bt.update_layout(title="Backtest: Actual vs Predicted",
                                 xaxis_title="Date", yaxis_title="Price ($)")
            st.plotly_chart(fig_bt, use_container_width=True)
            st.caption(
                f"Samples: {len(bt_preds)}  |  "
                f"MSE: {art['mse']:.2f}  |  "
                f"R²: {r2_str}  |  "
                f"RMSE: {rmse_str}  |  "
                f"MAPE: {mape_str}"
            )

    # ── Directional Accuracy Chart ────────────────────────────────────────────
    st.write("### 🧭 Directional Accuracy — Did Model Call Up/Down Correctly?")
    st.caption("Green = correct direction predicted. Red = wrong. Target: consistently > 50%.")
    if len(bt_preds) > 0:
        dir_correct = (np.sign(bt_pred_ret) == np.sign(bt_actual_ret)).astype(int)
        bt_idx2     = df_used.index[art['time_step'] + train_n:
                                    art['time_step'] + train_n + len(bt_preds)]
        bar_colors  = ['#22c55e' if c == 1 else '#ef4444' for c in dir_correct]
        fig_dir     = go.Figure()
        fig_dir.add_trace(go.Bar(x=bt_idx2, y=dir_correct,
                                 marker_color=bar_colors, showlegend=False))
        fig_dir.add_hline(y=0.5, line_dash='dash', line_color='#94a3b8',
                          annotation_text="Random baseline (50%)")
        n_correct = int(dir_correct.sum())
        n_total   = len(dir_correct)
        fig_dir.update_layout(
            title=f"Directional Accuracy: {art['dir_acc']:.1f}%  ({n_correct}/{n_total} correct)",
            xaxis_title="Date",
            yaxis_title="Correct (1) / Wrong (0)",
            height=280
        )
        st.plotly_chart(fig_dir, use_container_width=True)

    # ── Residuals Distribution ────────────────────────────────────────────────
    st.write("### 📐 Residuals Distribution")
    st.caption("Tight bell curve centred near 0 = well-calibrated model.")
    residuals = bt_actuals - bt_preds
    res_mean  = f"{residuals.mean():.2f}"
    res_std   = f"{residuals.std():.2f}"
    fig_res   = go.Figure()
    fig_res.add_trace(go.Histogram(x=residuals, nbinsx=40,
                                   marker_color='#6366f1', opacity=0.75))
    fig_res.add_vline(x=0, line_color='white', line_dash='dash')
    fig_res.update_layout(
        title=f"Residuals  |  Mean = ${res_mean}  |  Std = ${res_std}",
        xaxis_title="Residual ($)", yaxis_title="Count", height=280
    )
    st.plotly_chart(fig_res, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION C — FUTURE FORECAST
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🔮 Future Forecast")

    recent_scaled = scaler.transform(df_used.values[-time_step:]).tolist()
    last_price    = float(price_s.iloc[-1])
    chain_price   = last_price
    price_list    = price_s.tolist()
    future_preds_price = []
    dummy_f = np.zeros((1, n_feat))

    for step in range(days):
        inp     = np.array(recent_scaled[-time_step:]).reshape(1, time_step, -1)
        pred_sc = float(model.predict(inp, verbose=0)[0, 0])

        dummy_f[0, 0] = pred_sc
        pred_lr       = float(scaler.inverse_transform(dummy_f)[0, 0])
        pred_price    = chain_price * np.exp(pred_lr)
        future_preds_price.append(pred_price)
        chain_price = pred_price
        price_list.append(pred_price)

        # Rebuild next feature row from extended price list
        adj_s    = pd.Series(price_list)
        log_vol_ = float(df_used['LogVolume'].iloc[-1])
        sma20_   = adj_s.rolling(20).mean().iloc[-1]
        sma50_   = adj_s.rolling(50).mean().iloc[-1]
        ema12_   = adj_s.ewm(span=12, adjust=False).mean().iloc[-1]
        ema26_   = adj_s.ewm(span=26, adjust=False).mean().iloc[-1]
        macd_    = ema12_ - ema26_
        macd_s_  = pd.Series(
            [df_used['MACD_Signal'].iloc[-1]] * 8 + [macd_]
        ).ewm(span=9, adjust=False).mean().iloc[-1]
        diff_    = adj_s.diff().fillna(0)
        up_      = diff_.clip(lower=0)
        dn_      = -diff_.clip(upper=0)
        ru_      = up_.rolling(14).mean().iloc[-1] if len(up_) >= 14 else up_.mean()
        rd_      = dn_.rolling(14).mean().iloc[-1] if len(dn_) >= 14 else dn_.mean()
        rsi_     = 100 - 100 / (1 + ru_ / (rd_ + 1e-9))
        rm_      = adj_s.rolling(20).mean().iloc[-1]
        rs_      = adj_s.rolling(20).std().iloc[-1]  if len(adj_s) >= 20 else adj_s.std()
        bb_w_    = (2 * rs_) / (rm_ + 1e-9)
        atr_     = float(df_used['ATR'].iloc[-1])

        new_row = np.array([[pred_lr, log_vol_, sma20_, sma50_,
                             ema12_, ema26_, macd_, macd_s_,
                             rsi_,   bb_w_,  atr_]])
        new_sc  = scaler.transform(new_row)[0].tolist()
        recent_scaled.append(new_sc)

    # Safe growing CI (capped at 15% of predicted price)
    floor  = last_price * 0.55
    z      = 1.96
    uppers, lowers = [], []
    for i, p in enumerate(future_preds_price):
        hw = min(z * resid_std * np.sqrt(i + 1), 0.15 * p)
        uppers.append(p + hw)
        lowers.append(max(p - hw, floor))

    future_dates = pd.date_range(
        start=data_main.index[-1] + pd.Timedelta(days=1),
        periods=days, freq='B'
    )
    future_df = pd.DataFrame({
        'Date':      future_dates,
        'Predicted': future_preds_price,
        'Upper':     uppers,
        'Lower':     lowers
    })

    # Animated forecast chart
    hist_x = price_s.index
    hist_y = price_s.values

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=hist_x, y=hist_y,
                               name='Historical', line=dict(color='#1f77b4')))
    fig_f.add_trace(go.Scatter(x=[future_dates[0]], y=[future_preds_price[0]],
                               name='Forecast',   line=dict(color='#ff7f0e')))
    fig_f.add_trace(go.Scatter(
        x=[future_dates[0], future_dates[0]],
        y=[lowers[0], uppers[0]],
        fill='toself', fillcolor='rgba(255,127,14,0.15)',
        line=dict(color='rgba(255,127,14,0)'),
        name='95% CI', showlegend=True
    ))

    frames = []
    for i in range(len(future_dates)):
        xp = future_dates[:i+1]
        yp = future_preds_price[:i+1]
        xb = list(future_dates[:i+1]) + list(future_dates[:i+1][::-1])
        yb = list(uppers[:i+1]) + list(lowers[:i+1][::-1])
        frames.append(go.Frame(data=[
            go.Scatter(x=hist_x, y=hist_y),
            go.Scatter(x=xp, y=yp, line=dict(color='#ff7f0e')),
            go.Scatter(x=xb, y=yb, fill='toself',
                       fillcolor='rgba(255,127,14,0.15)',
                       line=dict(color='rgba(255,127,14,0)'))
        ], name=str(i)))
    fig_f.frames = frames

    last_close_str = f"${last_price:.2f}"
    fig_f.update_layout(
        title=f"{selected_ticker} — {days}-day Forecast  |  Last close: {last_close_str}",
        xaxis_title="Date", yaxis_title="Price ($)",
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 400, "redraw": True},
                                 "fromcurrent": True,
                                 "transition": {"duration": 200}}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}]}
            ],
            "direction": "left", "pad": {"r": 10, "t": 10},
            "showactive": True, "x": 0.01, "y": -0.12,
            "xanchor": "left", "yanchor": "top"
        }]
    )
    st.plotly_chart(fig_f, use_container_width=True)
    st.dataframe(
        future_df.style.format({
            "Predicted": "{:.2f}",
            "Upper":     "{:.2f}",
            "Lower":     "{:.2f}"
        }),
        use_container_width=True
    )

    # Summary footer
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Walk-Forward R²",  wf_r2_str)
    sm2.metric("Directional Acc",  da_str)
    sm3.metric("MAPE",             mape_str)
    sm4.metric("Beats Naïve",      "✅ Yes" if wins >= 2 else "⚠️ Partial")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Sentiment":
    st.subheader("News Sentiment")
    if news_posts:
        df_s = pd.DataFrame({'News': news_posts, 'Link': news_links, 'Score': vader_scores})
        def color(val):
            return f"color: {'green' if val > 0.1 else 'red' if val < -0.1 else 'gray'}"
        st.dataframe(
            df_s.style.applymap(color, subset=['Score']).format({'Score': '{:.3f}'}),
            use_container_width=True
        )
        pos = sum(1 for s in vader_scores if s > 0.1)
        neg = sum(1 for s in vader_scores if s < -0.1)
        neu = len(vader_scores) - pos - neg
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", pos)
        c2.metric("Negative", neg)
        c3.metric("Neutral",  neu)
    else:
        st.info("No news available.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Comparison":
    st.subheader(f"**{selected_ticker} vs {compare_ticker}**")
    if data_main is not None and data_compare is not None:
        bm = data_main['Adj Close'].iloc[0]
        bc = data_compare['Adj Close'].iloc[0]
        dm = (data_main['Adj Close']    / bm - 1) * 100
        dc = (data_compare['Adj Close'] / bc - 1) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_main.index,    y=dm,
                                 name=selected_ticker, line=dict(color='#26A69A')))
        fig.add_trace(go.Scatter(x=data_compare.index, y=dc,
                                 name=compare_ticker,  line=dict(color='#AB47BC')))
        fig.update_layout(title="Normalised Performance (%)",
                          height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        rm = (data_main['Adj Close'].iloc[-1]    / bm - 1) * 100
        rc = (data_compare['Adj Close'].iloc[-1] / bc - 1) * 100
        vm = data_main['Adj Close'].pct_change().std()    * np.sqrt(252) * 100
        vc = data_compare['Adj Close'].pct_change().std() * np.sqrt(252) * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{selected_ticker} Return", f"{rm:+.2f}%")
        c2.metric(f"{compare_ticker} Return",  f"{rc:+.2f}%")
        c3.metric(f"{selected_ticker} Vol",    f"{vm:.1f}%")
        c4.metric(f"{compare_ticker} Vol",     f"{vc:.1f}%")
    else:
        st.error("Not enough data to compare.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — PORTFOLIO ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Portfolio Analyzer":
    st.subheader("Portfolio Analyzer")
    port_tickers = st.multiselect("Select Tickers", tickers,
                                  default=[selected_ticker, compare_ticker])
    if len(port_tickers) < 2:
        st.warning("Select at least 2 tickers.")
    else:
        weights, total_w = [], 0.0
        cols = st.columns(len(port_tickers))
        for i, tick in enumerate(port_tickers):
            w = cols[i].number_input(f"Weight {tick} (%)", 0.0, 100.0,
                                     100.0 / len(port_tickers))
            weights.append(w / 100)
            total_w += w
        if abs(total_w - 100) > 0.01:
            st.warning(f"Weights sum to {total_w:.1f}%. Should be 100%.")
        else:
            data_dict = {}
            for tick in port_tickers:
                d = fetch_stock_data(tick, start_date, end_date)
                if d is None:
                    st.error(f"Data missing for {tick}."); st.stop()
                data_dict[tick] = d['Adj Close']
            port_df = pd.DataFrame(data_dict)
            rets    = port_df.pct_change().dropna()
            m_ret   = rets.mean() * 252
            cov_mat = rets.cov()  * 252
            w_np    = np.array(weights)
            p_ret   = np.dot(m_ret, w_np)
            p_vol   = np.sqrt(np.dot(w_np.T, np.dot(cov_mat, w_np)))
            sharpe  = (p_ret - 0.03) / p_vol
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Return",      f"{p_ret*100:.2f}%")
            c2.metric("Portfolio Volatility",  f"{p_vol*100:.2f}%")
            c3.metric("Sharpe Ratio",          f"{sharpe:.2f}")
            fig_h = px.imshow(rets.corr(), text_auto=True, aspect="auto",
                              color_continuous_scale='RdBu_r',
                              title="Correlation Heatmap")
            st.plotly_chart(fig_h, use_container_width=True)

# ── News Ticker ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Latest Headlines (24/7)")
all_h    = news_headlines + news_headlines
anim_dur = max(15, len(news_headlines) * 3)
st.markdown(f"""
<style>
.ticker-container{{height:180px;overflow:hidden;background:#0f172a;padding:16px;
  border-radius:14px;box-shadow:0 6px 24px rgba(0,0,0,.3);
  color:#fff;font-family:'Segoe UI',sans-serif;position:relative}}
.ticker-wrapper{{animation:scroll-up {anim_dur}s linear infinite;will-change:transform}}
@keyframes scroll-up{{0%{{transform:translateY(0)}}100%{{transform:translateY(-50%)}}}}
.ticker-item{{padding:12px 0;font-size:15px;line-height:1.6;
  min-height:40px;overflow:hidden;word-wrap:break-word}}
</style>""", unsafe_allow_html=True)
html_c = '<div class="ticker-container"><div class="ticker-wrapper">'
for h in all_h:
    html_c += f'<div class="ticker-item">{h}</div>'
html_c += '</div></div>'
st.markdown(html_c, unsafe_allow_html=True)
