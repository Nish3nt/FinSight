# =============================================================================
#  FinSight — app.py  (ENHANCED — LLM Insights + Better Portfolio + Comparison)
#  Model  : Multi-feature LSTM  |  Target : Log Returns
#  LLM    : Groq llama-3.3-70b-versatile (hardcoded key)
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

nltk.download('vader_lexicon', quiet=True)
sia          = SentimentIntensityAnalyzer()
current_date = datetime.now().date()
st.set_page_config(page_title="FinSight", layout="wide")

# ── Groq Config ───────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
GROQ_MODEL   = "openai/gpt-oss-120b"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

def call_groq(prompt, max_tokens=600):
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json"
        }
        payload = {
            "model":       GROQ_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  max_tokens,
            "temperature": 0.4
        }
        r = requests.post(GROQ_URL, headers=headers,
                          json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content'].strip()
        return f"__ERROR__: {r.status_code} — {r.text}"
    except Exception as e:
        return f"__ERROR__: {e}"

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
.ai-box{background:linear-gradient(135deg,#0f172a,#1e1b4b);
        border:1px solid #4338ca;border-radius:12px;padding:18px 20px;
        margin-top:12px;font-size:13px;color:#c7d2fe;line-height:1.8}
.ai-box-title{font-size:11px;font-weight:700;color:#818cf8;
              text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.rec-box-buy{background:#052e16;border:1px solid #166534;border-radius:12px;
             padding:18px 20px;margin-top:10px}
.rec-box-hold{background:#1c1917;border:1px solid #78350f;border-radius:12px;
              padding:18px 20px;margin-top:10px}
.rec-box-avoid{background:#1a0a0a;border:1px solid #7f1d1d;border-radius:12px;
               padding:18px 20px;margin-top:10px}
.ins-card{background:#0f172a;border-left:3px solid #6366f1;
          border-radius:0 8px 8px 0;padding:12px 16px;font-size:13px;
          color:#cbd5e1;margin-bottom:8px;line-height:1.6}
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
if start_date > end_date:
    st.error("Start date must be before end date."); st.stop()
if end_date > current_date:
    end_date = current_date

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
    up = delta.clip(lower=0); dn = -delta.clip(upper=0)
    df['RSI']         = 100 - 100/(1+up.rolling(14).mean()/(dn.rolling(14).mean()+1e-9))
    rm = df['Adj Close'].rolling(20).mean(); rs = df['Adj Close'].rolling(20).std()
    df['BB_Width']    = (2*rs)/(rm+1e-9)
    hl = df['High']-df['Low']
    hc = (df['High']-df['Adj Close'].shift()).abs()
    lc = (df['Low'] -df['Adj Close'].shift()).abs()
    df['ATR']         = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    df['LogVolume']   = np.log1p(df['Volume'])
    cols = ['LogReturn','LogVolume','SMA20','SMA50','EMA12','EMA26',
            'MACD','MACD_Signal','RSI','BB_Width','ATR']
    return df[cols].dropna(), df['Adj Close']

def compute_drawdown(series):
    roll_max = series.cummax()
    return (series - roll_max) / roll_max * 100

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab = option_menu(None,
    ["Data & Viz", "Predictions", "Sentiment", "Comparison", "Portfolio Analyzer"],
    icons=["table", "graph-up", "chat-dots", "arrow-left-right", "pie-chart"],
    orientation="horizontal")

# ==============================================================================
#  TAB 1 — DATA & VIZ
# ==============================================================================
if tab == "Data & Viz":
    st.subheader(f"**{selected_ticker}** - Price History")
    if data_main is not None:
        st.dataframe(data_main.tail(100), use_container_width=True)
        st.download_button("Download CSV", data_main.to_csv().encode(), f"{selected_ticker}.csv")
        fig = px.line(data_main, x=data_main.index, y='Adj Close',
                      title=f"{selected_ticker} - Adjusted Close Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data available. Try expanding date range.")


# ==============================================================================
#  TAB 2 — PREDICTIONS  (original logic + LLM insights added at bottom)
# ==============================================================================
elif tab == "Predictions":
    st.subheader("Price Forecast - Multi-feature LSTM | Industry-Grade Evaluation")

    if data_main is None:
        st.error("Not enough data. Expand date range or choose another ticker.")
        st.stop()

    c1, c2 = st.columns([2, 1])
    with c1:
        days       = st.slider("Forecast horizon (trading days)", 1, 30, 7)
        time_step  = st.slider("Lookback window (days)", 60, 180, 90, step=10)
        epochs     = st.slider("Training epochs", 20, 150, 80, step=5)
        batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
        retrain    = st.checkbox("Force retrain model", value=False)
    with c2:
        st.markdown("""
        <div class="model-box">
        <b>11-Feature LSTM</b><br><br>
        <b>Trend:</b> SMA20, SMA50, EMA12, EMA26<br>
        <b>Momentum:</b> MACD, Signal, RSI<br>
        <b>Volatility:</b> BB Width, ATR<br>
        <b>Other:</b> Log Return, Log Volume<br><br>
        <b>Evaluation:</b><br>
        Walk-Forward R2 (5 folds)<br>
        Directional Accuracy<br>
        MAPE and RMSE<br>
        vs Naive Baseline<br><br>
        Target R2 > 0.82 | DA > 55%
        </div>""", unsafe_allow_html=True)

    df_features, price_series = compute_features(data_main)
    if len(df_features) < time_step + 30:
        st.error(f"Not enough rows. Need >= {time_step+30}, got {len(df_features)}."); st.stop()

    @st.cache_resource(ttl=24*3600)
    def train_model(ticker, start_str, end_str, time_step, epochs, batch_size, retrain_flag):
        t0 = time.time(); training_time = datetime.now()
        raw = yf.download(ticker, start=start_str, end=end_str, progress=False)
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if 'Adj Close' not in raw.columns and 'Close' in raw.columns:
            raw['Adj Close'] = raw['Close']
        df_feat, price_s = compute_features(raw)
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaled = scaler.fit_transform(df_feat.values)
        X, y = [], []
        for i in range(len(scaled)-time_step):
            X.append(scaled[i:i+time_step,:]); y.append(scaled[i+time_step,0])
        X = np.array(X); y = np.array(y)
        n = X.shape[0]; train_n = int(n*0.80)
        X_tr,y_tr = X[:train_n],y[:train_n]
        X_te,y_te = X[train_n:],y[train_n:]
        n_feat = X.shape[2]
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(time_step,n_feat)),
            Dropout(0.2), LSTM(64), Dropout(0.15),
            Dense(32, activation='relu'), Dense(1)
        ])
        model.compile(optimizer=Adam(0.001), loss='mse')
        cbs, val_split = [], 0.0
        if len(X_tr) > 20:
            cbs = [EarlyStopping(monitor='val_loss',patience=12,
                                  restore_best_weights=True,verbose=0),
                   ReduceLROnPlateau(monitor='val_loss',factor=0.5,
                                     patience=6,min_lr=1e-6,verbose=0)]
            val_split = 0.1
        history = model.fit(X_tr,y_tr,epochs=epochs,batch_size=batch_size,
                            validation_split=val_split,callbacks=cbs,verbose=0)
        bt_pp,bt_ap,bt_pr,bt_ar = [],[],[],[]
        dummy_row = np.zeros((1,n_feat))
        for i in range(len(X_te)):
            gi = time_step+train_n+i
            ps = float(model.predict(X_te[i:i+1],verbose=0)[0,0])
            dummy_row[0,0] = ps
            plr = float(scaler.inverse_transform(dummy_row)[0,0])
            alr = float(df_feat['LogReturn'].iloc[gi])
            pp  = float(price_s.iloc[gi-1])*np.exp(plr)
            ap  = float(price_s.iloc[gi])
            bt_pp.append(pp); bt_ap.append(ap)
            bt_pr.append(plr); bt_ar.append(alr)
        bt_pp=np.array(bt_pp); bt_ap=np.array(bt_ap)
        bt_pr=np.array(bt_pr); bt_ar=np.array(bt_ar)
        wf,fs=[],max(10,len(bt_pp)//5)
        for f in range(5):
            s=f*fs; e=min(s+fs,len(bt_pp))
            if e-s<5: break
            wf.append(float(r2_score(bt_ap[s:e],bt_pp[s:e])))
        wf_r2  = float(np.mean(wf)) if wf else 0.0
        mse_v  = float(mean_squared_error(bt_ap,bt_pp))
        r2_v   = float(r2_score(bt_ap,bt_pp))
        rmse_v = float(np.sqrt(mse_v))
        mape_v = float(np.mean(np.abs((bt_ap-bt_pp)/(np.abs(bt_ap)+1e-9)))*100)
        da_v   = float(np.mean(np.sign(bt_pr)==np.sign(bt_ar))*100)
        np_=bt_ap[:-1]; na_=bt_ap[1:]
        n_r2=float(r2_score(na_,np_))
        n_mape=float(np.mean(np.abs((na_-np_)/(np.abs(na_)+1e-9)))*100)
        n_rmse=float(np.sqrt(mean_squared_error(na_,np_)))
        rs_v=float(np.std(bt_ap-bt_pp))
        return dict(model=model,scaler=scaler,df_feat=df_feat,price_series=price_s,
                    time_step=time_step,train_n=train_n,n_feat=n_feat,
                    bt_pp=bt_pp,bt_ap=bt_ap,bt_pr=bt_pr,bt_ar=bt_ar,
                    history=history.history,training_time=training_time,
                    training_secs=time.time()-t0,epochs=epochs,batch_size=batch_size,
                    mse=mse_v,r2=r2_v,rmse=rmse_v,mape=mape_v,da=da_v,
                    wf_r2=wf_r2,wf_list=wf,n_r2=n_r2,n_mape=n_mape,n_rmse=n_rmse,
                    resid_std=rs_v)

    ph = st.empty()
    with ph.container():
        st.markdown('<div class="skel-card"></div>',unsafe_allow_html=True)
        st.markdown('<div class="skel-card"></div>',unsafe_allow_html=True)
    try:
        art = train_model(selected_ticker,str(start_date),str(end_date),
                          time_step,epochs,batch_size,retrain)
    finally:
        ph.empty()

    model    = art['model'];    scaler   = art['scaler']
    df_used  = art['df_feat'];  price_s  = art['price_series']
    train_n  = art['train_n'];  n_feat   = art['n_feat']
    bt_preds = art['bt_pp'];    bt_actuals = art['bt_ap']
    bt_pred_ret = art['bt_pr']; bt_actual_ret = art['bt_ar']
    history  = art['history'];  training_time = art['training_time']
    resid_std = art['resid_std']
    beat_r2  = art['r2']   > art['n_r2']
    beat_mape= art['mape'] < art['n_mape']
    beat_rmse= art['rmse'] < art['n_rmse']
    wins     = sum([beat_r2,beat_mape,beat_rmse])

    def ccls(val,gt,wt,higher=True):
        if higher: return "good" if val>=gt else "warn" if val>=wt else "bad"
        return "good" if val<=gt else "warn" if val<=wt else "bad"

    wf_r2_str=f"{art['wf_r2']:.3f}"; r2_str=f"{art['r2']:.3f}"
    da_str=f"{art['da']:.1f}%"; mape_str=f"{art['mape']:.2f}%"
    rmse_str=f"${art['rmse']:.2f}"
    last_price = float(price_s.iloc[-1])

    age = datetime.now()-training_time
    age_str=f"{age.days}d {age.seconds//3600}h {(age.seconds%3600)//60}m"
    st.markdown(f"""<div class="info-bar">
    <b>Model</b>: 2-layer LSTM | <b>Features</b>: {n_feat} |
    <b>Lookback</b>: {art['time_step']}d | <b>Epochs</b>: {art['epochs']} |
    <b>Batch</b>: {art['batch_size']} | <b>Trained</b>: {training_time.strftime('%Y-%m-%d %H:%M')} |
    <b>Age</b>: {age_str} | <b>Cached</b>: {'Yes' if not retrain else 'No'}
    </div>""", unsafe_allow_html=True)

    st.markdown("## Model Evaluation Dashboard")
    st.caption("Industry-standard metrics measured on test data the model never saw during training.")

    wf_cls=ccls(art['wf_r2'],0.80,0.65); r2_cls=ccls(art['r2'],0.80,0.65)
    da_cls=ccls(art['da'],58,52);         mp_cls=ccls(art['mape'],3,5,higher=False)

    m1,m2,m3,m4,m5 = st.columns(5)
    for col,lbl,val,cls,sub in [
        (m1,"Walk-Forward R2",wf_r2_str,wf_cls,"5-fold rolling windows"),
        (m2,"Standard R2",r2_str,r2_cls,"Test set variance explained"),
        (m3,"Directional Accuracy",da_str,da_cls,"Random baseline = 50%"),
        (m4,"MAPE",mape_str,mp_cls,"Mean Abs % Error (lower=better)"),
        (m5,"RMSE",rmse_str,"","Avg $ error per day"),
    ]:
        col.markdown(f"""<div class="metric-card">
          <div class="metric-label">{lbl}</div>
          <div class="metric-value {cls}">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("### vs Naive Baseline")
    st.caption("Naive = tomorrow's price equals today. Any useful model must beat this.")

    vc_="good" if wins==3 else "warn" if wins>=2 else "bad"
    vt_=("Beats baseline on all 3 metrics" if wins==3
         else f"Beats baseline on {wins}/3 metrics" if wins>=2
         else "Does not beat baseline")
    b1,b2,b3,b4=st.columns(4)
    r2c_="good" if beat_r2 else "bad"
    mpc_="good" if beat_mape else "bad"
    rmc_="good" if beat_rmse else "bad"
    with b1:
        st.markdown("""<div class="baseline-box">
          <div class="metric-label">Metric</div>
          <div style="color:#94a3b8;margin-top:10px">R2</div>
          <div style="color:#94a3b8;margin-top:10px">MAPE</div>
          <div style="color:#94a3b8;margin-top:10px">RMSE</div>
        </div>""",unsafe_allow_html=True)
    with b2:
        st.markdown(f"""<div class="baseline-box">
          <div class="metric-label">Our LSTM</div>
          <div class="{r2c_}" style="margin-top:10px">{art['r2']:.3f}</div>
          <div class="{mpc_}" style="margin-top:10px">{art['mape']:.2f}%</div>
          <div class="{rmc_}" style="margin-top:10px">${art['rmse']:.2f}</div>
        </div>""",unsafe_allow_html=True)
    with b3:
        st.markdown(f"""<div class="baseline-box">
          <div class="metric-label">Naive Baseline</div>
          <div style="color:#e2e8f0;margin-top:10px">{art['n_r2']:.3f}</div>
          <div style="color:#e2e8f0;margin-top:10px">{art['n_mape']:.2f}%</div>
          <div style="color:#e2e8f0;margin-top:10px">${art['n_rmse']:.2f}</div>
        </div>""",unsafe_allow_html=True)
    with b4:
        st.markdown(f"""<div class="baseline-box">
          <div class="metric-label">Verdict</div>
          <div class="{vc_}" style="margin-top:10px;font-size:13px;font-weight:600">
            {vt_}</div>
        </div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    if art['wf_list']:
        st.markdown("### Walk-Forward R2 Per Fold")
        wfc=['#22c55e' if v>=0.80 else '#f59e0b' if v>=0.65 else '#ef4444'
             for v in art['wf_list']]
        fig_wf=go.Figure()
        for fn,fv,fc in zip([f"Fold {i+1}" for i in range(len(art['wf_list']))],
                             art['wf_list'],wfc):
            fig_wf.add_trace(go.Bar(x=[fn],y=[fv],marker_color=fc,
                                    text=[f"{fv:.3f}"],textposition='outside',showlegend=False))
        fig_wf.add_hline(y=0.80,line_dash='dash',line_color='#22c55e',annotation_text="Target 0.80")
        fig_wf.add_hline(y=0.65,line_dash='dot',line_color='#f59e0b',annotation_text="Acceptable 0.65")
        fig_wf.update_layout(yaxis_range=[min(min(art['wf_list'])-0.05,-0.05),1.05],height=300)
        st.plotly_chart(fig_wf,use_container_width=True)

    st.markdown("## Model Performance Charts")
    pc1,pc2=st.columns(2)
    with pc1:
        st.write("### Training Loss Curve")
        ep_r=list(range(1,len(history.get('loss',[]))+1))
        fig_l=go.Figure()
        fig_l.add_trace(go.Scatter(x=ep_r,y=history.get('loss',[]),
                                   mode='lines+markers',name='Train Loss',line=dict(color='#1f77b4')))
        if 'val_loss' in history:
            fig_l.add_trace(go.Scatter(x=ep_r,y=history['val_loss'],
                                       mode='lines+markers',name='Val Loss',line=dict(color='#ff7f0e')))
        fig_l.update_layout(title="Loss Curve",xaxis_title="Epoch",yaxis_title="MSE Loss")
        st.plotly_chart(fig_l,use_container_width=True)
    with pc2:
        st.write("### Backtest: Actual vs Predicted")
        if len(bt_preds)>0:
            bt_start=art['time_step']+train_n
            bt_idx=df_used.index[bt_start:bt_start+len(bt_preds)]
            fig_bt=go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt_idx,y=bt_actuals,name='Actual',line=dict(color='#1f77b4')))
            fig_bt.add_trace(go.Scatter(x=bt_idx,y=bt_preds,name='Predicted',line=dict(color='#ff7f0e')))
            fig_bt.update_layout(title="Backtest",xaxis_title="Date",yaxis_title="Price ($)")
            st.plotly_chart(fig_bt,use_container_width=True)

    if len(bt_preds)>0:
        dir_correct=(np.sign(bt_pred_ret)==np.sign(bt_actual_ret)).astype(int)
        bt_idx2=df_used.index[art['time_step']+train_n:art['time_step']+train_n+len(bt_preds)]
        bar_colors=['#22c55e' if c==1 else '#ef4444' for c in dir_correct]
        fig_dir=go.Figure()
        fig_dir.add_trace(go.Bar(x=bt_idx2,y=dir_correct,marker_color=bar_colors,showlegend=False))
        fig_dir.add_hline(y=0.5,line_dash='dash',line_color='#94a3b8',annotation_text="Random (50%)")
        fig_dir.update_layout(
            title=f"Directional Accuracy: {art['da']:.1f}% ({int(dir_correct.sum())}/{len(dir_correct)} correct)",
            height=280)
        st.plotly_chart(fig_dir,use_container_width=True)

    residuals=bt_actuals-bt_preds
    fig_res=go.Figure()
    fig_res.add_trace(go.Histogram(x=residuals,nbinsx=40,marker_color='#6366f1',opacity=0.75))
    fig_res.add_vline(x=0,line_color='white',line_dash='dash')
    fig_res.update_layout(
        title=f"Residuals | Mean=${residuals.mean():.2f} | Std=${residuals.std():.2f}",
        xaxis_title="Residual ($)",yaxis_title="Count",height=280)
    st.plotly_chart(fig_res,use_container_width=True)

    # ── FUTURE FORECAST ───────────────────────────────────────────────────────
    st.markdown("## Future Forecast")
    recent_scaled=scaler.transform(df_used.values[-time_step:]).tolist()
    chain_price=last_price; price_list=price_s.tolist()
    future_preds_price=[]; dummy_f=np.zeros((1,n_feat))

    for step in range(days):
        inp=np.array(recent_scaled[-time_step:]).reshape(1,time_step,-1)
        pred_sc=float(model.predict(inp,verbose=0)[0,0])
        dummy_f[0,0]=pred_sc
        pred_lr=float(scaler.inverse_transform(dummy_f)[0,0])
        pred_price=chain_price*np.exp(pred_lr)
        future_preds_price.append(pred_price)
        chain_price=pred_price; price_list.append(pred_price)
        adj_s=pd.Series(price_list)
        log_vol_=float(df_used['LogVolume'].iloc[-1])
        sma20_=adj_s.rolling(20).mean().iloc[-1]
        sma50_=adj_s.rolling(50).mean().iloc[-1]
        ema12_=adj_s.ewm(span=12,adjust=False).mean().iloc[-1]
        ema26_=adj_s.ewm(span=26,adjust=False).mean().iloc[-1]
        macd_=ema12_-ema26_
        macd_s_=pd.Series([df_used['MACD_Signal'].iloc[-1]]*8+[macd_]).ewm(span=9,adjust=False).mean().iloc[-1]
        diff_=adj_s.diff().fillna(0); up_=diff_.clip(lower=0); dn_=-diff_.clip(upper=0)
        ru_=up_.rolling(14).mean().iloc[-1] if len(up_)>=14 else up_.mean()
        rd_=dn_.rolling(14).mean().iloc[-1] if len(dn_)>=14 else dn_.mean()
        rsi_=100-100/(1+ru_/(rd_+1e-9))
        rm_=adj_s.rolling(20).mean().iloc[-1]; rs_=adj_s.rolling(20).std().iloc[-1] if len(adj_s)>=20 else adj_s.std()
        bb_w_=(2*rs_)/(rm_+1e-9); atr_=float(df_used['ATR'].iloc[-1])
        new_row=np.array([[pred_lr,log_vol_,sma20_,sma50_,ema12_,ema26_,macd_,macd_s_,rsi_,bb_w_,atr_]])
        new_sc=scaler.transform(new_row)[0].tolist()
        recent_scaled.append(new_sc)

    floor=last_price*0.55; z=1.96; uppers=[]; lowers=[]
    for i,p in enumerate(future_preds_price):
        hw=min(z*resid_std*np.sqrt(i+1),0.15*p)
        uppers.append(p+hw); lowers.append(max(p-hw,floor))

    future_dates=pd.date_range(start=data_main.index[-1]+pd.Timedelta(days=1),periods=days,freq='B')
    future_df=pd.DataFrame({'Date':future_dates,'Predicted':future_preds_price,
                             'Upper':uppers,'Lower':lowers})
    future_df['Change %']=[f"{(p-last_price)/last_price*100:+.2f}%" for p in future_preds_price]

    hist_x=price_s.index; hist_y=price_s.values
    fig_f=go.Figure()
    fig_f.add_trace(go.Scatter(x=hist_x,y=hist_y,name='Historical',line=dict(color='#1f77b4')))
    fig_f.add_trace(go.Scatter(x=[future_dates[0]],y=[future_preds_price[0]],
                               name='Forecast',line=dict(color='#ff7f0e')))
    fig_f.add_trace(go.Scatter(x=[future_dates[0],future_dates[0]],y=[lowers[0],uppers[0]],
                               fill='toself',fillcolor='rgba(255,127,14,0.15)',
                               line=dict(color='rgba(255,127,14,0)'),name='95% CI'))
    frames=[]
    for i in range(len(future_dates)):
        xp=future_dates[:i+1]; yp=future_preds_price[:i+1]
        xb=list(future_dates[:i+1])+list(future_dates[:i+1][::-1])
        yb=list(uppers[:i+1])+list(lowers[:i+1][::-1])
        frames.append(go.Frame(data=[go.Scatter(x=hist_x,y=hist_y),
                                      go.Scatter(x=xp,y=yp,line=dict(color='#ff7f0e')),
                                      go.Scatter(x=xb,y=yb,fill='toself',
                                                 fillcolor='rgba(255,127,14,0.15)',
                                                 line=dict(color='rgba(255,127,14,0)'))],name=str(i)))
    fig_f.frames=frames
    fig_f.update_layout(
        title=f"{selected_ticker} - {days}-day Forecast | Last: ${last_price:.2f}",
        xaxis_title="Date",yaxis_title="Price ($)",
        updatemenus=[{"type":"buttons","buttons":[
            {"label":"Play","method":"animate",
             "args":[None,{"frame":{"duration":400,"redraw":True},"fromcurrent":True}]},
            {"label":"Pause","method":"animate",
             "args":[[None],{"frame":{"duration":0,"redraw":False},"mode":"immediate"}]}
        ],"direction":"left","pad":{"r":10,"t":10},
         "showactive":True,"x":0.01,"y":-0.12,"xanchor":"left","yanchor":"top"}])
    st.plotly_chart(fig_f,use_container_width=True)
    st.dataframe(future_df.style.format({"Predicted":"${:.2f}","Upper":"${:.2f}","Lower":"${:.2f}"}),
                 use_container_width=True)

    sm1,sm2,sm3,sm4=st.columns(4)
    sm1.metric("Walk-Forward R2",wf_r2_str)
    sm2.metric("Directional Acc",da_str)
    sm3.metric("MAPE",mape_str)
    sm4.metric("Beats Naive","Yes" if wins>=2 else "Partial")

    # ── LLM FORECAST ANALYSIS ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 AI Forecast Analysis")
    st.caption("Powered by Llama 3.3 70B via Groq — interpreting your model results in plain English")

    if st.button("Generate AI Analysis of Forecast", type="primary", key="pred_ai"):
        final_price = future_preds_price[-1]
        pct_change  = (final_price - last_price) / last_price * 100
        direction   = "upward" if pct_change > 0 else "downward"
        with st.spinner("Analysing with Llama 3.3 70B..."):
            prompt = f"""You are a professional stock analyst. Analyse this LSTM model forecast and provide a concise, insightful interpretation.

Stock: {selected_ticker}
Current Price: ${last_price:.2f}
Forecast Horizon: {days} trading days
Predicted Price in {days} days: ${final_price:.2f} ({pct_change:+.2f}%)
Direction: {direction}

Model Quality Metrics:
- Directional Accuracy: {art['da']:.1f}% (how often model correctly calls UP/DOWN)
- MAPE: {art['mape']:.2f}% (average price prediction error)
- Walk-Forward R2: {art['wf_r2']:.3f} (model explanatory power, 1.0 = perfect)
- RMSE: ${art['rmse']:.2f} (average dollar error per day)
- Beats Naive Baseline: {wins}/3 metrics

Forecast range: Low ${lowers[-1]:.2f} - High ${uppers[-1]:.2f}

Provide exactly 3 sections:
1. FORECAST INTERPRETATION (2-3 sentences): What does this prediction mean for {selected_ticker}?
2. MODEL RELIABILITY (2 sentences): How much should a trader trust this forecast based on the metrics?
3. KEY RISKS (2-3 bullet points): What could invalidate this forecast?

Be direct, quantitative, and professional. No generic disclaimers."""

            response = call_groq(prompt, max_tokens=500)

        if response.startswith("__ERROR__"):
            st.error(f"AI analysis failed: {response}")
        else:
            sections = response.split('\n\n')
            for section in sections:
                if section.strip():
                    lines = section.strip().split('\n', 1)
                    title = lines[0].strip()
                    body  = lines[1].strip() if len(lines) > 1 else lines[0].strip()
                    st.markdown(f"""<div class="ai-box">
                      <div class="ai-box-title">{title}</div>
                      {body}
                    </div>""", unsafe_allow_html=True)


# ==============================================================================
#  TAB 3 — SENTIMENT
# ==============================================================================
elif tab == "Sentiment":
    st.subheader("News Sentiment")
    if news_posts:
        df_s=pd.DataFrame({'News':news_posts,'Link':news_links,'Score':vader_scores})
        def color(val):
            return f"color: {'green' if val>0.1 else 'red' if val<-0.1 else 'gray'}"
        st.dataframe(df_s.style.map(color,subset=['Score']).format({'Score':'{:.3f}'}),
                     use_container_width=True)
        pos=sum(1 for s in vader_scores if s>0.1)
        neg=sum(1 for s in vader_scores if s<-0.1)
        neu=len(vader_scores)-pos-neg
        c1,c2,c3=st.columns(3)
        c1.metric("Positive",pos); c2.metric("Negative",neg); c3.metric("Neutral",neu)
    else:
        st.info("No news available.")

# ==============================================================================
#  TAB 4 — COMPARISON  (+ AI-powered stock recommendation)
# ==============================================================================
elif tab == "Comparison":
    st.subheader(f"**{selected_ticker} vs {compare_ticker}**")
    if data_main is None or data_compare is None:
        st.error("Not enough data to compare."); st.stop()

    bm=data_main['Adj Close'].iloc[0]; bc=data_compare['Adj Close'].iloc[0]
    dm=(data_main['Adj Close']/bm-1)*100
    dc=(data_compare['Adj Close']/bc-1)*100
    rm=float(dm.iloc[-1]); rc=float(dc.iloc[-1])
    vm=float(data_main['Adj Close'].pct_change().std()*np.sqrt(252)*100)
    vc=float(data_compare['Adj Close'].pct_change().std()*np.sqrt(252)*100)

    # Extra metrics for recommendation
    ra_m = rm/vm if vm>0 else 0
    ra_c = rc/vc if vc>0 else 0
    dd_m = compute_drawdown(data_main['Adj Close'])
    dd_c = compute_drawdown(data_compare['Adj Close'])
    max_dd_m = float(dd_m.min())
    max_dd_c = float(dd_c.min())

    # Normalised chart
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data_main.index,y=dm,name=selected_ticker,
                             line=dict(color='#26A69A')))
    fig.add_trace(go.Scatter(x=data_compare.index,y=dc,name=compare_ticker,
                             line=dict(color='#AB47BC')))
    fig.add_hline(y=0,line_dash='dot',line_color='#475569')
    fig.update_layout(title="Normalised Performance (%)",height=400)
    st.plotly_chart(fig,use_container_width=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric(f"{selected_ticker} Return",f"{rm:+.2f}%")
    c2.metric(f"{compare_ticker} Return", f"{rc:+.2f}%")
    c3.metric(f"{selected_ticker} Vol",   f"{vm:.1f}%")
    c4.metric(f"{compare_ticker} Vol",    f"{vc:.1f}%")

    # Rolling 12M
    st.markdown("#### Rolling 12-Month Return")
    roll_m=data_main['Adj Close'].pct_change(252)*100
    roll_c=data_compare['Adj Close'].pct_change(252)*100
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=data_main.index,y=roll_m,name=selected_ticker,
                              line=dict(color='#26A69A'),fill='tozeroy',
                              fillcolor='rgba(38,166,154,0.08)'))
    fig2.add_trace(go.Scatter(x=data_compare.index,y=roll_c,name=compare_ticker,
                              line=dict(color='#AB47BC'),fill='tozeroy',
                              fillcolor='rgba(171,71,188,0.08)'))
    fig2.add_hline(y=0,line_color='#475569',line_dash='dash')
    fig2.update_layout(height=300)
    st.plotly_chart(fig2,use_container_width=True)

    # Drawdown side by side with Risk-Return
    cc1,cc2=st.columns(2)
    with cc1:
        st.markdown("**Drawdown from Peak**")
        fig3=go.Figure()
        fig3.add_trace(go.Scatter(x=data_main.index,y=dd_m,name=selected_ticker,
                                  line=dict(color='#26A69A'),fill='tozeroy',
                                  fillcolor='rgba(38,166,154,0.12)'))
        fig3.add_trace(go.Scatter(x=data_compare.index,y=dd_c,name=compare_ticker,
                                  line=dict(color='#AB47BC'),fill='tozeroy',
                                  fillcolor='rgba(171,71,188,0.12)'))
        fig3.update_layout(height=280,yaxis_title="Drawdown (%)")
        st.plotly_chart(fig3,use_container_width=True)
    with cc2:
        st.markdown("**Risk vs Return**")
        fig4=go.Figure()
        fig4.add_trace(go.Scatter(
            x=[vm,vc],y=[rm,rc],mode='markers+text',
            text=[selected_ticker,compare_ticker],textposition='top center',
            marker=dict(size=22,color=['#26A69A','#AB47BC'],symbol='diamond')))
        fig4.update_layout(height=280,
                           xaxis_title="Volatility (Risk) %",
                           yaxis_title="Total Return %")
        st.plotly_chart(fig4,use_container_width=True)

    # Summary table
    st.markdown("#### Summary Table")
    summary_df=pd.DataFrame({
        'Metric':       ['Total Return','Ann. Volatility','Max Drawdown','Risk-Adj Return'],
        selected_ticker:[f"{rm:+.2f}%",f"{vm:.1f}%",f"{max_dd_m:.1f}%",f"{ra_m:.2f}x"],
        compare_ticker: [f"{rc:+.2f}%",f"{vc:.1f}%",f"{max_dd_c:.1f}%",f"{ra_c:.2f}x"],
        'Winner':       [selected_ticker if rm>rc else compare_ticker,
                         selected_ticker if vm<vc else compare_ticker,
                         selected_ticker if max_dd_m>max_dd_c else compare_ticker,
                         selected_ticker if ra_m>ra_c else compare_ticker]
    })
    st.dataframe(summary_df,use_container_width=True,hide_index=True)

    # ── AI STOCK RECOMMENDATION ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 AI Stock Recommendation")
    st.caption("Llama 3.3 70B analyses all metrics and tells you which stock to pick and why")

    if st.button("Get AI Recommendation", type="primary", key="comp_ai"):
        with st.spinner("Analysing with Llama 3.3 70B..."):
            prompt = f"""You are a senior equity analyst with 20 years of experience.
Compare these two stocks and give a clear investment recommendation.

Stock A: {selected_ticker}
- Total Return (period): {rm:+.2f}%
- Annualised Volatility: {vm:.1f}%
- Max Drawdown: {max_dd_m:.1f}%
- Risk-Adjusted Return: {ra_m:.2f}x

Stock B: {compare_ticker}
- Total Return (period): {rc:+.2f}%
- Annualised Volatility: {vc:.1f}%
- Max Drawdown: {max_dd_c:.1f}%
- Risk-Adjusted Return: {ra_c:.2f}x

Period analysed: {start_date} to {end_date}

Respond in exactly this format:

RECOMMENDATION: [Stock ticker] is the stronger pick

REASON (3 bullet points with specific numbers from the data above):
- [Point 1]
- [Point 2]
- [Point 3]

WHO SHOULD CONSIDER {selected_ticker}: [1 sentence describing investor profile]
WHO SHOULD CONSIDER {compare_ticker}: [1 sentence describing investor profile]

RISK WARNING: [1 sentence about what could change this recommendation]

Be specific, cite the numbers, and be decisive. Do not hedge or say "it depends on your goals" as the opening."""

            response = call_groq(prompt, max_tokens=500)

        if response.startswith("__ERROR__"):
            st.error(f"AI recommendation failed: {response}")
        else:
            # Determine box style from response
            rec_upper = response.upper()
            if selected_ticker.upper() in rec_upper[:80]:
                box_cls = "rec-box-buy"
                rec_icon = "✅"
            elif compare_ticker.upper() in rec_upper[:80]:
                box_cls = "rec-box-hold"
                rec_icon = "✅"
            else:
                box_cls = "rec-box-hold"
                rec_icon = "⚖️"

            st.markdown(f"""<div class="{box_cls}">
              <div style="font-size:11px;color:#a3a3a3;text-transform:uppercase;
                          letter-spacing:.08em;margin-bottom:10px">
                AI RECOMMENDATION — {GROQ_MODEL}</div>
              <div style="font-size:14px;color:#e2e8f0;line-height:1.8;
                          white-space:pre-wrap">{rec_icon} {response}</div>
            </div>""", unsafe_allow_html=True)


# ==============================================================================
#  TAB 5 — PORTFOLIO ANALYZER  (clean, impressive, LLM insights)
# ==============================================================================
elif tab == "Portfolio Analyzer":
    st.subheader("Portfolio Analyzer — Professional Suite")

    port_tickers = st.multiselect("Select Portfolio Stocks", tickers,
                                   default=[selected_ticker, compare_ticker])
    if len(port_tickers) < 2:
        st.warning("Select at least 2 tickers."); st.stop()

    weights, total_w = [], 0.0
    wcols = st.columns(len(port_tickers))
    for i, tick in enumerate(port_tickers):
        w = wcols[i].number_input(f"Weight {tick} (%)", 0.0, 100.0,
                                   round(100.0 / len(port_tickers), 2),
                                   key=f"w_{tick}")
        weights.append(w / 100); total_w += w

    bar_col = '#22c55e' if abs(total_w - 100) < 0.1 else '#ef4444'
    st.markdown(
        f'<div style="background:#1e293b;border-radius:6px;height:10px;margin-bottom:4px">'
        f'<div style="width:{min(total_w,100):.1f}%;height:10px;border-radius:6px;'
        f'background:{bar_col}"></div></div>'
        f'<div style="font-size:12px;color:{bar_col};margin-bottom:14px">'
        f'Total: {total_w:.1f}%</div>', unsafe_allow_html=True)

    if abs(total_w - 100) > 0.1:
        st.warning(f"Weights must sum to 100%. Currently {total_w:.1f}%"); st.stop()

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
    sharpe   = (p_ret - 0.03) / p_vol if p_vol > 0 else 0.0

    port_daily = (rets * w_np).sum(axis=1)
    downside   = port_daily[port_daily < 0].std() * np.sqrt(252)
    sortino    = (p_ret - 0.03) / downside if downside > 0 else 0.0

    port_idx   = (1 + port_daily).cumprod()
    roll_max_p = port_idx.cummax()
    dd_port    = (port_idx - roll_max_p) / roll_max_p * 100
    max_dd_p   = float(dd_port.min())
    port_cum   = port_idx * 100 - 100
    var_95     = float(np.percentile(port_daily, 5) * 100)
    cvar_95    = float(
        port_daily[port_daily <= np.percentile(port_daily, 5)].mean() * 100)

    # Beta vs SPY
    beta  = 1.0
    spy_d = fetch_stock_data('SPY', start_date, end_date)
    if spy_d is not None:
        spy_ret_s  = spy_d['Adj Close'].pct_change().dropna()
        pa = port_daily.reindex(spy_ret_s.index).dropna()
        sa = spy_ret_s.reindex(pa.index).dropna()
        pa = pa.reindex(sa.index).dropna()
        if len(pa) > 30:
            beta = float(np.cov(pa, sa)[0, 1] / np.var(sa))

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown("#### Performance Analytics")
    pk1,pk2,pk3,pk4,pk5,pk6 = st.columns(6)
    for col, lbl, val, color in [
        (pk1, "Annual Return",    f"{p_ret*100:.2f}%",
         "#22c55e" if p_ret > 0.08 else "#f59e0b"),
        (pk2, "Ann. Volatility",  f"{p_vol*100:.2f}%", "#94a3b8"),
        (pk3, "Sharpe Ratio",     f"{sharpe:.2f}",
         "#22c55e" if sharpe > 1 else "#f59e0b" if sharpe > 0.5 else "#ef4444"),
        (pk4, "Sortino Ratio",    f"{sortino:.2f}",
         "#22c55e" if sortino > 1.5 else "#f59e0b" if sortino > 0.8 else "#ef4444"),
        (pk5, "Max Drawdown",     f"{max_dd_p:.1f}%",
         "#ef4444" if max_dd_p < -30 else "#f59e0b"),
        (pk6, "VaR 95% Daily",    f"{var_95:.2f}%", "#ef4444"),
    ]:
        col.markdown(f"""<div class="metric-card">
          <div class="metric-label">{lbl}</div>
          <div class="metric-value" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 1: Portfolio vs SPY ──────────────────────────────────────────────
    st.markdown("#### Portfolio vs S&P 500 Benchmark")
    st.caption("SPY is the standard benchmark. Beating it consistently is what professional fund managers aim for.")
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values,
                               name='Your Portfolio', line=dict(color='#6366f1', width=2)))
    if spy_d is not None:
        spy_cum = (1 + spy_d['Adj Close'].pct_change().dropna()).cumprod() * 100 - 100
        spy_cum = spy_cum.reindex(port_cum.index, method='ffill').dropna()
        fig_b.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum.values,
                                   name='S&P 500 (SPY)',
                                   line=dict(color='#94a3b8', width=1.5, dash='dash')))
        pf = float(port_cum.iloc[-1]); sf = float(spy_cum.iloc[-1])
        beat = pf > sf
        st.markdown(
            f"<div class='ins-card'>"
            f"{'✅ Portfolio outperformed' if beat else '📉 Portfolio underperformed'}"
            f" S&P 500 by <b>{abs(pf - sf):.1f}pp</b> over the period.</div>",
            unsafe_allow_html=True)
    fig_b.add_hline(y=0, line_dash='dot', line_color='#475569')
    fig_b.update_layout(height=360, yaxis_title="Cumulative Return (%)")
    st.plotly_chart(fig_b, use_container_width=True)

    # ── Chart 2: Efficient Frontier ────────────────────────────────────────────
    st.markdown("#### Efficient Frontier")
    st.caption("3,000 random weight combinations. Brighter colour = better Sharpe Ratio. Red star = your portfolio. Gold star = optimal mix.")

    n_sim = 3000
    s_rets, s_vols, s_sharpes, s_wts = [], [], [], []
    for _ in range(n_sim):
        w_  = np.random.dirichlet(np.ones(n_assets))
        r_  = float(np.dot(m_ret, w_))
        v_  = float(np.sqrt(np.dot(w_.T, np.dot(cov_mat, w_))))
        s_  = (r_ - 0.03) / v_ if v_ > 0 else 0
        s_rets.append(r_ * 100); s_vols.append(v_ * 100)
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
        x=[p_vol * 100], y=[p_ret * 100], mode='markers+text',
        marker=dict(size=18, color='red', symbol='star'),
        text=['Your Portfolio'], textposition='top center', name='Yours'))
    best_idx = int(np.argmax(s_sharpes))
    fig_ef.add_trace(go.Scatter(
        x=[s_vols[best_idx]], y=[s_rets[best_idx]], mode='markers+text',
        marker=dict(size=18, color='#fbbf24', symbol='star'),
        text=['Optimal'], textposition='top center', name='Max Sharpe'))
    fig_ef.update_layout(height=460,
                         xaxis_title="Risk (Volatility %)",
                         yaxis_title="Expected Return (%)")
    st.plotly_chart(fig_ef, use_container_width=True)

    best_w   = s_wts[best_idx]
    opt_hint = " | ".join([f"{port_tickers[j]}: {best_w[j]*100:.1f}%"
                           for j in range(n_assets)])
    st.markdown(
        f"<div class='ins-card'>Optimal weights (Sharpe {s_sharpes[best_idx]:.2f}): "
        f"<b>{opt_hint}</b>. Your current Sharpe: <b>{sharpe:.2f}</b>.</div>",
        unsafe_allow_html=True)

    # ── Chart 3: Monte Carlo ───────────────────────────────────────────────────
    st.markdown("#### Monte Carlo Simulation — 1-Year Outlook")
    st.caption("500 simulated future portfolio paths. Starting value = $100. Shows the realistic range of outcomes over the next 252 trading days.")

    n_mc       = 500
    n_days_mc  = 252
    start_val  = 100.0
    daily_mean = float(port_daily.mean())
    daily_std  = float(port_daily.std())

    mc_paths = np.zeros((n_mc, n_days_mc))
    for i in range(n_mc):
        sim_r        = np.random.normal(daily_mean, daily_std, n_days_mc)
        mc_paths[i]  = start_val * (1 + sim_r).cumprod()

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
    for path, name, color, width in [
        (np.percentile(mc_paths, 50, axis=0), 'Median',   '#6366f1', 2.5),
        (np.percentile(mc_paths, 95, axis=0), 'Best 5%',  '#22c55e', 2),
        (np.percentile(mc_paths, 5,  axis=0), 'Worst 5%', '#ef4444', 2),
    ]:
        fig_mc.add_trace(go.Scatter(
            x=list(range(n_days_mc)), y=path,
            name=name, line=dict(color=color, width=width)))
    fig_mc.update_layout(height=380,
                         xaxis_title="Trading Days",
                         yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig_mc, use_container_width=True)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Median Outcome (1Y)", f"${p50_:.2f}",
               f"{(p50_ - start_val) / start_val * 100:+.1f}%")
    mc2.metric("Best Case (95th %)",  f"${p95_:.2f}",
               f"{(p95_ - start_val) / start_val * 100:+.1f}%")
    mc3.metric("Worst Case (5th %)",  f"${p5_:.2f}",
               f"{(p5_ - start_val) / start_val * 100:+.1f}%")

    # ── Correlation Heatmap ────────────────────────────────────────────────────
    st.markdown("#### Correlation Matrix")
    st.caption("+1 = always move together (no diversification). 0 = independent. -1 = opposite (perfect hedge).")
    fig_h = px.imshow(rets.corr(), text_auto=True, aspect='auto',
                      color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig_h.update_layout(height=300)
    st.plotly_chart(fig_h, use_container_width=True)

    # ── Individual analytics table ─────────────────────────────────────────────
    st.markdown("#### Individual Stock Analytics")
    contrib_ret  = [float(m_ret[t] * w_np[i] * 100) for i, t in enumerate(port_tickers)]
    contrib_risk = [float(np.dot(cov_mat.iloc[i].values, w_np) / p_vol * w_np[i] * 100)
                    for i in range(n_assets)]
    indiv_df = pd.DataFrame({
        'Ticker':          port_tickers,
        'Weight':          [f"{w*100:.1f}%" for w in weights],
        'Ann. Return':     [f"{float(m_ret[t])*100:.2f}%" for t in port_tickers],
        'Ann. Volatility': [f"{float(np.sqrt(cov_mat.loc[t, t]))*100:.2f}%" for t in port_tickers],
        'Return Contrib.': [f"{r:.2f}%" for r in contrib_ret],
        'Risk Contrib.':   [f"{r:.2f}%" for r in contrib_risk],
    })
    st.dataframe(indiv_df, use_container_width=True, hide_index=True)

    # ── LLM AI PORTFOLIO INSIGHTS ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 AI Portfolio Insights")
    st.caption("Powered by Llama 3.3 70B via Groq — professional-grade portfolio analysis in seconds")

    if st.button("Generate AI Portfolio Insights", type="primary", key="port_ai"):
        corr_matrix = rets.corr()
        max_pair = ('', '', 0.0); min_pair = ('', '', 1.0)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                v = float(corr_matrix.iloc[i, j])
                if v > max_pair[2]: max_pair = (port_tickers[i], port_tickers[j], v)
                if v < min_pair[2]: min_pair = (port_tickers[i], port_tickers[j], v)

        with st.spinner("Analysing portfolio with Llama 3.3 70B..."):
            prompt = f"""You are a senior portfolio manager at a top-tier asset management firm.
Provide a professional analysis of this portfolio with 4 specific, data-driven insights.

Portfolio Composition:
{chr(10).join([f"- {port_tickers[i]}: {weights[i]*100:.1f}% weight" for i in range(n_assets)])}

Performance Metrics:
- Expected Annual Return: {p_ret*100:.2f}%
- Annual Volatility: {p_vol*100:.2f}%
- Sharpe Ratio: {sharpe:.2f} (above 1.0 is considered good)
- Sortino Ratio: {sortino:.2f}
- Beta vs S&P 500: {beta:.2f}
- Max Drawdown: {max_dd_p:.1f}%
- Daily VaR (95%): {var_95:.2f}%
- CVaR (95%): {cvar_95:.2f}%

Diversification:
- Most correlated pair: {max_pair[0]} & {max_pair[1]} (r={max_pair[2]:.2f})
- Least correlated pair: {min_pair[0]} & {min_pair[1]} (r={min_pair[2]:.2f})

Optimisation:
- Current Sharpe: {sharpe:.2f}
- Maximum achievable Sharpe: {s_sharpes[best_idx]:.2f}
- Optimal weights: {opt_hint}

Monte Carlo (1-year, $100 starting value):
- Median outcome: ${p50_:.2f}
- Best case (95th percentile): ${p95_:.2f}
- Worst case (5th percentile): ${p5_:.2f}

Provide exactly 4 insights in this format:

INSIGHT 1: [Title]
[2-3 sentences with specific numbers and one clear action]

INSIGHT 2: [Title]
[2-3 sentences with specific numbers and one clear action]

INSIGHT 3: [Title]
[2-3 sentences with specific numbers and one clear action]

INSIGHT 4: [Title]
[2-3 sentences with specific numbers and one clear action]

Be direct, use the numbers, focus on what the investor should actually do."""

            response = call_groq(prompt, max_tokens=700)

        if response.startswith("__ERROR__"):
            st.error(f"AI insights failed: {response}")
        else:
            raw_insights = [p.strip() for p in response.split('\n\n') if p.strip()]
            for insight in raw_insights:
                lines = insight.split('\n', 1)
                title = lines[0].strip().lstrip('INSIGHT 1234: ').strip()
                body  = lines[1].strip() if len(lines) > 1 else insight
                st.markdown(f"""<div class="ai-box">
                  <div class="ai-box-title">{title}</div>
                  {body}
                </div>""", unsafe_allow_html=True)

# ==============================================================================
#  NEWS TICKER — bottom of every page
# ==============================================================================
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
