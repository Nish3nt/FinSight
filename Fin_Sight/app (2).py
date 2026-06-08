# =============================================================================
#  FinSight — app.py  (Single File — Complete)
#  LSTM Stock Predictor + Groq LLM Analysis + Portfolio Suite
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
import time
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# TensorFlow import with full fallback chain
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
except Exception:
    try:
        from tf_keras.models import Sequential
        from tf_keras.layers import LSTM, Dense, Dropout
        from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tf_keras.optimizers import Adam
    except Exception:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras.optimizers import Adam

# =============================================================================
#  CONSTANTS
# =============================================================================
GROQ_API_KEY = "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"
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
FINNHUB_KEY  = "d6qgus9r01qhcrmk4od0d6qgus9r01qhcrmk4odg"

nltk.download('vader_lexicon', quiet=True)
sia          = SentimentIntensityAnalyzer()
current_date = datetime.now().date()

# =============================================================================
#  PAGE CONFIG
# =============================================================================
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
.bad{color:#ef4444!important}
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
.rec-box{background:#052e16;border:1px solid #166534;
         border-radius:12px;padding:18px 20px;margin-top:10px}
.ins-card{background:#0f172a;border-left:3px solid #6366f1;
          border-radius:0 8px 8px 0;padding:12px 16px;font-size:13px;
          color:#cbd5e1;margin-bottom:8px;line-height:1.6}
</style>
""", unsafe_allow_html=True)

st.title("**FinSight**: Real-Time Stock Intelligence")

# =============================================================================
#  TICKER LIST
# =============================================================================
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

# =============================================================================
#  SIDEBAR
# =============================================================================
st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Main Stock", tickers,
                                        index=tickers.index('AAPL'))
compare_ticker  = st.sidebar.selectbox("Compare With", tickers,
                                        index=tickers.index('MSFT'))
start_date = st.sidebar.date_input("Start Date",
                                    pd.to_datetime('2010-01-01').date())
end_date   = st.sidebar.date_input("End Date", current_date)
if start_date > end_date:
    st.error("Start date must be before end date."); st.stop()
if end_date > current_date:
    end_date = current_date

# =============================================================================
#  HELPER FUNCTIONS
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
        return f"__ERROR__: {r.status_code}"
    except Exception as e:
        return f"__ERROR__: {e}"

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
        return ["No news."], [], []
    except:
        return ["Error fetching news."], [], []

@st.cache_data(ttl=300)
def compute_vader(posts):
    return [sia.polarity_scores(p)['compound'] for p in posts]

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

def compute_features(raw):
    df = raw.copy()
    df['LogReturn']   = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['SMA20']       = df['Adj Close'].rolling(20).mean()
    df['SMA50']       = df['Adj Close'].rolling(50).mean()
    df['EMA12']       = df['Adj Close'].ewm(span=12, adjust=False).mean()
    df['EMA26']       = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['Adj Close'].diff()
    up = delta.clip(lower=0); dn = -delta.clip(upper=0)
    df['RSI']      = 100 - 100/(1+up.rolling(14).mean()/(dn.rolling(14).mean()+1e-9))
    rm = df['Adj Close'].rolling(20).mean()
    rs = df['Adj Close'].rolling(20).std()
    df['BB_Width'] = (2*rs)/(rm+1e-9)
    hl = df['High']-df['Low']
    hc = (df['High']-df['Adj Close'].shift()).abs()
    lc = (df['Low'] -df['Adj Close'].shift()).abs()
    df['ATR']      = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    df['LogVolume']= np.log1p(df['Volume'])
    cols = ['LogReturn','LogVolume','SMA20','SMA50','EMA12','EMA26',
            'MACD','MACD_Signal','RSI','BB_Width','ATR']
    return df[cols].dropna(), df['Adj Close']

def compute_drawdown(series):
    return (series - series.cummax()) / series.cummax() * 100

@st.cache_resource(ttl=24*3600)
def train_model(ticker, start_str, end_str,
                time_step, epochs, batch_size, retrain_flag):
    t0 = time.time(); training_time = datetime.now()
    raw = yf.download(ticker, start=start_str, end=end_str, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
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
    bt_pp,bt_ap,bt_pr,bt_ar=[],[],[],[]
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
    wf_r2=float(np.mean(wf)) if wf else 0.0
    mse_v=float(mean_squared_error(bt_ap,bt_pp))
    r2_v=float(r2_score(bt_ap,bt_pp))
    rmse_v=float(np.sqrt(mse_v))
    mape_v=float(np.mean(np.abs((bt_ap-bt_pp)/(np.abs(bt_ap)+1e-9)))*100)
    da_v=float(np.mean(np.sign(bt_pr)==np.sign(bt_ar))*100)
    np_=bt_ap[:-1]; na_=bt_ap[1:]
    n_r2=float(r2_score(na_,np_))
    n_mape=float(np.mean(np.abs((na_-np_)/(np.abs(na_)+1e-9)))*100)
    n_rmse=float(np.sqrt(mean_squared_error(na_,np_)))
    rs_v=float(np.std(bt_ap-bt_pp))
    return dict(
        model=model, scaler=scaler, df_feat=df_feat, price_series=price_s,
        time_step=time_step, train_n=train_n, n_feat=n_feat,
        bt_pp=bt_pp, bt_ap=bt_ap, bt_pr=bt_pr, bt_ar=bt_ar,
        history=history.history, training_time=training_time,
        training_secs=time.time()-t0, epochs=epochs, batch_size=batch_size,
        mse=mse_v, r2=r2_v, rmse=rmse_v, mape=mape_v, da=da_v,
        wf_r2=wf_r2, wf_list=wf,
        n_r2=n_r2, n_mape=n_mape, n_rmse=n_rmse, resid_std=rs_v
    )

# =============================================================================
#  LOAD DATA
# =============================================================================
news_headlines, news_posts, news_links = get_news(selected_ticker)
vader_scores = compute_vader(news_posts)
data_main    = fetch_stock_data(selected_ticker, start_date, end_date)
data_compare = fetch_stock_data(compare_ticker,  start_date, end_date)

# =============================================================================
#  TABS
# =============================================================================
tab = option_menu(None,
    ["Data & Viz","Predictions","Sentiment","Comparison","Portfolio Analyzer"],
    icons=["table","graph-up","chat-dots","arrow-left-right","pie-chart"],
    orientation="horizontal")


# ==============================================================================
#  TAB 1 — DATA & VIZ
# ==============================================================================
if tab == "Data & Viz":
    st.subheader(f"**{selected_ticker}** - Price History")
    if data_main is not None:
        st.dataframe(data_main.tail(100), use_container_width=True)
        st.download_button("Download CSV", data_main.to_csv().encode(),
                           f"{selected_ticker}.csv")
        fig = px.line(data_main, x=data_main.index, y='Adj Close',
                      title=f"{selected_ticker} - Adjusted Close Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data available. Try expanding date range.")

# ==============================================================================
#  TAB 2 — PREDICTIONS
# ==============================================================================
elif tab == "Predictions":
    st.subheader("Price Forecast - Multi-feature LSTM | Industry-Grade Evaluation")
    if data_main is None:
        st.error("Not enough data. Expand date range."); st.stop()

    c1, c2 = st.columns([2,1])
    with c1:
        days       = st.slider("Forecast horizon (trading days)", 1, 30, 7)
        time_step  = st.slider("Lookback window (days)", 60, 180, 90, step=10)
        epochs     = st.slider("Training epochs", 20, 150, 80, step=5)
        batch_size = st.selectbox("Batch size", [16,32,64], index=1)
        retrain    = st.checkbox("Force retrain model", value=False)
    with c2:
        st.markdown("""<div class="model-box">
        <b>11-Feature LSTM</b><br><br>
        <b>Trend:</b> SMA20, SMA50, EMA12, EMA26<br>
        <b>Momentum:</b> MACD, Signal, RSI<br>
        <b>Volatility:</b> BB Width, ATR<br>
        <b>Other:</b> Log Return, Log Volume<br><br>
        Walk-Forward R2 | Directional Accuracy<br>
        MAPE | RMSE | vs Naive Baseline<br><br>
        Target R2 > 0.82 | DA > 55%
        </div>""", unsafe_allow_html=True)

    df_features, price_series = compute_features(data_main)
    if len(df_features) < time_step + 30:
        st.error(f"Need >= {time_step+30} rows. Got {len(df_features)}."); st.stop()

    ph = st.empty()
    with ph.container():
        st.markdown('<div class="skel-card"></div>', unsafe_allow_html=True)
        st.markdown('<div class="skel-card"></div>', unsafe_allow_html=True)
    try:
        art = train_model(selected_ticker, str(start_date), str(end_date),
                          time_step, epochs, batch_size, retrain)
    finally:
        ph.empty()

    model=art['model']; scaler=art['scaler']; df_used=art['df_feat']
    price_s=art['price_series']; train_n=art['train_n']; n_feat=art['n_feat']
    bt_preds=art['bt_pp']; bt_actuals=art['bt_ap']
    bt_pred_ret=art['bt_pr']; bt_actual_ret=art['bt_ar']
    history=art['history']; training_time=art['training_time']
    resid_std=art['resid_std']; last_price=float(price_s.iloc[-1])
    beat_r2=art['r2']>art['n_r2']; beat_mape=art['mape']<art['n_mape']
    beat_rmse=art['rmse']<art['n_rmse']; wins=sum([beat_r2,beat_mape,beat_rmse])

    def ccls(v,g,w,hi=True):
        if hi: return "good" if v>=g else "warn" if v>=w else "bad"
        return "good" if v<=g else "warn" if v<=w else "bad"

    wf_r2_str=f"{art['wf_r2']:.3f}"; r2_str=f"{art['r2']:.3f}"
    da_str=f"{art['da']:.1f}%"; mape_str=f"{art['mape']:.2f}%"
    rmse_str=f"${art['rmse']:.2f}"
    age=datetime.now()-training_time
    age_str=f"{age.days}d {age.seconds//3600}h {(age.seconds%3600)//60}m"

    st.markdown(f"""<div class="info-bar">
    <b>Model</b>: 2-layer LSTM | <b>Features</b>: {n_feat} |
    <b>Lookback</b>: {art['time_step']}d | <b>Epochs</b>: {art['epochs']} |
    <b>Batch</b>: {art['batch_size']} |
    <b>Trained</b>: {training_time.strftime('%Y-%m-%d %H:%M')} |
    <b>Age</b>: {age_str} | <b>Cached</b>: {'Yes' if not retrain else 'No'}
    </div>""", unsafe_allow_html=True)

    st.markdown("## Model Evaluation Dashboard")
    st.caption("Metrics measured on the 20% test set the model never saw during training.")

    wf_cls=ccls(art['wf_r2'],0.80,0.65); r2_cls=ccls(art['r2'],0.80,0.65)
    da_cls=ccls(art['da'],58,52); mp_cls=ccls(art['mape'],3,5,hi=False)
    m1,m2,m3,m4,m5=st.columns(5)
    for col,lbl,val,cls,sub in [
        (m1,"Walk-Forward R2",wf_r2_str,wf_cls,"5-fold rolling windows"),
        (m2,"Standard R2",r2_str,r2_cls,"Test variance explained"),
        (m3,"Directional Accuracy",da_str,da_cls,"Random = 50%"),
        (m4,"MAPE",mape_str,mp_cls,"Mean Abs % Error"),
        (m5,"RMSE",rmse_str,"","Avg $ error/day"),
    ]:
        col.markdown(f"""<div class="metric-card">
          <div class="metric-label">{lbl}</div>
          <div class="metric-value {cls}">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("### vs Naive Baseline")
    vc_="good" if wins==3 else "warn" if wins>=2 else "bad"
    vt_=("Beats all 3 metrics" if wins==3
         else f"Beats {wins}/3 metrics" if wins>=2 else "Does not beat baseline")
    b1,b2,b3,b4=st.columns(4)
    with b1:
        st.markdown("""<div class="baseline-box">
          <div class="metric-label">Metric</div>
          <div style="color:#94a3b8;margin-top:10px">R2</div>
          <div style="color:#94a3b8;margin-top:10px">MAPE</div>
          <div style="color:#94a3b8;margin-top:10px">RMSE</div>
        </div>""", unsafe_allow_html=True)
    with b2:
        r2c_="good" if beat_r2 else "bad"
        mpc_="good" if beat_mape else "bad"
        rmc_="good" if beat_rmse else "bad"
        st.markdown(f"""<div class="baseline-box">
          <div class="metric-label">Our LSTM</div>
          <div class="{r2c_}" style="margin-top:10px">{art['r2']:.3f}</div>
          <div class="{mpc_}" style="margin-top:10px">{art['mape']:.2f}%</div>
          <div class="{rmc_}" style="margin-top:10px">${art['rmse']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with b3:
        st.markdown(f"""<div class="baseline-box">
          <div class="metric-label">Naive Baseline</div>
          <div style="color:#e2e8f0;margin-top:10px">{art['n_r2']:.3f}</div>
          <div style="color:#e2e8f0;margin-top:10px">{art['n_mape']:.2f}%</div>
          <div style="color:#e2e8f0;margin-top:10px">${art['n_rmse']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with b4:
        st.markdown(f"""<div class="baseline-box">
          <div class="metric-label">Verdict</div>
          <div class="{vc_}" style="margin-top:10px;font-size:13px;font-weight:600">
            {vt_}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    if art['wf_list']:
        st.markdown("### Walk-Forward R2 Per Fold")
        wfc=['#22c55e' if v>=0.80 else '#f59e0b' if v>=0.65 else '#ef4444'
             for v in art['wf_list']]
        fig_wf=go.Figure()
        for fn,fv,fc in zip([f"Fold {i+1}" for i in range(len(art['wf_list']))],
                             art['wf_list'],wfc):
            fig_wf.add_trace(go.Bar(x=[fn],y=[fv],marker_color=fc,
                                    text=[f"{fv:.3f}"],textposition='outside',
                                    showlegend=False))
        fig_wf.add_hline(y=0.80,line_dash='dash',line_color='#22c55e',
                         annotation_text="Target 0.80")
        fig_wf.add_hline(y=0.65,line_dash='dot',line_color='#f59e0b',
                         annotation_text="Acceptable 0.65")
        fig_wf.update_layout(
            yaxis_range=[min(min(art['wf_list'])-0.05,-0.05),1.05],height=300)
        st.plotly_chart(fig_wf,use_container_width=True)

    st.markdown("## Model Performance Charts")
    pc1,pc2=st.columns(2)
    with pc1:
        st.write("### Training Loss Curve")
        ep_r=list(range(1,len(history.get('loss',[]))+1))
        fig_l=go.Figure()
        fig_l.add_trace(go.Scatter(x=ep_r,y=history.get('loss',[]),
                                   mode='lines+markers',name='Train',
                                   line=dict(color='#1f77b4')))
        if 'val_loss' in history:
            fig_l.add_trace(go.Scatter(x=ep_r,y=history['val_loss'],
                                       mode='lines+markers',name='Val',
                                       line=dict(color='#ff7f0e')))
        fig_l.update_layout(title="Loss Curve",
                            xaxis_title="Epoch",yaxis_title="MSE Loss")
        st.plotly_chart(fig_l,use_container_width=True)
    with pc2:
        st.write("### Backtest: Actual vs Predicted")
        if len(bt_preds)>0:
            bt_start=art['time_step']+train_n
            bt_idx=df_used.index[bt_start:bt_start+len(bt_preds)]
            fig_bt=go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt_idx,y=bt_actuals,name='Actual',
                                        line=dict(color='#1f77b4')))
            fig_bt.add_trace(go.Scatter(x=bt_idx,y=bt_preds,name='Predicted',
                                        line=dict(color='#ff7f0e')))
            fig_bt.update_layout(title="Backtest",
                                 xaxis_title="Date",yaxis_title="Price ($)")
            st.plotly_chart(fig_bt,use_container_width=True)

    if len(bt_preds)>0:
        dir_correct=(np.sign(bt_pred_ret)==np.sign(bt_actual_ret)).astype(int)
        bt_idx2=df_used.index[art['time_step']+train_n:
                               art['time_step']+train_n+len(bt_preds)]
        fig_dir=go.Figure()
        fig_dir.add_trace(go.Bar(x=bt_idx2,y=dir_correct,
                                 marker_color=['#22c55e' if c==1 else '#ef4444'
                                               for c in dir_correct],
                                 showlegend=False))
        fig_dir.add_hline(y=0.5,line_dash='dash',line_color='#94a3b8',
                          annotation_text="Random (50%)")
        fig_dir.update_layout(
            title=f"Directional Accuracy: {art['da']:.1f}% "
                  f"({int(dir_correct.sum())}/{len(dir_correct)} correct)",
            height=280)
        st.plotly_chart(fig_dir,use_container_width=True)

    residuals=bt_actuals-bt_preds
    fig_res=go.Figure()
    fig_res.add_trace(go.Histogram(x=residuals,nbinsx=40,
                                   marker_color='#6366f1',opacity=0.75))
    fig_res.add_vline(x=0,line_color='white',line_dash='dash')
    fig_res.update_layout(
        title=f"Residuals | Mean=${residuals.mean():.2f} | Std=${residuals.std():.2f}",
        xaxis_title="Residual ($)",yaxis_title="Count",height=280)
    st.plotly_chart(fig_res,use_container_width=True)

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
        macd_s_=pd.Series([df_used['MACD_Signal'].iloc[-1]]*8+[macd_]
                           ).ewm(span=9,adjust=False).mean().iloc[-1]
        diff_=adj_s.diff().fillna(0)
        up_=diff_.clip(lower=0); dn_=-diff_.clip(upper=0)
        ru_=up_.rolling(14).mean().iloc[-1] if len(up_)>=14 else up_.mean()
        rd_=dn_.rolling(14).mean().iloc[-1] if len(dn_)>=14 else dn_.mean()
        rsi_=100-100/(1+ru_/(rd_+1e-9))
        rm_=adj_s.rolling(20).mean().iloc[-1]
        rs_=adj_s.rolling(20).std().iloc[-1] if len(adj_s)>=20 else adj_s.std()
        bb_w_=(2*rs_)/(rm_+1e-9); atr_=float(df_used['ATR'].iloc[-1])
        new_row=np.array([[pred_lr,log_vol_,sma20_,sma50_,ema12_,ema26_,
                           macd_,macd_s_,rsi_,bb_w_,atr_]])
        recent_scaled.append(scaler.transform(new_row)[0].tolist())

    floor=last_price*0.55; z=1.96; uppers=[]; lowers=[]
    for i,p in enumerate(future_preds_price):
        hw=min(z*resid_std*np.sqrt(i+1),0.15*p)
        uppers.append(p+hw); lowers.append(max(p-hw,floor))

    future_dates=pd.date_range(
        start=data_main.index[-1]+pd.Timedelta(days=1),periods=days,freq='B')
    future_df=pd.DataFrame({'Date':future_dates,'Predicted':future_preds_price,
                             'Upper':uppers,'Lower':lowers})
    future_df['Change %']=[f"{(p-last_price)/last_price*100:+.2f}%"
                            for p in future_preds_price]

    hist_x=price_s.index; hist_y=price_s.values
    fig_f=go.Figure()
    fig_f.add_trace(go.Scatter(x=hist_x,y=hist_y,name='Historical',
                               line=dict(color='#1f77b4')))
    fig_f.add_trace(go.Scatter(x=[future_dates[0]],y=[future_preds_price[0]],
                               name='Forecast',line=dict(color='#ff7f0e')))
    fig_f.add_trace(go.Scatter(
        x=[future_dates[0],future_dates[0]],y=[lowers[0],uppers[0]],
        fill='toself',fillcolor='rgba(255,127,14,0.15)',
        line=dict(color='rgba(255,127,14,0)'),name='95% CI'))
    frames=[]
    for i in range(len(future_dates)):
        xp=future_dates[:i+1]; yp=future_preds_price[:i+1]
        xb=list(future_dates[:i+1])+list(future_dates[:i+1][::-1])
        yb=list(uppers[:i+1])+list(lowers[:i+1][::-1])
        frames.append(go.Frame(data=[
            go.Scatter(x=hist_x,y=hist_y),
            go.Scatter(x=xp,y=yp,line=dict(color='#ff7f0e')),
            go.Scatter(x=xb,y=yb,fill='toself',
                       fillcolor='rgba(255,127,14,0.15)',
                       line=dict(color='rgba(255,127,14,0)'))
        ],name=str(i)))
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
    st.dataframe(future_df.style.format(
        {"Predicted":"${:.2f}","Upper":"${:.2f}","Lower":"${:.2f}"}),
        use_container_width=True)

    sm1,sm2,sm3,sm4=st.columns(4)
    sm1.metric("Walk-Forward R2",wf_r2_str)
    sm2.metric("Directional Acc",da_str)
    sm3.metric("MAPE",mape_str)
    sm4.metric("Beats Naive","Yes" if wins>=2 else "Partial")

    st.markdown("---")
    st.markdown("### AI Forecast Analysis")
    st.caption(f"Powered by {GROQ_MODEL} via Groq")
    if st.button("Generate AI Analysis",type="primary",key="pred_ai"):
        final_price=future_preds_price[-1]
        pct_change=(final_price-last_price)/last_price*100
        with st.spinner("Analysing with Llama 3.3 70B..."):
            prompt=f"""You are a professional stock analyst. Analyse this LSTM model forecast.
Stock: {selected_ticker} | Current: ${last_price:.2f}
Forecast {days} days ahead: ${final_price:.2f} ({pct_change:+.2f}%)
Confidence Range: Low ${lowers[-1]:.2f} - High ${uppers[-1]:.2f}
Model Quality: DA={art['da']:.1f}%, MAPE={art['mape']:.2f}%, WF-R2={art['wf_r2']:.3f}, Beats Naive={wins}/3
Provide exactly 3 sections:
1. FORECAST INTERPRETATION: What does this mean for {selected_ticker}?
2. MODEL RELIABILITY: How much should a trader trust this based on the metrics?
3. KEY RISKS: 2-3 bullet points of what could invalidate this forecast.
Be direct, cite the numbers, be professional."""
            response=call_groq(prompt,max_tokens=500)
        if response.startswith("__ERROR__"):
            st.error(f"AI analysis failed: {response}")
        else:
            for section in [s for s in response.split('\n\n') if s.strip()]:
                lines=section.strip().split('\n',1)
                title=lines[0].strip()
                body=lines[1].strip() if len(lines)>1 else lines[0].strip()
                st.markdown(f"""<div class="ai-box">
                  <div class="ai-box-title">{title}</div>{body}
                </div>""",unsafe_allow_html=True)

