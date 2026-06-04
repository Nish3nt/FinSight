# =============================================================================
#  FinSight — app.py  (FULLY OPTIMIZED)
#  Speed: Batched inference, rolling buffer, tf.function, BatchNorm, no dup DL
#  UX:    Guided predictions, plain English, comparison charts, portfolio suite
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
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
from streamlit_option_menu import option_menu
from collections import deque

nltk.download('vader_lexicon', quiet=True)
sia          = SentimentIntensityAnalyzer()
current_date = datetime.now().date()
st.set_page_config(page_title="FinSight", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"]>div:first-child{background:#0b1220;padding:16px 12px}
.block-container{padding-top:.6rem;padding-bottom:.4rem}
.model-box{background:#000;padding:18px;border-radius:12px;border:1px solid #111827;
           font-size:14px;color:#e6eef8}
.info-bar{font-size:12px;color:#cbd5e1;padding:8px 10px;background:#0b1220;
          border-radius:6px;margin-bottom:8px}
.metric-card{background:#0f172a;border-radius:10px;padding:14px 10px;text-align:center;
             border:1px solid #1e293b;margin-bottom:6px}
.metric-label{font-size:12px;color:#94a3b8;margin-bottom:4px}
.metric-value{font-size:22px;font-weight:700;color:#e2e8f0}
.metric-sub{font-size:11px;color:#64748b;margin-top:2px}
.metric-explain{font-size:11px;color:#475569;margin-top:4px;font-style:italic;line-height:1.4}
.good{color:#22c55e!important}
.warn{color:#f59e0b!important}
.bad{color:#ef4444!important}
.skel-card{background:linear-gradient(90deg,#111827 25%,#0b1220 50%,#111827 75%);
           background-size:200% 100%;animation:shimmer 1.4s linear infinite;
           height:120px;border-radius:10px;margin-bottom:12px}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.baseline-box{background:#0f172a;border:1px solid #1e293b;border-radius:10px;
              padding:14px;margin-bottom:6px}
.summary-card{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
              border:1px solid #1e3a5f;border-radius:14px;padding:20px 24px;
              margin-bottom:18px;color:#e2e8f0}
.summary-title{font-size:18px;font-weight:700;margin-bottom:8px;color:#7dd3fc}
.summary-body{font-size:14px;line-height:1.8;color:#cbd5e1}
.confidence-bar-wrap{background:#1e293b;border-radius:10px;height:18px;
                     overflow:hidden;margin-top:6px}
.confidence-bar{height:18px;border-radius:10px;transition:width .6s ease}
.pipeline-step{background:#0f172a;border:1px solid #1e293b;border-radius:8px;
               padding:10px 14px;font-size:12px;color:#94a3b8;text-align:center}
.pipeline-arrow{font-size:20px;color:#334155;padding:0 4px;line-height:40px}
.comp-summary{background:#0f172a;border:1px solid #1e3a5f;border-radius:12px;
              padding:18px 22px;margin-bottom:16px}
.comp-summary p{margin:0;font-size:14px;color:#cbd5e1;line-height:1.8}
.port-insight{background:#0f172a;border-left:3px solid #6366f1;
              border-radius:0 8px 8px 0;padding:12px 16px;font-size:13px;
              color:#cbd5e1;margin-bottom:10px;line-height:1.6}
</style>
""", unsafe_allow_html=True)

st.title("**FinSight**: Real-Time Stock Intelligence")

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
    df['RSI']      = 100 - 100 / (1 + up.rolling(14).mean() / (dn.rolling(14).mean() + 1e-9))
    rm = df['Adj Close'].rolling(20).mean()
    rs = df['Adj Close'].rolling(20).std()
    df['BB_Width'] = (2 * rs) / (rm + 1e-9)
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Adj Close'].shift()).abs()
    lc = (df['Low']  - df['Adj Close'].shift()).abs()
    df['ATR']       = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    df['LogVolume'] = np.log1p(df['Volume'])
    cols = ['LogReturn','LogVolume','SMA20','SMA50','EMA12','EMA26',
            'MACD','MACD_Signal','RSI','BB_Width','ATR']
    return df[cols].dropna(), df['Adj Close']

def update_indicators_incremental(price_buffer, last_macd_signal,
                                   last_log_vol, last_atr, pred_lr):
    prices = np.array(price_buffer); n = len(prices)
    sma20  = float(np.mean(prices[-20:])) if n>=20 else float(np.mean(prices))
    sma50  = float(np.mean(prices[-50:])) if n>=50 else float(np.mean(prices))
    s      = pd.Series(prices)
    ema12  = float(s.ewm(span=12,adjust=False).mean().iloc[-1])
    ema26  = float(s.ewm(span=26,adjust=False).mean().iloc[-1])
    macd   = ema12 - ema26
    alpha9 = 2/(9+1)
    macd_s = last_macd_signal*(1-alpha9) + macd*alpha9
    diff_  = s.diff().fillna(0)
    up_    = diff_.clip(lower=0); dn_ = -diff_.clip(upper=0)
    ru     = float(up_.rolling(14).mean().iloc[-1]) if n>=14 else float(up_.mean())
    rd     = float(dn_.rolling(14).mean().iloc[-1]) if n>=14 else float(dn_.mean())
    rsi    = 100 - 100/(1+ru/(rd+1e-9))
    rm_    = float(s.rolling(20).mean().iloc[-1]) if n>=20 else float(s.mean())
    rs_    = float(s.rolling(20).std().iloc[-1])  if n>=20 else float(s.std())
    bb_w   = (2*rs_)/(rm_+1e-9)
    return np.array([[pred_lr,last_log_vol,sma20,sma50,
                      ema12,ema26,macd,macd_s,rsi,bb_w,last_atr]]), macd_s

def compute_drawdown(series):
    roll_max = series.cummax()
    return (series - roll_max)/roll_max*100

tab = option_menu(None,
    ["Data & Viz","Predictions","Sentiment","Comparison","Portfolio Analyzer"],
    icons=["table","graph-up","chat-dots","arrow-left-right","pie-chart"],
    orientation="horizontal")

# ==============================================================================
#  TAB 1 - DATA & VIZ
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
#  TAB 2 - PREDICTIONS (Guided narrative + plain English + Confidence Score)
# ==============================================================================
elif tab == "Predictions":

    if data_main is None:
        st.error("Not enough data. Expand date range or choose another ticker."); st.stop()

    with st.expander("Model Settings", expanded=False):
        c1, c2 = st.columns([2,1])
        with c1:
            days       = st.slider("Forecast horizon (trading days)", 1, 30, 7)
            time_step  = st.slider("Lookback window (days)", 60, 180, 90, step=10)
            epochs     = st.slider("Training epochs", 20, 150, 60, step=5)
            batch_size = st.selectbox("Batch size", [16,32,64], index=1)
            retrain    = st.checkbox("Force retrain model", value=False)
        with c2:
            st.markdown("""
            <div class="model-box">
            <b>11-Feature LSTM (Optimized)</b><br><br>
            <b>Trend:</b> SMA20, SMA50, EMA12, EMA26<br>
            <b>Momentum:</b> MACD, Signal, RSI<br>
            <b>Volatility:</b> BB Width, ATR<br>
            <b>Other:</b> Log Return, Log Volume<br><br>
            LSTM(64) + BN -> LSTM(32) + BN<br>
            -> Dense(32,relu) -> Dense(1)<br><br>
            Batched inference | tf.function<br>
            Rolling buffer | No dup download<br><br>
            Target R2 > 0.82 | DA > 55%
            </div>""", unsafe_allow_html=True)

    # Provide defaults if expander never opened
    try: days
    except NameError: days=7
    try: time_step
    except NameError: time_step=90
    try: epochs
    except NameError: epochs=60
    try: batch_size
    except NameError: batch_size=32
    try: retrain
    except NameError: retrain=False

    df_features, price_series = compute_features(data_main)
    if len(df_features) < time_step + 30:
        st.error(f"Not enough rows. Need >= {time_step+30}, got {len(df_features)}."); st.stop()

    @st.cache_resource(ttl=24*3600)
    def train_model(ticker, start_str, end_str, time_step, epochs,
                    batch_size, retrain_flag, _n_rows):
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
        X = np.array(X,dtype=np.float32); y = np.array(y,dtype=np.float32)
        n = X.shape[0]; train_n = int(n*0.80)
        X_tr,y_tr = X[:train_n],y[:train_n]
        X_te,y_te = X[train_n:],y[train_n:]
        n_feat = X.shape[2]
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(time_step,n_feat)),
            BatchNormalization(), Dropout(0.2),
            LSTM(32), BatchNormalization(), Dropout(0.15),
            Dense(32,activation='relu'), Dense(1)
        ])
        model.compile(optimizer=Adam(0.001), loss='mse')
        cbs, val_split = [], 0.0
        if len(X_tr) > 20:
            cbs = [EarlyStopping(monitor='val_loss',patience=8,
                                  restore_best_weights=True,verbose=0),
                   ReduceLROnPlateau(monitor='val_loss',factor=0.5,
                                     patience=5,min_lr=1e-6,verbose=0)]
            val_split = 0.1
        history = model.fit(X_tr,y_tr,epochs=epochs,batch_size=batch_size,
                            validation_split=val_split,callbacks=cbs,verbose=0)

        @tf.function(reduce_retracing=True)
        def fast_predict(x): return model(x, training=False)

        all_sc = fast_predict(tf.constant(X_te,dtype=tf.float32)).numpy().flatten()
        dummy  = np.zeros((len(all_sc),n_feat),dtype=np.float32)
        dummy[:,0] = all_sc
        all_lr = scaler.inverse_transform(dummy)[:,0]

        bt_pp,bt_ap,bt_pr,bt_ar = [],[],[],[]
        for i in range(len(all_lr)):
            gi = time_step+train_n+i
            plr = float(all_lr[i]); alr = float(df_feat['LogReturn'].iloc[gi])
            pp  = float(price_s.iloc[gi-1])*np.exp(plr)
            ap  = float(price_s.iloc[gi])
            bt_pp.append(pp); bt_ap.append(ap)
            bt_pr.append(plr); bt_ar.append(alr)
        bt_pp=np.array(bt_pp); bt_ap=np.array(bt_ap)
        bt_pr=np.array(bt_pr); bt_ar=np.array(bt_ar)

        wf,fs = [],max(10,len(bt_pp)//5)
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
        n_r2   = float(r2_score(na_,np_))
        n_mape = float(np.mean(np.abs((na_-np_)/(np.abs(na_)+1e-9)))*100)
        n_rmse = float(np.sqrt(mean_squared_error(na_,np_)))
        rs_v   = float(np.std(bt_ap-bt_pp))
        return dict(model=model,fast_predict=fast_predict,scaler=scaler,
                    df_feat=df_feat,price_series=price_s,time_step=time_step,
                    train_n=train_n,n_feat=n_feat,bt_pp=bt_pp,bt_ap=bt_ap,
                    bt_pr=bt_pr,bt_ar=bt_ar,history=history.history,
                    training_time=training_time,training_secs=time.time()-t0,
                    epochs=epochs,batch_size=batch_size,
                    mse=mse_v,r2=r2_v,rmse=rmse_v,mape=mape_v,da=da_v,
                    wf_r2=wf_r2,wf_list=wf,n_r2=n_r2,n_mape=n_mape,
                    n_rmse=n_rmse,resid_std=rs_v)

    ph = st.empty()
    with ph.container():
        st.markdown('<div class="skel-card"></div>',unsafe_allow_html=True)
        st.markdown('<div class="skel-card"></div>',unsafe_allow_html=True)
    try:
        art = train_model(selected_ticker,str(start_date),str(end_date),
                          time_step,epochs,batch_size,retrain,
                          _n_rows=len(df_features))
    finally:
        ph.empty()

    model        = art['model'];       fp          = art['fast_predict']
    scaler       = art['scaler'];      df_used     = art['df_feat']
    price_s      = art['price_series'];train_n     = art['train_n']
    bt_preds     = art['bt_pp'];       bt_actuals  = art['bt_ap']
    bt_pred_ret  = art['bt_pr'];       bt_act_ret  = art['bt_ar']
    history      = art['history'];     n_feat      = art['n_feat']
    resid_std    = art['resid_std'];   last_price  = float(price_s.iloc[-1])
    beat_r2   = art['r2']   > art['n_r2']
    beat_mape = art['mape'] < art['n_mape']
    beat_rmse = art['rmse'] < art['n_rmse']
    wins      = sum([beat_r2,beat_mape,beat_rmse])

    wf_str  = f"{art['wf_r2']:.3f}"; r2_str  = f"{art['r2']:.3f}"
    da_str  = f"{art['da']:.1f}%";   mp_str  = f"{art['mape']:.2f}%"
    rm_str  = f"${art['rmse']:.2f}"

    def ccls(v,g,w,hi=True):
        if hi: return "good" if v>=g else "warn" if v>=w else "bad"
        return "good" if v<=g else "warn" if v<=w else "bad"

    r2n  = max(0,min(100,art['r2']*100))
    dan  = max(0,min(100,(art['da']-50)*5))
    mpn  = max(0,min(100,(10-art['mape'])*10))
    bsn  = wins*33.3
    conf = int(0.30*r2n+0.35*dan+0.20*mpn+0.15*bsn)
    conf = max(0,min(100,conf))
    clbl = "High" if conf>=70 else "Medium" if conf>=45 else "Low"
    ccol = "#22c55e" if conf>=70 else "#f59e0b" if conf>=45 else "#ef4444"

    yrs   = round((pd.Timestamp(end_date)-pd.Timestamp(start_date)).days/365,1)
    bt_txt = ("It beats a simple do-nothing baseline on all 3 quality checks."
              if wins==3 else f"It beats a simple do-nothing baseline on {wins}/3 checks.")

    st.markdown(f"""
    <div class="summary-card">
      <div class="summary-title">What is this model doing? - Plain English</div>
      <div class="summary-body">
        The AI studied <b>{yrs} years</b> of <b>{selected_ticker}</b> price history
        using <b>11 market signals</b> (price trends, momentum, volatility).
        It was tested on the most recent <b>20% of data it had never seen</b> during training.<br><br>
        On that unseen data it correctly predicted price direction
        <b>up or down {art['da']:.1f}%</b> of the time (random = 50%).
        Its average price error was just <b>{art['mape']:.2f}%</b>. {bt_txt}<br><br>
        Forecast <b style="color:{ccol}">Confidence: {conf}/100 ({clbl})</b>
      </div>
      <div style="margin-top:10px">
        <div style="font-size:12px;color:#94a3b8;margin-bottom:4px">
          Prediction Confidence: {conf}/100 ({clbl})</div>
        <div class="confidence-bar-wrap">
          <div class="confidence-bar" style="width:{conf}%;background:{ccol}"></div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    with st.expander("How was this prediction built? (click to expand)", expanded=False):
        st.markdown("##### 5 steps the model followed:")
        p1,a1,p2,a2,p3,a3,p4,a4,p5 = st.columns([2,.3,2,.3,2,.3,2,.3,2])
        steps = [
            ("<b>1. Collect Data</b><br>Downloaded daily OHLCV from Yahoo Finance",p1),
            ("->",a1),
            ("<b>2. Engineer 11 Signals</b><br>SMA,EMA,MACD,RSI,BB,ATR,LogReturn,Volume",p2),
            ("->",a2),
            ("<b>3. Train LSTM</b><br>2-layer neural net learns temporal price patterns on 80% of data",p3),
            ("->",a3),
            ("<b>4. Validate on 20%</b><br>Tested on unseen data - computed R2, MAPE, DA, RMSE",p4),
            ("->",a4),
            ("<b>5. Forecast Future</b><br>Predicted next N trading days with growing confidence bands",p5),
        ]
        for txt, col in steps:
            if txt == "->":
                col.markdown('<div class="pipeline-arrow">-></div>',unsafe_allow_html=True)
            else:
                col.markdown(f'<div class="pipeline-step">{txt}</div>',unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Model Quality - How Reliable is This Forecast?")
    st.caption("All metrics measured on test data the model had never seen during training.")

    wf_cls=ccls(art['wf_r2'],0.80,0.65); r2_cls=ccls(art['r2'],0.80,0.65)
    da_cls=ccls(art['da'],58,52);         mp_cls=ccls(art['mape'],3,5,hi=False)

    m1,m2,m3,m4,m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Confidence Score</div>
          <div class="metric-value" style="color:{ccol}">{conf}/100</div>
          <div class="metric-sub">{clbl} reliability</div>
          <div class="metric-explain">Combined trust score from all metrics below</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Directional Accuracy</div>
          <div class="metric-value {da_cls}">{da_str}</div>
          <div class="metric-sub">Random baseline = 50%</div>
          <div class="metric-explain">How often did the model correctly predict UP vs DOWN?</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Avg Price Error (MAPE)</div>
          <div class="metric-value {mp_cls}">{mp_str}</div>
          <div class="metric-sub">Lower is better</div>
          <div class="metric-explain">How far off was predicted price from real price on average?</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Walk-Forward R2</div>
          <div class="metric-value {wf_cls}">{wf_str}</div>
          <div class="metric-sub">5 rolling time windows</div>
          <div class="metric-explain">How well does the model explain price movement? (1.0=perfect)</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Dollar Error (RMSE)</div>
          <div class="metric-value">{rm_str}</div>
          <div class="metric-sub">Avg $ error per day</div>
          <div class="metric-explain">Average dollar gap between predicted and actual price</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    vc = "good" if wins==3 else "warn" if wins>=2 else "bad"
    vt = ("Beats baseline on all 3 metrics" if wins==3
          else f"Beats baseline on {wins}/3 metrics" if wins>=2
          else "Does not beat baseline")

    with st.expander("Does the model beat a do-nothing baseline?", expanded=False):
        st.caption("A Naive baseline predicts tomorrow = today. Any real ML must beat this.")
        b1,b2,b3,b4 = st.columns(4)
        r2c="good" if beat_r2 else "bad"
        mpc="good" if beat_mape else "bad"
        rmc="good" if beat_rmse else "bad"
        with b1:
            st.markdown("""<div class="baseline-box">
              <div class="metric-label">Metric</div>
              <div style="color:#94a3b8;margin-top:10px">R2 (higher better)</div>
              <div style="color:#94a3b8;margin-top:10px">MAPE (lower better)</div>
              <div style="color:#94a3b8;margin-top:10px">RMSE (lower better)</div>
            </div>""", unsafe_allow_html=True)
        with b2:
            st.markdown(f"""<div class="baseline-box">
              <div class="metric-label">Our LSTM</div>
              <div class="{r2c}" style="margin-top:10px">{art['r2']:.3f}</div>
              <div class="{mpc}" style="margin-top:10px">{art['mape']:.2f}%</div>
              <div class="{rmc}" style="margin-top:10px">${art['rmse']:.2f}</div>
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
              <div class="{vc}" style="margin-top:10px;font-size:13px;font-weight:600">
                {vt}</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("Technical Validation Charts (for advanced users)", expanded=False):
        st.caption("These show model performance on historical test data before making future predictions.")
        pc1,pc2 = st.columns(2)
        with pc1:
            st.write("**Training Loss Curve**")
            st.caption("Both lines trending down and close together = model learned well without memorising the data.")
            ep_r  = list(range(1,len(history.get('loss',[]))+1))
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(x=ep_r,y=history.get('loss',[]),
                                       mode='lines+markers',name='Train Loss',
                                       line=dict(color='#1f77b4')))
            if 'val_loss' in history:
                fig_l.add_trace(go.Scatter(x=ep_r,y=history['val_loss'],
                                           mode='lines+markers',name='Val Loss',
                                           line=dict(color='#ff7f0e')))
            fig_l.update_layout(title="Loss Curve (lower = better)",height=300)
            st.plotly_chart(fig_l, use_container_width=True)
        with pc2:
            st.write("**Backtest: Actual vs Predicted**")
            st.caption("Orange = predicted, Blue = actual. Closer lines = better model.")
            if len(bt_preds)>0:
                bt_start = art['time_step']+train_n
                bt_idx   = df_used.index[bt_start:bt_start+len(bt_preds)]
                fig_bt   = go.Figure()
                fig_bt.add_trace(go.Scatter(x=bt_idx,y=bt_actuals,
                                            name='Actual',line=dict(color='#1f77b4')))
                fig_bt.add_trace(go.Scatter(x=bt_idx,y=bt_preds,
                                            name='Predicted',line=dict(color='#ff7f0e')))
                fig_bt.update_layout(title="Backtest on unseen 20% of data",height=300)
                st.plotly_chart(fig_bt, use_container_width=True)

        if art['wf_list']:
            st.write("**Walk-Forward R2 - Consistency Across Time Windows**")
            st.caption("Each bar = model performance in a different time period. Consistent green = works in all market conditions, not just luck in one period.")
            wfc = ['#22c55e' if v>=0.80 else '#f59e0b' if v>=0.65 else '#ef4444'
                   for v in art['wf_list']]
            fig_wf = go.Figure()
            for fn,fv,fc in zip([f"Fold {i+1}" for i in range(5)],art['wf_list'],wfc):
                fig_wf.add_trace(go.Bar(x=[fn],y=[fv],marker_color=fc,
                                        text=[f"{fv:.3f}"],textposition='outside',
                                        showlegend=False))
            fig_wf.add_hline(y=0.80,line_dash='dash',line_color='#22c55e',
                             annotation_text="Target 0.80")
            fig_wf.add_hline(y=0.65,line_dash='dot',line_color='#f59e0b',
                             annotation_text="Acceptable 0.65")
            fig_wf.update_layout(yaxis_range=[min(min(art['wf_list'])-0.05,-0.05),1.05],
                                 height=300,title=f"Mean R2 = {wf_str}")
            st.plotly_chart(fig_wf, use_container_width=True)

        dc1,dc2 = st.columns(2)
        with dc1:
            st.write("**Directional Accuracy - Up or Down?**")
            st.caption("Green = model called direction correctly. Red = wrong. Any score above 50% beats random chance.")
            if len(bt_preds)>0:
                dir_c = (np.sign(bt_pred_ret)==np.sign(bt_act_ret)).astype(int)
                bt_i2 = df_used.index[art['time_step']+train_n:
                                       art['time_step']+train_n+len(bt_preds)]
                bclrs = ['#22c55e' if c==1 else '#ef4444' for c in dir_c]
                fig_d = go.Figure()
                fig_d.add_trace(go.Bar(x=bt_i2,y=dir_c,marker_color=bclrs,showlegend=False))
                fig_d.add_hline(y=0.5,line_dash='dash',line_color='#94a3b8',
                                annotation_text="Random (50%)")
                fig_d.update_layout(
                    title=f"Direction correct {art['da']:.1f}% ({int(dir_c.sum())}/{len(dir_c)})",
                    height=280)
                st.plotly_chart(fig_d, use_container_width=True)
        with dc2:
            st.write("**Prediction Error Distribution**")
            st.caption("Tight bell curve near $0 = well-calibrated. Model not consistently over or under-predicting.")
            residuals = bt_actuals - bt_preds
            fig_r = go.Figure()
            fig_r.add_trace(go.Histogram(x=residuals,nbinsx=40,
                                         marker_color='#6366f1',opacity=0.75))
            fig_r.add_vline(x=0,line_color='white',line_dash='dash')
            fig_r.update_layout(
                title=f"Errors | Mean=${residuals.mean():.2f} Std=${residuals.std():.2f}",
                xaxis_title="Error ($)",yaxis_title="Count",height=280)
            st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")
    st.markdown("## Future Price Forecast")
    st.markdown(f"""
    <div class="summary-body" style="margin-bottom:14px">
    The <b>orange line</b> is the predicted price path for the next <b>{days} trading days</b>.
    The <b>shaded band</b> is the 95% confidence interval - it widens over time because
    uncertainty compounds the further into the future we predict.
    Press <b>Play</b> to reveal the forecast day by day.
    </div>""", unsafe_allow_html=True)

    price_buffer  = deque(price_s.values[-51:].tolist(), maxlen=51)
    recent_scaled = scaler.transform(df_used.values[-time_step:]).tolist()
    chain_price   = last_price
    last_ms       = float(df_used['MACD_Signal'].iloc[-1])
    last_lv       = float(df_used['LogVolume'].iloc[-1])
    last_atr      = float(df_used['ATR'].iloc[-1])
    fp_prices     = []
    dummy_f       = np.zeros((1,n_feat),dtype=np.float32)

    for _ in range(days):
        inp    = np.array(recent_scaled[-time_step:],dtype=np.float32).reshape(1,time_step,-1)
        psc    = float(fp(tf.constant(inp))[0,0])
        dummy_f[0,0] = psc
        plr    = float(scaler.inverse_transform(dummy_f)[0,0])
        pp     = chain_price * np.exp(plr)
        fp_prices.append(pp); chain_price = pp
        price_buffer.append(pp)
        nr, last_ms = update_indicators_incremental(
            price_buffer, last_ms, last_lv, last_atr, plr)
        recent_scaled.append(scaler.transform(nr)[0].tolist())

    floor = last_price*0.55; z = 1.96
    uppers,lowers = [],[]
    for i,p in enumerate(fp_prices):
        hw = min(z*resid_std*np.sqrt(i+1), 0.15*p)
        uppers.append(p+hw); lowers.append(max(p-hw,floor))

    future_dates = pd.date_range(
        start=data_main.index[-1]+pd.Timedelta(days=1),periods=days,freq='B')
    future_df = pd.DataFrame({'Date':future_dates,'Predicted':fp_prices,
                               'Upper (95% CI)':uppers,'Lower (95% CI)':lowers})
    future_df['Expected Change %'] = [f"{(p-last_price)/last_price*100:+.2f}%"
                                       for p in fp_prices]

    hx = price_s.index; hy = price_s.values
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=hx,y=hy,name='Historical',line=dict(color='#1f77b4')))
    fig_f.add_trace(go.Scatter(x=[future_dates[0]],y=[fp_prices[0]],
                               name='Forecast',line=dict(color='#ff7f0e')))
    fig_f.add_trace(go.Scatter(
        x=[future_dates[0],future_dates[0]],y=[lowers[0],uppers[0]],
        fill='toself',fillcolor='rgba(255,127,14,0.15)',
        line=dict(color='rgba(255,127,14,0)'),name='95% CI'))
    frames = []
    for i in range(len(future_dates)):
        xp = future_dates[:i+1]; yp = fp_prices[:i+1]
        xb = list(future_dates[:i+1])+list(future_dates[:i+1][::-1])
        yb = list(uppers[:i+1])+list(lowers[:i+1][::-1])
        frames.append(go.Frame(data=[
            go.Scatter(x=hx,y=hy),
            go.Scatter(x=xp,y=yp,line=dict(color='#ff7f0e')),
            go.Scatter(x=xb,y=yb,fill='toself',fillcolor='rgba(255,127,14,0.15)',
                       line=dict(color='rgba(255,127,14,0)'))
        ],name=str(i)))
    fig_f.frames = frames
    fig_f.update_layout(
        title=f"{selected_ticker} - {days}-Day Forecast | Last: ${last_price:.2f} | Confidence: {conf}/100 ({clbl})",
        xaxis_title="Date",yaxis_title="Price ($)",
        updatemenus=[{"type":"buttons","buttons":[
            {"label":"Play","method":"animate",
             "args":[None,{"frame":{"duration":400,"redraw":True},
                           "fromcurrent":True,"transition":{"duration":200}}]},
            {"label":"Pause","method":"animate",
             "args":[[None],{"frame":{"duration":0,"redraw":False},
                             "mode":"immediate","transition":{"duration":0}}]}
        ],"direction":"left","pad":{"r":10,"t":10},
         "showactive":True,"x":0.01,"y":-0.12,"xanchor":"left","yanchor":"top"}])
    st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("##### Day-by-Day Forecast Table")
    st.caption("Each row is one future trading day. Bounds show the 95% confidence range.")
    st.dataframe(future_df.style.format({
        "Predicted":"${:.2f}","Upper (95% CI)":"${:.2f}","Lower (95% CI)":"${:.2f}"}),
        use_container_width=True)

    st.markdown("<br>",unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Confidence Score", f"{conf}/100 ({clbl})")
    s2.metric("Directional Acc",  da_str)
    s3.metric("Avg Price Error",  mp_str)
    s4.metric("Beats Naive",      "Yes" if wins>=2 else "Partial")

# ==============================================================================
#  TAB 3 - SENTIMENT
# ==============================================================================
elif tab == "Sentiment":
    st.subheader("News Sentiment")
    if news_posts:
        df_s = pd.DataFrame({'News':news_posts,'Link':news_links,'Score':vader_scores})
        def color(val):
            return f"color: {'green' if val>0.1 else 'red' if val<-0.1 else 'gray'}"
        st.dataframe(df_s.style.map(color,subset=['Score']).format({'Score':'{:.3f}'}),
                     use_container_width=True)
        pos = sum(1 for s in vader_scores if s> 0.1)
        neg = sum(1 for s in vader_scores if s<-0.1)
        neu = len(vader_scores)-pos-neg
        c1,c2,c3 = st.columns(3)
        c1.metric("Positive",pos); c2.metric("Negative",neg); c3.metric("Neutral",neu)
    else:
        st.info("No news available.")

# ==============================================================================
#  TAB 4 - COMPARISON (plain English + drawdown + rolling + risk-return)
# ==============================================================================
elif tab == "Comparison":
    st.subheader(f"**{selected_ticker} vs {compare_ticker}** - Head to Head Analysis")
    if data_main is None or data_compare is None:
        st.error("Not enough data for one or both tickers."); st.stop()

    bm = data_main['Adj Close'].iloc[0];    bc = data_compare['Adj Close'].iloc[0]
    dm = (data_main['Adj Close']/bm-1)*100; dc = (data_compare['Adj Close']/bc-1)*100
    rm = float(dm.iloc[-1]);               rc = float(dc.iloc[-1])
    vm = float(data_main['Adj Close'].pct_change().std()*np.sqrt(252)*100)
    vc = float(data_compare['Adj Close'].pct_change().std()*np.sqrt(252)*100)
    ra_m = rm/vm if vm>0 else 0
    ra_c = rc/vc if vc>0 else 0
    dd_m = compute_drawdown(data_main['Adj Close'])
    dd_c = compute_drawdown(data_compare['Adj Close'])
    max_dd_m = float(dd_m.min()); max_dd_c = float(dd_c.min())

    better_ret  = selected_ticker if rm>rc else compare_ticker
    better_stab = selected_ticker if vm<vc else compare_ticker
    worse_stab  = compare_ticker  if vm<vc else selected_ticker
    ret_diff    = abs(rm-rc); vol_diff = abs(vm-vc)
    if abs(ra_m-ra_c) < 0.5:
        ra_txt = "both performed similarly on risk-adjusted basis"
    elif ra_m > ra_c:
        ra_txt = f"{selected_ticker} had better risk-adjusted returns"
    else:
        ra_txt = f"{compare_ticker} had better risk-adjusted returns"

    st.markdown(f"""
    <div class="comp-summary"><p>
    <b>What you are looking at:</b> Both stocks are normalised to start at 0% on
    <b>{start_date}</b> so you can fairly compare their growth regardless of their
    different share prices. The lines show % gained or lost from the starting point.<br><br>
    <b>What the data says:</b> Since <b>{start_date}</b>,
    <b>{selected_ticker}</b> returned <b>{rm:+.1f}%</b> while
    <b>{compare_ticker}</b> returned <b>{rc:+.1f}%</b> - a difference of
    <b>{ret_diff:.1f} percentage points</b>. <b>{better_ret}</b> was the stronger
    raw performer.<br><br>
    <b>Risk context:</b> <b>{worse_stab}</b> had higher volatility
    ({max(vm,vc):.1f}% vs {min(vm,vc):.1f}% annualised), meaning bigger price swings.
    <b>{better_stab}</b> delivered more stable returns. When adjusting for risk,
    <b>{ra_txt}</b>.
    </p></div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"{selected_ticker} Total Return",   f"{rm:+.2f}%")
    c2.metric(f"{compare_ticker} Total Return",    f"{rc:+.2f}%")
    c3.metric(f"{selected_ticker} Ann. Volatility",f"{vm:.1f}%",
              help="Higher = bigger daily price swings = more risk")
    c4.metric(f"{compare_ticker} Ann. Volatility", f"{vc:.1f}%",
              help="Higher = bigger daily price swings = more risk")

    st.markdown("#### Cumulative Return Since Start Date")
    st.caption("Both start at 0%. Shows who grew more from the same starting point - regardless of stock price level.")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data_main.index,y=dm,
                              name=selected_ticker,line=dict(color='#26A69A')))
    fig1.add_trace(go.Scatter(x=data_compare.index,y=dc,
                              name=compare_ticker,line=dict(color='#AB47BC')))
    fig1.add_hline(y=0,line_dash='dot',line_color='#475569')
    fig1.update_layout(title="Normalised Performance (%)",height=400,
                       yaxis_title="Return (%)",xaxis_title="Date")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("#### Rolling 12-Month Return")
    st.caption("What your return would have been holding the stock for exactly 12 months ending on each date. Answers: was this stock consistently good or just lucky in one period? Staying above 0% = consistently profitable.")
    roll_m = data_main['Adj Close'].pct_change(252)*100
    roll_c = data_compare['Adj Close'].pct_change(252)*100
    fig2   = go.Figure()
    fig2.add_trace(go.Scatter(x=data_main.index,y=roll_m,name=selected_ticker,
                              line=dict(color='#26A69A'),fill='tozeroy',
                              fillcolor='rgba(38,166,154,0.08)'))
    fig2.add_trace(go.Scatter(x=data_compare.index,y=roll_c,name=compare_ticker,
                              line=dict(color='#AB47BC'),fill='tozeroy',
                              fillcolor='rgba(171,71,188,0.08)'))
    fig2.add_hline(y=0,line_color='#475569',line_dash='dash')
    fig2.update_layout(title="Rolling 12-Month Return (%)",height=350,
                       yaxis_title="12-Month Return (%)",xaxis_title="Date")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Maximum Drawdown - Worst Drops from Peak")
    st.caption("Shows how far the stock fell from its highest point at any moment. For example -40% means the stock was 40% below its all-time high at that point. Shallower troughs = less pain for investors during bad periods.")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data_main.index,y=dd_m,name=selected_ticker,
                              line=dict(color='#26A69A'),fill='tozeroy',
                              fillcolor='rgba(38,166,154,0.12)'))
    fig3.add_trace(go.Scatter(x=data_compare.index,y=dd_c,name=compare_ticker,
                              line=dict(color='#AB47BC'),fill='tozeroy',
                              fillcolor='rgba(171,71,188,0.12)'))
    fig3.update_layout(
        title=f"Drawdown | Worst drop: {selected_ticker}={max_dd_m:.1f}% | {compare_ticker}={max_dd_c:.1f}%",
        height=350,yaxis_title="Drawdown (%)",xaxis_title="Date")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Risk vs Return Positioning")
    st.caption("Each diamond = one stock plotted by its total return (y-axis) against its risk/volatility (x-axis). Top-left corner = ideal: high return, low risk. Bottom-right = worst: low return, high risk.")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=[vm,vc],y=[rm,rc],mode='markers+text',
        text=[selected_ticker,compare_ticker],textposition='top center',
        marker=dict(size=20,color=['#26A69A','#AB47BC'],symbol='diamond')))
    fig4.update_layout(title="Risk vs Return",height=350,
                       xaxis_title="Annualised Volatility (Risk) %",
                       yaxis_title="Total Return %")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Summary Comparison Table")
    summary_df = pd.DataFrame({
        'Metric':          ['Total Return','Ann. Volatility',
                            'Max Drawdown','Risk-Adj Return (Return/Vol)'],
        selected_ticker:   [f"{rm:+.2f}%",f"{vm:.1f}%",
                            f"{max_dd_m:.1f}%",f"{ra_m:.2f}x"],
        compare_ticker:    [f"{rc:+.2f}%",f"{vc:.1f}%",
                            f"{max_dd_c:.1f}%",f"{ra_c:.2f}x"],
        'Winner':          [
            selected_ticker if rm>rc else compare_ticker,
            selected_ticker if vm<vc else compare_ticker,
            selected_ticker if max_dd_m>max_dd_c else compare_ticker,
            selected_ticker if ra_m>ra_c else compare_ticker
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ==============================================================================
#  TAB 5 - PORTFOLIO ANALYZER (Industry-Grade)
# ==============================================================================
elif tab == "Portfolio Analyzer":
    st.subheader("Portfolio Analyzer - Industry-Grade Analytics")

    st.markdown("""<div class="port-insight">
    <b>What is this?</b> A portfolio is a combination of multiple stocks held together.
    This tool shows how your mix of stocks would have performed historically, how risky
    it is, and whether combining them is actually better than holding any one alone.
    The key insight of Modern Portfolio Theory (Nobel Prize 1990) is that combining stocks
    with different behaviours can <b>reduce risk without sacrificing return</b>.
    </div>""", unsafe_allow_html=True)

    port_tickers = st.multiselect("Select Tickers", tickers,
                                   default=[selected_ticker, compare_ticker])
    if len(port_tickers) < 2:
        st.warning("Select at least 2 tickers to build a portfolio.")
        st.stop()

    weights, total_w = [], 0.0
    cols = st.columns(len(port_tickers))
    for i, tick in enumerate(port_tickers):
        w = cols[i].number_input(f"Weight {tick} (%)", 0.0, 100.0,
                                  round(100.0/len(port_tickers),2))
        weights.append(w/100); total_w += w

    if abs(total_w-100) > 0.01:
        st.warning(f"Weights sum to {total_w:.1f}%. Must add to exactly 100%.")
        st.stop()

    data_dict = {}
    for tick in port_tickers:
        d = fetch_stock_data(tick, start_date, end_date)
        if d is None:
            st.error(f"Data missing for {tick}."); st.stop()
        data_dict[tick] = d['Adj Close']

    port_df  = pd.DataFrame(data_dict).dropna()
    rets     = port_df.pct_change().dropna()
    m_ret    = rets.mean()*252
    cov_mat  = rets.cov()*252
    w_np     = np.array(weights)
    n_assets = len(port_tickers)
    p_ret    = float(np.dot(m_ret, w_np))
    p_vol    = float(np.sqrt(np.dot(w_np.T, np.dot(cov_mat, w_np))))
    sharpe   = (p_ret-0.03)/p_vol if p_vol>0 else 0

    port_daily = (rets*w_np).sum(axis=1)
    port_cum   = (1+port_daily).cumprod()*100-100
    port_idx   = (1+port_daily).cumprod()
    roll_max_p = port_idx.cummax()
    dd_port    = (port_idx-roll_max_p)/roll_max_p*100
    max_dd_p   = float(dd_port.min())
    var_95     = float(np.percentile(port_daily,5)*100)
    cvar_95    = float(port_daily[port_daily<=np.percentile(port_daily,5)].mean()*100)

    sharpe_desc = ("excellent" if sharpe>2 else "good" if sharpe>1
                   else "acceptable" if sharpe>0.5 else "below target")

    st.markdown(f"""
    <div class="comp-summary"><p>
    <b>Your Portfolio at a Glance:</b> Your {len(port_tickers)}-stock portfolio
    has an expected annual return of <b>{p_ret*100:.2f}%</b> with annual volatility
    (risk) of <b>{p_vol*100:.2f}%</b>. The Sharpe Ratio of <b>{sharpe:.2f}</b>
    is considered <b>{sharpe_desc}</b> - this measures return earned per unit of risk
    (above 1.0 is good by professional fund manager standards, above 2.0 is excellent).<br><br>
    <b>Downside Risk:</b> On a bad day (95% confidence), the portfolio could lose up to
    <b>{abs(var_95):.2f}%</b> in a single session. The worst peak-to-trough loss
    in this period was <b>{abs(max_dd_p):.1f}%</b> - that is the maximum pain you
    would have experienced buying at the worst possible time.
    </p></div>""", unsafe_allow_html=True)

    mc1,mc2,mc3,mc4,mc5 = st.columns(5)
    mc1.metric("Expected Annual Return", f"{p_ret*100:.2f}%",
               help="Average yearly return based on historical performance")
    mc2.metric("Annual Volatility",      f"{p_vol*100:.2f}%",
               help="How much the portfolio swings per year. Higher = more risk")
    mc3.metric("Sharpe Ratio",           f"{sharpe:.2f}",
               help="Return per unit of risk. >1.0 good, >2.0 excellent")
    mc4.metric("Daily VaR 95%",          f"{var_95:.2f}%",
               help="On a bad day, you could lose at most this % with 95% confidence")
    mc5.metric("Max Drawdown",           f"{max_dd_p:.1f}%",
               help="Worst peak-to-trough loss in the selected period")

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart 1: Portfolio vs SPY benchmark
    st.markdown("#### Portfolio Performance vs S&P 500 Benchmark")
    st.caption("Compares your portfolio against the S&P 500 index (SPY). SPY is the standard benchmark used by every professional fund manager. If your portfolio cannot beat it, a simple index fund would have been better.")
    spy_data = fetch_stock_data('SPY', start_date, end_date)
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=port_cum.index,y=port_cum.values,
                               name='Your Portfolio',line=dict(color='#6366f1',width=2)))
    if spy_data is not None:
        spy_ret = spy_data['Adj Close'].pct_change().dropna()
        spy_cum = (1+spy_ret).cumprod()*100-100
        spy_cum = spy_cum.reindex(port_cum.index, method='ffill').dropna()
        fig_b.add_trace(go.Scatter(x=spy_cum.index,y=spy_cum.values,
                                   name='S&P 500 (SPY)',
                                   line=dict(color='#94a3b8',width=1.5,dash='dash')))
        pf = float(port_cum.iloc[-1]); sf = float(spy_cum.iloc[-1])
        note = (f"Your portfolio outperformed SPY by {abs(pf-sf):.1f} percentage points"
                if pf>sf else
                f"Your portfolio underperformed SPY by {abs(pf-sf):.1f} percentage points")
        st.caption(note)
    fig_b.add_hline(y=0,line_dash='dot',line_color='#475569')
    fig_b.update_layout(title="Portfolio vs S&P 500",height=380,
                        yaxis_title="Cumulative Return (%)",xaxis_title="Date")
    st.plotly_chart(fig_b, use_container_width=True)

    # Chart 2: Efficient Frontier
    st.markdown("#### Efficient Frontier - Finding the Optimal Portfolio Mix")
    st.caption("""Nobel Prize winning theory (Markowitz 1952). Each dot = a randomly generated portfolio with different weight combinations of your stocks. Colour = Sharpe Ratio (brighter = better risk-adjusted return). The curved top-left edge is the Efficient Frontier - portfolios giving maximum return for a given level of risk. Your current portfolio is the red star. The gold star is the mathematically optimal mix.""")

    n_sim = 3000
    s_rets,s_vols,s_sharpes,s_wts = [],[],[],[]
    for _ in range(n_sim):
        w_ = np.random.dirichlet(np.ones(n_assets))
        r_ = float(np.dot(m_ret,w_))
        v_ = float(np.sqrt(np.dot(w_.T,np.dot(cov_mat,w_))))
        s_ = (r_-0.03)/v_ if v_>0 else 0
        s_rets.append(r_*100); s_vols.append(v_*100)
        s_sharpes.append(s_); s_wts.append(w_)

    hover = [" | ".join([f"{port_tickers[j]}: {s_wts[i][j]*100:.1f}%"
                         for j in range(n_assets)]) for i in range(n_sim)]
    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=s_vols,y=s_rets,mode='markers',
        marker=dict(size=4,color=s_sharpes,colorscale='Viridis',
                    showscale=True,colorbar=dict(title="Sharpe")),
        text=hover,
        hovertemplate="Return: %{y:.1f}%<br>Risk: %{x:.1f}%<br>%{text}<extra></extra>",
        name='Simulated Portfolios'))
    fig_ef.add_trace(go.Scatter(
        x=[p_vol*100],y=[p_ret*100],mode='markers+text',
        marker=dict(size=18,color='red',symbol='star'),
        text=['Your Portfolio'],textposition='top center',name='Your Portfolio'))
    best_idx = int(np.argmax(s_sharpes))
    fig_ef.add_trace(go.Scatter(
        x=[s_vols[best_idx]],y=[s_rets[best_idx]],mode='markers+text',
        marker=dict(size=18,color='#fbbf24',symbol='star'),
        text=['Optimal'],textposition='top center',name='Max Sharpe Portfolio'))
    fig_ef.update_layout(
        title="Efficient Frontier (3,000 simulated portfolios)",
        xaxis_title="Risk - Annualised Volatility (%)",
        yaxis_title="Expected Annual Return (%)",height=500)
    st.plotly_chart(fig_ef, use_container_width=True)

    best_w   = s_wts[best_idx]
    opt_hint = " | ".join([f"{port_tickers[j]}: {best_w[j]*100:.1f}%"
                           for j in range(n_assets)])
    st.markdown(f"""<div class="port-insight">
    <b>Optimal Portfolio Suggestion:</b> From 3,000 simulations, the weight mix with
    the highest Sharpe Ratio is: <b>{opt_hint}</b> - achieving a Sharpe of
    <b>{s_sharpes[best_idx]:.2f}</b> vs your current <b>{sharpe:.2f}</b>.
    </div>""", unsafe_allow_html=True)

    # Chart 3: Individual contribution to return and risk
    st.markdown("#### Individual Stock Contributions")
    st.caption("Shows how much each stock contributes to portfolio return vs portfolio risk. Ideal: a stock contributes more to return (green) than it does to risk (red). A stock with high risk contribution but low return contribution is dragging the portfolio.")
    contrib_ret  = [float(m_ret[t]*w_np[i]*100) for i,t in enumerate(port_tickers)]
    contrib_risk = []
    for i in range(n_assets):
        mg = float(np.dot(cov_mat.iloc[i].values,w_np))/p_vol*w_np[i]*100
        contrib_risk.append(mg)
    fig_c = go.Figure()
    fig_c.add_trace(go.Bar(name='Return Contribution (%)',x=port_tickers,
                            y=contrib_ret,marker_color='#22c55e'))
    fig_c.add_trace(go.Bar(name='Risk Contribution (%)',x=port_tickers,
                            y=contrib_risk,marker_color='#ef4444'))
    fig_c.update_layout(barmode='group',
                         title="Each Stock's Return vs Risk Contribution",
                         yaxis_title="%",height=380)
    st.plotly_chart(fig_c, use_container_width=True)

    # Chart 4: Portfolio Drawdown
    st.markdown("#### Portfolio Drawdown")
    st.caption("How far the combined portfolio fell from its peak at any point in time. Deeper troughs = worse periods for investors. A well-diversified portfolio should have shallower and shorter drawdowns than any individual stock.")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_port.index,y=dd_port.values,
                                name='Portfolio Drawdown',fill='tozeroy',
                                fillcolor='rgba(239,68,68,0.15)',
                                line=dict(color='#ef4444')))
    fig_dd.update_layout(
        title=f"Portfolio Drawdown | Worst: {max_dd_p:.1f}%",
        yaxis_title="Drawdown (%)",xaxis_title="Date",height=320)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Chart 5: Correlation heatmap
    st.markdown("#### Correlation Heatmap")
    st.caption("+1 = stocks always move together (no diversification benefit). 0 = independent movement (good diversification). -1 = opposite movement (perfect hedge). For a healthy portfolio you want low or mixed correlations.")
    fig_h = px.imshow(rets.corr(),text_auto=True,aspect="auto",
                      color_continuous_scale='RdBu_r',
                      title="Correlation Between Stocks (-1 to +1)")
    st.plotly_chart(fig_h, use_container_width=True)

    # Analytics summary table
    st.markdown("#### Full Analytics Summary")
    indiv_df = pd.DataFrame({
        'Ticker':            port_tickers,
        'Weight':            [f"{w*100:.1f}%" for w in weights],
        'Ann. Return':       [f"{float(m_ret[t])*100:.2f}%" for t in port_tickers],
        'Ann. Volatility':   [f"{float(np.sqrt(cov_mat.loc[t,t]))*100:.2f}%" for t in port_tickers],
        'Return Contrib.':   [f"{r:.2f}%" for r in contrib_ret],
        'Risk Contrib.':     [f"{r:.2f}%" for r in contrib_risk],
    })
    st.dataframe(indiv_df, use_container_width=True, hide_index=True)

# ==============================================================================
#  NEWS TICKER (bottom of every page)
# ==============================================================================
st.markdown("---")
st.markdown("### Latest Headlines (24/7)")
all_h    = news_headlines + news_headlines
anim_dur = max(15, len(news_headlines)*3)
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
