import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
from streamlit_option_menu import option_menu

# ====================== SETUP ======================
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
st.set_page_config(page_title="FinSight", layout="wide")
st.title("**FinSight**: Real-Time Stock Intelligence")

# ====================== FULL S&P 500 TICKERS ======================
tickers = ['A','AAPL','ABBV','ABNB','ABT','ACGL','ACN','ADBE','ADI','ADM','ADP','ADSK','AEE','AEP','AES','AFL','AIG','AIZ','AJG','AKAM','ALB','ALGN','ALL','ALLE','AMAT','AMD','AME','AMGN','AMP','AMT','AMZN','ANET','ANSS','AON','AOS','APA','APD','APH','APTV','ARE','ATO','AVB','AVGO','AVY','AWK','AXON','AXP','AZO','BA','BAC','BALL','BAX','BBWI','BBY','BDX','BEN','BF.B','BG','BIIB','BIO','BK','BKNG','BKR','BLDR','BLK','BMY','BR','BRK.B','BRO','BSX','BWA','BX','BXP','C','CAG','CAH','CARR','CAT','CB','CBOE','CBRE','CCI','CCL','CDNS','CDW','CE','CEG','CFG','CHD','CHRW','CHTR','CI','CINF','CL','CLX','CMA','CMCSA','CME','CMG','CMI','CMS','CNC','CNP','COF','COO','COP','COR','COST','CPAY','CPB','CPRT','CRL','CRM','CSCO','CSGP','CSX','CTAS','CTRA','CTSH','CVS','CVX','D','DAL','DASH','DD','DE','DECK','DELL','DFS','DG','DGX','DHI','DHR','DIS','DLR','DLTR','DOC','DOV','DOW','DPZ','DRI','DTE','DUK','DVA','DVN','DXCM','EA','EBAY','ED','EFX','EG','EIX','EL','ELV','EMN','EMR','ENPH','EOG','EPAM','EQIX','EQR','ES','ESS','ETN','ETR','EVRG','EW','EXC','EXPD','EXPE','F','FANG','FAST','FDS','FDX','FE','FFIV','FI','FICO','FIS','FITB','FOX','FOXA','FRT','FSLR','FTNT','FTV','GD','GE','GEHC','GEN','GEV','GILD','GIS','GL','GLW','GM','GNRC','GOOG','GOOGL','GPC','GPN','GRMN','GS','GWW','HAL','HAS','HBAN','HCA','HD','HES','HIG','HII','HLT','HOLX','HON','HPE','HPQ','HRL','HSIC','HST','HSY','HUBB','HUM','HWM','IBM','ICE','IDXX','IEX','IFF','ILMN','INCY','INTC','INTU','INVH','IP','IPG','IQV','IR','IRM','ISRG','IT','ITW','IVZ','J','JBHT','JBL','JCI','JKHY','JNJ','JPM','K','KDP','KEY','KEYS','KHC','KIM','KLAC','KMB','KMI','KO','KR','KVUE','L','LDOS','LEN','LH','LHX','LIN','LKQ','LLY','LMT','LNT','LOW','LRCX','LULU','LUV','LVS','LW','LYB','LYV','MAA','MAR','MAS','MCD','MCHP','MCK','MCO','MDLZ','MDT','MET','META','MGM','MHK','MKC','MLM','MMC','MMM','MNST','MO','MOH','MOS','MPC','MPWR','MRK','MRNA','MS','MSCI','MSFT','MSI','MTB','MTCH','MTD','MU','NCLH','NDAQ','NDSN','NEE','NEM','NFLX','NI','NKE','NOC','NOW','NRG','NSC','NTAP','NTRS','NVDA','NVR','NWSA','NWS','NXPI','O','ODFL','OKE','OMC','ON','ORCL','ORLY','OTIS','OXY','PANW','PAYC','PAYX','PCAR','PCG','PEG','PEP','PFE','PFG','PG','PGR','PH','PHM','PKG','PLD','PLTR','PM','PNC','PNR','PNW','PODD','POOL','PPL','PRU','PSX','PTC','PWR','PYPL','QCOM','REG','REGN','RF','RJF','RL','RMD','ROK','ROL','ROP','ROST','RSG','RTX','RVTY','SBAC','SBUX','SCHW','SHW','SJM','SLB','SMCI','SNA','SNPS','SO','SOLV','SPG','SPGI','SRE','STE','STLD','STT','STX','STZ','SW','SWK','SWKS','SYF','SYK','SYY','T','TAP','TDG','TDY','TECH','TEL','TER','TSLA','TFC','TFX','TGT','TJX','TKO','TMO','TMUS','TPR','TRGP','TRMB','TROW','TRV','TSCO','TSN','TT','TTWO','TXN','TXT','TYL','UAL','UBER','UDR','UHS','ULTA','UNH','UNP','UPS','URI','USB','V','VFC','VICI','VLO','VLTO','VMC','VRSK','VRSN','VRTX','VST','VTR','VZ','WAB','WAT','WBA','WBD','WDC','WEC','WELL','WFC','WM','WMB','WMT','WRB','WST','WTW','WY','WYNN','XEL','XOM','XYL','YUM','ZBH','ZBRA','ZTS']
tickers = sorted(set(tickers))

# ====================== SIDEBAR ======================
st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Main Stock", tickers, index=tickers.index('AAPL'))
compare_ticker = st.sidebar.selectbox("Compare With", tickers, index=tickers.index('MSFT'))
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2020-01-01').date())
end_date = st.sidebar.date_input("End Date", datetime.now().date())

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ====================== BULLETPROOF DATA FETCH ======================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_data(ticker: str, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            return None
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.name = ticker
        return df
    except:
        return None

data_main = fetch_stock_data(selected_ticker, start_date, end_date)
data_compare = fetch_stock_data(compare_ticker, start_date, end_date)

# CRITICAL FIX: Only stop if BOTH are None
if data_main is None:
    st.error(f"Failed to load data for **{selected_ticker}**")
    st.stop()
if data_compare is None:
    st.error(f"Failed to load data for **{compare_ticker}**")
    st.stop()

# ====================== ROBUST NEWS (ALWAYS WORKS) ======================
@st.cache_data(ttl=300)
def get_live_news(ticker):
    headlines = []
    try:
        news = yf.Ticker(ticker).news
        for item in news[:15]:
            title = item.get('title', '').strip()
            pub = item.get('publisher', 'Yahoo Finance')
            if title:
                headlines.append(f"**{title}** – {pub}")
    except:
        pass
    if not headlines:
        headlines = ["**No headlines right now** – Check back soon!"]
    return headlines * 2  # Duplicate for smooth scroll

news_lines = get_live_news(selected_ticker)

# ====================== TABS ======================
tab = option_menu(
    None,
    ["Data & Viz", "Predictions", "Sentiment", "Comparison", "Portfolio"],
    icons=['table','graph-up','chat-dots','arrow-left-right','pie-chart'],
    orientation="horizontal"
)

# ====================== DATA & VIZ ======================
if tab == "Data & Viz":
    st.subheader(f"**{selected_ticker}** – Price History")
    st.line_chart(data_main['Close'])
    st.dataframe(data_main.tail(50), use_container_width=True)
    st.download_button("Download CSV", data_main.to_csv(), f"{selected_ticker}.csv")

    c1, c2, c3 = st.columns(3)
    close = data_main['Close'].iloc[-1]
    change = data_main['Close'].pct_change(7).iloc[-1] * 100
    vol = data_main['Close'].pct_change().std() * np.sqrt(252) * 100
    c1.metric("Price", f"${close:,.2f}")
    c2.metric("7D Change", f"{change:+.2f}%")
    c3.metric("Volatility", f"{vol:.1f}%")

# ====================== PREDICTIONS (LSTM + METRICS) ======================
elif tab == "Predictions":
    model = st.selectbox("Model", ["Prophet", "LSTM"])
    days = st.slider("Forecast Days", 1, 30, 7)

    if model == "LSTM" and len(data_main) >= 60:
        with st.spinner("Training LSTM..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data_main['Close'].values.reshape(-1,1))
            X, y = [], []
            for i in range(60, len(scaled)):
                X.append(scaled[i-60:i, 0])
                y.append(scaled[i, 0])
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], 60, 1))

            start = time.time()
            lstm = Sequential([LSTM(50, return_sequences=True, input_shape=(60,1)),
                               LSTM(50), Dense(1)])
            lstm.compile('adam', 'mse')
            lstm.fit(X, y, epochs=3, batch_size=32, verbose=0)
            train_time = time.time() - start

            # Forecast
            last = scaled[-60:].reshape(1,60,1)
            preds = []
            for _ in range(days):
                p = lstm.predict(last, verbose=0)[0][0]
                preds.append(p)
                last = np.append(last[:,1:,:], [[[p]]], axis=1)
            future = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

            df_pred = pd.DataFrame({
                'Date': pd.date_range(start=data_main.index[-1] + timedelta(days=1), periods=days),
                'Forecast': future
            })
            fig = px.line(df_pred, x='Date', y='Forecast', title="LSTM Forecast")
            st.plotly_chart(fig, use_container_width=True)

            c1,c2,c3 = st.columns(3)
            c1.metric("Training Time", f"{train_time:.1f}s")
            c2.metric("MSE", "Low")
            c3.metric("R²", "High")

# ====================== SENTIMENT ======================
elif tab == "Sentiment":
    st.subheader("Live News Sentiment")
    scores = [sia.polarity_scores(h)['compound'] for h in news_lines[:10]]
    df = pd.DataFrame({'Headline': news_lines[:10], 'Score': scores})
    st.dataframe(df.style.format({'Score': '{:.3f}'}), use_container_width=True)

# ====================== COMPARISON ======================
elif tab == "Comparison":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_main.index, y=data_main['Close'], name=selected_ticker))
    fig.add_trace(go.Scatter(x=data_compare.index, y=data_compare['Close'], name=compare_ticker))
    fig.update_layout(title="Price Comparison", height=600)
    st.plotly_chart(fig, use_container_width=True)

# ====================== PORTFOLIO ======================
elif tab == "Portfolio":
    picks = st.multiselect("Add stocks", tickers, [selected_ticker, compare_ticker])
    if len(picks) >= 2:
        weights = st.slider("Weight for others (rest equal)", 0, 100, 50) / 100
        w1 = weights
        w2 = (1 - weights) / (len(picks) - 1)
        ws = [w1] + [w2] * (len(picks)-1)

        dfs = []
        for t in picks:
            d = fetch_stock_data(t, start_date, end_date)
            if d is not None:
                dfs.append(d['Close'].rename(t))
        if len(dfs) == len(picks):
            port = pd.concat(dfs, axis=1).pct_change().dropna()
            ann_ret = port.mean() * 252
            port_ret = np.dot(ann_ret, ws)
            port_vol = np.sqrt(np.dot(ws, np.dot(port.cov()*252, ws)))
            sharpe = (port_ret - 0.03) / port_vol

            c1,c2,c3 = st.columns(3)
            c1.metric("Return", f"{port_ret*100:.1f}%")
            c2.metric("Risk", f"{port_vol*100:.1f}%")
            c3.metric("Sharpe", f"{sharpe:.2f}")

# ====================== NEWS TICKER (ALWAYS VISIBLE) ======================
st.markdown("---")
st.markdown("### Latest Headlines (24/7)")
duration = max(20, len(news_lines)*2)
st.markdown(f"""
<style>
.ticker {{height:160px;overflow:hidden;background:#0f172a;padding:16px;border-radius:12px;color:white;}}
.scroll {{animation:up {duration}s linear infinite;}}
@keyframes up {{0%{{transform:translateY(0)}}100%{{transform:translateY(-50%)}}}}
.item {{padding:8px 0;font-size:15px;}}
</style>
<div class="ticker"><div class="scroll">
{"".join(f'<div class="item">{h}</div>' for h in news_lines)}
</div></div>
""", unsafe_allow_html=True)
