# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import time
from streamlit_option_menu import option_menu

# ==================== INIT ====================
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
st.set_page_config(page_title="FinSight", layout="wide")
st.title("**FinSight**: Real-Time Stock Intelligence")

# ==================== TICKERS ====================
tickers = ['A','AAPL','ABBV','ABNB','ABT','ACGL','ACN','ADBE','ADI','ADM','ADP','ADSK','AEE','AEP','AES','AFL','AIG','AIZ','AJG','AKAM','ALB','ALGN','ALL','ALLE','AMAT','AMD','AME','AMGN','AMP','AMT','AMZN','ANET','ANSS','AON','AOS','APA','APD','APH','APTV','ARE','ATO','AVB','AVGO','AVY','AWK','AXON','AXP','AZO','BA','BAC','BALL','BAX','BBWI','BBY','BDX','BEN','BF.B','BG','BIIB','BIO','BK','BKNG','BKR','BLDR','BLK','BMY','BR','BRK.B','BRO','BSX','BWA','BX','BXP','C','CAG','CAH','CARR','CAT','CB','CBOE','CBRE','CCI','CCL','CDNS','CDW','CE','CEG','CFG','CHD','CHRW','CHTR','CI','CINF','CL','CLX','CMA','CMCSA','CME','CMG','CMI','CMS','CNC','CNP','COF','COO','COP','COR','COST','CPAY','CPB','CPRT','CRL','CRM','CSCO','CSGP','CSX','CTAS','CTRA','CTSH','CVS','CVX','D','DAL','DASH','DD','DE','DECK','DELL','DFS','DG','DGX','DHI','DHR','DIS','DLR','DLTR','DOC','DOV','DOW','DPZ','DRI','DTE','DUK','DVA','DVN','DXCM','EA','EBAY','ED','EFX','EG','EIX','EL','ELV','EMN','EMR','ENPH','EOG','EPAM','EQIX','EQR','ES','ESS','ETN','ETR','EVRG','EW','EXC','EXPD','EXPE','F','FANG','FAST','FDS','FDX','FE','FFIV','FI','FICO','FIS','FITB','FOX','FOXA','FRT','FSLR','FTNT','FTV','GD','GE','GEHC','GEN','GEV','GILD','GIS','GL','GLW','GM','GNRC','GOOG','GOOGL','GPC','GPN','GRMN','GS','GWW','HAL','HAS','HBAN','HCA','HD','HES','HIG','HII','HLT','HOLX','HON','HPE','HPQ','HRL','HSIC','HST','HSY','HUBB','HUM','HWM','IBM','ICE','IDXX','IEX','IFF','ILMN','INCY','INTC','INTU','INVH','IP','IPG','IQV','IR','IRM','ISRG','IT','ITW','IVZ','J','JBHT','JBL','JCI','JKHY','JNJ','JPM','K','KDP','KEY','KEYS','KHC','KIM','KLAC','KMB','KMI','KO','KR','KVUE','L','LDOS','LEN','LH','LHX','LIN','LKQ','LLY','LMT','LNT','LOW','LRCX','LULU','LUV','LVS','LW','LYB','LYV','MAA','MAR','MAS','MCD','MCHP','MCK','MCO','MDLZ','MDT','MET','META','MGM','MHK','MKC','MLM','MMC','MMM','MNST','MO','MOH','MOS','MPC','MPWR','MRK','MRNA','MS','MSCI','MSFT','MSI','MTB','MTCH','MTD','MU','NCLH','NDAQ','NDSN','NEE','NEM','NFLX','NI','NKE','NOC','NOW','NRG','NSC','NTAP','NTRS','NVDA','NVR','NWSA','NWS','NXPI','O','ODFL','OKE','OMC','ON','ORCL','ORLY','OTIS','OXY','PANW','PAYC','PAYX','PCAR','PCG','PEG','PEP','PFE','PFG','PG','PGR','PH','PHM','PKG','PLD','PLTR','PM','PNC','PNR','PNW','PODD','POOL','PPL','PRU','PSX','PTC','PWR','PYPL','QCOM','REG','REGN','RF','RJF','RL','RMD','ROK','ROL','ROP','ROST','RSG','RTX','RVTY','SBAC','SBUX','SCHW','SHW','SJM','SLB','SMCI','SNA','SNPS','SO','SOLV','SPG','SPGI','SRE','STE','STLD','STT','STX','STZ','SW','SWK','SWKS','SYF','SYK','SYY','T','TAP','TDG','TDY','TECH','TEL','TER','TSLA','TFC','TFX','TGT','TJX','TKO','TMO','TMUS','TPR','TRGP','TRMB','TROW','TRV','TSCO','TSN','TT','TTWO','TXN','TXT','TYL','UAL','UBER','UDR','UHS','ULTA','UNH','UNP','UPS','URI','USB','V','VFC','VICI','VLO','VLTO','VMC','VRSK','VRSN','VRTX','VST','VTR','VZ','WAB','WAT','WBA','WBD','WDC','WEC','WELL','WFC','WM','WMB','WMT','WRB','WST','WTW','WY','WYNN','XEL','XOM','XYL','YUM','ZBH','ZBRA','ZTS']
tickers = sorted(set(tickers))

# ==================== SIDEBAR ====================
st.sidebar.header("Controls")
selected = st.sidebar.selectbox("Main Stock", tickers, index=tickers.index('AAPL'))
compare = st.sidebar.selectbox("Compare With", tickers, index=tickers.index('MSFT'))
start = st.sidebar.date_input("Start", pd.to_datetime("2020-01-01").date())
end = st.sidebar.date_input("End", datetime.now().date())

if start >= end:
    st.error("Start date must be before end date.")
    st.stop()

# ==================== DATA (BULLETPROOF) ====================
@st.cache_data(ttl=600)
def get_data(ticker):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty or len(df) < 20:
        return None
    df = df[['Open','High','Low','Close','Volume']].copy()
    df.index = pd.to_datetime(df.index).date
    return df

df1 = get_data(selected)
df2 = get_data(compare)

if df1 is None:
    st.error(f"Cannot load **{selected}**")
    st.stop()
if df2 is None:
    st.error(f"Cannot load **{compare}**")
    st.stop()

# ==================== REAL NEWS (ALWAYS WORKS) ====================
@st.cache_data(ttl=180)
def fetch_news(ticker):
    try:
        news = yf.Ticker(ticker).news
        lines = []
        for item in news[:12]:
            t = item.get('title', '').strip()
            p = item.get('publisher', 'Yahoo')
            if t:
                lines.append(f"**{t}** – {p}")
        return lines or ["**Market quiet** – No major headlines."]
    except:
        return ["**Market quiet** – Try again soon."]

news = fetch_news(selected)
news_scroll = news + news  # Smooth scroll

# ==================== TABS ====================
tab = option_menu(None, ["Data & Viz","Predictions","Sentiment","Comparison","Portfolio"],
                  icons=['table','graph-up','chat','exchange','pie-chart'],
                  orientation="horizontal")

# ==================== DATA & VIZ ====================
if tab == "Data & Viz":
    st.subheader(f"**{selected}** – Price History")
    st.line_chart(df1['Close'])
    st.dataframe(df1.tail(30), use_container_width=True)
    st.download_button("Download CSV", df1.to_csv(), f"{selected}.csv")

    close = float(df1['Close'].iloc[-1])
    change = float(df1['Close'].pct_change(7).iloc[-1] * 100)
    vol = float(df1['Close'].pct_change().std() * np.sqrt(252) * 100)

    c1,c2,c3 = st.columns(3)
    c1.metric("Price", f"${close:,.2f}")
    c2.metric("7D Change", f"{change:+.2f}%")
    c3.metric("Volatility", f"{vol:.1f}%")

    st.plotly_chart(go.Figure(go.Candlestick(
        x=df1.index, open=df1['Open'], high=df1['High'],
        low=df1['Low'], close=df1['Close']
    )).update_layout(title="Candlestick", height=500), use_container_width=True)

# ==================== SENTIMENT ====================
elif tab == "Sentiment":
    st.subheader("Live News Sentiment")
    scores = [sia.polarity_scores(h)['compound'] for h in news[:10]]
    sentiment_df = pd.DataFrame({
        "Headline": news[:10],
        "Score": scores
    })
    st.dataframe(
        sentiment_df.style.format({"Score": "{:.3f}"})
        .map(lambda x: f"color: {'lime' if x>0.1 else 'red' if x<-0.1 else 'gray'}", subset=['Score']),
        use_container_width=True
    )

    pos = sum(1 for s in scores if s > 0.1)
    neg = sum(1 for s in scores if s < -0.1)
    neu = len(scores) - pos - neg
    c1,c2,c3 = st.columns(3)
    c1.metric("Positive", pos)
    c2.metric("Negative", neg)
    c3.metric("Neutral", neu)

# ==================== COMPARISON ====================
elif tab == "Comparison":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1.index, y=df1['Close'], name=selected, line=dict(color="#26A69A")))
    fig.add_trace(go.Scatter(x=df2.index, y=df2['Close'], name=compare, line=dict(color="#AB47BC")))
    fig.update_layout(title="Price Comparison", height=600)
    st.plotly_chart(fig, use_container_width=True)

# ==================== PORTFOLIO ====================
elif tab == "Portfolio":
    picks = st.multiselect("Stocks", tickers, [selected, compare])
    if len(picks) >= 2:
        weights = []
        cols = st.columns(len(picks))
        for i, t in enumerate(picks):
            w = cols[i].number_input(f"{t} %", 0, 100, int(100/len(picks)))
            weights.append(w/100)
        if abs(sum(weights)-1) > 0.01:
            st.warning("Weights must sum to 100%")
        else:
            dfs = [get_data(t)['Close'] for t in picks if get_data(t) is not None]
            if len(dfs) == len(picks):
                rets = pd.concat(dfs, axis=1).pct_change().dropna()
                ann_ret = rets.mean() * 252
                cov = rets.cov() * 252
                port_ret = np.dot(ann_ret, weights)
                port_vol = np.sqrt(np.dot(np.array(weights), np.dot(cov, weights)))
                sharpe = (port_ret - 0.03) / port_vol
                c1,c2,c3 = st.columns(3)
                c1.metric("Return", f"{port_ret*100:.1f}%")
                c2.metric("Risk", f"{port_vol*100:.1f}%")
                c3.metric("Sharpe", f"{sharpe:.2f}")

# ==================== NEWS TICKER (BOTTOM) ====================
st.markdown("---")
st.markdown("### Latest Headlines (24/7)")
duration = max(20, len(news_scroll)*2)
st.markdown(f"""
<style>
.ticker {{height:160px;overflow:hidden;background:#0f172a;padding:12px;border-radius:12px;color:white;font-size:15px;}}
.scroll {{animation:up {duration}s linear infinite;}}
@keyframes up {{0%{{transform:translateY(0)}}100%{{transform:translateY(-50%)}}}}
.item {{padding:6px 0;}}
</style>
<div class="ticker"><div class="scroll">
{"".join(f'<div class="item">{h}</div>' for h in news_scroll)}
</div></div>
""", unsafe_allow_html=True)
