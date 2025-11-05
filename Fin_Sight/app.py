import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from datetime import datetime
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

# ====================== 503 TICKERS ======================
tickers = ['A','AAPL','ABBV','ABNB','ABT','ACGL','ACN','ADBE','ADI','ADM','ADP','ADSK','AEE','AEP','AES','AFL','AIG','AIZ','AJG','AKAM','ALB','ALGN','ALL','ALLE','AMAT','AMD','AME','AMGN','AMP','AMT','AMZN','ANET','ANSS','AON','AOS','APA','APD','APH','APTV','ARE','ATO','AVB','AVGO','AVY','AWK','AXON','AXP','AZO','BA','BAC','BALL','BAX','BBWI','BBY','BDX','BEN','BF.B','BG','BIIB','BIO','BK','BKNG','BKR','BLDR','BLK','BMY','BR','BRK.B','BRO','BSX','BWA','BX','BXP','C','CAG','CAH','CARR','CAT','CB','CBOE','CBRE','CCI','CCL','CDNS','CDW','CE','CEG','CFG','CHD','CHRW','CHTR','CI','CINF','CL','CLX','CMA','CMCSA','CME','CMG','CMI','CMS','CNC','CNP','COF','COO','COP','COR','COST','CPAY','CPB','CPRT','CRL','CRM','CSCO','CSGP','CSX','CTAS','CTRA','CTSH','CVS','CVX','D','DAL','DASH','DD','DE','DECK','DELL','DFS','DG','DGX','DHI','DHR','DIS','DLR','DLTR','DOC','DOV','DOW','DPZ','DRI','DTE','DUK','DVA','DVN','DXCM','EA','EBAY','ED','EFX','EG','EIX','EL','ELV','EMN','EMR','ENPH','EOG','EPAM','EQIX','EQR','ES','ESS','ETN','ETR','EVRG','EW','EXC','EXPD','EXPE','F','FANG','FAST','FDS','FDX','FE','FFIV','FI','FICO','FIS','FITB','FOX','FOXA','FRT','FSLR','FTNT','FTV','GD','GE','GEHC','GEN','GEV','GILD','GIS','GL','GLW','GM','GNRC','GOOG','GOOGL','GPC','GPN','GRMN','GS','GWW','HAL','HAS','HBAN','HCA','HD','HES','HIG','HII','HLT','HOLX','HON','HPE','HPQ','HRL','HSIC','HST','HSY','HUBB','HUM','HWM','IBM','ICE','IDXX','IEX','IFF','ILMN','INCY','INTC','INTU','INVH','IP','IPG','IQV','IR','IRM','ISRG','IT','ITW','IVZ','J','JBHT','JBL','JCI','JKHY','JNJ','JPM','K','KDP','KEY','KEYS','KHC','KIM','KLAC','KMB','KMI','KO','KR','KVUE','L','LDOS','LEN','LH','LHX','LIN','LKQ','LLY','LMT','LNT','LOW','LRCX','LULU','LUV','LVS','LW','LYB','LYV','MAA','MAR','MAS','MCD','MCHP','MCK','MCO','MDLZ','MDT','MET','META','MGM','MHK','MKC','MLM','MMC','MMM','MNST','MO','MOH','MOS','MPC','MPWR','MRK','MRNA','MS','MSCI','MSFT','MSI','MTB','MTCH','MTD','MU','NCLH','NDAQ','NDSN','NEE','NEM','NFLX','NI','NKE','NOC','NOW','NRG','NSC','NTAP','NTRS','NVDA','NVR','NWSA','NWS','NXPI','O','ODFL','OKE','OMC','ON','ORCL','ORLY','OTIS','OXY','PANW','PAYC','PAYX','PCAR','PCG','PEG','PEP','PFE','PFG','PG','PGR','PH','PHM','PKG','PLD','PLTR','PM','PNC','PNR','PNW','PODD','POOL','PPL','PRU','PSX','PTC','PWR','PYPL','QCOM','REG','REGN','RF','RJF','RL','RMD','ROK','ROL','ROP','ROST','RSG','RTX','RVTY','SBAC','SBUX','SCHW','SHW','SJM','SLB','SMCI','SNA','SNPS','SO','SOLV','SPG','SPGI','SRE','STE','STLD','STT','STX','STZ','SW','SWK','SWKS','SYF','SYK','SYY','T','TAP','TDG','TDY','TECH','TEL','TER','TSLA','TFC','TFX','TGT','TJX','TKO','TMO','TMUS','TPR','TRGP','TRMB','TROW','TRV','TSCO','TSN','TT','TTWO','TXN','TXT','TYL','UAL','UBER','UDR','UHS','ULTA','UNH','UNP','UPS','URI','USB','V','VFC','VICI','VLO','VLTO','VMC','VRSK','VRSN','VRTX','VST','VTR','VZ','WAB','WAT','WBA','WBD','WDC','WEC','WELL','WFC','WM','WMB','WMT','WRB','WST','WTW','WY','WYNN','XEL','XOM','XYL','YUM','ZBH','ZBRA','ZTS']
tickers = sorted(set(tickers))

# ====================== SIDEBAR ======================
st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Main Stock", tickers, index=tickers.index('AAPL'))
compare_ticker = st.sidebar.selectbox("Compare With", tickers, index=tickers.index('MSFT'))
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2018-01-01').date())
end_date = st.sidebar.date_input("End Date", datetime.now().date())

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ====================== ROBUST NEWS FETCHER ======================
@st.cache_data(ttl=180)
def get_news(ticker):
    headlines, posts, links = [], [], []
    try:
        # 1. Yahoo Finance (always works)
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news
        for item in news[:20]:
            title = item.get('title', '').strip()
            pub = item.get('publisher', 'Yahoo Finance').strip()
            url = item.get('link', '#')
            if title:
                hl = f"**{title}** – {pub}"
                headlines.append(hl)
                posts.append(f"{title} – {pub}")
                links.append(url)
    except:
        pass

    # 2. NewsAPI fallback (if you add your key later)
    try:
        api_key = st.secrets.get("NEWSAPI_KEY")
        if api_key:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{ticker} stock',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10,
                'apiKey': api_key
            }
            r = requests.get(url, params=params, timeout=8)
            if r.status_code == 200:
                for art in r.json().get('articles', []):
                    t = art.get('title', '').strip()
                    s = art.get('source', {}).get('name', 'News')
                    u = art.get('url', '#')
                    if t:
                        hl = f"**{t}** – {s}"
                        headlines.append(hl)
                        posts.append(f"{t} – {s}")
                        links.append(u)
    except:
        pass

    # Final fallback
    if not headlines:
        headlines = ["**Market quiet today.** No major headlines."]
        posts = ["Market quiet today."]
        links = ["#"]

    return headlines, posts, links

news_headlines, news_posts, news_links = get_news(selected_ticker)
vader_scores = [sia.polarity_scores(p)['compound'] for p in news_posts]

# ====================== SAFE DATA LOADER ======================
@st.cache_data(ttl=600)
def fetch_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty or len(df) < 30:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df['Adj Close'] = df.get('Adj Close', df['Close'])
    return df[['Open','High','Low','Close','Adj Close','Volume']]

data_main = fetch_stock_data(selected_ticker, start_date, end_date)
data_compare = fetch_stock_data(compare_ticker, start_date, end_date)
if not data_main or not data_compare:
    st.stop()

# ====================== TABS ======================
tab = option_menu(None,
    options=["Data & Viz","Predictions","Sentiment","Comparison","Portfolio Analyzer"],
    icons=["table","graph-up","chat-dots","arrow-left-right","pie-chart"],
    orientation="horizontal"
)

# ====================== DATA & VIZ ======================
if tab == "Data & Viz":
    st.subheader(f"**{selected_ticker}** – Price History")
    st.dataframe(data_main.tail(100))  # Fixed: removed deprecated arg
    st.download_button("Download CSV", data_main.to_csv().encode(), f"{selected_ticker}.csv")
    fig = px.line(data_main, x=data_main.index, y='Adj Close', title="Price Trend")
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    close = data_main['Close'].iloc[-1]
    change = data_main['Close'].pct_change(7).iloc[-1]*100
    vol = data_main['Close'].pct_change().std()*np.sqrt(252)*100
    c1.metric("Price", f"${close:,.2f}")
    c2.metric("7D Δ", f"{change:+.2f}%")
    c3.metric("Volatility", f"{vol:.1f}%")

    st.plotly_chart(go.Figure(go.Candlestick(x=data_main.index,
        open=data_main['Open'], high=data_main['High'],
        low=data_main['Low'], close=data_main['Close']))
        .update_layout(title="Candlestick", height=600), use_container_width=True)

# ====================== PREDICTIONS + LSTM METRICS ======================
elif tab == "Predictions":
    st.subheader("Price Forecast")
    model = st.selectbox("Model", ["Prophet", "LSTM"])
    days = st.slider("Days Ahead", 1, 30, 7)

    if model == "Prophet":
        with st.spinner("Running Prophet..."):
            df_p = data_main.reset_index()[['Date','Adj Close']].rename(columns={'Date':'ds','Adj Close':'y'})
            m = Prophet(); m.fit(df_p)
            future = m.make_future_dataframe(periods=days)
            forecast = m.predict(future)
            st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)

    elif model == "LSTM":
        with st.spinner("Training LSTM (3 epochs)..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data_main['Adj Close'].values.reshape(-1,1))
            train_len = int(len(scaled)*0.8)
            train, test = scaled[:train_len], scaled[train_len:]

            def create_seq(data, seq=60):
                X, y = [], []
                for i in range(seq, len(data)):
                    X.append(data[i-seq:i, 0])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)

            X_train, y_train = create_seq(train)
            X_test, y_test = create_seq(test)
            X_train = X_train.reshape((X_train.shape[0], 60, 1))
            X_test = X_test.reshape((X_test.shape[0], 60, 1))

            start = time.time()
            lstm = Sequential([LSTM(50, return_sequences=True, input_shape=(60,1)),
                               LSTM(50), Dense(1)])
            lstm.compile('adam','mse')
            lstm.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
            train_time = time.time() - start

            pred = lstm.predict(X_test, verbose=0)
            pred = scaler.inverse_transform(pred)
            actual = scaler.inverse_transform(y_test.reshape(-1,1))
            mse = mean_squared_error(actual, pred)
            r2 = r2_score(actual, pred)

            # Future forecast
            last = scaled[-60:].reshape(1,60,1)
            future = []
            for _ in range(days):
                p = lstm.predict(last, verbose=0)[0][0]
                future.append(p)
                last = np.append(last[:,1:,:], [[[p]]], axis=1)
            future = scaler.inverse_transform(np.array(future).reshape(-1,1)).flatten()

            pred_df = pd.DataFrame({
                'Date': pd.date_range(start=data_main.index[-1]+pd.Timedelta(days=1), periods=days),
                'Predicted': future
            })
            st.plotly_chart(px.line(pred_df, x='Date', y='Predicted', title="LSTM Forecast"), use_container_width=True)

            c1,c2,c3 = st.columns(3)
            c1.metric("MSE", f"{mse:.3f}")
            c2.metric("R²", f"{r2:.3f}")
            c3.metric("Training", f"{train_time:.1f}s")

# ====================== SENTIMENT (FIXED) ======================
elif tab == "Sentiment":
    st.subheader("Live News Sentiment")
    df = pd.DataFrame({'Headline': news_posts, 'Link': news_links, 'Score': vader_scores})
    # Fixed: use .map instead of deprecated .applymap
    styled = df.style.map(
        lambda x: f"color: {'lime' if x>0.1 else 'red' if x<-0.1 else 'gray'}",
        subset=['Score']
    ).format({'Score': '{:.3f}'})
    st.dataframe(styled, use_container_width=True)  # Fixed

    pos = sum(1 for s in vader_scores if s > 0.1)
    neg = sum(1 for s in vader_scores if s < -0.1)
    neu = len(vader_scores) - pos - neg
    c1,c2,c3 = st.columns(3)
    c1.metric("Positive", pos, delta=None)
    c2.metric("Negative", neg)
    c3.metric("Neutral", neu)

# ====================== COMPARISON & PORTFOLIO ======================
elif tab == "Comparison":
    st.subheader(f"{selected_ticker} vs {compare_ticker}")
    perf1 = (data_main['Adj Close']/data_main['Adj Close'].iloc[0] - 1)*100
    perf2 = (data_compare['Adj Close']/data_compare['Adj Close'].iloc[0] - 1)*100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_main.index, y=perf1, name=selected_ticker, line=dict(color='#26A69A')))
    fig.add_trace(go.Scatter(x=data_compare.index, y=perf2, name=compare_ticker, line=dict(color='#AB47BC')))
    fig.update_layout(title="Performance (%)", height=600)
    st.plotly_chart(fig, use_container_width=True)

elif tab == "Portfolio Analyzer":
    st.subheader("Build Your Portfolio")
    picks = st.multiselect("Add stocks", tickers, default=[selected_ticker, compare_ticker])
    if len(picks) >= 2:
        cols = st.columns(len(picks))
        weights = [cols[i].number_input(f"{t} %", 0, 100, int(100/len(picks)), key=t)/100 for i,t in enumerate(picks)]
        if abs(sum(weights)-1) < 0.02:
            dfs = [fetch_stock_data(t, start_date, end_date)['Adj Close'] for t in picks]
            rets = pd.concat(dfs, axis=1).pct_change().dropna()
            rets.columns = picks
            ann_ret = rets.mean()*252
            cov = rets.cov()*252
            port_ret = np.dot(ann_ret, weights)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
            sharpe = (port_ret - 0.03)/port_vol

            c1,c2,c3 = st.columns(3)
            c1.metric("Return", f"{port_ret*100:.1f}%")
            c2.metric("Risk", f"{port_vol*100:.1f}%")
            c3.metric("Sharpe", f"{sharpe:.2f}")

            st.plotly_chart(px.imshow(rets.corr(), title="Correlation Heatmap",
                                    color_continuous_scale='RdBu_r', text_auto=True),
                           use_container_width=True)

# ====================== NEWS TICKER (ALWAYS SHOWS) ======================
st.markdown("---")
st.markdown("### Latest Headlines (24/7)")
all_lines = news_headlines + news_headlines
duration = max(20, len(news_headlines)*3)

st.markdown(f"""
<style>
.ticker-box {{height:180px;overflow:hidden;background:#0f172a;padding:16px;border-radius:14px;color:white;}}
.ticker {{animation:scroll {duration}s linear infinite;}}
@keyframes scroll {{0%{{transform:translateY(0)}}100%{{transform:translateY(-50%)}}}}
.item {{padding:10px 0;font-size:15px;}}
</style>
<div class="ticker-box"><div class="ticker">
{"".join(f'<div class="item">{h}</div>' for h in all_lines)}
</div></div>
""", unsafe_allow_html=True)
