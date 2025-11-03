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
from streamlit_option_menu import option_menu
import time

# ====================== INITIAL SETUP ======================
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
current_date = datetime.now().date()

st.set_page_config(page_title="FinSight", layout="wide")
st.title("**FinSight**: Real-Time Stock Intelligence")

# ====================== HERO SECTION (IMPRESSIVE AT FIRST GLANCE) ======================
def hero_globe(selected_ticker):
    # ---- 1. Get market-cap of top 15 tickers (fast cache) ----
    top_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
                   'NFLX', 'AMD', 'JPM', 'V', 'XOM', 'BRK-B', 'UNH', 'MA']
    caps = {}
    for t in top_tickers:
        try:
            info = yf.Ticker(t).info
            caps[t] = info.get('marketCap', 0) or 0
        except:
            caps[t] = 0
    caps[selected_ticker] = caps.get(selected_ticker, 0)

    df = pd.DataFrame({
        'Ticker': list(caps.keys()),
        'MarketCap': list(caps.values())
    })
    df = df[df['MarketCap'] > 0].sort_values('MarketCap', ascending=False)

    # ---- 2. 3-D globe (size = market-cap) ----
    # Earth coordinates (randomly placed on a sphere)
    np.random.seed(42)
    lat = np.random.uniform(-90, 90, len(df))
    lon = np.random.uniform(-180, 180, len(df))
    size = np.sqrt(df['MarketCap']) / 1e5   # scale for visibility

    fig = go.Figure(data=go.Scatter3d(
        x=lon, y=lat, z=np.zeros_like(lon),
        mode='markers+text',
        marker=dict(size=size, color=size, colorscale='Viridis', opacity=0.9,
                    showscale=True, colorbar=dict(title="Market Cap")),
        text=df['Ticker'],
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>Cap: $%{marker.color:,.0f}B"
    ))

    fig.update_layout(
        title="Global Market-Cap Landscape (3-D)",
        scene=dict(
            xaxis_title='Longitude', yaxis_title='Latitude', zaxis_title='',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene_bgcolor="#0f172a"
    )
    return fig

# Show hero only once per ticker change
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None

if st.session_state.last_ticker != selected_ticker:
    st.session_state.last_ticker = selected_ticker
    # Force a tiny pause so the globe renders before other content
    placeholder = st.empty()
    with placeholder.container():
        st.plotly_chart(hero_globe(selected_ticker), use_container_width=True)
    time.sleep(0.8)        # let the animation start
    placeholder.empty()    # remove placeholder – globe stays rendered

# ====================== SIDEBAR ======================
st.sidebar.header("Controls")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
           'NFLX', 'AMD', 'JPM', 'V', 'XOM']
selected_ticker = st.sidebar.selectbox("Main Stock", tickers)
compare_ticker = st.sidebar.selectbox("Compare With", tickers, index=1)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2018-01-01').date())
end_date = st.sidebar.date_input("End Date", current_date)

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()
if end_date > current_date:
    end_date = current_date
    st.warning(f"End date capped at today: {current_date}")

# ====================== FETCH TICKER OBJECT ======================
@st.cache_resource(ttl=300)
def get_ticker(ticker):
    return yf.Ticker(ticker)

ticker_obj = get_ticker(selected_ticker)

# ====================== FETCH NEWS (NewsAPI + Yahoo Fallback) ======================
@st.cache_data(ttl=300)
def get_news(ticker):
    try:
        api_key = st.secrets.get("NEWSAPI_KEY") or "d848a496d874401b9e2129a71adb57ba"
        if api_key and api_key != "YOUR_NEWSAPI_KEY":
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{ticker} stock OR {ticker} earnings OR {ticker} news',
                'sortBy': 'publishedAt',
                'pageSize': 15,
                'language': 'en',
                'apiKey': api_key
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                j = r.json()
                articles = j.get('articles', [])
                if articles:
                    headlines, posts, links = [], [], []
                    for art in articles:
                        title = art.get('title', '').strip()
                        source = art.get('source', {}).get('name', 'Source').strip()
                        url = art.get('url', '#')
                        if title:
                            hl = f"**{title}** – {source}"
                            headlines.append(hl)
                            posts.append(f"{title} – {source}")
                            links.append(url)
                    return headlines, posts, links
    except:
        st.warning("NewsAPI failed. Using Yahoo Finance.")

    # Fallback: Yahoo Finance
    try:
        news = ticker_obj.news
        if not news or len(news) == 0:
            return ["No recent headlines."], ["No recent headlines."], ["#"]
        headlines, posts, links = [], [], []
        for item in news[:15]:
            title = item.get('title', '').strip()
            pub = item.get('publisher', 'Source').strip()
            url = item.get('link', '#')
            if title:
                hl = f"**{title}** – {pub}"
                headlines.append(hl)
                posts.append(f"{title} – {pub}")
                links.append(url)
        return headlines if headlines else ["Market quiet."], posts, links
    except:
        return ["News feed unavailable."], ["News feed unavailable."], ["#"]

news_headlines, news_posts, news_links = get_news(selected_ticker)

# ====================== COMPUTE VADER SENTIMENT ======================
@st.cache_data(ttl=300)
def compute_vader_sentiment(posts):
    return [sia.polarity_scores(post)['compound'] for post in posts]

vader_scores = compute_vader_sentiment(news_posts)

# ====================== FETCH STOCK DATA ======================
@st.cache_data(ttl=600)
def fetch_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            st.error(f"No data for {ticker}.")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        required = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        df = df[required].dropna(how='all')
        if len(df) < 30:
            st.warning("Need 30+ days of data.")
            return None
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

data_main = fetch_stock_data(selected_ticker, start_date, end_date)
data_compare = fetch_stock_data(compare_ticker, start_date, end_date)

if data_main is None or data_compare is None:
    st.stop()

# ====================== KPI BANNER (real-time) ======================
col1, col2, col3, col4 = st.columns(4)
try:
    price = data_main['Close'].iloc[-1]
    change_7d = (price - data_main['Close'].iloc[-8]) / data_main['Close'].iloc[-8] * 100
    vol = data_main['Close'].pct_change().std() * np.sqrt(252) * 100
    market_cap = ticker_obj.info.get('marketCap', 0) / 1e9
    col1.metric("Price", f"${price:,.2f}")
    col2.metric("7D Δ", f"{change_7d:+.2f}%")
    col3.metric("Volatility", f"{vol:.1f}%")
    col4.metric("Market Cap", f"${market_cap:,.1f}B")
except:
    col1.metric("Price", "N/A")

# ====================== TABS ======================
tab = option_menu(
    menu_title=None,
    options=["Data & Viz", "Predictions", "Sentiment", "Comparison"],
    icons=["table", "graph-up", "chat-dots", "arrow-left-right"],
    orientation="horizontal"
)

# ====================== DATA & VIZ ======================
if tab == "Data & Viz":
    st.subheader(f"**{selected_ticker}** – Price History")
    st.dataframe(data_main.tail(100), use_container_width=True)
    st.download_button("Download CSV", data_main.to_csv().encode(), f"{selected_ticker}.csv")

    fig = px.line(data_main, x=data_main.index, y='Adj Close', title="Price Trend")
    st.plotly_chart(fig, use_container_width=True)

    fig_c = go.Figure(go.Candlestick(x=data_main.index,
                                     open=data_main['Open'],
                                     high=data_main['High'],
                                     low=data_main['Low'],
                                     close=data_main['Close']))
    st.plotly_chart(fig_c.update_layout(title="Candlestick", height=600), use_container_width=True)

# ====================== PREDICTIONS ======================
elif tab == "Predictions":
    st.subheader("Price Forecast")
    model = st.selectbox("Model", ["Prophet", "LSTM"])
    days = st.slider("Days", 1, 30, 7)

    if model == "Prophet" and len(data_main) >= 30:
        with st.spinner("Running Prophet..."):
            df_p = data_main.reset_index()[['Date', 'Adj Close']].rename(columns={'Date': 'ds', 'Adj Close': 'y'})
            m = Prophet()
            m.fit(df_p)
            future = m.make_future_dataframe(periods=days)
            forecast = m.predict(future)
            pred = forecast[['ds', 'yhat']].tail(days).rename(columns={'ds': 'Date', 'yhat': 'Predicted'})
            fig = plot_plotly(m, forecast)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pred.style.format({"Predicted": "{:.2f}"}))

    elif model == "LSTM" and len(data_main) >= 60:
        with st.spinner("Training LSTM..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data_main['Adj Close'].values.reshape(-1,1))
            X, y = [], []
            for i in range(60, len(scaled)):
                X.append(scaled[i-60:i, 0])
                y.append(scaled[i, 0])
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], 60, 1))
            lstm = Sequential([LSTM(50, return_sequences=True, input_shape=(60,1)),
                               LSTM(50), Dense(1)])
            lstm.compile('adam', 'mse')
            lstm.fit(X, y, epochs=3, batch_size=32, verbose=0)
            last = scaled[-60:].reshape(1,60,1)
            preds = []
            for _ in range(days):
                p = lstm.predict(last, verbose=0)[0][0]
                preds.append(p)
                last = np.append(last[:,1:,:], [[[p]]], axis=1)
            pred_vals = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            pred_df = pd.DataFrame({'Date': pd.date_range(start=data_main.index[-1]+pd.Timedelta(days=1), periods=days),
                                   'Predicted': pred_vals})
            fig = px.line(pred_df, x='Date', y='Predicted', title="LSTM Forecast")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pred_df.style.format({"Predicted": "{:.2f}"}))

    else:
        st.error("Not enough data.")

# ====================== SENTIMENT ======================
elif tab == "Sentiment":
    st.subheader("News Sentiment")
    df = pd.DataFrame({'News': news_posts, 'Link': news_links, 'Score': vader_scores})

    def color(val):
        return f"color: {'green' if val > 0.1 else 'red' if val < -0.1 else 'gray'}"
    st.dataframe(df.style.applymap(color, subset=['Score']).format({'Score': '{:.3f}'}), use_container_width=True)

    pos = sum(1 for s in vader_scores if s > 0.1)
    neg = sum(1 for s in vader_scores if s < -0.1)
    neu = len(vader_scores) - pos - neg
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", pos)
    c2.metric("Negative", neg)
    c3.metric("Neutral", neu)

    st.caption("Real-time sentiment from news headlines (24/7)")

# ====================== COMPARISON ======================
elif tab == "Comparison":
    st.subheader(f"**{selected_ticker} vs {compare_ticker}**")
    base_main = data_main['Adj Close'].iloc[0]
    base_compare = data_compare['Adj Close'].iloc[0]
    df_main = (data_main['Adj Close'] / base_main - 1) * 100
    df_compare = (data_compare['Adj Close'] / base_compare - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_main.index, y=df_main, name=selected_ticker,
                             line=dict(color='#26A69A')))
    fig.add_trace(go.Scatter(x=data_compare.index, y=df_compare, name=compare_ticker,
                             line=dict(color='#AB47BC')))
    fig.update_layout(title="Performance (%)", height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    ret_main = (data_main['Adj Close'].iloc[-1] / base_main - 1) * 100
    ret_compare = (data_compare['Adj Close'].iloc[-1] / base_compare - 1) * 100
    vol_main = data_main['Adj Close'].pct_change().std() * np.sqrt(252) * 100
    vol_compare = data_compare['Adj Close'].pct_change().std() * np.sqrt(252) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{selected_ticker} Return", f"{ret_main:+.2f}%")
    c2.metric(f"{compare_ticker} Return", f"{ret_compare:+.2f}%")
    c3.metric(f"{selected_ticker} Vol", f"{vol_main:.1f}%")
    c4.metric(f"{compare_ticker} Vol", f"{vol_compare:.1f}%")

# ====================== AUTO-SCROLLING NEWS TICKER (BOTTOM) ======================
st.markdown("---")
st.markdown("### Latest Headlines (24/7)")

all_headlines = news_headlines + news_headlines
animation_duration = max(15, len(news_headlines) * 3)

st.markdown(f"""
<style>
.ticker-container {{
    height: 180px; overflow: hidden; background: #0f172a; padding: 16px;
    border-radius: 14px; box-shadow: 0 6px 24px rgba(0,0,0,0.3);
    color: white; font-family: 'Segoe UI', sans-serif; position: relative;
}}
.ticker-wrapper {{ animation: scroll-up {animation_duration}s linear infinite; }}
@keyframes scroll-up {{ 0% {{ transform: translateY(0); }} 100% {{ transform: translateY(-50%); }} }}
.ticker-item {{ padding: 12px 0; font-size: 15px; line-height: 1.6; min-height: 40px; }}
</style>
""", unsafe_allow_html=True)

html_content = '<div class="ticker-container"><div class="ticker-wrapper">'
for h in all_headlines:
    html_content += f'<div class="ticker-item">{h}</div>'
html_content += '</div></div>'
st.markdown(html_content, unsafe_allow_html=True)
