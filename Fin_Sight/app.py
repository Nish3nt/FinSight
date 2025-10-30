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

# ====================== INITIAL SETUP ======================
nltk.download('vader_lexicon', quiet=True)
current_date = datetime.now().date()

# App Title
st.set_page_config(page_title="FinSight", layout="wide")
st.title("FinSight: Advanced Stock Analysis Dashboard")

# ====================== SIDEBAR (MUST BE FIRST) ======================
st.sidebar.header("User Input")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'JPM', 'V', 'XOM']
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2018-01-01').date())
end_date = st.sidebar.date_input("End Date", current_date)

# Validate dates
if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()
if end_date > current_date:
    end_date = current_date
    st.warning(f"End date set to today: {current_date}")

# ====================== FETCH TICKER OBJECT ======================
@st.cache_resource(ttl=300)
def get_ticker_obj(ticker):
    return yf.Ticker(ticker)

ticker_obj = get_ticker_obj(selected_ticker)

# ====================== NEWS TICKER (100% SAFE) ======================
st.markdown("""
<style>
.news-ticker {
    background: linear-gradient(90deg, #1e3a8a, #1e40af);
    color: white;
    padding: 12px;
    border-radius: 10px;
    overflow: hidden;
    white-space: nowrap;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    font-weight: 500;
}
.news-ticker:hover .ticker-content {
    animation-play-state: paused;
}
@keyframes scroll {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}
.ticker-content {
    display: inline-block;
    animation: scroll 45s linear infinite;
    padding-left: 100%;
}
</style>
""", unsafe_allow_html=True)

try:
    news_items = ticker_obj.news[:10]
    if news_items and isinstance(news_items, list):
        headlines = []
        for item in news_items:
            title = item.get('title', '').strip()
            publisher = item.get('publisher', 'Unknown').strip()
            if title:
                headlines.append(f"{title} ({publisher})")
        if headlines:
            headline_str = "  •  ".join(headlines)
            st.markdown(f'<div class="news-ticker"><span class="ticker-content">{headline_str}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="news-ticker"><span class="ticker-content">No recent news available.</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="news-ticker"><span class="ticker-content">No news feed available.</span></div>', unsafe_allow_html=True)
except Exception as e:
    st.markdown('<div class="news-ticker"><span class="ticker-content">News temporarily unavailable.</span></div>', unsafe_allow_html=True)

# ====================== FETCH PRICE DATA ======================
@st.cache_data(ttl=600)
def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            st.error(f"No price data for {ticker}.")
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].dropna(how='all')
        if len(df) < 30:
            st.warning("Need at least 30 days of data.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

data = fetch_data(selected_ticker, start_date, end_date)

if data.empty:
    st.stop()

# Full OHLC
full_data = data.copy()

# ====================== SENTIMENT NEWS (SAFE FALLBACK) ======================
@st.cache_data(ttl=300)
def get_sentiment_news(ticker):
    try:
        # Try Alpha Vantage
        api_key = "D8VCWYUPOFJR8D52"
        if api_key != "your_alphavantage_key_here":
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                json_data = resp.json()
                feed = json_data.get('feed', [])
                if feed:
                    posts, links = [], []
                    for art in feed[:10]:
                        title = art.get('title', 'No title')
                        summary = art.get('summary', '')[:120]
                        posts.append(f"{title}: {summary}...")
                        links.append(art.get('url', '#'))
                    return posts, links
        # Fallback to yfinance
        news = ticker_obj.news[:10]
        posts, links = [], []
        for n in news:
            title = n.get('title', 'No title')
            pub = n.get('publisher', 'Unknown')
            posts.append(f"{title} - {pub}")
            links.append(n.get('link', '#'))
        return posts, links
    except:
        return ["News unavailable (check connection or API key)."], ['#']

sample_posts, links = get_sentiment_news(selected_ticker)

# ====================== TABS ======================
tab = option_menu(
    menu_title=None,
    options=["Data & Viz", "Predictions", "Sentiment"],
    icons=["table", "graph-up", "chat-dots"],
    orientation="horizontal"
)

# ====================== DATA & VIZ TAB ======================
if tab == "Data & Viz":
    st.header(f"{selected_ticker} - Historical Data")
    st.dataframe(data.tail(100), use_container_width=True)
    st.download_button("Download CSV", data.to_csv().encode(), f"{selected_ticker}.csv", "text/csv")

    st.subheader("Price Trend")
    fig = px.line(data, x=data.index, y='Adj Close', title=f"{selected_ticker} Adjusted Close")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    try:
        close = data['Close'].iloc[-1]
        change_7d = (close - data['Close'].iloc[-8]) / data['Close'].iloc[-8] * 100 if len(data) > 7 else 0
        vol = data['Close'].pct_change().std() * np.sqrt(252) * 100
        volume = data['Volume'].iloc[-1]
        col1.metric("Price", f"${close:,.2f}")
        col2.metric("7D Change", f"{change_7d:+.2f}%")
        col3.metric("Volatility", f"{vol:.1f}%")
        col4.metric("Volume", f"{volume:,.0f}")
    except:
        col1.metric("Price", "$0.00")

    st.subheader("Candlestick Chart")
    if all(c in data.columns for c in ['Open', 'High', 'Low', 'Close']):
        fig_c = go.Figure(data=[go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close']
        )])
        fig_c.update_layout(title=f"{selected_ticker} OHLC", height=600)
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("OHLC data not available.")

# ====================== PREDICTIONS TAB ======================
elif tab == "Predictions":
    st.header("Price Forecast")
    model = st.selectbox("Model", ["Prophet", "LSTM"])
    days = st.slider("Days Ahead", 1, 30, 7)

    pred_df = pd.DataFrame()

    if model == "Prophet" and len(data) >= 30:
        with st.spinner("Training Prophet..."):
            df_p = data.reset_index()[['Date', 'Adj Close']].rename(columns={'Date': 'ds', 'Adj Close': 'y'})
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            m.fit(df_p)
            future = m.make_future_dataframe(periods=days)
            forecast = m.predict(future)
            pred_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
            pred_df.columns = ['Date', 'Predicted', 'Lower', 'Upper']
            fig = plot_plotly(m, forecast)
            st.plotly_chart(fig, use_container_width=True)

    elif model == "LSTM" and len(data) >= 60:
        with st.spinner("Training LSTM..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data['Adj Close'].values.reshape(-1,1))
            X, y = [], []
            for i in range(60, len(scaled)):
                X.append(scaled[i-60:i, 0])
                y.append(scaled[i, 0])
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], 60, 1))
            lstm = Sequential([LSTM(50, return_sequences=True, input_shape=(60,1)),
                               LSTM(50), Dense(1)])
            lstm.compile(optimizer='adam', loss='mse')
            lstm.fit(X, y, epochs=3, batch_size=32, verbose=0)
            last = scaled[-60:].reshape(1,60,1)
            preds = []
            for _ in range(days):
                p = lstm.predict(last, verbose=0)[0][0]
                preds.append(p)
                last = np.append(last[:,1:,:], [[[p]]], axis=1)
            pred_vals = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            pred_df['Date'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days)
            pred_df['Predicted'] = pred_vals
            fig = px.line(pred_df, x='Date', y='Predicted', title="LSTM Forecast")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Not enough data.")

    if not pred_df.empty:
        st.dataframe(pred_df.style.format({"Predicted": "{:.2f}"}))

# ====================== SENTIMENT TAB ======================
elif tab == "Sentiment":
    st.header("News Sentiment Analysis")
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(post)['compound'] for post in sample_posts]
    df = pd.DataFrame({'News': sample_posts, 'Link': links, 'Score': scores})

    def color(val):
        return f"color: {'green' if val > 0.1 else 'red' if val < -0.1 else 'gray'}"
    st.dataframe(df.style.applymap(color, subset=['Score']).format({'Score': '{:.3f}'}), use_container_width=True)

    pos = sum(1 for s in scores if s > 0.1)
    neg = sum(1 for s in scores if s < -0.1)
    neu = len(scores) - pos - neg
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", pos)
    c2.metric("Negative", neg)
    c3.metric("Neutral", neu)

    st.caption("Sentiment based on recent news. Green = Positive, Red = Negative.")
