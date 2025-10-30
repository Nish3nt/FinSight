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

# ====================== SETUP ======================
nltk.download('vader_lexicon', quiet=True)
current_date = datetime.now().date()

st.set_page_config(page_title="FinSight", layout="wide")
st.title("**FinSight**: Real-Time Stock Intelligence")

# ====================== SIDEBAR ======================
st.sidebar.header("Controls")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'JPM', 'V', 'XOM']
selected_ticker = st.sidebar.selectbox("Stock Ticker", tickers)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2018-01-01').date())
end_date = st.sidebar.date_input("End Date", current_date)

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()
if end_date > current_date:
    end_date = current_date
    st.warning(f"End date set to today: {current_date}")

# ====================== TICKER & NEWS ======================
@st.cache_resource(ttl=300)
def get_ticker(ticker):
    return yf.Ticker(ticker)

ticker_obj = get_ticker(selected_ticker)

# Fetch news safely
def get_news():
    try:
        news = ticker_obj.news
        if not news or len(news) == 0:
            return ["No news available at this time."]
        headlines = []
        for item in news[:15]:  # Top 15
            title = item.get('title', '').strip()
            pub = item.get('publisher', 'Source').strip()
            if title:
                headlines.append(f"**{title}** – {pub}")
        return headlines if headlines else ["Market quiet. No major headlines."]
    except:
        return ["News feed temporarily down."]
news_headlines = get_news()

# ====================== DATA FETCH (BULLETPROOF) ======================
@st.cache_data(ttl=600)
def fetch_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            st.error("No data found.")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        required = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        df = df[required].dropna(how='all')
        if len(df) < 30:
            st.warning("Need 30+ days.")
            return None
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

data = fetch_stock_data(selected_ticker, start_date, end_date)
if data is None:
    st.stop()

full_data = data.copy()

# ====================== SENTIMENT ======================
@st.cache_data(ttl=300)
def get_sentiment_news():
    try:
        news = ticker_obj.news[:10]
        posts = [f"{n.get('title','News')} – {n.get('publisher','Source')}" for n in news]
        links = [n.get('link','#') for n in news]
        return posts, links
    except:
        return ["News unavailable."], ['#']

sample_posts, links = get_sentiment_news()

# ====================== TABS ======================
tab = option_menu(
    menu_title=None,
    options=["Data & Viz", "Predictions", "Sentiment"],
    icons=["table", "graph-up", "chat-dots"],
    orientation="horizontal"
)

# ====================== MAIN CONTENT ======================
if tab == "Data & Viz":
    st.subheader(f"**{selected_ticker}** – Price History")
    st.dataframe(data.tail(100), use_container_width=True)
    st.download_button("Download CSV", data.to_csv().encode(), f"{selected_ticker}.csv")

    st.plotly_chart(px.line(data, x=data.index, y='Adj Close', title="Price Trend"), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    try:
        close = data['Close'].iloc[-1]
        change = (close - data['Close'].iloc[-8]) / data['Close'].iloc[-8] * 100 if len(data) > 7 else 0
        vol = data['Close'].pct_change().std() * np.sqrt(252) * 100
        c1.metric("Price", f"${close:,.2f}")
        c2.metric("7D Change", f"{change:+.2f}%")
        c3.metric("Volatility", f"{vol:.1f}%")
    except:
        c1.metric("Price", "N/A")

    fig_c = go.Figure(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']))
    st.plotly_chart(fig_c.update_layout(title="Candlestick", height=600), use_container_width=True)

elif tab == "Predictions":
    st.subheader("Price Forecast")
    model = st.selectbox("Model", ["Prophet", "LSTM"])
    days = st.slider("Days", 1, 30, 7)

    if model == "Prophet" and len(data) >= 30:
        with st.spinner("Running Prophet..."):
            df_p = data.reset_index()[['Date', 'Adj Close']].rename(columns={'Date': 'ds', 'Adj Close': 'y'})
            m = Prophet()
            m.fit(df_p)
            future = m.make_future_dataframe(periods=days)
            forecast = m.predict(future)
            pred = forecast[['ds', 'yhat']].tail(days)
            pred.columns = ['Date', 'Predicted']
            fig = plot_plotly(m, forecast)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pred.style.format({"Predicted": "{:.2f}"}))

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
            lstm = Sequential([LSTM(50, return_sequences=True, input_shape=(60,1)), LSTM(50), Dense(1)])
            lstm.compile('adam', 'mse')
            lstm.fit(X, y, epochs=3, batch_size=32, verbose=0)
            last = scaled[-60:].reshape(1,60,1)
            preds = []
            for _ in range(days):
                p = lstm.predict(last, verbose=0)[0][0]
                preds.append(p)
                last = np.append(last[:,1:,:], [[[p]]], axis=1)
            pred_vals = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            pred_df = pd.DataFrame({'Date': pd.date_range(start=data.index[-1]+pd.Timedelta(days=1), periods=days), 'Predicted': pred_vals})
            st.plotly_chart(px.line(pred_df, x='Date', y='Predicted', title="LSTM Forecast"), use_container_width=True)
            st.dataframe(pred_df.style.format({"Predicted": "{:.2f}"}))

    else:
        st.error("Not enough data.")

elif tab == "Sentiment":
    st.subheader("News Sentiment")
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(p)['compound'] for p in sample_posts]
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

# ====================== VERTICAL NEWS TICKER (BOTTOM) ======================
st.markdown("---")
st.markdown("### Latest Market Headlines")

# CSS for vertical scroll
st.markdown("""
<style>
.news-container {
    height: 220px;
    overflow: hidden;
    background: #0f172a;
    padding: 12px;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
.news-scroll {
    animation: scroll-up 25s linear infinite;
}
@keyframes scroll-up {
    0% { transform: translateY(0); }
    100% { transform: translateY(-100%); }
}
.news-item {
    padding: 8px 0;
    border-bottom: 1px solid #334155;
    font-size: 14px;
    line-height: 1.5;
}
.news-item:last-child {
    border-bottom: none;
}
</style>
""", unsafe_allow_html=True)

# Duplicate headlines for seamless loop
all_headlines = news_headlines + news_headlines

with st.container():
    st.markdown('<div class="news-container">', unsafe_allow_html=True)
    st.markdown('<div class="news-scroll">', unsafe_allow_html=True)
    for h in all_headlines:
        st.markdown(f'<div class="news-item">{h}</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

st.caption("News sourced from Yahoo Finance • Updates every 5 minutes")
