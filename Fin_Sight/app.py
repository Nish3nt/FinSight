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

st.set_page_config(page_title="FinSight", layout="wide")
st.title("**FinSight**: Advanced Stock Analysis Dashboard")

# ====================== SIDEBAR (FIRST) ======================
st.sidebar.header("User Input")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'JPM', 'V', 'XOM']
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2018-01-01').date())
end_date = st.sidebar.date_input("End Date", current_date)

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()
if end_date > current_date:
    end_date = current_date
    st.warning(f"End date capped at today: {current_date}")

# ====================== SAFE TICKER OBJECT ======================
@st.cache_resource(ttl=300)
def get_ticker(ticker):
    return yf.Ticker(ticker)

ticker_obj = get_ticker(selected_ticker)

# ====================== NEWS TICKER (SAFE) ======================
st.markdown("""
<style>
.news-ticker {
    background: linear-gradient(90deg, #1e40af, #3b82f6);
    color: white;
    padding: 14px;
    border-radius: 12px;
    overflow: hidden;
    white-space: nowrap;
    margin: 15px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    font-size: 15px;
    font-weight: 500;
}
.news-ticker:hover .ticker-content { animation-play-state: paused; }
@keyframes scroll {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}
.ticker-content {
    display: inline-block;
    animation: scroll 50s linear infinite;
    padding-left: 100%;
}
</style>
""", unsafe_allow_html=True)

try:
    news = ticker_obj.news[:10]
    headlines = []
    for item in news:
        title = item.get('title', '').strip()
        pub = item.get('publisher', 'Source').strip()
        if title:
            headlines.append(f"{title} ({pub})")
    text = "  •  ".join(headlines) if headlines else "No recent news."
    st.markdown(f'<div class="news-ticker"><span class="ticker-content">{text}</span></div>', unsafe_allow_html=True)
except:
    st.markdown('<div class="news-ticker"><span class="ticker-content">News feed unavailable.</span></div>', unsafe_allow_html=True)

# ====================== BULLETPROOF DATA FETCH ======================
@st.cache_data(ttl=600)
def fetch_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            st.error(f"No data for {ticker}. Try a different ticker.")
            return None

        # --- STEP 1: Flatten MultiIndex ---
        if isinstance(df.columns, pd.MultiIndex):
            # Case: ('Close', 'AAPL')
            df.columns = df.columns.droplevel(1)  # Remove ticker level

        # --- STEP 2: Ensure required columns ---
        required = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}. Using available data.")
            if 'Adj Close' not in df.columns:
                if 'Close' in df.columns:
                    df['Adj Close'] = df['Close']
                else:
                    st.error("No price data available.")
                    return None

        # --- STEP 3: Select & Clean ---
        df = df[required].dropna(how='all')
        if len(df) < 30:
            st.warning("Need 30+ days of data.")
            return None

        df.index.name = 'Date'
        return df

    except Exception as e:
        st.error(f"Data error: {str(e)}")
        return None

data = fetch_stock_data(selected_ticker, start_date, end_date)
if data is None:
    st.stop()

# Use same data for OHLC
full_data = data.copy()

# ====================== SENTIMENT (SAFE) ======================
@st.cache_data(ttl=300)
def get_sentiment_news(ticker):
    try:
        api_key = "D8VCWYUPOFJR8D52"
        if api_key != "your_alphavantage_key_here":
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                j = r.json()
                feed = j.get('feed', [])
                if feed:
                    return [
                        f"{a.get('title','')}: {a.get('summary','')[:120]}..."
                        for a in feed[:10]
                    ], [a.get('url','#') for a in feed[:10]]
        # yfinance fallback
        news = ticker_obj.news[:10]
        return [
            f"{n.get('title','No title')} - {n.get('publisher','Source')}"
            for n in news
        ], [n.get('link','#') for n in news]
    except:
        return ["News unavailable."], ['#']

sample_posts, links = get_sentiment_news(selected_ticker)

# ====================== TABS ======================
tab = option_menu(
    menu_title=None,
    options=["Data & Viz", "Predictions", "Sentiment"],
    icons=["table", "graph-up", "chat-dots"],
    orientation="horizontal"
)

# ====================== DATA & VIZ ======================
if tab == "Data & Viz":
    st.header(f"**{selected_ticker}** - Price History")
    st.dataframe(data.tail(100), use_container_width=True)
    st.download_button("Download CSV", data.to_csv().encode(), f"{selected_ticker}.csv", "text/csv")

    st.subheader("Price Trend")
    fig = px.line(data, x=data.index, y='Adj Close', title="Adjusted Close Price")
    fig.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    try:
        close = data['Close'].iloc[-1]
        change_7d = (close - data['Close'].iloc[-8]) / data['Close'].iloc[-8] * 100 if len(data) > 7 else 0
        vol = data['Close'].pct_change().std() * np.sqrt(252) * 100
        c1.metric("Price", f"${close:,.2f}")
        c2.metric("7D Δ", f"{change_7d:+.2f}%")
        c3.metric("Volatility", f"{vol:.1f}%")
        c4.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
    except:
        c1.metric("Price", "N/A")

    st.subheader("Candlestick Chart")
    fig_c = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close']
    )])
    fig_c.update_layout(title=f"{selected_ticker} OHLC", height=600, template="plotly_white")
    st.plotly_chart(fig_c, use_container_width=True)

# ====================== PREDICTIONS ======================
elif tab == "Predictions":
    st.header("Future Price Forecast")
    model = st.selectbox("Model", ["Prophet", "LSTM"])
    days = st.slider("Days Ahead", 1, 30, 7)

    if model == "Prophet" and len(data) >= 30:
        with st.spinner("Training Prophet..."):
            df_p = data.reset_index()[['Date', 'Adj Close']].rename(columns={'Date': 'ds', 'Adj Close': 'y'})
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            m.fit(df_p)
            future = m.make_future_dataframe(periods=days)
            forecast = m.predict(future)
            pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
            pred.columns = ['Date', 'Predicted', 'Lower', 'Upper']
            fig = plot_plotly(m, forecast)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pred.style.format({"Predicted": "{:.2f}", "Lower": "{:.2f}", "Upper": "{:.2f}"}))

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
            lstm.compile('adam', 'mse')
            lstm.fit(X, y, epochs=3, batch_size=32, verbose=0)
            last = scaled[-60:].reshape(1,60,1)
            preds = []
            for _ in range(days):
                p = lstm.predict(last, verbose=0)[0][0]
                preds.append(p)
                last = np.append(last[:,1:,:], [[[p]]], axis=1)
            pred_vals = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            pred_df = pd.DataFrame({'Date': pd.date_range(start=data.index[-1]+pd.Timedelta(days=1), periods=days),
                                    'Predicted': pred_vals})
            fig = px.line(pred_df, x='Date', y='Predicted', title="LSTM Forecast")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pred_df.style.format({"Predicted": "{:.2f}"}))

    else:
        st.error("Not enough data for selected model.")

# ====================== SENTIMENT ======================
elif tab == "Sentiment":
    st.header("News Sentiment")
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

    st.caption("Sentiment from latest news. Uses VADER NLP.")
