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

# Download NLTK data for VADER
nltk.download('vader_lexicon', quiet=True)

# Set current date for validation
current_date = datetime.now().date()

# Title of the app
st.title("FinSight: Advanced Stock Analysis Dashboard")

# ==================== SIDEBAR INPUT (MUST BE BEFORE ANY USE OF selected_ticker) ====================
st.sidebar.header("User Input")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'JPM', 'V', 'XOM']
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2018-01-01').date())
end_date = st.sidebar.date_input("End Date", current_date)

# Validate date range
if start_date > end_date:
    st.error("Error: Start date must be before or equal to end date.")
    st.stop()
else:
    if end_date > current_date:
        end_date = current_date
        st.warning(f"End date set to today ({current_date}) as future dates are not available.")

# ==================== FETCH TICKER OBJECT FOR NEWS TICKER (NOW SAFE) ====================
ticker_obj = yf.Ticker(selected_ticker)

# ==================== NEWS TICKER (SCROLLING HEADLINES) ====================
st.markdown("""
<style>
.news-ticker {
    background-color: #f0f2f6;
    padding: 12px;
    border-radius: 8px;
    overflow: hidden;
    white-space: nowrap;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    font-family: Arial, sans-serif;
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
    animation: scroll 40s linear infinite;
    color: #333;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

news = ticker_obj.news[:10]  # Top 10 recent news
if news:
    headlines = " | ".join([f"{article['title']} ({article.get('publisher', 'Source')})" for article in news])
    st.markdown(f'<div class="news-ticker"><span class="ticker-content">{headlines}</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="news-ticker"><span class="ticker-content">No recent news available for this stock.</span></div>', unsafe_allow_html=True)

# ==================== FETCH HISTORICAL DATA ====================
@st.cache_data
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=False)
        if data.empty:
            st.error(f"No data returned for {ticker}. Check ticker or date range.")
            return pd.DataFrame()
        # Flatten MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
            if all((col, ticker) in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']):
                data = pd.DataFrame({
                    'Open': data[('Open', ticker)],
                    'High': data[('High', ticker)],
                    'Low': data[('Low', ticker)],
                    'Close': data[('Close', ticker)],
                    'Adj Close': data[('Adj Close', ticker)]
                })
            else:
                data = data.xs('Adj Close', level=0, axis=1).to_frame(name='Adj Close') if 'Adj Close' in data.columns.levels[0] else pd.DataFrame()
        elif all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']):
            data = data[['Open', 'High', 'Low', 'Close', 'Adj Close']]
        else:
            data = data[['Adj Close']] if 'Adj Close' in data.columns else pd.DataFrame()
        if data.empty or len(data) < 30:
            st.warning(f"Insufficient data ({len(data)} days) for {ticker}. Need at least 30 days.")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

data = fetch_data(selected_ticker, start_date, end_date)

if not data.empty:
    # Fetch full OHLC data
    def fetch_full_data(ticker, start, end, max_retries=3):
        for attempt in range(max_retries):
            try:
                full_data = yf.download(ticker, start=start, end=end, auto_adjust=False)
                if full_data.empty:
                    raise ValueError("Full data empty.")
                if isinstance(full_data.columns, pd.MultiIndex):
                    field_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
                    new_data = pd.DataFrame(index=full_data.index)
                    for col in field_cols:
                        if (col, ticker) in full_data.columns:
                            new_data[col] = full_data[(col, ticker)]
                        else:
                            new_data[col] = full_data.xs(col, level=0, axis=1)
                    full_data = new_data
                elif all(col in full_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']):
                    full_data = full_data[['Open', 'High', 'Low', 'Close', 'Adj Close']]
                else:
                    raise ValueError("Missing OHLC columns.")
                return full_data
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Using Adj Close only: {str(e)}")
                    return data.copy()
                st.warning(f"Retry {attempt + 1}/{max_retries} for OHLC data.")
        return data.copy()

    full_data = fetch_full_data(selected_ticker, start_date, end_date)

    # ==================== SENTIMENT DATA (Alpha Vantage + Fallback) ====================
    @st.cache_data(ttl=300)
    def fetch_real_time_sentiment(ticker):
        try:
            api_key = "D8VCWYUPOFJR8D52"
            if not api_key or api_key == "your_alphavantage_key_here":
                st.warning("Alpha Vantage API key missing. Using yfinance news.")
                raise ValueError("No API key")
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                raise ValueError(f"API error: {response.status_code}")
            data = response.json()
            if 'feed' not in data or not data.get('feed'):
                raise ValueError("No feed in response")
            posts, links = [], []
            for article in data['feed'][:10]:
                title = article.get('title', 'No title')
                summary = article.get('summary', '')[:100] + "..."
                posts.append(f"{title}: {summary}")
                links.append(article.get('url', '#'))
            return posts, links, 0.0
        except Exception as e:
            st.warning(f"Alpha Vantage failed: {str(e)}. Using yfinance news.")
            news = ticker_obj.news[:10]
            posts = [f"{n.get('title','No title')} - {n.get('publisher','Unknown')}" for n in news]
            links = [n.get('link', '#') for n in news]
            return posts, links, 0.0

    sample_posts, links, _ = fetch_real_time_sentiment(selected_ticker)

    # ==================== TABS ====================
    selected_tab = option_menu(
        menu_title=None,
        options=["Data & Viz", "Predictions", "Sentiment"],
        icons=["table", "graph-up", "chat-dots"],
        orientation="horizontal"
    )

    # ==================== DATA & VIZ TAB ====================
    if selected_tab == "Data & Viz":
        st.header(f"Stock Data for {selected_ticker}")
        st.dataframe(data)
        csv = data.to_csv().encode('utf-8')
        st.download_button("Download Data", csv, f"{selected_ticker}_data.csv", "text/csv")

        st.header("Price Chart")
        fig = px.line(data, x=data.index, y='Adj Close', title=f'{selected_ticker} Adjusted Close')
        st.plotly_chart(fig)

        st.header("Key Metrics")
        col1, col2, col3 = st.columns(3)
        try:
            price = float(full_data['Close'].iloc[-1]) if 'Close' in full_data.columns else float(data['Adj Close'].iloc[-1])
            change_7d = ((price - full_data['Close'].iloc[-8]) / full_data['Close'].iloc[-8] * 100) if len(full_data) > 7 else 0.0
            volatility = full_data['Close'].pct_change().std() * np.sqrt(252) * 100 if len(full_data) > 1 else 0.0
            col1.metric("Current Price", f"${price:.2f}")
            col2.metric("7-Day Change", f"{change_7d:.2f}%")
            col3.metric("Annual Volatility", f"{volatility:.2f}%")
        except Exception:
            col1.metric("Current Price", "$0.00")
            col2.metric("7-Day Change", "0.00%")
            col3.metric("Annual Volatility", "0.00%")

        st.header("Candlestick Chart")
        if all(col in full_data.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig_candle = go.Figure(data=[go.Candlestick(
                x=full_data.index, open=full_data['Open'], high=full_data['High'],
                low=full_data['Low'], close=full_data['Close']
            )])
            fig_candle.update_layout(title=f'{selected_ticker} OHLC')
            st.plotly_chart(fig_candle)
        else:
            st.info("OHLC data not available.")

    # ==================== PREDICTIONS TAB ====================
    elif selected_tab == "Predictions":
        st.header("Stock Price Prediction")
        model_type = st.selectbox("Model", ["Prophet", "LSTM"])
        future_days = st.slider("Days Ahead", 1, 30, 5)
        pred_df = pd.DataFrame()

        if model_type == "Prophet" and len(data) >= 30:
            with st.spinner("Training Prophet..."):
                df_p = data.reset_index().rename(columns={data.index.name: 'ds', 'Adj Close': 'y'})
                m = Prophet(daily_seasonality=True, yearly_seasonality=True)
                m.fit(df_p)
                future = m.make_future_dataframe(periods=future_days)
                forecast = m.predict(future)
                pred_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_days)
                pred_df.columns = ['Date', 'Predicted', 'Lower', 'Upper']
                fig = plot_plotly(m, forecast)
                st.plotly_chart(fig)
        elif model_type == "LSTM" and len(data) >= 60:
            with st.spinner("Training LSTM..."):
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))
                X, y = [], []
                for i in range(60, len(scaled)):
                    X.append(scaled[i-60:i, 0])
                    y.append(scaled[i, 0])
                X, y = np.array(X), np.array(y)
                X = X.reshape((X.shape[0], X.shape[1], 1))
                model = Sequential([LSTM(50, return_sequences=True, input_shape=(60,1)),
                                    LSTM(50), Dense(1)])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                last = scaled[-60:].reshape(1,60,1)
                preds = []
                for _ in range(future_days):
                    p = model.predict(last, verbose=0)[0][0]
                    preds.append(p)
                    last = np.append(last[:,1:,:], [[[p]]], axis=1)
                pred_df['Predicted'] = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
                pred_df['Date'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
                fig = px.line(pred_df, x='Date', y='Predicted', title="LSTM Prediction")
                st.plotly_chart(fig)
        else:
            st.error("Not enough data for selected model.")

        if not pred_df.empty:
            st.dataframe(pred_df.style.format({'Predicted': '{:.2f}'}))

    # ==================== SENTIMENT TAB ====================
    elif selected_tab == "Sentiment":
        st.header("News Sentiment Analysis")
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(p)['compound'] for p in sample_posts]
        df_sent = pd.DataFrame({'News': sample_posts, 'Link': links, 'Score': scores})

        def color_score(val):
            return f"color: {'green' if val > 0.1 else 'red' if val < -0.1 else 'gray'}"
        st.dataframe(df_sent.style.applymap(color_score, subset=['Score']).format({'Score': '{:.2f}'}))

        pos = len([s for s in scores if s > 0.1])
        neg = len([s for s in scores if s < -0.1])
        neu = len(scores) - pos - neg
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", pos)
        col2.metric("Negative", neg)
        col3.metric("Neutral", neu)

else:
    st.error("No data available. Try a longer date range or different ticker.")
