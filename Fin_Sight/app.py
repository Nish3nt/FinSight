import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
import os
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

# Sidebar for user input
st.sidebar.header("User Input")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
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

    # Fetch data
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
                    # Fallback: Use xs to extract level 0 (field names)
                    data = data.xs('Adj Close', level=0, axis=1).to_frame(name='Adj Close') if 'Adj Close' in data.columns.levels[0] else pd.DataFrame()
            elif all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']):
                data = data[['Open', 'High', 'Low', 'Close', 'Adj Close']]
            else:
                st.warning(f"Incomplete data for {ticker}. Using available columns.")
                data = data[['Adj Close']] if 'Adj Close' in data.columns else pd.DataFrame()
            return data
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

    data = fetch_data(selected_ticker, start_date, end_date)

    if not data.empty:
        # Fetch full OHLC data with fixed MultiIndex handling
        def fetch_full_data(ticker, start, end, max_retries=3):
            for attempt in range(max_retries):
                try:
                    full_data = yf.download(ticker, start=start, end=end, auto_adjust=False)
                    if full_data.empty:
                        raise ValueError("Full data empty.")
                    # Fixed MultiIndex handling: Try tuple access, fallback to xs
                    if isinstance(full_data.columns, pd.MultiIndex):
                        field_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
                        new_data = pd.DataFrame(index=full_data.index)
                        for col in field_cols:
                            if (col, ticker) in full_data.columns:
                                new_data[col] = full_data[(col, ticker)]
                            else:
                                # Fallback: Extract from level 0
                                new_data[col] = full_data.xs(col, level=0, axis=1)
                        full_data = new_data
                    elif all(col in full_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']):
                        full_data = full_data[['Open', 'High', 'Low', 'Close', 'Adj Close']]
                    else:
                        raise ValueError("Missing OHLC columns.")
                    return full_data
                except Exception as e:
                    if attempt == max_retries - 1:
                        st.warning(f"Failed to fetch OHLC data after {max_retries} attempts: {str(e)}. Using Adj Close data.")
                        return data.copy()
                    st.warning(f"Retry {attempt + 1}/{max_retries} for OHLC data: {str(e)}")

        full_data = fetch_full_data(selected_ticker, start_date, end_date)

        # Fetch real-time sentiment from Alpha Vantage (hardcoded key)
        @st.cache_data(ttl=300)  # Cache for 5 minutes
        def fetch_real_time_sentiment(ticker):
            try:
                # Replace with your Alpha Vantage API key
                api_key = "D8VCWYUPOFJR8D52"  # Insert your key here (e.g., "ABC123XYZ")
                if not api_key or api_key == "your_alphavantage_key_here":
                    st.warning("Please replace 'your_alphavantage_key_here' with your Alpha Vantage API key in the code.")
                    raise ValueError("No valid API key provided.")
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
                response = requests.get(url)
                if response.status_code != 200:
                    raise ValueError(f"Alpha Vantage API error: {response.status_code}")
                data = response.json()
                if 'feed' not in data:
                    raise ValueError("No sentiment data returned.")
                
                # Extract sentiment scores and articles
                posts = []
                links = []
                sentiment_scores = []
                articles = data['feed'][:10]
                for article in articles:
                    title = article.get('title', 'No title')
                    summary = article.get('summary', '')
                    post = f"{title}: {summary[:100]}..."  # Shortened for display
                    posts.append(post)
                    links.append(article.get('url', '#'))
                    
                    # Extract sentiment score (e.g., from ticker_sentiment)
                    ticker_sent = article.get('ticker_sentiment', [])
                    score = 0.0
                    if ticker_sent:
                        for ts in ticker_sent:
                            if ts['ticker'] == ticker:
                                score = ts.get('ticker_sentiment_score', 0.0)  # Average score
                    sentiment_scores.append(score)
                
                # Average sentiment score
                avg_score = np.mean(sentiment_scores) if sentiment_scores else 0.0
                
                # Fallback if no articles
                if not posts:
                    posts = [f"Average sentiment score for {ticker}: {avg_score}"]
                    links = ['#']
                
                return posts, links, avg_score
            except Exception as e:
                st.warning(f"Alpha Vantage failed: {str(e)}. Using yfinance fallback.")
                # Fallback to yfinance news
                ticker_obj = yf.Ticker(ticker)
                news = ticker_obj.news[:10]
                posts = [f"{article.get('title', 'No title')} - {article.get('publisher', 'Unknown')}" for article in news if article.get('title')]
                links = [article.get('link', '#') for article in news]
                avg_score = 0.0  # Default for fallback
                return posts if posts else ["Sample: Neutral market news."], ['#'], avg_score

        sample_posts, links, avg_sentiment_from_api = fetch_real_time_sentiment(selected_ticker)

        # Multi-page layout with tabs
        selected_tab = option_menu(
            menu_title=None,
            options=["Data & Viz", "Predictions", "Sentiment", "Insights"],
            icons=["table", "graph-up", "chat-dots", "lightbulb"],
            orientation="horizontal"
        )

        if selected_tab == "Data & Viz":
            st.header(f"Stock Data for {selected_ticker}")
            st.dataframe(data)
            csv = data.to_csv().encode('utf-8')
            st.download_button("Download Data", csv, f"{selected_ticker}_data.csv", "text/csv")
            
            # Stock Price Visualization
            st.header("Stock Price Visualization")
            fig = px.line(data, x=data.index, y='Adj Close', title=f'{selected_ticker} Adjusted Close Price')
            st.plotly_chart(fig)

            # KPI Metrics Dashboard
            st.header("Key Performance Indicators")
            col1, col2, col3 = st.columns(3)
            try:
                current_price = float(full_data['Close'].iloc[-1]) if 'Close' in full_data.columns and not full_data.empty else float(data['Adj Close'].iloc[-1])
                change_7d = ((current_price - float(full_data['Close'].iloc[-8])) / float(full_data['Close'].iloc[-8]) * 100 
                            if len(full_data) > 7 and 'Close' in full_data.columns else 0.0)
                volatility = (full_data['Close'].pct_change().std() * np.sqrt(252) * 100 
                            if 'Close' in full_data.columns and len(full_data) > 1 else 0.0)
                col1.metric("Current Price", f"${current_price:.2f}")
                col2.metric("7-Day Change", f"{change_7d:.2f}%")
                col3.metric("Annual Volatility", f"{volatility:.2f}%")
            except (ValueError, IndexError, TypeError) as e:
                st.warning(f"Error calculating metrics: {str(e)}. Using default values.")
                col1.metric("Current Price", "$0.00")
                col2.metric("7-Day Change", "0.00%")
                col3.metric("Annual Volatility", "0.00%")

            # OHLC Candlestick Chart
            st.header("OHLC Candlestick Chart")
            if all(col in full_data.columns for col in ['Open', 'High', 'Low', 'Close']):
                fig_candle = go.Figure(data=[go.Candlestick(x=full_data.index,
                                                           open=full_data['Open'],
                                                           high=full_data['High'],
                                                           low=full_data['Low'],
                                                           close=full_data['Close'])])
                fig_candle.update_layout(title=f'{selected_ticker} OHLC Prices')
                st.plotly_chart(fig_candle)
            else:
                st.warning("OHLC data unavailable. Candlestick chart not displayed due to data limitations.")

        elif selected_tab == "Predictions":
            st.header("Stock Price Prediction")
            model_type = st.selectbox("Model Type", ["Prophet", "LSTM"])
            future_days = st.slider("Predict for next N days", 1, 30, 5)
            pred_df = pd.DataFrame()

            if model_type == "Prophet":
                df_prophet = data.reset_index().rename(columns={'Date': 'ds', 'Adj Close': 'y'})
                with st.spinner("Training Prophet model..."):
                    try:
                        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
                        m.fit(df_prophet)
                        future = m.make_future_dataframe(periods=future_days)
                        forecast = m.predict(future)
                        pred_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_days)
                        pred_df.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
                        pred_df['Predicted Price'] = pred_df['Predicted Price'].astype(float)
                        pred_df['Lower Bound'] = pred_df['Lower Bound'].astype(float)
                        pred_df['Upper Bound'] = pred_df['Upper Bound'].astype(float)
                        fig_pred = plot_plotly(m, forecast)
                        # Customize colors: Teal for prediction, light teal for confidence
                        fig_pred.update_traces(line_color='#26A69A', fill_color='#B2DFDB', name='Prophet Prediction')
                        st.plotly_chart(fig_pred)
                    except Exception as e:
                        st.error(f"Prophet training failed: {str(e)}. Try a different date range.")
            else:  # LSTM
                if len(data) < 60:
                    st.error("Need at least 60 days of data for LSTM.")
                else:
                    with st.spinner("Training LSTM model..."):
                        try:
                            scaler = MinMaxScaler()
                            scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))
                            time_step = 60
                            X, y = [], []
                            for i in range(time_step, len(scaled_data)):
                                X.append(scaled_data[i-time_step:i, 0])
                                y.append(scaled_data[i, 0])
                            X, y = np.array(X), np.array(y)
                            X = X.reshape((X.shape[0], X.shape[1], 1))
                            
                            model = Sequential()
                            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
                            model.add(LSTM(50))
                            model.add(Dense(1))
                            model.compile(optimizer='adam', loss='mean_squared_error')
                            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                            
                            # Predict
                            predictions = []
                            last_seq = scaled_data[-time_step:].reshape(1, time_step, 1)
                            for _ in range(future_days):
                                pred = model.predict(last_seq, verbose=0)[0][0]
                                predictions.append(pred)
                                last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)
                            pred_df['Predicted Price'] = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                            pred_df['Date'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
                            
                            # Customize color: Purple for LSTM prediction
                            fig_pred = px.line(pred_df, x='Date', y='Predicted Price', title='LSTM Future Price Predictions', color_discrete_sequence=['#AB47BC'])
                            st.plotly_chart(fig_pred)
                        except Exception as e:
                            st.error(f"LSTM training failed: {str(e)}. Using fallback prediction.")
                            last_price = float(data['Adj Close'].iloc[-1])
                            pred_df['Predicted Price'] = [last_price] * future_days
                            pred_df['Date'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
                            fig_pred = px.line(pred_df, x='Date', y='Predicted Price', title='LSTM Future Price Predictions', color_discrete_sequence=['#AB47BC'])
                            st.plotly_chart(fig_pred)
            
            if not pred_df.empty:
                st.dataframe(pred_df.style.format({'Predicted Price': '{:.2f}', 'Lower Bound': '{:.2f}', 'Upper Bound': '{:.2f}'}))

        elif selected_tab == "Sentiment":
            st.header("Real-Time Sentiment Analysis (Alpha Vantage News)")
            sia = SentimentIntensityAnalyzer()
            sentiments = [sia.polarity_scores(text)['compound'] for text in sample_posts]
            sentiments_df = pd.DataFrame({'Post/News': sample_posts, 'Link': links, 'Sentiment Score': sentiments})
            
            def color_sentiment(val):
                color = 'green' if val > 0.1 else 'red' if val < -0.1 else 'gray'
                return f'color: {color}'
            st.dataframe(sentiments_df.style.applymap(color_sentiment, subset=['Sentiment Score']).format({'Sentiment Score': '{:.2f}'}))

            pos = len([s for s in sentiments if s > 0.1])
            neg = len([s for s in sentiments if s < -0.1])
            neu = len(sentiments) - pos - neg
            st.metric("Positive Count", pos)
            st.metric("Negative Count", neg)
            st.metric("Neutral Count", neu)
            st.caption(f"Real-time data from Alpha Vantage news for '{selected_ticker}'. Refresh for updates.")

        elif selected_tab == "Insights":
            st.header("Insights")
            avg_sentiment = avg_sentiment_from_api  # Use Alpha Vantage's direct score
            if avg_sentiment > 0.1:
                sentiment_note = "🟢 Bullish sentiment detected—consider long positions if predictions align."
            elif avg_sentiment < -0.1:
                sentiment_note = "🔴 Bearish sentiment—watch for downside risk in forecasts."
            else:
                sentiment_note = "⚪ Neutral sentiment—rely on technical indicators."
            if avg_sentiment == 0.0:
                st.warning("Fallback data used; real-time sentiment unavailable.")
            st.write(f"💡 Average sentiment score: {avg_sentiment:.2f}")
            st.write(sentiment_note)
            st.write("Real-time news sentiment can amplify ML predictions—e.g., high positive scores narrow confidence intervals.")
    else:
        st.error("No data available for the selected parameters.")
