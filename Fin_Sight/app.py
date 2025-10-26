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
else:
    # Adjust end_date if it exceeds current date
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
            
            if isinstance(data.columns, pd.MultiIndex):
                if ('Adj Close', ticker) in data.columns:
                    data = data['Adj Close'][ticker].to_frame(name='Adj Close')
                else:
                    st.error(f"No 'Adj Close' column found for {ticker}. Available columns: {data.columns.tolist()}")
                    return pd.DataFrame()
            elif 'Adj Close' in data.columns:
                data = data[['Adj Close']]
            else:
                st.error(f"No 'Adj Close' column found for {ticker}. Available columns: {data.columns.tolist()}")
                return pd.DataFrame()
            
            return data
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

    data = fetch_data(selected_ticker, start_date, end_date)

    if not data.empty:
        # Fetch full OHLC data for viz
        try:
            full_data = yf.download(selected_ticker, start=start_date, end=end_date)
        except Exception as e:
            st.error(f"Error fetching OHLC data: {str(e)}. Using Adj Close data only.")
            full_data = data  # Fallback to Adj Close data

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
            
            # Stock Price Visualization (line chart)
            st.header("Stock Price Visualization")
            fig = px.line(data, x=data.index, y='Adj Close', title=f'{selected_ticker} Adjusted Close Price')
            st.plotly_chart(fig)

            # KPI Metrics Dashboard
            st.header("Key Performance Indicators")
            col1, col2, col3 = st.columns(3)
            try:
                current_price = full_data['Close'].iloc[-1].item() if 'Close' in full_data.columns and not full_data.empty else data['Adj Close'].iloc[-1].item()
                change_7d = ((current_price - full_data['Close'].iloc[-8].item()) / full_data['Close'].iloc[-8].item() * 100 
                            if len(full_data) > 7 and 'Close' in full_data.columns else 0)
                volatility = (full_data['Close'].pct_change().std() * np.sqrt(252) * 100 
                            if 'Close' in full_data.columns and len(full_data) > 1 else 0)
                col1.metric("Current Price", f"${current_price:.2f}")
                col2.metric("7-Day Change", f"{change_7d:.2f}%")
                col3.metric("Annual Volatility", f"{volatility:.2f}%")
            except (TypeError, IndexError) as e:
                st.warning(f"Error calculating metrics: {str(e)}. Using default values.")
                col1.metric("Current Price", "$0.00")
                col2.metric("7-Day Change", "0.00%")
                col3.metric("Annual Volatility", "0.00%")

            # OHLC Candlestick Chart
            st.header("OHLC Candlestick Chart")
            if 'Open' in full_data.columns and 'High' in full_data.columns and 'Low' in full_data.columns and 'Close' in full_data.columns:
                fig_candle = go.Figure(data=[go.Candlestick(x=full_data.index,
                                                           open=full_data['Open'],
                                                           high=full_data['High'],
                                                           low=full_data['Low'],
                                                           close=full_data['Close'])])
                fig_candle.update_layout(title=f'{selected_ticker} OHLC Prices')
                st.plotly_chart(fig_candle)
            else:
                st.warning("OHLC data unavailable. Candlestick chart not displayed.")

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
                        fig_pred = plot_plotly(m, forecast)
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
                            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                            
                            # Predict
                            predictions = []
                            last_seq = scaled_data[-time_step:].reshape(1, time_step, 1)
                            for _ in range(future_days):
                                pred = model.predict(last_seq, verbose=0)[0][0]
                                predictions.append(pred)
                                last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)
                            pred_df['Predicted Price'] = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                            pred_df['Date'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
                            
                            fig_pred = px.line(pred_df, x='Date', y='Predicted Price', title='LSTM Future Price Predictions')
                            st.plotly_chart(fig_pred)
                        except Exception as e:
                            st.error(f"LSTM training failed: {str(e)}. Using fallback prediction.")
                            pred_df['Predicted Price'] = [data['Adj Close'].iloc[-1].item()] * future_days
                            pred_df['Date'] = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
            
            if not pred_df.empty:
                st.dataframe(pred_df)
            else:
                st.warning("No predictions available.")

        elif selected_tab == "Sentiment":
            st.header("Sentiment Analysis")
            try:
                ticker_obj = yf.Ticker(selected_ticker)
                news = ticker_obj.news[:10]  # Fetch top 10 recent news articles
                if not news:
                    raise ValueError("No news available.")
                sample_news = [f"{article['title']} - {article.get('publisher', 'Unknown')}" for article in news]
                links = [article['link'] for article in news]
                sia = SentimentIntensityAnalyzer()
                sentiments = [sia.polarity_scores(text)['compound'] for text in sample_news]
                sentiments_df = pd.DataFrame({'News': sample_news, 'Link': links, 'Sentiment Score': sentiments})
                
                # Color-code scores
                def color_sentiment(val):
                    color = 'green' if val > 0.1 else 'red' if val < -0.1 else 'gray'
                    return f'color: {color}'
                st.dataframe(sentiments_df.style.applymap(color_sentiment, subset=['Sentiment Score']))
                
                pos = sentiments_df[sentiments_df['Sentiment Score'] > 0.1].shape[0]
                neg = sentiments_df[sentiments_df['Sentiment Score'] < -0.1].shape[0]
                neu = sentiments_df.shape[0] - pos - neg
                st.write(f"Positive news count: {pos}")
                st.write(f"Negative news count: {neg}")
                st.write(f"Neutral news count: {neu}")
            except Exception as e:
                st.warning(f"Failed to fetch news: {str(e)}. Using sample data.")
                sample_news = [
                    "Apple releases new iPhone with great features.",
                    "Microsoft faces antitrust lawsuit.",
                    "Google AI advancements boost stock.",
                    "Amazon reports record profits.",
                    "Tesla recalls vehicles due to safety issues."
                ]
                sia = SentimentIntensityAnalyzer()
                sentiments = [sia.polarity_scores(text)['compound'] for text in sample_news]
                sentiments_df = pd.DataFrame({'News': sample_news, 'Sentiment Score': sentiments})
                st.dataframe(sentiments_df)

        elif selected_tab == "Insights":
            st.header("Insights")
            avg_sentiment = sentiments_df['Sentiment Score'].mean() if 'sentiments_df' in locals() else 0
            st.write(f"💡 Average news sentiment: {avg_sentiment:.2f}. Positive sentiment often correlates with price uptrends.")
            st.write("Combining ML forecasts with sentiment can predict volatility—e.g., negative news may widen confidence bounds.")
    else:
        st.error("No data available for the selected parameters.")
