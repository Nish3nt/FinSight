import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
import os
from datetime import datetime

# Download NLTK data for VADER
nltk.download('vader_lexicon', quiet=True)

# Set current date for validation
current_date = datetime.now().date()

# Title of the app
st.title("FinSight: Stock Analysis and Prediction Dashboard")

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
            # Fetch data for a single ticker
            data = yf.download(ticker, start=start, end=end, auto_adjust=False)
            if data.empty:
                st.error(f"No data returned for {ticker}. Check ticker or date range.")
                return pd.DataFrame()
            
            # Handle different yfinance output formats
            if isinstance(data, pd.Series):
                # If Series (e.g., single column like Adj Close), convert to DataFrame
                data = pd.DataFrame(data, columns=['Adj Close'])
            elif isinstance(data.columns, pd.MultiIndex):
                # Handle MultiIndex (e.g., [('Adj Close', 'AAPL')])
                if ('Adj Close', ticker) in data.columns:
                    data = data['Adj Close'][ticker].to_frame(name='Adj Close')
                else:
                    st.error(f"No 'Adj Close' column found for {ticker}. Available columns: {data.columns.tolist()}")
                    return pd.DataFrame()
            elif 'Adj Close' in data.columns:
                # Handle flat DataFrame with 'Adj Close' column
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
        st.header(f"Stock Data for {selected_ticker}")
        st.dataframe(data.head())

        # Data Visualization
        st.header("Stock Price Visualization")
        fig = px.line(data, x=data.index, y='Adj Close', title=f'{selected_ticker} Adjusted Close Price')
        st.plotly_chart(fig)

        # Model Training (Assuming we train on the fly or load saved model)
        model_file = f'{selected_ticker}_rf_model.pkl'
        if os.path.exists(model_file):
            rf_model = joblib.load(model_file)
            st.success("Loaded pre-trained Random Forest model.")
        else:
            st.info("Training Random Forest model...")
            # Simple feature engineering (e.g., lagged prices)
            df = data.copy()
            df['Lag1'] = df['Adj Close'].shift(1)
            df.dropna(inplace=True)
            if len(df) < 2:
                st.error("Insufficient data to train model.")
            else:
                X = df[['Lag1']]
                y = df['Adj Close']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                
                # Save model
                joblib.dump(rf_model, model_file)
                st.success("Trained and saved Random Forest model.")

                # Model Evaluation
                st.header("Model Performance")
                y_pred = rf_model.predict(X_test)
                rf_r2 = r2_score(y_test, y_pred)
                rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rf_mae = mean_absolute_error(y_test, y_pred)
                st.write(f"Random Forest → R²: {rf_r2:.2f}, RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}")

        # Predictions
        st.header("Stock Price Prediction")
        if 'rf_model' in locals():
            last_price = data['Adj Close'].iloc[-1]
            future_days = st.slider("Predict for next N days", 1, 30, 5)
            
            # Simple prediction using last price as input
            predictions = []
            current_price = last_price
            for _ in range(future_days):
                pred = rf_model.predict([[current_price]])[0]
                predictions.append(pred)
                current_price = pred
            
            pred_df = pd.DataFrame({
                'Date': pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days),
                'Predicted Price': predictions
            })
            st.dataframe(pred_df)
            
            # Plot predictions
            fig_pred = px.line(pred_df, x='Date', y='Predicted Price', title='Future Price Predictions')
            st.plotly_chart(fig_pred)
        else:
            st.warning("No model available for predictions.")

        # Sentiment Analysis (Dummy news for demo)
        st.header("Sentiment Analysis")
        sample_news = [
            "Apple releases new iPhone with great features.",
            "Microsoft faces antitrust lawsuit.",
            "Google AI advancements boost stock.",
            "Amazon reports record profits.",
            "Tesla recalls vehicles due to safety issues."
        ]
        sia = SentimentIntensityAnalyzer()
        sentiments = [sia.polarity_scores(text)['compound'] for text in sample_news]
        sentiments_df = pd.DataFrame({'News': sample_news, 'compound': sentiments})
        
        pos = sentiments_df[sentiments_df['compound'] > 0].shape[0]
        neg = sentiments_df[sentiments_df['compound'] < 0].shape[0]
        neu = sentiments_df[sentiments_df['compound'] == 0].shape[0]
        
        st.write(f"Positive news count: {pos}")
        st.write(f"Negative news count: {neg}")
        st.write(f"Neutral news count: {neu}")
        
        # Insights
        st.header("Insights")
        st.write("💡 Combining stock prediction with sentiment shows how news affects price movement.")
    else:
        st.error("No data available for the selected parameters.")
