import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FinSight - Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 FinSight: Stock Price Prediction</h1>', unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_stock_model.pkl')
        return model
    except:
        st.error("Model file not found. Please ensure 'best_stock_model.pkl' is in the same directory.")
        return None

# Calculate technical indicators
def calculate_indicators(df, stock_name):
    """Calculate MA10, MA50, and RSI for a given stock"""
    # Moving Averages
    df[f'{stock_name}_MA10'] = df[stock_name].rolling(window=10).mean()
    df[f'{stock_name}_MA50'] = df[stock_name].rolling(window=50).mean()
    
    # RSI Calculation
    delta = df[stock_name].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df[f'{stock_name}_RSI'] = 100 - (100 / (1 + rs))
    
    return df

# Fetch stock data
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return data['Adj Close']
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Main app
def main():
    model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/bullish.png", width=150)
        st.markdown("## Settings")
        
        # Stock selection
        stock_options = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        selected_stock = st.selectbox(
            "Select Stock",
            stock_options,
            help="Choose a stock to analyze"
        )
        
        # Date range
        st.markdown("### Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Predictions", "📉 Technical Analysis", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Stock Dashboard</h2>', unsafe_allow_html=True)
        
        if predict_button or st.session_state.get('initial_load', True):
            st.session_state['initial_load'] = False
            
            with st.spinner(f'Fetching data for {selected_stock}...'):
                # Fetch data for all stocks
                tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
                data = pd.DataFrame()
                
                for ticker in tickers:
                    stock_data = fetch_stock_data(ticker, start_date, end_date)
                    if stock_data is not None:
                        data[ticker] = stock_data
                
                if data.empty:
                    st.error("Failed to fetch stock data. Please try again.")
                    return
                
                # Calculate indicators for selected stock
                data = calculate_indicators(data, selected_stock)
                
                # Store in session state
                st.session_state['data'] = data
                st.session_state['selected_stock'] = selected_stock
        
        # Display data if available
        if 'data' in st.session_state:
            data = st.session_state['data']
            selected_stock = st.session_state['selected_stock']
            
            # Current metrics
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = data[selected_stock].iloc[-1]
            prev_price = data[selected_stock].iloc[-2]
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            with col1:
                st.metric(
                    label="Current Price",
                    value=f"${current_price:.2f}",
                    delta=f"{change:.2f} ({change_pct:.2f}%)"
                )
            
            with col2:
                st.metric(
                    label="MA10",
                    value=f"${data[f'{selected_stock}_MA10'].iloc[-1]:.2f}"
                )
            
            with col3:
                st.metric(
                    label="MA50",
                    value=f"${data[f'{selected_stock}_MA50'].iloc[-1]:.2f}"
                )
            
            with col4:
                rsi_value = data[f'{selected_stock}_RSI'].iloc[-1]
                rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                st.metric(
                    label=f"RSI ({rsi_status})",
                    value=f"{rsi_value:.2f}"
                )
            
            # Price chart
            st.markdown("### Price History")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[selected_stock],
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[f'{selected_stock}_MA10'],
                mode='lines',
                name='MA10',
                line=dict(color='#ff7f0e', width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[f'{selected_stock}_MA50'],
                mode='lines',
                name='MA50',
                line=dict(color='#2ca02c', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title=f'{selected_stock} Stock Price',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent data table
            st.markdown("### Recent Data")
            st.dataframe(
                data[[selected_stock, f'{selected_stock}_MA10', f'{selected_stock}_MA50', f'{selected_stock}_RSI']].tail(10),
                use_container_width=True
            )
    
    with tab2:
        st.markdown('<h2 class="sub-header">Price Predictions</h2>', unsafe_allow_html=True)
        
        if model is None:
            st.warning("Model not loaded. Cannot make predictions.")
        elif 'data' in st.session_state:
            data = st.session_state['data']
            selected_stock = st.session_state['selected_stock']
            
            # Prepare features for prediction
            features = [selected_stock, f'{selected_stock}_MA10', f'{selected_stock}_MA50', f'{selected_stock}_RSI']
            latest_data = data[features].dropna().tail(30)
            
            if len(latest_data) > 0:
                # Make predictions
                X = latest_data[features].values
                predictions = model.predict(X)
                
                # Create prediction dataframe
                pred_df = pd.DataFrame({
                    'Date': latest_data.index,
                    'Actual': latest_data[selected_stock].values,
                    'Predicted': predictions
                })
                
                # Calculate metrics
                mae = np.mean(np.abs(pred_df['Actual'] - pred_df['Predicted']))
                rmse = np.sqrt(np.mean((pred_df['Actual'] - pred_df['Predicted'])**2))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error", f"${mae:.2f}")
                with col2:
                    st.metric("Root Mean Squared Error", f"${rmse:.2f}")
                
                # Prediction chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=pred_df['Date'],
                    y=pred_df['Actual'],
                    mode='lines+markers',
                    name='Actual Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=pred_df['Date'],
                    y=pred_df['Predicted'],
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'{selected_stock} - Actual vs Predicted Prices',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Next day prediction
                st.markdown("### Next Day Prediction")
                next_day_pred = predictions[-1]
                current_price = pred_df['Actual'].iloc[-1]
                expected_change = next_day_pred - current_price
                expected_change_pct = (expected_change / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric(
                        "Predicted Next Day Price",
                        f"${next_day_pred:.2f}",
                        delta=f"{expected_change:.2f} ({expected_change_pct:.2f}%)"
                    )
                with col3:
                    recommendation = "🟢 BUY" if expected_change > 0 else "🔴 SELL" if expected_change < 0 else "🟡 HOLD"
                    st.metric("Recommendation", recommendation)
                
                # Show prediction table
                st.markdown("### Prediction Details")
                st.dataframe(pred_df.tail(10), use_container_width=True)
            else:
                st.warning("Not enough data to make predictions. Please adjust the date range.")
        else:
            st.info("Please fetch stock data first from the Dashboard tab.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Technical Analysis</h2>', unsafe_allow_html=True)
        
        if 'data' in st.session_state:
            data = st.session_state['data']
            selected_stock = st.session_state['selected_stock']
            
            # RSI Chart
            st.markdown("### RSI (Relative Strength Index)")
            fig_rsi = go.Figure()
            
            fig_rsi.add_trace(go.Scatter(
                x=data.index,
                y=data[f'{selected_stock}_RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='#9467bd', width=2)
            ))
            
            # Add overbought/oversold lines
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            
            fig_rsi.update_layout(
                title=f'{selected_stock} RSI',
                xaxis_title='Date',
                yaxis_title='RSI',
                height=400,
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Volume analysis (if available)
            st.markdown("### Moving Average Analysis")
            
            # Calculate signals
            ma_signal = "Bullish" if data[f'{selected_stock}_MA10'].iloc[-1] > data[f'{selected_stock}_MA50'].iloc[-1] else "Bearish"
            rsi_signal = "Overbought" if data[f'{selected_stock}_RSI'].iloc[-1] > 70 else "Oversold" if data[f'{selected_stock}_RSI'].iloc[-1] < 30 else "Neutral"
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**MA Crossover Signal:** {ma_signal}")
                st.write("MA10 > MA50 indicates an upward trend (Bullish)")
            with col2:
                st.info(f"**RSI Signal:** {rsi_signal}")
                st.write("RSI > 70: Overbought | RSI < 30: Oversold")
            
            # Statistics
            st.markdown("### Statistical Summary")
            stats_df = data[[selected_stock, f'{selected_stock}_MA10', f'{selected_stock}_MA50', f'{selected_stock}_RSI']].describe()
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("Please fetch stock data first from the Dashboard tab.")
    
    with tab4:
        st.markdown('<h2 class="sub-header">About FinSight</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🎯 Project Overview
        
        FinSight is an advanced stock price prediction application that leverages machine learning 
        to forecast stock prices based on historical data and technical indicators.
        
        ### 📊 Features
        
        - **Real-time Data**: Fetches live stock data from Yahoo Finance
        - **Technical Indicators**: Calculates MA10, MA50, and RSI
        - **ML Predictions**: Uses trained Random Forest model for price predictions
        - **Interactive Visualizations**: Dynamic charts with Plotly
        - **Multiple Stocks**: Supports AAPL, MSFT, GOOGL, AMZN, and TSLA
        
        ### 🤖 Model Information
        
        - **Algorithm**: Random Forest Regressor
        - **Features**: Current Price, MA10, MA50, RSI
        - **Target**: Next day's closing price
        - **Performance**: R² Score: ~0.76, RMSE: ~$9.63
        
        ### 📈 Technical Indicators
        
        1. **MA10 (10-day Moving Average)**: Short-term trend indicator
        2. **MA50 (50-day Moving Average)**: Medium-term trend indicator
        3. **RSI (Relative Strength Index)**: Momentum oscillator (0-100)
           - RSI > 70: Overbought condition
           - RSI < 30: Oversold condition
        
        ### ⚠️ Disclaimer
        
        This application is for educational and informational purposes only. 
        The predictions should not be considered as financial advice. 
        Always conduct your own research and consult with financial professionals 
        before making investment decisions.
        
        ### 👨‍💻 Developer
        
        Built with ❤️ using:
        - Python
        - Streamlit
        - Scikit-learn
        - Plotly
        - yfinance
        
        ---
        
        **Version**: 1.0.0 | **Last Updated**: October 2025
        """)
        
        # Add some stock market quotes
        st.markdown("### 💡 Investment Wisdom")
        quotes = [
            "\"The stock market is a device for transferring money from the impatient to the patient.\" - Warren Buffett",
            "\"In investing, what is comfortable is rarely profitable.\" - Robert Arnott",
            "\"The four most dangerous words in investing are: 'This time it's different.'\" - Sir John Templeton",
            "\"Risk comes from not knowing what you're doing.\" - Warren Buffett"
        ]
        import random
        st.info(random.choice(quotes))

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with Streamlit | Data from Yahoo Finance</p>
        <p>© 2025 FinSight. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()
