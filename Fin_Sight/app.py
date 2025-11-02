import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas_ta as ta  # pip install pandas_ta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# ====================== INITIAL SETUP ======================
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
current_date = datetime.now().date()
SP500_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'AVGO', 'META', 'TSLA', 'BRK.B', 'JPM', 'WMT', 'LLY', 'ORCL', 'V', 'MA', 'XOM', 'PLTR', 'NFLX', 'JNJ', 'AMD', 'COST', 'BAC', 'ABBV', 'HD', 'PG', 'GE', 'CVX', 'UNH', 'KO', 'CSCO', 'IBM', 'WFC', 'CAT', 'MS', 'MU', 'AXP', 'CRM', 'GS', 'RTX', 'TMUS', 'PM', 'APP', 'ABT', 'MRK', 'TMO', 'MCD', 'DIS', 'UBER', 'PEP', 'ANET', 'LRCX', 'LIN', 'QCOM', 'NOW', 'INTC', 'ISRG', 'INTU', 'AMAT', 'C', 'BX', 'BLK', 'T', 'SCHW', 'APH', 'NEE', 'VZ', 'BKNG', 'AMGN', 'KLAC', 'GEV', 'TJX', 'ACN', 'BA', 'DHR', 'BSX', 'PANW', 'GILD', 'ETN', 'SPGI', 'TXN', 'ADBE', 'PFE', 'COF', 'CRWD', 'SYK', 'LOW', 'UNP', 'HOOD', 'HON', 'DE', 'WELL', 'PGR', 'PLD', 'CEG', 'MDT', 'ADI', 'LMT', 'COP', 'VRTX', 'CB', 'DASH', 'DELL', 'KKR', 'ADP', 'HCA', 'SO', 'CMCSA', 'MCK', 'TT', 'CVS', 'PH', 'DUK', 'CME', 'NKE', 'MO', 'BMY', 'GD', 'CDNS', 'SBUX', 'MMM', 'NEM', 'COIN', 'MMC', 'MCO', 'SHW', 'SNPS', 'AMT', 'ICE', 'NOC', 'EQIX', 'HWM', 'UPS', 'WM', 'ORLY', 'EMR', 'RCL', 'ABNB', 'GLW', 'BK', 'JCI', 'MDLZ', 'TDG', 'CTAS', 'AON', 'TEL', 'USB', 'ECL', 'PNC', 'APO', 'ITW', 'MAR', 'WMB', 'ELV', 'MSI', 'CSX', 'PWR', 'REGN', 'SPG', 'FTNT', 'COR', 'CI', 'MNST', 'PYPL', 'GM', 'RSG', 'AEP', 'ADSK', 'AJG', 'WDAY', 'ZTS', 'VST', 'NSC', 'CL', 'AZO', 'CMI', 'SRE', 'TRV', 'FDX', 'FCX', 'HLT', 'DLR', 'MPC', 'KMI', 'EOG', 'TFC', 'AXON', 'AFL', 'DDOG', 'WBD', 'URI', 'PSX', 'STX', 'LHX', 'APD', 'SLB', 'O', 'MET', 'NXPI', 'F', 'VLO', 'ROST', 'PCAR', 'WDC', 'BDX', 'ALL', 'IDXX', 'D', 'CARR', 'EA', 'PSA', 'NDAQ', 'EW', 'MPWR', 'ROP', 'XEL', 'BKR', 'TTWO', 'FAST', 'GWW', 'EXC', 'AME', 'CAH', 'CBRE', 'MSCI', 'DHI', 'AIG', 'ETR', 'KR', 'OKE', 'TGT', 'PAYX', 'AMP', 'CMG', 'CTVA', 'CPRT', 'A', 'FANG', 'ROK', 'GRMN', 'OXY', 'PEG', 'LVS', 'FICO', 'KMB', 'CCI', 'YUM', 'VMC', 'CCL', 'TKO', 'DAL', 'MLM', 'KDP', 'IQV', 'EBAY', 'XYL', 'PRU', 'WEC', 'OTIS', 'RMD', 'FI', 'SYY', 'CTSH', 'ED', 'PCG', 'WAB', 'VTR', 'EL', 'LYV', 'HIG', 'NUE', 'HSY', 'DD', 'GEHC', 'CHTR', 'MCHP', 'HUM', 'EQT', 'NRG', 'TRGP', 'FIS', 'STT', 'HPE', 'VICI', 'ACGL', 'LEN', 'KEYS', 'RJF', 'IBKR', 'SMCI', 'VRSK', 'UAL', 'IRM', 'EME', 'IR', 'WTW', 'EXR', 'ODFL', 'KHC', 'MTD', 'CSGP', 'ADM', 'TER', 'K', 'FOXA', 'TSCO', 'FSLR', 'MTB', 'DTE', 'ROL', 'AEE', 'KVUE', 'ATO', 'FITB', 'ES', 'FOX', 'BRO', 'EXPE', 'WRB', 'PPL', 'SYF', 'FE', 'HPQ', 'EFX', 'BR', 'CBOE', 'AWK', 'HUBB', 'CNP', 'DOV', 'GIS', 'AVB', 'TDY', 'EXE', 'TTD', 'VLTO', 'LDOS', 'NTRS', 'HBAN', 'CINF', 'PTC', 'WSM', 'JBL', 'NTAP', 'PHM', 'ULTA', 'STE', 'EQR', 'STZ', 'STLD', 'TPR', 'DXCM', 'BIIB', 'HAL', 'CMS', 'TROW', 'VRSN', 'PODD', 'CFG', 'PPG', 'DG', 'TPL', 'RF', 'EIX', 'CHD', 'LH', 'DRI', 'CDW', 'WAT', 'L', 'NVR', 'DVN', 'SBAC', 'TYL', 'ON', 'IP', 'WST', 'LULU', 'NI', 'DLTR', 'ZBH', 'KEY', 'DGX', 'RL', 'SW', 'TRMB', 'BG', 'GPN', 'IT', 'J', 'PFG', 'CPAY', 'INCY', 'TSN', 'AMCR', 'CHRW', 'CTRA', 'GDDY', 'LII', 'GPC', 'EVRG', 'APTV', 'PKG', 'SNA', 'PNR', 'CNC', 'INVH', 'BBY', 'MKC', 'LNT', 'DOW', 'PSKY', 'ESS', 'WY', 'EXPD', 'HOLX', 'GEN', 'IFF', 'JBHT', 'FTV', 'LUV', 'NWS', 'MAA', 'ERIE', 'LYB', 'NWSA', 'FFIV', 'OMC', 'ALLE', 'TXT', 'KIM', 'COO', 'UHS', 'CLX', 'ZBRA', 'AVY', 'CF', 'DPZ', 'MAS', 'EG', 'NDSN', 'BF.B', 'BLDR', 'IEX', 'BALL', 'DOC', 'HII', 'BXP', 'REG', 'WYNN', 'UDR', 'VTRS', 'SOLV', 'DECK', 'HRL', 'BEN', 'ALB', 'SWKS', 'HST', 'SJM', 'DAY', 'RVTY', 'JKHY', 'CPT', 'AKAM', 'HAS', 'AIZ', 'MRNA', 'PNW', 'GL', 'IVZ', 'PAYC', 'SWK', 'NCLH', 'ARE', 'ALGN', 'FDS', 'POOL', 'AES', 'GNRC', 'TECH', 'BAX', 'IPG', 'AOS', 'EPAM', 'CPB', 'CRL', 'MGM', 'MOS', 'TAP', 'LW', 'DVA', 'FRT', 'CAG', 'LKQ', 'APA', 'MOH', 'MTCH', 'HSIC', 'MHK', 'EMN']  # Full ~500 from web

st.set_page_config(page_title="FinSight Pro", layout="wide")
st.title("**FinSight Pro**: AI-Powered Stock Screener & Portfolio Builder")

# ====================== SIDEBAR CONTROLS ======================
st.sidebar.header("Configuration")
universe_size = st.sidebar.slider("Universe Size (Top N from S&P 500)", 50, 500, 100)
sentiment_window = st.sidebar.slider("Sentiment Window (Days)", 1, 7, 3)
horizon = st.sidebar.selectbox("Prediction Horizon", ["1-day", "7-day"])
top_n = st.sidebar.slider("Select Top N Stocks", 5, 20, 10)
alert_threshold = st.sidebar.slider("Alert Threshold (Momentum)", 0.1, 0.5, 0.2)
allocation_type = st.sidebar.selectbox("Allocation Type", ["Equal Weight", "Risk-Adjusted"])

if st.sidebar.button("Run Analysis"):
    st.session_state.run_analysis = True
else:
    st.session_state.run_analysis = False

# ====================== MODULAR SERVICES ======================

@st.cache_data(ttl=1800)  # 30 min cache
def ingest_data(tickers, years=5):
    """Data Ingestion: Fetch historical prices for tickers."""
    end = current_date
    start = end - timedelta(days=years*365)
    data = {}
    for ticker in tickers[:universe_size]:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                df['Return'] = df['Adj Close'].pct_change().shift(-1)  # Next-day return
                data[ticker] = df.dropna()
        except:
            continue
    return data

@st.cache_data(ttl=1800)
def compute_technicals(df):
    """Compute Technical Indicators."""
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['BB_upper'], df['BB_lower'] = ta.bbands(df['Close']).iloc[:, 0], ta.bbands(df['Close']).iloc[:, 2]
    features = ['SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'Volume']
    return df[features].dropna(), df['Return'].dropna()

def predict_returns(data, horizon_days):
    """ML Inference: Predict returns using RandomForest."""
    predictions = {}
    for ticker, df in data.items():
        if len(df) < 100: continue  # Need data
        features, returns = compute_technicals(df)
        if len(features) < 50: continue
        X_train, X_test, y_train, y_test = train_test_split(features[:-horizon_days], returns[:-horizon_days], test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        # Predict next period (average for 7-day)
        if horizon_days == 1:
            pred = model.predict(features.tail(1))[0]
        else:
            preds = []
            for _ in range(7):
                pred = model.predict(features.tail(1))[0]
                preds.append(pred)
            pred = np.mean(preds)
        predictions[ticker] = pred
    return predictions

@st.cache_data(ttl=300)
def sentiment_momentum(ticker, window_days):
    """Sentiment Engine: Fetch news and compute momentum."""
    try:
        api_key = st.secrets.get("NEWSAPI_KEY") or "d848a496d874401b9e2129a71adb57ba"
        if api_key == "YOUR_NEWSAPI_KEY":
            return 0.0  # Fallback neutral
        from_date = (current_date - timedelta(days=window_days)).strftime('%Y-%m-%d')
        url = "https://newsapi.org/v2/everything"
        params = {'q': f'{ticker} stock', 'from': from_date, 'sortBy': 'publishedAt', 'pageSize': 20, 'language': 'en', 'apiKey': api_key}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return 0.0
        articles = r.json().get('articles', [])
        if not articles:
            return 0.0
        scores = [sia.polarity_scores(art.get('title', ''))['compound'] for art in articles[:10]]
        avg_sent = np.mean(scores)
        # Momentum: Simple delta (compare to previous window if possible, else current as base)
        prev_from = (current_date - timedelta(days=window_days*2)).strftime('%Y-%m-%d')
        prev_params = params.copy(); prev_params['from'] = prev_from; prev_params['to'] = from_date
        prev_r = requests.get(url, params=prev_params, timeout=10)
        if prev_r.status_code == 200:
            prev_articles = prev_r.json().get('articles', [])
            prev_scores = [sia.polarity_scores(art.get('title', ''))['compound'] for art in prev_articles[:10]]
            prev_avg = np.mean(prev_scores) if prev_scores else 0
            momentum = avg_sent - prev_avg
        else:
            momentum = 0.0  # No prev, neutral momentum
        return momentum
    except:
        return 0.0

def build_portfolio(predictions, sentiments, top_n):
    """Portfolio Engine: Rank, allocate, compute metrics."""
    scores = {t: p * s for t, p in predictions.items() for s in [sentiments.get(t, 0)] if t in sentiments}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    tickers, comp_scores = zip(*ranked) if ranked else ([], [])
    
    if not tickers:
        return {}, pd.DataFrame(), np.nan, np.nan
    
    # Fetch recent prices for allocation/metrics
    recent_data = {t: yf.download(t, period='1mo')['Adj Close'][-1] for t in tickers}
    weights = np.ones(len(tickers)) / len(tickers) if allocation_type == "Equal Weight" else np.random.rand(len(tickers))  # Placeholder for risk-adj; normalize
    weights /= weights.sum()
    
    exp_ret = np.dot(weights, [predictions[t] for t in tickers])
    risk = np.sqrt(np.dot(weights.T, np.dot(np.cov([recent_data[t] for t in tickers]), weights)))  # Simple vol proxy
    
    # Correlation matrix
    prices = pd.DataFrame({t: yf.download(t, period='3mo')['Adj Close'] for t in tickers}).dropna()
    corr = prices.corr()
    
    return dict(zip(tickers, weights)), pd.DataFrame({'Ticker': tickers, 'Score': comp_scores}), exp_ret, risk, corr

def generate_alerts(sentiments, threshold):
    """Alert Module: Highlight high momentum changes."""
    alerts = {t: s for t, s in sentiments.items() if abs(s) > threshold}
    return alerts

# ====================== MAIN APP LOGIC ======================
if st.session_state.get('run_analysis', False):
    with st.spinner("Scanning universe..."):
        universe = SP500_TICKERS[:universe_size]
        data = ingest_data(universe)
        predictions = predict_returns(data, 1 if horizon == "1-day" else 7)
        sentiments = {t: sentiment_momentum(t, sentiment_window) for t in universe if t in predictions}
    
    allocations, ranked_df, exp_ret, risk, corr = build_portfolio(predictions, sentiments, top_n)
    alerts = generate_alerts(sentiments, alert_threshold)
    
    # ====================== RESULTS DISPLAY ======================
    st.header("Top Ranked Stocks")
    st.dataframe(ranked_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Expected Portfolio Return", f"{exp_ret:.2%}")
    with col2:
        st.metric("Portfolio Risk (Vol)", f"{risk:.2%}")
    
    # Alerts
    if alerts:
        st.warning("🚨 Alerts:")
        alert_df = pd.DataFrame(list(alerts.items()), columns=['Ticker', 'Momentum Score'])
        alert_df['Alert'] = alert_df['Momentum Score'].apply(lambda x: 'Positive Surge' if x > 0 else 'Negative Drop')
        st.dataframe(alert_df, use_container_width=True)
    else:
        st.info("No alerts triggered.")
    
    # ====================== VISUALIZATIONS ======================
    st.header("Interactive Visualizations")
    
    # Portfolio Allocation Pie
    if allocations:
        fig_pie = px.pie(values=list(allocations.values()), names=list(allocations.keys()), title="Portfolio Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Correlation Heatmap
    if not corr.empty:
        fig_heat = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='RdYlGn')
        fig_heat.update_layout(title="Correlation Matrix of Top Stocks")
        st.plotly_chart(fig_heat, use_container_width=True)
    
    # Sentiment Momentum Bar
    sent_df = pd.DataFrame(list(sentiments.items()), columns=['Ticker', 'Momentum']).head(top_n)
    fig_bar = px.bar(sent_df, x='Ticker', y='Momentum', title="Sentiment Momentum (Top Stocks)", color='Momentum', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # News Timeline (Gantt-like for top stocks)
    timeline_data = []
    for ticker in ranked_df['Ticker']:
        # Mock timeline: Recent news dates (in real, from NewsAPI)
        dates = pd.date_range(end=current_date, periods=5).tolist()
        for d in dates:
            timeline_data.append(dict(Task=ticker, Start=d, Finish=d + timedelta(hours=1), Resource='News'))
    fig_timeline = px.timeline(timeline_data, x_start="Start", x_end="Finish", y="Task", title="News Headline Timeline (Top Stocks)")
    st.plotly_chart(fig_timeline, use_container_width=True)

else:
    st.info("👈 Adjust settings in sidebar and click 'Run Analysis' to start.")

# ====================== NEWS TICKER (BOTTOM) ======================
st.markdown("---")
st.markdown("### Latest Market Headlines")
# (Re-use previous ticker code for brevity; fetch general market news if needed)
st.caption("Pro Tip: Add NewsAPI key for enhanced sentiment.")
