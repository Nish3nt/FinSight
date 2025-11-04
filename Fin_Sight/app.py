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
from sklearn.metrics import mean_squared_error, r2_score
import time
from streamlit_option_menu import option_menu

# ====================== INITIAL SETUP ======================
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
current_date = datetime.now().date()

# ====================== DARK / LIGHT MODE TOGGLE ======================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

theme_icon = "Moon" if st.session_state.theme == "dark" else "Sun"
st.sidebar.button(f"{theme_icon} Toggle Theme", on_click=toggle_theme, use_container_width=True)

# Apply theme
if st.session_state.theme == "dark":
    st.markdown("""
    <style>
    .stApp { background: #0f172a; color: white; }
    .ticker-container { background: #1e293b; }
    section[data-testid="stSidebar"] { background: #1e293b; }
    .stSelectbox > div > div { background: #1e293b; color: white; }
    </style>
    """, unsafe_allow_html=True)
    plotly_template = "plotly_dark"
else:
    st.markdown("""
    <style>
    .stApp { background: #f8fafc; color: #1e293b; }
    .ticker-container { background: #e2e8f0; color: #1e293b; }
    section[data-testid="stSidebar"] { background: #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)
    plotly_template = "plotly_white"

st.set_page_config(page_title="FinSight", layout="wide")
st.title("**FinSight**: Real-Time Stock Intelligence")

# ====================== FULL S&P 500 TICKERS ======================
tickers = [
    'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES',
    'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT',
    'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK',
    'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF.B', 'BG', 'BIIB', 'BIO',
    'BK', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK.B', 'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG',
    'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CFG', 'CHD',
    'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF',
    'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPB', 'CPRT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTRA',
    'CTSH', 'CVS', 'CVX', 'D', 'DAL', 'DASH', 'DD', 'DE', 'DECK', 'DELL', 'DFS', 'DG', 'DGX', 'DHI', 'DHR',
    'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY',
    'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'ES', 'ESS',
    'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'F', 'FANG', 'FAST', 'FDS', 'FDX', 'FE', 'FFIV', 'FI',
    'FICO', 'FIS', 'FITB', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GE', 'GEHC', 'GEN', 'GEV',
    'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS',
    'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY',
    'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP',
    'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM',
    'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN',
    'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV',
    'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC',
    'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MS', 'MSCI', 'MSFT',
    'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC',
    'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NVDA', 'NVR', 'NWSA', 'NWS', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC',
    'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG',
    'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPL', 'PRU',
    'PSX', 'PTC', 'PWR', 'PYPL', 'QCOM', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST',
    'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNPS', 'SO', 'SOLV',
    'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T',
    'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TSLA', 'TFC', 'TFX', 'TGT', 'TJX', 'TKO', 'TMO', 'TMUS',
    'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER',
    'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VICI', 'VLO', 'VLTO', 'VMC',
    'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC',
    'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS'
]
tickers = sorted(set(tickers))

# ====================== SIDEBAR CONTROLS ======================
st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Main Stock", tickers, index=tickers.index('AAPL') if 'AAPL' in tickers else 0)
compare_ticker = st.sidebar.selectbox("Compare With", tickers, index=tickers.index('MSFT') if 'MSFT' in tickers else 1)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2018-01-01').date())
end_date = st.sidebar.date_input("End Date", current_date)

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()
if end_date > current_date:
    end_date = current_date
    st.warning(f"End date capped at today: {current_date}")

# ====================== DATA FETCHING ======================
@st.cache_resource(ttl=300)
def get_ticker(ticker): return yf.Ticker(ticker)
ticker_obj = get_ticker(selected_ticker)

@st.cache_data(ttl=300)
def get_news(ticker):
    try:
        api_key = st.secrets.get("NEWSAPI_KEY") or "d848a496d874401b9e2129a71adb57ba"
        if api_key != "YOUR_NEWSAPI_KEY":
            url = "https://newsapi.org/v2/everything"
            params = {'q': f'{ticker} stock OR {ticker} earnings OR {ticker} news', 'sortBy': 'publishedAt', 'pageSize': 15, 'language': 'en', 'apiKey': api_key}
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                arts = r.json().get('articles', [])
                if arts:
                    h, p, l = [], [], []
                    for a in arts:
                        t = a.get('title', '').strip()
                        s = a.get('source', {}).get('name', 'Source')
                        u = a.get('url', '#')
                        if t:
                            hl = f"**{t}** – {s}"
                            h.append(hl); p.append(f"{t} – {s}"); l.append(u)
                    return h, p, l
    except:
        st.warning("NewsAPI failed. Using Yahoo Finance.")
    try:
        news = ticker_obj.news
        if not news:
            return ["No recent headlines."], ["No recent headlines."], ["#"]
        h, p, l = [], [], []
        for item in news[:15]:
            t = item.get('title', '').strip()
            pub = item.get('publisher', 'Source').strip()
            u = item.get('link', '#')
            if t:
                hl = f"**{t}** – {pub}"
                h.append(hl); p.append(f"{t} – {pub}"); l.append(u)
        return h if h else ["Market quiet."], p, l
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

# ====================== TABS ======================
tab = option_menu(
    menu_title=None,
    options=["Data & Viz", "Predictions", "Sentiment", "Comparison", "Portfolio Analyzer"],
    icons=["table", "graph-up", "chat-dots", "arrow-left-right", "pie-chart"],
    orientation="horizontal"
)

# ====================== DATA & VIZ ======================
if tab == "Data & Viz":
    st.subheader(f"**{selected_ticker}** – Price History")
    st.dataframe(data_main.tail(100), use_container_width=True)
    st.download_button("Download CSV", data_main.to_csv().encode(), f"{selected_ticker}.csv")
    fig = px.line(data_main, x=data_main.index, y='Adj Close', title="Price Trend", template=plotly_template)
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    try:
        close = data_main['Close'].iloc[-1]
        change = (close - data_main['Close'].iloc[-8]) / data_main['Close'].iloc[-8] * 100 if len(data_main) > 7 else 0
        vol = data_main['Close'].pct_change().std() * np.sqrt(252) * 100
        c1.metric("Price", f"${close:,.2f}")
        c2.metric("7D Δ", f"{change:+.2f}%")
        c3.metric("Volatility", f"{vol:.1f}%")
    except:
        c1.metric("Price", "N/A")
    fig_c = go.Figure(go.Candlestick(x=data_main.index, open=data_main['Open'], high=data_main['High'], low=data_main['Low'], close=data_main['Close']))
    st.plotly_chart(fig_c.update_layout(title="Candlestick", height=600, template=plotly_template), use_container_width=True)

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
            pred = forecast[['ds', 'yhat']].tail(days)
            pred.columns = ['Date', 'Predicted']
            fig = plot_plotly(m, forecast)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pred.style.format({"Predicted": "{:.2f}"}))
    elif model == "LSTM" and len(data_main) >= 60:
        with st.spinner("Training LSTM..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data_main['Adj Close'].values.reshape(-1,1))
            train_size = int(len(scaled) * 0.8)
            train, test = scaled[:train_size], scaled[train_size:]
            def create_dataset(data, time_step=60):
                X, y = [], []
                for i in range(len(data) - time_step):
                    X.append(data[i:(i + time_step), 0])
                    y.append(data[i + time_step, 0])
                return np.array(X), np.array(y)
            X_train, y_train = create_dataset(train)
            X_test, y_test = create_dataset(test)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            start_time = time.time()
            lstm = Sequential([LSTM(50, return_sequences=True, input_shape=(60,1)), LSTM(50), Dense(1)])
            lstm.compile('adam', 'mse')
            lstm.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
            training_time = time.time() - start_time
            test_pred = lstm.predict(X_test, verbose=0)
            test_pred_inv = scaler.inverse_transform(test_pred)
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
            mse = mean_squared_error(y_test_inv, test_pred_inv)
            r2 = r2_score(y_test_inv, test_pred_inv)
            last = scaled[-60:].reshape(1,60,1)
            preds = []
            for _ in range(days):
                p = lstm.predict(last, verbose=0)[0][0]
                preds.append(p)
                last = np.append(last[:,1:,:], [[[p]]], axis=1)
            pred_vals = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            pred_df = pd.DataFrame({'Date': pd.date_range(start=data_main.index[-1]+pd.Timedelta(days=1), periods=days), 'Predicted': pred_vals})
            fig = px.line(pred_df, x='Date', y='Predicted', title="LSTM Forecast", template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pred_df.style.format({"Predicted": "{:.2f}"}))
            c1, c2, c3 = st.columns(3)
            c1.metric("MSE", f"{mse:.3f}")
            c2.metric("R²", f"{r2:.2f}")
            c3.metric("Training Time", f"{training_time:.1f}s")
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
    fig.add_trace(go.Scatter(x=data_main.index, y=df_main, name=selected_ticker, line=dict(color='#26A69A')))
    fig.add_trace(go.Scatter(x=data_compare.index, y=df_compare, name=compare_ticker, line=dict(color='#AB47BC')))
    fig.update_layout(title="Performance (%)", height=600, template=plotly_template)
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

# ====================== PORTFOLIO ANALYZER ======================
elif tab == "Portfolio Analyzer":
    st.subheader("Portfolio Analyzer")
    portfolio_tickers = st.multiselect("Select Tickers", tickers, default=[selected_ticker, compare_ticker])
    if len(portfolio_tickers) < 2:
        st.warning("Select at least 2 tickers.")
    else:
        weights = []
        cols = st.columns(len(portfolio_tickers))
        total_weight = 0
        for i, tick in enumerate(portfolio_tickers):
            w = cols[i].number_input(f"Weight {tick} (%)", 0.0, 100.0, 100.0 / len(portfolio_tickers))
            weights.append(w / 100)
            total_weight += w
        if abs(total_weight - 100) > 0.01:
            st.warning(f"Weights sum to {total_weight:.1f}%. Should be 100%.")
        else:
            data_dict = {}
            for tick in portfolio_tickers:
                data = fetch_stock_data(tick, start_date, end_date)
                if data is None:
                    st.error(f"Data missing for {tick}.")
                    st.stop()
                data_dict[tick] = data['Adj Close']
            portfolio_df = pd.DataFrame(data_dict)
            returns = portfolio_df.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            weights_np = np.array(weights)
            port_return = np.dot(mean_returns, weights_np)
            port_vol = np.sqrt(np.dot(weights_np.T, np.dot(cov_matrix, weights_np)))
            risk_free = 0.03  # Assume 3%
            sharpe = (port_return - risk_free) / port_vol
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Return", f"{port_return * 100:.2f}%")
            c2.metric("Portfolio Volatility", f"{port_vol * 100:.2f}%")
            c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            corr_matrix = returns.corr()
            fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Heatmap")
            st.plotly_chart(fig_heat, use_container_width=True)

# ====================== AUTO-SCROLLING NEWS TICKER (BOTTOM) ======================
st.markdown("---")
st.markdown("### Latest Headlines (24/7)")
all_headlines = news_headlines + news_headlines
animation_duration = max(15, len(news_headlines) * 3)
st.markdown(f"""
<style>
.ticker-container {{
    height: 180px;
    overflow: hidden;
    background: #0f172a;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.3);
    color: white;
    font-family: 'Segoe UI', sans-serif;
    position: relative;
}}
.ticker-wrapper {{
    animation: scroll-up {animation_duration}s linear infinite;
    will-change: transform;
}}
@keyframes scroll-up {{
    0% {{ transform: translateY(0); }}
    100% {{ transform: translateY(-50%); }}
}}
.ticker-item {{
    padding: 12px 0;
    font-size: 15px;
    line-height: 1.6;
    min-height: 40px; /* Consistent spacing */
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: normal; /* Allow wrapping for long titles */
    word-wrap: break-word;
}}
</style>
""", unsafe_allow_html=True)
html_content = '<div class="ticker-container"><div class="ticker-wrapper">'
for h in all_headlines:
    html_content += f'<div class="ticker-item">{h}</div>'
html_content += '</div></div>'
st.markdown(html_content, unsafe_allow_html=True)
