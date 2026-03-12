# app.py
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
from streamlit_option_menu import option_menu

# ====================== INITIAL SETUP ======================
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
current_date = datetime.now().date()
st.set_page_config(page_title="FinSight", layout="wide")
st.title("**FinSight**: Real-Time Stock Intelligence")

# ====================== FULL S&P 500 TICKERS (same list you provided) ======================
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

# ====================== SIDEBAR ======================
st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Main Stock", tickers, index=tickers.index('AAPL') if 'AAPL' in tickers else 0)
compare_ticker = st.sidebar.selectbox("Compare With", tickers, index=tickers.index('MSFT') if 'MSFT' in tickers else 1)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2010-01-01').date())
end_date = st.sidebar.date_input("End Date", current_date)

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()
if end_date > current_date:
    end_date = current_date
    st.warning(f"End date capped at today: {current_date}")

# ====================== FETCH TICKER OBJECT ======================
@st.cache_resource(ttl=300)
def get_ticker(ticker):
    return yf.Ticker(ticker)
ticker_obj = get_ticker(selected_ticker)

# ====================== FETCH NEWS (NewsAPI + Yahoo Fallback) ======================
@st.cache_data(ttl=300)
def get_news(ticker):
    try:
        api_key = st.secrets.get("NEWSAPI_KEY") or ""
        if api_key and api_key != "YOUR_NEWSAPI_KEY":
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{ticker} stock OR {ticker} earnings OR {ticker} news',
                'sortBy': 'publishedAt',
                'pageSize': 10,
                'language': 'en',
                'apiKey': api_key
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                j = r.json()
                articles = j.get('articles', [])
                if articles:
                    headlines = []
                    posts = []
                    links = []
                    for art in articles:
                        title = art.get('title', '').strip()
                        source = art.get('source', {}).get('name', 'Source').strip()
                        url = art.get('url', '#')
                        if title:
                            hl = f"**{title}** – {source}"
                            headlines.append(hl)
                            posts.append(f"{title} – {source}")
                            links.append(url)
                    return headlines, posts, links
    except:
        pass
    # Fallback: Yahoo Finance
    try:
        news = ticker_obj.news
        if not news or len(news) == 0:
            return ["No recent headlines."], ["No recent headlines."], ["#"]
        headlines = []
        posts = []
        links = []
        for item in news[:10]:
            title = item.get('title', '').strip()
            pub = item.get('publisher', 'Source').strip()
            url = item.get('link', '#')
            if title:
                hl = f"**{title}** – {pub}"
                headlines.append(hl)
                posts.append(f"{title} – {pub}")
                links.append(url)
        return headlines if headlines else ["Market quiet."], posts, links
    except:
        return ["News feed unavailable."], ["News feed unavailable."], ["#"]

news_headlines, news_posts, news_links = get_news(selected_ticker)

# ====================== COMPUTE VADER SENTIMENT ======================
@st.cache_data(ttl=300)
def compute_vader_sentiment(posts):
    return [sia.polarity_scores(post)['compound'] for post in posts]
vader_scores = compute_vader_sentiment(news_posts)

# ====================== FETCH STOCK DATA (includes Volume now) ======================
@st.cache_data(ttl=600)
def fetch_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        # ensure columns present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        # include Volume as well
        required = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in required:
            if col not in df.columns:
                df[col] = np.nan
        df = df[required].dropna(subset=['Adj Close'])
        if len(df) < 90:
            return None
        return df
    except Exception as e:
        return None

data_main = fetch_stock_data(selected_ticker, start_date, end_date)
data_compare = fetch_stock_data(compare_ticker, start_date, end_date)
if data_main is None or data_compare is None:
    st.warning("Not enough data for selected date range. Increase range or pick another ticker.")

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
    if data_main is not None:
        st.dataframe(data_main.tail(100), use_container_width=True)
        st.download_button("Download CSV", data_main.to_csv().encode(), f"{selected_ticker}.csv")
        fig = px.line(data_main, x=data_main.index, y='Adj Close', title="Price Trend")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data to display.")

# ====================== PREDICTIONS (MULTI-FEATURE LSTM, CACHED PER TICKER) ======================
elif tab == "Predictions":
    st.subheader("Price Forecast — Multi-feature LSTM (cached per ticker)")

    if data_main is None:
        st.error("Not enough data to run predictions. Try expanding date range or choosing a different ticker.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            days = st.slider("Forecast days", 1, 30, 7)
            time_step = st.slider("Lookback (days / time_step)", 60, 180, 90, step=10)
            epochs = st.slider("Epochs (training - first run)", 5, 100, 40, step=5)
            batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
            retrain = st.checkbox("Retrain model (force training now)", value=False)
        with col2:
            # --- Short user-facing model controls & how-to info ---
            st.markdown("### Model Controls")
            st.info(
                """
**How forecasting works**

This section runs a **multi-feature LSTM deep learning model** to predict future stock prices.

The model learns patterns from historical data using:

• Adjusted Close price  
• Trading Volume  
• SMA20 (20-day moving average)  
• SMA50 (50-day moving average)  
• RSI (momentum indicator)

**How to use**

1. Select how many **future days** you want to forecast.  
2. Adjust the **lookback window** (how many past days the model studies). Larger values let the model learn longer trends.  
3. Increase **epochs** to train stronger (first run will take longer).  
4. Use **Retrain model** only when you want to rebuild the model for the selected configuration.

⚡ The model is **cached per stock + parameters**, so predictions are fast after the first run.
                """
            )

        # Feature engineering (SMA, RSI)
        df_feat = data_main.copy()
        df_feat['SMA20'] = df_feat['Adj Close'].rolling(window=20).mean()
        df_feat['SMA50'] = df_feat['Adj Close'].rolling(window=50).mean()

        # RSI (14)
        delta = df_feat['Adj Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-9)
        df_feat['RSI'] = 100.0 - (100.0 / (1.0 + rs))

        # Use these features
        features = ['Adj Close', 'Volume', 'SMA20', 'SMA50', 'RSI']
        df_features = df_feat[features].dropna().copy()
        if len(df_features) < (time_step + 10):
            st.error(f"Not enough rows after computing indicators. Need at least {time_step+10} rows; got {len(df_features)}.")
        else:
            # Train & cache (cache key includes retrain flag)
            @st.cache_resource(ttl=24*3600)
            def train_and_cache_model(ticker, start_str, end_str, time_step, epochs, batch_size, retrain_flag):
                """
                Trains an LSTM and returns artifacts:
                model, scaler, df_used, time_step, train_n, X_test, y_test,
                inv_preds (backtest), inv_actuals (backtest), history (training history),
                training_time (datetime), training_duration_secs, epochs, batch_size, features
                """
                t0 = time.time()
                training_time = datetime.now()

                # Fetch data fresh inside the cached function
                df = yf.download(ticker, start=start_str, end=end_str, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                if 'Adj Close' not in df.columns and 'Close' in df.columns:
                    df['Adj Close'] = df['Close']

                # Indicators
                df['SMA20'] = df['Adj Close'].rolling(20).mean()
                df['SMA50'] = df['Adj Close'].rolling(50).mean()
                delta = df['Adj Close'].diff()
                up = delta.clip(lower=0)
                down = -1 * delta.clip(upper=0)
                roll_up = up.rolling(14).mean()
                roll_down = down.rolling(14).mean()
                rs = roll_up / (roll_down + 1e-9)
                df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

                df = df[['Adj Close', 'Volume', 'SMA20', 'SMA50', 'RSI']].dropna()

                # scaler & sequences
                scaler_local = MinMaxScaler()
                scaled_all = scaler_local.fit_transform(df.values)  # (N, features)

                X = []
                y = []
                for i in range(len(scaled_all) - time_step):
                    X.append(scaled_all[i:i + time_step, :])
                    y.append(scaled_all[i + time_step, 0])  # Adj Close scaled
                X = np.array(X)
                y = np.array(y)

                n_samples = X.shape[0]
                train_n = int(n_samples * 0.8)
                X_train = X[:train_n]
                y_train = y[:train_n]
                X_test = X[train_n:]
                y_test = y[train_n:]

                n_features = X.shape[2]

                model_local = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(time_step, n_features)),
                    Dropout(0.2),
                    LSTM(64),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])
                model_local.compile(optimizer='adam', loss='mse')

                # Early stopping if enough training samples
                callbacks = []
                if len(X_train) > 20:
                    callbacks = [EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=0)]
                    validation_split = 0.1
                else:
                    validation_split = 0.0

                history = model_local.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=0
                )

                # Backtest predictions (invert scaled preds to real prices)
                inv_preds = []
                inv_actuals = []
                # df index mapping: y_test[0] corresponds to df index at position (time_step + train_n)
                for i in range(len(X_test)):
                    pred_scaled = model_local.predict(X_test[i:i+1], verbose=0)[0, 0]
                    # template: take last row of X_test[i] (scaled) and replace adj close with pred_scaled
                    template_scaled = X_test[i, -1, :].copy()
                    template_scaled[0] = pred_scaled
                    inv_full = scaler_local.inverse_transform(template_scaled.reshape(1, -1))
                    inv_preds.append(float(inv_full[0, 0]))
                    idx_actual = time_step + train_n + i
                    actual_price = df['Adj Close'].iloc[idx_actual]
                    inv_actuals.append(float(actual_price))

                inv_preds = np.array(inv_preds)
                inv_actuals = np.array(inv_actuals)

                training_duration = time.time() - t0

                return {
                    'model': model_local,
                    'scaler': scaler_local,
                    'df_raw': df,
                    'time_step': time_step,
                    'train_n': train_n,
                    'X_test': X_test,
                    'y_test': y_test,
                    'inv_preds': inv_preds,
                    'inv_actuals': inv_actuals,
                    'history': history.history,
                    'training_time': training_time,
                    'training_duration_secs': training_duration,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'features': ['Adj Close', 'Volume', 'SMA20', 'SMA50', 'RSI']
                }

            # Prepare strings to pass into cached function (so cache key is deterministic)
            start_str = str(start_date)
            end_str = str(end_date)

            # Train / load cached model
            cache_msg = "Loading model (cached)..." if not retrain else "Retraining model (forced)..."
            with st.spinner(cache_msg):
                model_artifacts = train_and_cache_model(
                    selected_ticker, start_str, end_str,
                    time_step, epochs, batch_size, retrain
                )

            # Extract artifacts
            model = model_artifacts['model']
            scaler_model = model_artifacts['scaler']
            df_used = model_artifacts['df_raw']
            train_n = model_artifacts['train_n']
            inv_preds = model_artifacts['inv_preds']
            inv_actuals = model_artifacts['inv_actuals']
            history = model_artifacts['history']
            # training_time and duration intentionally available in artifacts if you need them later

            # Compute metrics on backtest
            if inv_preds.size > 0:
                mse = mean_squared_error(inv_actuals, inv_preds)
                r2 = r2_score(inv_actuals, inv_preds)
            else:
                mse = None
                r2 = None

            # -------- Model Performance Panel --------
            st.markdown("## Model Performance")

            perf_col1, perf_col2 = st.columns([1, 1])

            # Loss curves
            with perf_col1:
                st.write("### Training Loss Curve")
                epochs_range = list(range(1, len(history.get('loss', [])) + 1))
                loss_trace = go.Scatter(x=epochs_range, y=history.get('loss', []), mode='lines+markers', name='loss')
                fig_loss = go.Figure(data=[loss_trace])
                if 'val_loss' in history:
                    val_trace = go.Scatter(x=epochs_range, y=history.get('val_loss', []), mode='lines+markers', name='val_loss')
                    fig_loss.add_trace(val_trace)
                    fig_loss.update_layout(title="Loss & Validation Loss", xaxis_title="Epoch", yaxis_title="Loss")
                else:
                    fig_loss.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
                st.plotly_chart(fig_loss, use_container_width=True)

            # Backtest plot: predicted vs actual
            with perf_col2:
                st.write("### Backtest: Predictions vs Actual")
                if len(inv_preds) > 0:
                    # compute dates for backtest: indices correspond to df_used index at pos (time_step + train_n ..)
                    backtest_start_idx = model_artifacts['time_step'] + model_artifacts['train_n']
                    backtest_indices = df_used.index[backtest_start_idx: backtest_start_idx + len(inv_preds)]
                    df_backtest = pd.DataFrame({
                        'Date': backtest_indices,
                        'Actual': inv_actuals,
                        'Predicted': inv_preds
                    }).set_index('Date')
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['Actual'], name='Actual'))
                    fig_bt.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['Predicted'], name='Predicted'))
                    fig_bt.update_layout(title="Backtest: Actual vs Predicted", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig_bt, use_container_width=True)
                    # Show small metrics
                    if mse is not None:
                        st.write(f"Backtest samples: {len(inv_preds)} — Test MSE: {mse:.3f} — Test R²: {r2:.3f}")
                    else:
                        st.write(f"Backtest samples: {len(inv_preds)}")
                else:
                    st.info("Not enough backtest samples to plot predictions vs actual.")

            # -------- Future Forecast (recursive multi-step) --------
            st.markdown("## Forecast (future days)")
            # Build seed recent data from df_used last time_step rows
            recent = df_used.iloc[-time_step:].copy()
            recent_adj = recent['Adj Close'].tolist()
            recent_vol = recent['Volume'].tolist()
            future_preds = []
            for step in range(days):
                sma20 = np.mean(recent_adj[-20:]) if len(recent_adj) >= 20 else np.mean(recent_adj)
                sma50 = np.mean(recent_adj[-50:]) if len(recent_adj) >= 50 else np.mean(recent_adj)
                # RSI on recent_adj
                window = pd.Series(recent_adj)
                delta_local = window.diff()
                up_local = delta_local.clip(lower=0).fillna(0)
                down_local = -1 * delta_local.clip(upper=0).fillna(0)
                roll_up_local = up_local.rolling(14).mean().iloc[-1] if len(up_local) >= 14 else up_local.mean()
                roll_down_local = down_local.rolling(14).mean().iloc[-1] if len(down_local) >= 14 else down_local.mean()
                rs_local = (roll_up_local / (roll_down_local + 1e-9)) if (roll_down_local + 1e-9) != 0 else 0.0
                rsi_local = 100.0 - (100.0 / (1.0 + rs_local))
                vol_local = recent_vol[-1] if len(recent_vol) > 0 else 0.0
                feat_row = np.array([[recent_adj[-1], vol_local, sma20, sma50, rsi_local]])
                feat_scaled = scaler_model.transform(feat_row)
                # prepare input window scaled
                scaled_full = scaler_model.transform(df_used.values[-time_step:])
                input_window = np.vstack([scaled_full])
                input_window[-1] = feat_scaled
                input_window = input_window.reshape(1, time_step, scaled_full.shape[1])
                pred_scaled = model.predict(input_window, verbose=0)[0, 0]
                template_scaled = feat_scaled.copy().reshape(1, -1)
                template_scaled[0, 0] = pred_scaled
                inv = scaler_model.inverse_transform(template_scaled)
                pred_price = float(inv[0, 0])
                future_preds.append(pred_price)
                recent_adj.append(pred_price)
                recent_vol.append(vol_local)

            future_dates = pd.date_range(start=data_main.index[-1] + pd.Timedelta(days=1), periods=days, freq='B')
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_preds})
            fig = px.line(future_df, x='Date', y='Predicted', title=f"{selected_ticker} — LSTM Forecast ({len(future_preds)} days)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(future_df.style.format({"Predicted": "{:.2f}"}), use_container_width=True)

            # final metrics display
            c1, c2, c3 = st.columns(3)
            if mse is not None:
                c1.metric("Test MSE", f"{mse:.3f}")
                c2.metric("Test R²", f"{r2:.3f}")
            else:
                c1.metric("Test MSE", "N/A")
                c2.metric("Test R²", "N/A")
            c3.metric("Model", "Multi-feature LSTM (cached)")

# ====================== SENTIMENT ======================
elif tab == "Sentiment":
    st.subheader("News Sentiment")
    if news_posts:
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
    else:
        st.info("No news available.")

# ====================== COMPARISON ======================
elif tab == "Comparison":
    st.subheader(f"**{selected_ticker} vs {compare_ticker}**")
    if data_main is not None and data_compare is not None:
        base_main = data_main['Adj Close'].iloc[0]
        base_compare = data_compare['Adj Close'].iloc[0]
        df_main = (data_main['Adj Close'] / base_main - 1) * 100
        df_compare = (data_compare['Adj Close'] / base_compare - 1) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_main.index, y=df_main, name=selected_ticker, line=dict(color='#26A69A')))
        fig.add_trace(go.Scatter(x=data_compare.index, y=df_compare, name=compare_ticker, line=dict(color='#AB47BC')))
        fig.update_layout(title="Performance (%)", height=600, template="plotly_white")
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
    else:
        st.error("Not enough data to compare.")

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
    min-height: 40px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: normal;
    word-wrap: break-word;
}}
</style>
""", unsafe_allow_html=True)
html_content = '<div class="ticker-container"><div class="ticker-wrapper">'
for h in all_headlines:
    html_content += f'<div class="ticker-item">{h}</div>'
html_content += '</div></div>'
st.markdown(html_content, unsafe_allow_html=True)
