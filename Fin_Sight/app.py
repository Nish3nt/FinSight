# ====================== app.py (STABLE FIX - Correct Forecast + Safe Architecture) ======================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
from streamlit_option_menu import option_menu

# ====================== INITIAL SETUP ======================
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
current_date = datetime.now().date()
st.set_page_config(page_title="FinSight", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] > div:first-child { background-color: #0b1220; padding: 16px 12px; }
[data-testid="stSidebar"] .css-1d391kg { color: #e6eef8; }
.block-container { padding-top: 0.6rem; padding-bottom: 0.4rem; }
.model-box { background-color: #000000; padding: 18px; border-radius: 12px; border: 1px solid #111827; font-size: 14px; color: #e6eef8; }
.compact-model-info { font-size:12px; color:#cbd5e1; padding:8px 6px; }
.skel-card { background: linear-gradient(90deg, #111827 25%, #0b1220 50%, #111827 75%); background-size: 200% 100%; animation: shimmer 1.4s linear infinite; height: 120px; border-radius: 10px; margin-bottom: 12px; }
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
</style>
""", unsafe_allow_html=True)

st.title("**FinSight**: Real-Time Stock Intelligence")

# ====================== S&P 500 TICKERS ======================
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
selected_ticker = st.sidebar.selectbox("Main Stock", tickers, index=tickers.index('AAPL'))
compare_ticker  = st.sidebar.selectbox("Compare With", tickers, index=tickers.index('MSFT'))
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2010-01-01').date())
end_date   = st.sidebar.date_input("End Date", current_date)
if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()
if end_date > current_date:
    end_date = current_date

# ====================== FETCH TICKER ======================
@st.cache_resource(ttl=300)
def get_ticker(ticker):
    return yf.Ticker(ticker)
ticker_obj = get_ticker(selected_ticker)

# ====================== FINNHUB NEWS ======================
@st.cache_data(ttl=300)
def get_news(ticker):
    api_key = "d6qgus9r01qhcrmk4od0d6qgus9r01qhcrmk4odg"
    try:
        to_date   = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={api_key}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            articles = r.json()
            headlines, posts, links = [], [], []
            for art in articles[:10]:
                title    = art.get('headline', '').strip()
                source   = art.get('source', 'Finnhub')
                url_link = art.get('url', '#')
                if title:
                    headlines.append(f"**{title}** – {source}")
                    posts.append(f"{title} – {source}")
                    links.append(url_link)
            return (headlines if headlines else ["No recent news."]), posts, links
        return [f"Finnhub API Error {r.status_code}"], [], []
    except Exception as e:
        return [f"Finnhub error: {str(e)}"], [], []

news_headlines, news_posts, news_links = get_news(selected_ticker)

# ====================== VADER SENTIMENT ======================
@st.cache_data(ttl=300)
def compute_vader_sentiment(posts):
    return [sia.polarity_scores(p)['compound'] for p in posts]
vader_scores = compute_vader_sentiment(news_posts)

# ====================== FETCH STOCK DATA ======================
@st.cache_data(ttl=600)
def fetch_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        required = ['Open','High','Low','Close','Adj Close','Volume']
        for col in required:
            if col not in df.columns: df[col] = np.nan
        df = df[required].dropna(subset=['Adj Close'])
        return df if len(df) >= 90 else None
    except:
        return None

data_main    = fetch_stock_data(selected_ticker, start_date, end_date)
data_compare = fetch_stock_data(compare_ticker,  start_date, end_date)

# ====================== TABS ======================
tab = option_menu(None,
    ["Data & Viz", "Predictions", "Sentiment", "Comparison", "Portfolio Analyzer"],
    icons=["table","graph-up","chat-dots","arrow-left-right","pie-chart"],
    orientation="horizontal")

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

# ====================== PREDICTIONS ======================
elif tab == "Predictions":
    st.subheader("Price Forecast — Multi-feature LSTM")

    if data_main is None:
        st.error("Not enough data. Try expanding date range or choosing a different ticker.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            days       = st.slider("Forecast days", 1, 30, 7)
            time_step  = st.slider("Lookback window (days)", 60, 180, 90, step=10)
            epochs     = st.slider("Training epochs", 10, 150, 60, step=5)
            batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
            retrain    = st.checkbox("⚠️ Force retrain model", value=False)
        with col2:
            st.markdown("""
            <div class="model-box">
            <b>Model Controls</b><br><br>
            <b>Features used:</b><br>
            • Adjusted Close price<br>
            • Trading Volume<br>
            • SMA20 &amp; SMA50<br>
            • RSI (momentum)<br><br>
            <b>Tips:</b><br>
            1. Use 60–100 epochs for best accuracy.<br>
            2. Lookback 90 days works well for most stocks.<br>
            3. Always tick <b>Force retrain</b> after changing any setting.<br><br>
            ⚡ Cached per stock — fast after first run.
            </div>""", unsafe_allow_html=True)

        # ====================== TRAIN MODEL ======================
        @st.cache_resource(ttl=24*3600)
        def train_and_cache_model(ticker, start_str, end_str, time_step, epochs, batch_size, retrain_flag):
            t0            = time.time()
            training_time = datetime.now()

            df = yf.download(ticker, start=start_str, end=end_str, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if 'Adj Close' not in df.columns and 'Close' in df.columns:
                df['Adj Close'] = df['Close']

            # Feature engineering
            df['SMA20'] = df['Adj Close'].rolling(20).mean()
            df['SMA50'] = df['Adj Close'].rolling(50).mean()
            delta   = df['Adj Close'].diff()
            up      = delta.clip(lower=0)
            down    = -1 * delta.clip(upper=0)
            roll_up = up.rolling(14).mean()
            roll_dn = down.rolling(14).mean()
            rs      = roll_up / (roll_dn + 1e-9)
            df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
            df = df[['Adj Close','Volume','SMA20','SMA50','RSI']].dropna()

            scaler_local = MinMaxScaler()
            scaled_all   = scaler_local.fit_transform(df.values)

            X, y = [], []
            for i in range(len(scaled_all) - time_step):
                X.append(scaled_all[i : i + time_step, :])
                y.append(scaled_all[i + time_step, 0])
            X = np.array(X)
            y = np.array(y)

            n_samples  = X.shape[0]
            train_n    = int(n_samples * 0.8)
            X_train, y_train = X[:train_n], y[:train_n]
            X_test,  y_test  = X[train_n:],  y[train_n:]
            n_features = X.shape[2]

            # -------------------------------------------------------
            # ARCHITECTURE: proven 2-layer LSTM — stable & accurate
            # (3-layer needs 150+ epochs; overkill for this use-case)
            # -------------------------------------------------------
            model_local = Sequential([
                LSTM(128, return_sequences=True, input_shape=(time_step, n_features)),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.15),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model_local.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

            callbacks = []
            val_split = 0.0
            if len(X_train) > 20:
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=12,
                                  restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=6, min_lr=1e-6, verbose=0)
                ]
                val_split = 0.1

            history = model_local.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=val_split,
                callbacks=callbacks,
                verbose=0
            )

            # Backtest
            inv_preds, inv_actuals = [], []
            for i in range(len(X_test)):
                pred_sc = model_local.predict(X_test[i:i+1], verbose=0)[0, 0]
                tmpl    = X_test[i, -1, :].copy()
                tmpl[0] = pred_sc
                inv_preds.append(float(scaler_local.inverse_transform(tmpl.reshape(1,-1))[0, 0]))
                inv_actuals.append(float(df['Adj Close'].iloc[time_step + train_n + i]))

            return {
                'model':                  model_local,
                'scaler':                 scaler_local,
                'df_raw':                 df,
                'time_step':              time_step,
                'train_n':                train_n,
                'X_test':                 X_test,
                'y_test':                 y_test,
                'inv_preds':              np.array(inv_preds),
                'inv_actuals':            np.array(inv_actuals),
                'history':                history.history,
                'training_time':          training_time,
                'training_duration_secs': time.time() - t0,
                'epochs':                 epochs,
                'batch_size':             batch_size,
                'features':               ['Adj Close','Volume','SMA20','SMA50','RSI']
            }

        ph = st.empty()
        with ph.container():
            st.markdown('<div class="skel-card"></div>', unsafe_allow_html=True)
            st.markdown('<div class="skel-card"></div>', unsafe_allow_html=True)

        try:
            model_artifacts = train_and_cache_model(
                selected_ticker, str(start_date), str(end_date),
                time_step, epochs, batch_size, retrain
            )
        finally:
            ph.empty()

        model         = model_artifacts['model']
        scaler_model  = model_artifacts['scaler']
        df_used       = model_artifacts['df_raw']
        train_n       = model_artifacts['train_n']
        inv_preds     = model_artifacts['inv_preds']
        inv_actuals   = model_artifacts['inv_actuals']
        history       = model_artifacts['history']
        training_time = model_artifacts['training_time']

        if inv_preds.size > 0:
            mse       = mean_squared_error(inv_actuals, inv_preds)
            r2        = r2_score(inv_actuals, inv_preds)
            residuals = inv_actuals - inv_preds
            resid_std = float(np.std(residuals))
        else:
            mse = r2 = None
            resid_std = float(
                df_used['Adj Close'].pct_change().std() * df_used['Adj Close'].iloc[-1]
            )

        # Compact info bar
        try:
            age     = datetime.now() - training_time
            age_str = f"{age.days}d {age.seconds//3600}h {(age.seconds%3600)//60}m"
            st.markdown(f"""
            <div class="compact-model-info">
            <b>Model Info</b> — Trained: {training_time.strftime('%Y-%m-%d %H:%M:%S')}
            &nbsp;|&nbsp; Age: {age_str}
            &nbsp;|&nbsp; Cached: {'Yes' if not retrain else 'No (forced)'}
            &nbsp;|&nbsp; Lookback: {model_artifacts['time_step']}d
            &nbsp;|&nbsp; Epochs: {model_artifacts['epochs']}
            &nbsp;|&nbsp; Batch: {model_artifacts['batch_size']}
            </div>""", unsafe_allow_html=True)
        except Exception:
            pass

        # ---- Performance panel ----
        st.markdown("## Model Performance")
        pc1, pc2 = st.columns(2)

        with pc1:
            st.write("### Training Loss Curve")
            ep_range = list(range(1, len(history.get('loss',[])) + 1))
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=ep_range, y=history.get('loss',[]),
                                          mode='lines+markers', name='Train Loss'))
            if 'val_loss' in history:
                fig_loss.add_trace(go.Scatter(x=ep_range, y=history['val_loss'],
                                              mode='lines+markers', name='Val Loss'))
            fig_loss.update_layout(title="Loss Curve", xaxis_title="Epoch", yaxis_title="MSE")
            st.plotly_chart(fig_loss, use_container_width=True)

        with pc2:
            st.write("### Backtest: Actual vs Predicted")
            if len(inv_preds) > 0:
                bt_start = model_artifacts['time_step'] + model_artifacts['train_n']
                bt_idx   = df_used.index[bt_start : bt_start + len(inv_preds)]
                df_bt    = pd.DataFrame({'Actual': inv_actuals, 'Predicted': inv_preds}, index=bt_idx)
                fig_bt   = go.Figure()
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Actual'],
                                            name='Actual',    line=dict(color='#1f77b4')))
                fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Predicted'],
                                            name='Predicted', line=dict(color='#ff7f0e')))
                fig_bt.update_layout(title="Backtest", xaxis_title="Date", yaxis_title="Price ($)")
                st.plotly_chart(fig_bt, use_container_width=True)
                if mse is not None:
                    st.write(f"Samples: {len(inv_preds)} | MSE: {mse:.3f} | R²: {r2:.3f}")
            else:
                st.info("Not enough test samples.")

        # ====================== FIXED RECURSIVE FORECAST ======================
        st.markdown("## Forecast (future days)")

        recent_adj = df_used['Adj Close'].tolist()
        recent_vol = df_used['Volume'].tolist()

        # Seed the sliding window with the last time_step scaled rows
        recent_scaled = scaler_model.transform(df_used.values[-time_step:]).tolist()

        future_preds = []
        for step in range(days):
            sma20 = np.mean(recent_adj[-20:]) if len(recent_adj) >= 20 else np.mean(recent_adj)
            sma50 = np.mean(recent_adj[-50:]) if len(recent_adj) >= 50 else np.mean(recent_adj)

            w_   = pd.Series(recent_adj[-30:])
            d_   = w_.diff()
            up_  = d_.clip(lower=0).fillna(0)
            dn_  = -1 * d_.clip(upper=0).fillna(0)
            ru_  = up_.rolling(14).mean().iloc[-1] if len(up_) >= 14 else up_.mean()
            rd_  = dn_.rolling(14).mean().iloc[-1] if len(dn_) >= 14 else dn_.mean()
            rsi_ = 100.0 - (100.0 / (1.0 + ru_ / (rd_ + 1e-9)))
            vol_ = recent_vol[-1] if recent_vol else 0.0

            feat_row    = np.array([[recent_adj[-1], vol_, sma20, sma50, rsi_]])
            feat_scaled = scaler_model.transform(feat_row)[0].tolist()

            # ✅ KEY FIX: slide window — drop oldest row, append newest
            recent_scaled.append(feat_scaled)
            recent_scaled = recent_scaled[-time_step:]

            inp      = np.array(recent_scaled).reshape(1, time_step, -1)
            pred_sc  = model.predict(inp, verbose=0)[0, 0]

            tmpl       = np.array(feat_scaled).reshape(1, -1).copy()
            tmpl[0, 0] = pred_sc
            pred_price = float(scaler_model.inverse_transform(tmpl)[0, 0])

            future_preds.append(pred_price)
            recent_adj.append(pred_price)
            recent_vol.append(vol_)

        # -------------------------------------------------------
        # SAFE growing confidence band
        # half-width capped at 20% of predicted price so the band
        # never collapses to negative values.
        # -------------------------------------------------------
        last_close  = float(df_used['Adj Close'].iloc[-1])
        floor_price = last_close * 0.50
        z = 1.96
        band_uppers, band_lowers = [], []
        for i, p in enumerate(future_preds):
            half = min(z * resid_std * np.sqrt(i + 1), 0.20 * p)
            band_uppers.append(p + half)
            band_lowers.append(max(p - half, floor_price))

        future_dates = pd.date_range(
            start=data_main.index[-1] + pd.Timedelta(days=1),
            periods=days, freq='B'
        )
        future_df = pd.DataFrame({
            'Date':      future_dates,
            'Predicted': future_preds,
            'Upper':     band_uppers,
            'Lower':     band_lowers
        })

        # Animated chart
        hist_x = df_used.index
        hist_y = df_used['Adj Close'].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_x, y=hist_y,
                                 name='Historical', line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=[future_dates[0]], y=[future_preds[0]],
                                 name='Predicted',   line=dict(color='#ff7f0e')))
        fig.add_trace(go.Scatter(
            x=[future_dates[0], future_dates[0]],
            y=[band_lowers[0],  band_uppers[0]],
            fill='toself', fillcolor='rgba(255,127,14,0.15)',
            line=dict(color='rgba(255,127,14,0)'),
            name='95% CI', showlegend=True
        ))

        frames = []
        for i in range(len(future_dates)):
            xp = future_dates[:i+1]
            yp = future_preds[:i+1]
            xb = list(future_dates[:i+1]) + list(future_dates[:i+1][::-1])
            yb = list(band_uppers[:i+1])  + list(band_lowers[:i+1][::-1])
            frames.append(go.Frame(data=[
                go.Scatter(x=hist_x, y=hist_y),
                go.Scatter(x=xp, y=yp, line=dict(color='#ff7f0e')),
                go.Scatter(x=xb, y=yb, fill='toself',
                           fillcolor='rgba(255,127,14,0.15)',
                           line=dict(color='rgba(255,127,14,0)'))
            ], name=str(i)))
        fig.frames = frames

        fig.update_layout(
            title=f"{selected_ticker} — {days}-day Forecast with 95% CI",
            xaxis_title="Date", yaxis_title="Price ($)",
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "▶ Play",  "method": "animate",
                     "args": [None, {"frame": {"duration": 400, "redraw": True},
                                     "fromcurrent": True,
                                     "transition": {"duration": 200}}]},
                    {"label": "⏸ Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}]}
                ],
                "direction": "left", "pad": {"r": 10, "t": 10},
                "showactive": True, "x": 0.01, "y": -0.12,
                "xanchor": "left", "yanchor": "top"
            }]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            future_df.style.format({"Predicted": "{:.2f}", "Upper": "{:.2f}", "Lower": "{:.2f}"}),
            use_container_width=True
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Test MSE", f"{mse:.3f}" if mse is not None else "N/A")
        c2.metric("Test R²",  f"{r2:.3f}"  if r2  is not None else "N/A")
        c3.metric("Model", "Multi-feature LSTM (Fixed)")

# ====================== SENTIMENT ======================
elif tab == "Sentiment":
    st.subheader("News Sentiment")
    if news_posts:
        df_s = pd.DataFrame({'News': news_posts, 'Link': news_links, 'Score': vader_scores})
        def color(val):
            return f"color: {'green' if val > 0.1 else 'red' if val < -0.1 else 'gray'}"
        st.dataframe(
            df_s.style.applymap(color, subset=['Score']).format({'Score': '{:.3f}'}),
            use_container_width=True
        )
        pos = sum(1 for s in vader_scores if s > 0.1)
        neg = sum(1 for s in vader_scores if s < -0.1)
        neu = len(vader_scores) - pos - neg
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", pos)
        c2.metric("Negative", neg)
        c3.metric("Neutral",  neu)
    else:
        st.info("No news available.")

# ====================== COMPARISON ======================
elif tab == "Comparison":
    st.subheader(f"**{selected_ticker} vs {compare_ticker}**")
    if data_main is not None and data_compare is not None:
        base_m = data_main['Adj Close'].iloc[0]
        base_c = data_compare['Adj Close'].iloc[0]
        df_m   = (data_main['Adj Close']    / base_m - 1) * 100
        df_c   = (data_compare['Adj Close'] / base_c - 1) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_main.index,    y=df_m, name=selected_ticker, line=dict(color='#26A69A')))
        fig.add_trace(go.Scatter(x=data_compare.index, y=df_c, name=compare_ticker,  line=dict(color='#AB47BC')))
        fig.update_layout(title="Normalised Performance (%)", height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        ret_m = (data_main['Adj Close'].iloc[-1]    / base_m - 1) * 100
        ret_c = (data_compare['Adj Close'].iloc[-1] / base_c - 1) * 100
        vol_m = data_main['Adj Close'].pct_change().std()    * np.sqrt(252) * 100
        vol_c = data_compare['Adj Close'].pct_change().std() * np.sqrt(252) * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{selected_ticker} Return", f"{ret_m:+.2f}%")
        c2.metric(f"{compare_ticker} Return",  f"{ret_c:+.2f}%")
        c3.metric(f"{selected_ticker} Vol",    f"{vol_m:.1f}%")
        c4.metric(f"{compare_ticker} Vol",     f"{vol_c:.1f}%")
    else:
        st.error("Not enough data to compare.")

# ====================== PORTFOLIO ANALYZER ======================
elif tab == "Portfolio Analyzer":
    st.subheader("Portfolio Analyzer")
    portfolio_tickers = st.multiselect("Select Tickers", tickers,
                                       default=[selected_ticker, compare_ticker])
    if len(portfolio_tickers) < 2:
        st.warning("Select at least 2 tickers.")
    else:
        weights = []
        cols    = st.columns(len(portfolio_tickers))
        total_w = 0
        for i, tick in enumerate(portfolio_tickers):
            w = cols[i].number_input(f"Weight {tick} (%)", 0.0, 100.0,
                                     100.0 / len(portfolio_tickers))
            weights.append(w / 100)
            total_w += w
        if abs(total_w - 100) > 0.01:
            st.warning(f"Weights sum to {total_w:.1f}%. Should be 100%.")
        else:
            data_dict = {}
            for tick in portfolio_tickers:
                d = fetch_stock_data(tick, start_date, end_date)
                if d is None:
                    st.error(f"Data missing for {tick}.")
                    st.stop()
                data_dict[tick] = d['Adj Close']
            port_df  = pd.DataFrame(data_dict)
            returns  = port_df.pct_change().dropna()
            m_ret    = returns.mean() * 252
            cov_mat  = returns.cov()  * 252
            w_np     = np.array(weights)
            p_ret    = np.dot(m_ret, w_np)
            p_vol    = np.sqrt(np.dot(w_np.T, np.dot(cov_mat, w_np)))
            sharpe   = (p_ret - 0.03) / p_vol
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Return",     f"{p_ret*100:.2f}%")
            c2.metric("Portfolio Volatility", f"{p_vol*100:.2f}%")
            c3.metric("Sharpe Ratio",        f"{sharpe:.2f}")
            fig_h = px.imshow(returns.corr(), text_auto=True, aspect="auto",
                              color_continuous_scale='RdBu_r', title="Correlation Heatmap")
            st.plotly_chart(fig_h, use_container_width=True)

# ====================== NEWS TICKER ======================
st.markdown("---")
st.markdown("### Latest Headlines (24/7)")
all_h    = news_headlines + news_headlines
anim_dur = max(15, len(news_headlines) * 3)
st.markdown(f"""
<style>
.ticker-container {{
    height:180px; overflow:hidden; background:#0f172a; padding:16px;
    border-radius:14px; box-shadow:0 6px 24px rgba(0,0,0,0.3);
    color:white; font-family:'Segoe UI',sans-serif; position:relative;
}}
.ticker-wrapper {{ animation: scroll-up {anim_dur}s linear infinite; will-change:transform; }}
@keyframes scroll-up {{ 0%{{transform:translateY(0)}} 100%{{transform:translateY(-50%)}} }}
.ticker-item {{
    padding:12px 0; font-size:15px; line-height:1.6; min-height:40px;
    overflow:hidden; text-overflow:ellipsis; word-wrap:break-word;
}}
</style>""", unsafe_allow_html=True)
html_c = '<div class="ticker-container"><div class="ticker-wrapper">'
for h in all_h:
    html_c += f'<div class="ticker-item">{h}</div>'
html_c += '</div></div>'
st.markdown(html_c, unsafe_allow_html=True)
