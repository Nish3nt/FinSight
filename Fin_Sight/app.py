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
from streamlit_chat import message
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ====================== INITIAL SETUP ======================
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
current_date = datetime.now().date()

st.set_page_config(page_title="FinSight", layout="wide")
st.title("**FinSight**: Real-Time Stock Intelligence")

# ====================== SIDEBAR ======================
st.sidebar.header("Controls")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
           'NFLX', 'AMD', 'JPM', 'V', 'XOM']
selected_ticker = st.sidebar.selectbox("Main Stock", tickers, index=0)
compare_ticker = st.sidebar.selectbox("Compare With", tickers, index=1)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2018-01-01').date())
end_date = st.sidebar.date_input("End Date", current_date)

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()
if end_date > current_date:
    end_date = current_date
    st.warning(f"End date capped at today: {current_date}")

# ====================== LOAD LIGHTWEIGHT LLM (phi-3-mini) ======================
@st.cache_resource
def load_chat_model():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    return pipe

chat_pipe = load_chat_model()

# ====================== DATA FETCHING (unchanged) ======================
@st.cache_resource(ttl=300)
def get_ticker(ticker):
    return yf.Ticker(ticker)

ticker_obj = get_ticker(selected_ticker)

@st.cache_data(ttl=300)
def get_news(ticker):
    try:
        api_key = st.secrets.get("NEWSAPI_KEY") or "d848a496d874401b9e2129a71adb57ba"
        if api_key and api_key != "YOUR_NEWSAPI_KEY":
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{ticker} stock OR {ticker} earnings OR {ticker} news',
                'sortBy': 'publishedAt',
                'pageSize': 15,
                'language': 'en',
                'apiKey': api_key
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                articles = r.json().get('articles', [])
                if articles:
                    headlines, posts, links = [], [], []
                    for art in articles:
                        title = art.get('title', '').strip()
                        source = art.get('source', {}).get('name', 'Source').strip()
                        url_link = art.get('url', '#')
                        if title:
                            headlines.append(f"**{title}** – {source}")
                            posts.append(f"{title} – {source}")
                            links.append(url_link)
                    return headlines, posts, links
    except Exception:
        st.warning("NewsAPI failed. Using Yahoo Finance.")

    try:
        news = ticker_obj.news
        if not news:
            return ["No recent headlines."], ["No recent headlines."], ["#"]
        headlines, posts, links = [], [], []
        for item in news[:15]:
            title = item.get('title', '').strip()
            pub = item.get('publisher', 'Source').strip()
            url_link = item.get('link', '#')
            if title:
                headlines.append(f"**{title}** – {pub}")
                posts.append(f"{title} – {pub}")
                links.append(url_link)
        return headlines or ["Market quiet."], posts, links
    except Exception:
        return ["News feed unavailable."], ["News feed unavailable."], ["#"]

news_headlines, news_posts, news_links = get_news(selected_ticker)

@st.cache_data(ttl=300)
def compute_vader_sentiment(posts):
    return [sia.polarity_scores(p)['compound'] for p in posts]

vader_scores = compute_vader_sentiment(news_posts)

@st.cache_data(ttl=600)
def fetch_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            st.error(f"No data for {ticker}.")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        required = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        df = df[required].dropna(how='all')
        if len(df) < 30:
            st.warning(f"Only {len(df)} days of data for {ticker}.")
        return df
    except Exception as e:
        st.error(f"Data error: {e}")
        return None

data_main = fetch_stock_data(selected_ticker, start_date, end_date)
data_compare = fetch_stock_data(compare_ticker, start_date, end_date)

if data_main is None or data_compare is None:
    st.stop()

# ====================== 3-D GLOBE (unchanged, cached) ======================
@st.cache_data(ttl=3600)
def hero_globe(selected_ticker):
    top_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
                   'NFLX', 'AMD', 'JPM', 'V', 'XOM', 'BRK-B', 'UNH', 'MA']
    caps = {}
    for t in top_tickers:
        try:
            caps[t] = yf.Ticker(t).info.get('marketCap', 0) or 0
        except:
            caps[t] = 0
    try:
        caps[selected_ticker] = yf.Ticker(selected_ticker).info.get('marketCap', 0) or 0
    except:
        caps[selected_ticker] = 0

    df = pd.DataFrame({'Ticker': list(caps.keys()), 'MarketCap': list(caps.values())})
    df = df[df['MarketCap'] > 0].sort_values('MarketCap', ascending=False).head(15)

    np.random.seed(42)
    lat = np.random.uniform(-90, 90, len(df))
    lon = np.random.uniform(-180, 180, len(df))
    size = np.sqrt(df['MarketCap']) / 1e5

    theta = np.deg2rad(90 - lat)
    phi = np.deg2rad(lon)
    r = 1.0
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(size=size, color=size, colorscale='Viridis', opacity=0.9,
                    showscale=True, colorbar=dict(title="Market Cap (B)")),
        text=df['Ticker'],
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>Cap: $%{marker.color:,.0f}B"
    ))
    fig.update_layout(
        title=f"3D Market-Cap Globe – {selected_ticker} highlighted",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)), aspectmode='cube'),
        height=400, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", scene_bgcolor="#0f172a"
    )
    return fig

if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None
if 'current_fig' not in st.session_state:
    st.session_state.current_fig = None

if st.session_state.last_ticker != selected_ticker:
    st.session_state.last_ticker = selected_ticker
    with st.spinner("Rendering 3D Globe..."):
        st.session_state.current_fig = hero_globe(selected_ticker)

if st.session_state.current_fig:
    st.plotly_chart(st.session_state.current_fig, use_container_width=True)

# ====================== KPI BANNER ======================
col1, col2, col3, col4 = st.columns(4)
try:
    price = data_main['Close'].iloc[-1]
    change_7d = (price - data_main['Close'].iloc[-8]) / data_main['Close'].iloc[-8] * 100 if len(data_main) > 7 else None
    vol = data_main['Close'].pct_change().std() * np.sqrt(252) * 100 if len(data_main) > 1 else None
    market_cap = ticker_obj.info.get('marketCap', 0) / 1e9
    col1.metric("Price", f"${price:,.2f}")
    col2.metric("7D Δ", f"{change_7d:+.2f}%" if change_7d else "N/A")
    col3.metric("Volatility", f"{vol:.1f}%" if vol else "N/A")
    col4.metric("Market Cap", f"${market_cap:,.1f}B")
except:
    col1.metric("Price", "N/A")

# ====================== TAB NAVIGATION ======================
tab = option_menu(
    menu_title=None,
    options=["Data & Viz", "Predictions", "Sentiment", "Comparison", "Chatbot"],
    icons=["table", "graph-up", "chat-dots", "arrow-left-right", "robot"],
    orientation="horizontal"
)

# ====================== CHATBOT LOGIC ======================
if tab == "Chatbot":
    st.subheader("FinSight Assistant – Ask Anything!")

    # Helper to extract context
    def build_context():
        info = ticker_obj.info
        fundamentals = f"""
        Company: {info.get('longName','N/A')}
        Sector: {info.get('sector','N/A')}
        Industry: {info.get('industry','N/A')}
        Market Cap: ${info.get('marketCap',0)/1e9:,.1f}B
        P/E: {info.get('trailingPE','N/A')}
        52W High: ${info.get('fiftyTwoWeekHigh','N/A'):.2f}
        52W Low: ${info.get('fiftyTwoWeekLow','N/A'):.2f}
        """
        recent_price = data_main['Close'].iloc[-1]
        recent_change = (data_main['Close'].iloc[-1] - data_main['Close'].iloc[-2]) / data_main['Close'].iloc[-2] * 100 if len(data_main) > 1 else 0
        sentiment_summary = f"Positive: {sum(1 for s in vader_scores if s>0.1)}, Negative: {sum(1 for s in vader_scores if s<-0.1)}"
        return f"""
        Current ticker: {selected_ticker}
        Latest price: ${recent_price:,.2f} ({recent_change:+.2f}% today)
        {fundamentals}
        Recent news sentiment: {sentiment_summary}
        """

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hello! I'm your FinSight assistant. Ask me anything about **{selected_ticker}**, forecasts, sentiment, or which stock to explore next!"}
        ]

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"user_{hash(msg['content'])}")
        else:
            message(msg["content"], key=f"bot_{hash(msg['content'])}")

    # User input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        message(prompt, is_user=True)

        # Build prompt for LLM
        context = build_context()
        full_prompt = f"""You are a helpful stock-analysis assistant. Use ONLY the following context to answer. 
        If the user asks to navigate, respond with **[[Go to TAB_NAME]]** (e.g., [[Go to Predictions]]). 
        Context:
        {context}
        User question: {prompt}
        Answer concisely and helpfully.
        """

        with st.spinner("Thinking..."):
            response = chat_pipe(full_prompt)[0]['generated_text'].split("Answer:")[-1].strip()
            # Clean up possible navigation tags
            navigation = None
            if "[[Go to" in response:
                import re
                nav_match = re.search(r'\[\[Go to (.*?)\]\]', response)
                if nav_match:
                    navigation = nav_match.group(1).strip()
                    response = response.replace(nav_match.group(0), "").strip()

            st.session_state.messages.append({"role": "assistant", "content": response})
            message(response)

            # Auto-navigate if requested
            if navigation:
                tab_map = {
                    "Data & Viz": 0,
                    "Predictions": 1,
                    "Sentiment": 2,
                    "Comparison": 3,
                    "Chatbot": 4,
                }
                if navigation in tab_map:
                    st.session_state.selected_tab = tab_map[navigation]
                    st.experimental_rerun()

# ====================== OTHER TABS (unchanged) ======================
if tab == "Data & Viz":
    st.subheader(f"**{selected_ticker}** – Price History")
    st.dataframe(data_main.tail(100), use_container_width=True)
    st.download_button("Download CSV", data_main.to_csv().encode('utf-8'), f"{selected_ticker}.csv")
    fig = px.line(data_main, x=data_main.index, y='Adj Close', title="Price Trend")
    st.plotly_chart(fig, use_container_width=True)
    fig_c = go.Figure(go.Candlestick(x=data_main.index, open=data_main['Open'],
                                     high=data_main['High'], low=data_main['Low'], close=data_main['Close']))
    st.plotly_chart(fig_c.update_layout(title="Candlestick", height=600), use_container_width=True)

elif tab == "Predictions":
    st.subheader("Price Forecast")
    model = st.selectbox("Model", ["Prophet", "LSTM"])
    days = st.slider("Days", 1, 30, 7)
    # ... (same as before – omitted for brevity)

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
    c1.metric("Positive", pos); c2.metric("Negative", neg); c3.metric("Neutral", neu)

elif tab == "Comparison":
    st.subheader(f"**{selected_ticker} vs {compare_ticker}**")
    base_main = data_main['Adj Close'].iloc[0]
    base_compare = data_compare['Adj Close'].iloc[0]
    df_main = (data_main['Adj Close'] / base_main - 1) * 100
    df_compare = (data_compare['Adj Close'] / base_compare - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_main.index, y=df_main, name=selected_ticker, line=dict(color='#26A69A')))
    fig.add_trace(go.Scatter(x=data_compare.index, y=df_compare, name=compare_ticker, line=dict(color='#AB47BC')))
    fig.update_layout(title="Performance (%)", height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ====================== TICKER (bottom) ======================
st.markdown("---")
st.markdown("### Latest Headlines (24/7)")
all_headlines = news_headlines + news_headlines
animation_duration = max(15, len(news_headlines) * 3)
st.markdown(f"""
<style>
.ticker-container {{height:180px;overflow:hidden;background:#0f172a;padding:16px;border-radius:14px;
    box-shadow:0 6px 24px rgba(0,0,0,0.3);color:white;font-family:'Segoe UI',sans-serif;}}
.ticker-wrapper {{animation:scroll-up {animation_duration}s linear infinite;}}
@keyframes scroll-up {{0%{{transform:translateY(0)}}100%{{transform:translateY(-50%)}}}}
.ticker-item {{padding:12px 0;font-size:15px;line-height:1.6;min-height:40px;}}
</style>
""", unsafe_allow_html=True)
html = '<div class="ticker-container"><div class="ticker-wrapper">'
for h in all_headlines:
    html += f'<div class="ticker-item">{h}</div>'
html += '</div></div>'
st.markdown(html, unsafe_allow_html=True)
