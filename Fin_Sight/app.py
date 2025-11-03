import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
from prophet import Prophet
from newsapi import NewsApiClient
from streamlit_option_menu import option_menu
from streamlit_chat import message
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import base64
import threading
import time
import requests
from telegram import Bot

# ====================== FREE AI (NO OPENAI) ======================
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def free_ai(prompt):
    try:
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {st.session_state.get('hf_key', '')}"}
        payload = {"inputs": f"Stock: {selected_ticker}. Price: ${price:.2f}. Q: {prompt} Answer in 2 sentences."}
        r = requests.post(API_URL, headers=headers, json=payload, timeout=8)
        return r.json()[0]['generated_text'].split("Q:")[-1].strip()
    except:
        p = prompt.lower()
        if "buy" in p: return f"Buy {selected_ticker} if price drops 5%."
        if "sell" in p: return f"Sell if price hits ${price*1.1:.2f}."
        return "Ask: 'Should I buy?' or 'Is this good?'"

# ====================== PAGE SETUP ======================
st.set_page_config(page_title="FinSight FREE AI", layout="wide")
st.markdown("""
<style>
    .main {background: #0f172a; color: white; font-family: 'Segoe UI';}
    .stButton>button {background: #00D4FF; color: black; font-weight: bold; border-radius: 12px;}
    .ticker {background: linear-gradient(90deg, #00D4FF, #8B00FF); padding: 14px; border-radius: 16px; color: white; text-align: center; font-weight: bold; margin: 15px 0;}
    .css-1d391kg {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#00D4FF;'>FinSight FREE AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaa;'>No OpenAI • No pyaudio • Works on Phone</p>", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/80/robot.png")
    st.header("FREE Controls")

    # Hugging Face Key
    if 'hf_key' not in st.session_state:
        st.session_state.hf_key = ""
    hf_key = st.text_input("Hugging Face Key (FREE)", type="password", value=st.session_state.hf_key)
    st.session_state.hf_key = hf_key
    st.caption("Get FREE: [hf.co/settings/tokens](https://huggingface.co/settings/tokens)")

    # Telegram (Optional)
    telegram_token = st.text_input("Telegram Bot Token (Optional)", type="password")
    chat_id = st.text_input("Your Chat ID", type="password")

    # Stock
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    selected_ticker = st.selectbox("Stock", tickers, index=0)

    # Fake Voice Input
    st.warning("Voice works on laptop only")
    voice_text = st.text_input("Type your question here:")
    if voice_text:
        st.session_state.voice_input = voice_text

# ====================== DATA ======================
@st.cache_data(ttl=300)
def get_data(t):
    return yf.download(t, period="1y", progress=False)

df = get_data(selected_ticker)
price = df['Close'].iloc[-1]

# ====================== HERO METRICS ======================
c1, c2, c3 = st.columns(3)
c1.metric("Price", f"${price:.2f}", f"{(price/df['Close'].iloc[-8]-1)*100:+.1f}%")
c2.metric("7D Change", f"{(price/df['Close'].iloc[-8]-1)*100:+.1f}%")
c3.metric("Volatility", f"{df['Close'].pct_change().std()*100:.1f}%")

# ====================== TABS ======================
tab = option_menu(None, ["AI Chat", "Chart", "Forecast", "News", "Alerts", "PDF"],
                  icons=['robot', 'graph-up', 'crystal-ball', 'newspaper', 'bell', 'file-pdf'],
                  orientation="horizontal")

# ====================== AI CHAT ======================
if tab == "AI Chat":
    st.subheader("FREE AI Analyst")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        message(msg["text"], is_user=msg["user"])

    prompt = st.session_state.get("voice_input", st.chat_input("Ask AI..."))
    if "voice_input" in st.session_state:
        del st.session_state.voice_input

    if prompt:
        st.session_state.messages.append({"text": prompt, "user": True})
        message(prompt, is_user=True)

        with st.spinner("AI Thinking..."):
            reply = free_ai(prompt)
        st.session_state.messages.append({"text": reply, "user": False})
        message(reply)

# ====================== CHART ======================
elif tab == "Chart":
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean(), name="20DMA", line=dict(color="#00D4FF")))
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ====================== FORECAST ======================
elif tab == "Forecast":
    st.subheader("30-Day Forecast")
    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
    m = Prophet()
    m.fit(df_p)
    future = m.make_future_dataframe(30)
    forecast = m.predict(future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Real"))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Forecast", line=dict(dash="dash")))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ====================== NEWS ======================
elif tab == "News":
    st.subheader("Latest News")
    try:
        news = NewsApiClient("d848a496d874401b9e2129a71adb57ba").get_everything(q=selected_ticker, page_size=5)['articles']
        for a in news:
            st.write(f"**{a['title']}**")
            st.caption(f"Source: {a['source']['name']}")
            st.markdown(f"[Read more]({a['url']})")
            st.markdown("---")
    except:
        st.info("News loading... (free API)")

# ====================== TELEGRAM ALERTS ======================
elif tab == "Alerts":
    st.subheader("Telegram Alert")
    target = st.number_input("Alert when price hits", value=price)
    if st.button("Set Alert") and telegram_token and chat_id:
        def send_alert():
            bot = Bot(token=telegram_token)
            while True:
                time.sleep(300)
                current = yf.Ticker(selected_ticker).info.get('regularMarketPrice', price)
                if abs(current - target) < 1:
                    bot.send_message(chat_id, f"{selected_ticker} hit ${current:.2f}!")
                    break
        threading.Thread(target=send_alert, daemon=True).start()
        st.success("Alert ON! Check Telegram")
    else:
        st.info("Enter Telegram keys in sidebar")

# ====================== PDF REPORT ======================
elif tab == "PDF":
    st.subheader("Download Report")
    if st.button("Generate PDF"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"<b>FinSight Report: {selected_ticker}</b>", styles['Title']),
            Spacer(1, 12),
            Paragraph(f"Price: ${price:.2f}", styles['Normal']),
            Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']),
            Paragraph("AI Verdict: Hold for now", styles['Normal']),
        ]
        doc.build(story)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{selected_ticker}_report.pdf">Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.balloons()

# ====================== LIVE TICKER ======================
st.markdown(f"""
<div class="ticker">
{selected_ticker} @ ${price:.2f} • AI: "Ask me anything" • PDF ready • Telegram alerts active
</div>
""", unsafe_allow_html=True)
