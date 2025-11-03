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

# ====================== 100% CLEAN ======================
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def free_ai(prompt):
    p = prompt.lower()
    if any(w in p for w in ["buy", "should"]): return f"Buy {selected_ticker} if price drops 5%."
    if "sell" in p: return f"Sell at ${price*1.1:.2f}."
    return f"{selected_ticker} is strong. Hold tight!"

st.set_page_config(page_title="FinSight", layout="wide")
st.markdown("""
<style>
    .main {background: #0f172a; color: white;}
    .stButton>button {background: #00D4FF; color: black; font-weight: bold;}
    .ticker {background: linear-gradient(90deg, #00D4FF, #8B00FF); padding: 14px; border-radius: 16px; color: white; text-align: center; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#00D4FF;'>FinSight FREE AI</h1>", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/80/robot.png")
    st.header("FREE AI")
    st.info("No keys needed!")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
    global selected_ticker
    selected_ticker = st.selectbox("Stock", tickers)

# ====================== DATA ======================
@st.cache_data(ttl=300)
def get_data(t): return yf.download(t, period="1y", progress=False)

df = get_data(selected_ticker)
price = df['Close'].iloc[-1]

# ====================== HERO ======================
c1, c2 = st.columns(2)
c1.metric("Price", f"${price:.2f}")
c2.metric("7D", f"{(price/df['Close'].iloc[-8]-1)*100:+.1f}%")

# ====================== TABS ======================
tab = option_menu(None, ["AI", "Chart", "Forecast", "News", "PDF"],
                  icons=['robot', 'graph-up', 'crystal-ball', 'newspaper', 'file-pdf'],
                  orientation="horizontal")

# ====================== AI ======================
if tab == "AI":
    st.subheader("FREE AI Chat")
    if "msgs" not in st.session_state: st.session_state.msgs = []
    for m in st.session_state.msgs:
        message(m["text"], is_user=m["user"])
    q = st.chat_input("Ask anything...")
    if q:
        st.session_state.msgs.append({"text": q, "user": True})
        message(q, is_user=True)
        reply = free_ai(q)
        st.session_state.msgs.append({"text": reply, "user": False})
        message(reply)

# ====================== CHART ======================
elif tab == "Chart":
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.update_layout(height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ====================== FORECAST ======================
elif tab == "Forecast":
    df_p = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
    m = Prophet()
    m.fit(df_p)
    future = m.make_future_dataframe(30)
    f = m.predict(future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Real"))
    fig.add_trace(go.Scatter(x=f['ds'], y=f['yhat'], name="Forecast"))
    st.plotly_chart(fig, use_container_width=True)

# ====================== NEWS ======================
elif tab == "News":
    try:
        news = NewsApiClient("d848a496d874401b9e2129a71adb57ba").get_everything(q=selected_ticker, page_size=3)['articles']
        for a in news:
            st.write(f"**{a['title']}**")
            st.markdown(f"[Read]({a['url']})")
    except:
        st.write("News loading...")

# ====================== PDF ======================
elif tab == "PDF":
    if st.button("Download PDF"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph(f"<b>{selected_ticker} Report</b>", styles['Title'])]
        story.append(Paragraph(f"Price: ${price:.2f}", styles['Normal']))
        doc.build(story)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download</a>'
        st.markdown(href, unsafe_allow_html=True)

# ====================== TICKER ======================
st.markdown(f"""
<div class="ticker">
{selected_ticker} @ ${price:.2f} • Ask AI: "Should I buy?"
</div>
""", unsafe_allow_html=True)
