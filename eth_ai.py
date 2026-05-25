import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import yfinance as yf

# ============================================================
# НАСТРОЙКИ СТРАНИЦЫ — МИНИМАЛИЗМ И СТРОГОСТЬ
# ============================================================

st.set_page_config(
    page_title="MULTI-ASSET BOTTOM DETECTOR",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Убираем дефолтный sidebar и лишние элементы
st.markdown("""
    <style>
        [data-testid="collapsedControl"] { display: none; }
        .main > div { padding-top: 0; }
        header { display: none; }
        footer { display: none; }
        .stApp { background-color: #0a0a0a; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #1a1a1a; padding: 5px; border-radius: 8px; }
        .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: 8px 16px; background-color: #1a1a1a; color: #888; }
        .stTabs [aria-selected="true"] { background-color: #00ff44; color: #000; font-weight: bold; }
        div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 600 !important; }
        div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; color: #666; }
        .stDataFrame { border: 1px solid #2a2a2a; border-radius: 12px; overflow: hidden; }
        .stButton button { background: linear-gradient(135deg, #00ff44 0%, #00cc33 100%); color: #000; font-weight: bold; border: none; padding: 8px 24px; border-radius: 6px; }
        .stButton button:hover { background: linear-gradient(135deg, #00ff44 0%, #00aa22 100%); color: #000; }
        h1, h2, h3, h4, h5, h6, p, span, div { font-family: 'Times New Roman', Times, serif !important; }
        .metric-card { background-color: #1a1a1a; border-radius: 12px; padding: 16px; border-left: 3px solid; margin: 8px 0; }
        .signal-card { background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%); border-radius: 16px; padding: 32px; text-align: center; margin: 24px 0; border: 1px solid #2a2a3e; }
        .section-title { font-size: 1rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #666; margin-bottom: 16px; border-bottom: 1px solid #2a2a2a; padding-bottom: 8px; }
    </style>
    <meta http-equiv="refresh" content="300">
""", unsafe_allow_html=True)

# ============================================================
# ЗАГОЛОВОК — ТОЛЬКО НАЗВАНИЕ
# ============================================================

st.title("MULTI-ASSET BOTTOM DETECTOR")
st.markdown("<p style='color: #666; font-size: 0.8rem; letter-spacing: 1px;'>PROFESSIONAL EDITION</p>", unsafe_allow_html=True)

# ============================================================
# 1. ФИНАЛЬНЫЙ СПИСОК АКТИВОВ
# ============================================================

CRYPTO_LIST = [
    "ETH", "BTC", "SOL", "ASTER", "IMX", "ZK", "FIL", "STX", "RENDER",
    "ONDO", "GRT", "CELO", "CRV", "TWT", "SUI", "APE", "ARKM", "ONE",
    "GOAT", "POL", "LINK", "UNI", "TRUMP", "ARC", "NEAR", "ALGO", "FLOCK"
]

STOCK_LIST = [
    "HIMS", "SIL", "GDX", "TSLA", "LIT", "ZM", "URA", "PLTR",
    "EWW", "BABA", "COIN", "NVDA", "SBER", "MTSS", "HEAD"
]

VETERAN_LIST = ["BTC", "ETH", "BNB", "XRP", "LTC", "ADA", "DOGE", "AAPL", "MSFT", "NVDA"]

COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "FIL": "filecoin",
    "LINK": "chainlink", "UNI": "uniswap", "NEAR": "near", "ALGO": "algorand",
    "GRT": "the-graph", "CRV": "curve-dao-token", "STX": "blockstack",
    "RENDER": "render-token", "ONDO": "ondo-finance", "SUI": "sui",
    "APE": "apecoin", "IMX": "immutable-x", "ZK": "zkSync", "TWT": "trust-wallet-token",
    "CELO": "celo", "ARKM": "arkham", "ONE": "harmony", "GOAT": "goat",
    "POL": "polygon", "TRUMP": "maga", "ARC": "arc", "FLOCK": "flock"
}

# ============================================================
# 2. ФУНКЦИИ
# ============================================================

@st.cache_data(ttl=600)
def get_coingecko_fundamentals(coin_id):
    try:
        api_key = st.secrets.get("COINGECKO_API_KEY")
        if not api_key:
            return None
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {"localization": "false", "market_data": "true", "x_cg_demo_api_key": api_key}
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            md = data.get("market_data", {})
            return {
                "price_usd": md.get("current_price", {}).get("usd", 0),
                "market_cap": md.get("market_cap", {}).get("usd", 0),
                "fully_diluted_valuation": md.get("fully_diluted_valuation", {}).get("usd", 0),
                "total_volume": md.get("total_volume", {}).get("usd", 0),
                "price_change_24h": md.get("price_change_percentage_24h", 0)
            }
    except:
        pass
    return None

def get_adaptive_thresholds(z_scores):
    if len(z_scores) < 30:
        return -1.8, 1.5
    lower = max(-3.0, min(-1.0, np.percentile(z_scores, 5)))
    upper = min(3.0, max(0.5, np.percentile(z_scores, 95)))
    return lower, upper

def get_signal_adaptive(z_score, lower_thr, upper_thr, is_veteran):
    if is_veteran:
        if z_score <= -1.8: return "STRONG BUY", "#00ff44"
        elif z_score <= -1.2: return "ACCUMULATE", "#88ff88"
        elif z_score >= 1.5: return "SELL", "#ff2200"
        else: return "HOLD", "#666666"
    else:
        if z_score <= lower_thr: return "STRONG BUY", "#00ff44"
        elif z_score <= lower_thr * 0.7: return "ACCUMULATE", "#88ff88"
        elif z_score >= upper_thr: return "SELL", "#ff2200"
        else: return "HOLD", "#666666"

@st.cache_data(ttl=600)
def load_crypto_data(symbol, days=500):
    try:
        if "CRYPTOCOMPARE_KEY" in st.secrets:
            api_key = st.secrets["CRYPTOCOMPARE_KEY"]
            url = "https://min-api.cryptocompare.com/data/v2/histoday"
            params = {"fsym": symbol, "tsym": "USD", "limit": days, "api_key": api_key}
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get("Response") == "Success":
                    raw = data["Data"]["Data"]
                    df = pd.DataFrame(raw)
                    df["date"] = pd.to_datetime(df["time"], unit='s')
                    df["close"] = df["close"].astype(float)
                    return df.sort_values("date").reset_index(drop=True)
    except:
        pass
    return None

@st.cache_data(ttl=600)
def load_stock_data(symbol, days=500):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df.reset_index()
            df = df.rename(columns={"Date": "date", "Close": "close"})
            return df[["date", "close"]]
    except:
        pass
    return None

def calculate_metrics_adaptive(df):
    if df is None or len(df) < 30:
        return None, None, None, None, None, None
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    mean_ret = df["returns"].mean()
    std_ret = df["returns"].std()
    df["z_score"] = (df["returns"] - mean_ret) / (std_ret + 1e-10)
    df = df.fillna(0)
    z_scores = df["z_score"].values
    lower, upper = get_adaptive_thresholds(z_scores)
    current_price = df["close"].iloc[-1]
    current_z = df["z_score"].iloc[-1]
    sensitivity = 1.5 if len(df) > 365 else 1.0
    current_prob = 1 / (1 + np.exp(current_z * sensitivity))
    confidence = min(100, len(df) / 365 * 100)
    return df, current_price, current_z, current_prob, confidence, (lower, upper)

# ============================================================
# 3. ИНТЕРФЕЙС — ТОЛЬКО ВЕРХНЯЯ СТРОКА
# ============================================================

col_selector, col_placeholder = st.columns([1, 3])

with col_selector:
    asset_type = st.radio("", ["CRYPTO", "STOCKS"], horizontal=True, label_visibility="collapsed")
    if asset_type == "CRYPTO":
        selected_asset = st.selectbox("", CRYPTO_LIST, label_visibility="collapsed")
    else:
        selected_asset = st.selectbox("", STOCK_LIST, label_visibility="collapsed")

with col_placeholder:
    st.markdown("---")

# ============================================================
# 4. ЗАГРУЗКА ДАННЫХ
# ============================================================

is_crypto = selected_asset in CRYPTO_LIST
is_veteran = selected_asset in VETERAN_LIST

fundamentals = None
if is_crypto and selected_asset in COINGECKO_IDS:
    coin_id = COINGECKO_IDS[selected_asset]
    fundamentals = get_coingecko_fundamentals(coin_id)

with st.spinner("LOADING DATA..."):
    if is_crypto:
        df = load_crypto_data(selected_asset)
    else:
        df = load_stock_data(selected_asset)

if df is None or len(df) < 30:
    st.warning("INSUFFICIENT DATA")
    st.stop()

df, price, z_score, prob, confidence, (lower, upper) = calculate_metrics_adaptive(df)
signal_text, signal_color = get_signal_adaptive(z_score, lower, upper, is_veteran)

# ============================================================
# 5. ОСНОВНЫЕ МЕТРИКИ — МИНИМАЛЬНО
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: #3b82f6;'>
        <div style='color: #666; font-size: 0.7rem; letter-spacing: 1px;'>PRICE</div>
        <div style='font-size: 1.8rem; font-weight: 600; color: #fff;'>${price:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: #38bdf8;'>
        <div style='color: #666; font-size: 0.7rem; letter-spacing: 1px;'>Z-SCORE</div>
        <div style='font-size: 1.8rem; font-weight: 600; color: #38bdf8;'>{z_score:+.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    prob_color = "#00ff44" if prob > 0.6 else "#eab308" if prob > 0.4 else "#ef4444"
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: {prob_color};'>
        <div style='color: #666; font-size: 0.7rem; letter-spacing: 1px;'>BOTTOM PROB.</div>
        <div style='font-size: 1.8rem; font-weight: 600; color: {prob_color};'>{prob*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: {signal_color};'>
        <div style='color: #666; font-size: 0.7rem; letter-spacing: 1px;'>SIGNAL</div>
        <div style='font-size: 1.8rem; font-weight: 600; color: {signal_color};'>{signal_text}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 6. СИГНАЛ КАРТА
# ============================================================

st.markdown(f"""
<div class='signal-card' style='border: 1px solid {signal_color}40;'>
    <div style='color: {signal_color}; font-size: 2.5rem; font-weight: 700; letter-spacing: 4px;'>{signal_text}</div>
    <div style='color: #666; font-size: 0.7rem; margin-top: 12px;'>ADAPTIVE Z-SCORE • {lower:.2f} / {upper:.2f} THRESHOLDS</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 7. ГРАФИК ЦЕНЫ — ЖИРНЫЕ ЦВЕТНЫЕ ПОЛОСЫ + ЧЁРНАЯ ЛИНИЯ
# ============================================================

st.markdown("<div class='section-title'>PRICE CHART</div>", unsafe_allow_html=True)

df_chart = df.tail(500).copy()

def get_color(z, lower, upper):
    if z <= lower: return "rgba(0, 255, 68, 0.4)"
    elif z <= lower * 0.7: return "rgba(68, 255, 68, 0.35)"
    elif z <= -0.5: return "rgba(136, 255, 136, 0.3)"
    elif z <= 0.5: return "rgba(136, 136, 136, 0.2)"
    elif z <= 1.2: return "rgba(255, 170, 102, 0.35)"
    elif z <= upper: return "rgba(255, 102, 68, 0.4)"
    else: return "rgba(255, 34, 0, 0.5)"

fig = go.Figure()

# Цветные полосы (area)
for i in range(len(df_chart) - 1):
    color = get_color(df_chart["z_score"].iloc[i], lower, upper)
    fig.add_trace(go.Scatter(
        x=[df_chart["date"].iloc[i], df_chart["date"].iloc[i+1]],
        y=[df_chart["close"].iloc[i], df_chart["close"].iloc[i+1]],
        mode='lines', line=dict(color=color, width=12),
        fill='tozeroy', fillcolor=color,
        showlegend=False, hoverinfo='skip'
    ))

# Чёрная тонкая линия (цена)
fig.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["close"],
    mode='lines', line=dict(color='#000000', width=1.5),
    name="PRICE", hovertemplate='%{y:,.2f}<extra></extra>'
))

fig.update_layout(
    height=380, template="plotly_dark", margin=dict(l=0, r=0, t=20, b=20),
    xaxis_title="", yaxis_title="", yaxis_type="log" if price > 100 else "linear",
    hovermode="x unified", showlegend=False,
    paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a"
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 8. Z-SCORE ГРАФИК — ЖИРНАЯ ЛИНИЯ
# ============================================================

st.markdown("<div class='section-title'>Z-SCORE & THRESHOLDS</div>", unsafe_allow_html=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["z_score"],
    mode='lines', name='Z-SCORE', line=dict(color='#00d4ff', width=3),
    fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'
))
fig2.add_hline(y=lower, line_dash="dash", line_color="#00ff44", line_width=2,
               annotation_text=f"BUY ({lower:.2f})", annotation_position="right")
fig2.add_hline(y=upper, line_dash="dash", line_color="#ff4422", line_width=2,
               annotation_text=f"SELL ({upper:.2f})", annotation_position="right")
fig2.add_hline(y=0, line_dash="dot", line_color="#444444")

fig2.update_layout(
    height=280, template="plotly_dark", margin=dict(l=0, r=0, t=20, b=20),
    xaxis_title="", yaxis_title="", paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a"
)
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 9. AI-АНАЛИЗ
# ============================================================

st.markdown("<div class='section-title'>AI ANALYSIS</div>", unsafe_allow_html=True)

if st.button("GENERATE AI ANALYSIS", type="primary"):
    with st.spinner("DEEPSEEK PROCESSING..."):
        # Опционально: здесь будет вызов AI
        st.info("AI analysis will appear here. Ensure DEEPSEEK_API_KEY is configured in Secrets.")
